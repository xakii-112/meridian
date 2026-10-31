// Multi-instrument matching engine: routes commands to books, runs the optional pre-trade risk
// gate, and folds every trade into a running 64-bit tape hash.
//
// The tape hash (FNV-style word hash over price, qty, maker cl_ord_id, taker cl_ord_id) is the
// engine's determinism fingerprint: the Python research simulator computes the same hash for the
// same order flow, and the replay tool checks the two agree.
#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

#include "meridian/event_buffer.hpp"
#include "meridian/order_book.hpp"
#include "meridian/risk.hpp"
#include "meridian/types.hpp"

namespace meridian {

inline constexpr std::uint64_t kTapeHashSeed = 14695981039346656037ull;
inline constexpr std::uint64_t kTapeHashPrime = 1099511628211ull;

constexpr std::uint64_t mix_trade(std::uint64_t h, Price price, Qty qty, ClOrdId maker, ClOrdId taker) noexcept {
  h = (h ^ static_cast<std::uint64_t>(price)) * kTapeHashPrime;
  h = (h ^ static_cast<std::uint64_t>(qty)) * kTapeHashPrime;
  h = (h ^ static_cast<std::uint64_t>(maker)) * kTapeHashPrime;
  h = (h ^ static_cast<std::uint64_t>(taker)) * kTapeHashPrime;
  return h;
}

class MatchingEngine {
 public:
  explicit MatchingEngine(RiskEngine* risk = nullptr, std::size_t event_capacity = std::size_t{1} << 12)
      : events_(event_capacity), risk_(risk) {}

  OrderBook& add_instrument(InstrumentSpec spec) {
    const SymbolId id = spec.id;
    if (id >= books_.size()) books_.resize(static_cast<std::size_t>(id) + 1);
    if (books_[id]) throw std::invalid_argument("MatchingEngine: duplicate symbol id");
    books_[id] = std::make_unique<OrderBook>(std::move(spec), events_);
    return *books_[id];
  }

  OrderBook* book(SymbolId id) noexcept { return id < books_.size() ? books_[id].get() : nullptr; }
  const OrderBook* book(SymbolId id) const noexcept { return id < books_.size() ? books_[id].get() : nullptr; }

  void submit(const NewOrder& c) {
    begin();
    OrderBook* b = book(c.symbol);
    if (b == nullptr) [[unlikely]] {
      reject_new(c, RejectReason::UnknownSymbol);
      return;
    }
    if (risk_ != nullptr) {
      NewOrder adjusted = c;
      if (const RejectReason r = risk_->check_new(adjusted, *b); r != RejectReason::None) {
        reject_new(c, r);
        return;
      }
      b->add(adjusted);
    } else {
      b->add(c);
    }
    finish();
  }

  void cancel(const CancelOrder& c) {
    begin();
    OrderBook* b = book(c.symbol);
    if (b == nullptr) [[unlikely]] {
      reject_op(EventType::CancelRejected, c.ts, c.symbol, c.order_id, c.owner, RejectReason::UnknownSymbol);
      return;
    }
    b->cancel(c);  // cancels always pass risk: they can only reduce exposure
    finish();
  }

  void replace(const ReplaceOrder& c) {
    begin();
    OrderBook* b = book(c.symbol);
    if (b == nullptr) [[unlikely]] {
      reject_op(EventType::ReplaceRejected, c.ts, c.symbol, c.order_id, c.owner, RejectReason::UnknownSymbol);
      return;
    }
    if (risk_ != nullptr) {
      OrderView v;
      if (b->find(c.order_id, v) && v.owner == c.owner) {
        if (const RejectReason r = risk_->check_replace(c, v, *b); r != RejectReason::None) {
          reject_op(EventType::ReplaceRejected, c.ts, c.symbol, c.order_id, c.owner, r);
          return;
        }
      }
    }
    b->replace(c);
    finish();
  }

  // Cancels all resting orders of `owner` (everyone if owner < 0) across all instruments.
  std::size_t mass_cancel(OwnerId owner, Timestamp ts, CancelReason reason = CancelReason::MassCancel) {
    begin();
    std::size_t n = 0;
    for (auto& b : books_)
      if (b) n += b->cancel_all(owner, ts, reason);
    finish();
    return n;
  }

  // Engaging the kill switch blocks new orders from `owner` and pulls all its resting orders.
  std::size_t set_kill_switch(OwnerId owner, bool engaged, Timestamp ts) {
    if (risk_ == nullptr) throw std::logic_error("MatchingEngine: kill switch requires a risk engine");
    risk_->set_kill(owner, engaged);
    return engaged ? mass_cancel(owner, ts, CancelReason::KillSwitch) : 0;
  }

  std::span<const Event> events() const noexcept { return events_.view(); }
  std::uint64_t tape_hash() const noexcept { return hash_; }
  std::int64_t trade_count() const noexcept { return trades_; }
  std::int64_t traded_volume() const noexcept { return volume_; }
  std::int64_t command_count() const noexcept { return commands_; }

 private:
  void begin() noexcept {
    events_.clear();
    ++commands_;
  }

  void finish() noexcept {
    for (const Event& e : events_.view()) {
      if (e.type == EventType::Trade) {
        hash_ = mix_trade(hash_, e.price, e.qty, e.cl_ord_id, e.contra_cl_ord_id);
        ++trades_;
        volume_ += e.qty;
      }
      if (risk_ != nullptr) risk_->on_event(e);
    }
  }

  void reject_new(const NewOrder& c, RejectReason r) {
    Event& e = events_.emit(EventType::Rejected, c.ts, c.symbol);
    e.cl_ord_id = c.cl_ord_id;
    e.owner = c.owner;
    e.side = c.side;
    e.price = c.price;
    e.qty = c.qty;
    e.reason = static_cast<std::uint8_t>(r);
  }

  void reject_op(EventType type, Timestamp ts, SymbolId symbol, OrderId id, OwnerId owner, RejectReason r) {
    Event& e = events_.emit(type, ts, symbol);
    e.order_id = id;
    e.owner = owner;
    e.reason = static_cast<std::uint8_t>(r);
  }

  EventBuffer events_;
  std::vector<std::unique_ptr<OrderBook>> books_;
  RiskEngine* risk_;
  std::uint64_t hash_ = kTapeHashSeed;
  std::int64_t trades_ = 0;
  std::int64_t volume_ = 0;
  std::int64_t commands_ = 0;
};

}  // namespace meridian
