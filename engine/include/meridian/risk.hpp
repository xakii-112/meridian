// Pre-trade risk gate (in the spirit of SEC 15c3-5 market-access controls).
//
// Checks run before an order reaches the book, in a fixed order: participant known, kill switch,
// message throttle (token bucket on sequencer time), max order size, price collar (market orders
// are converted to collared limits - "market protection"), max notional, max open orders, and
// worst-case position (position + all resting same-side quantity + this order).
//
// Exposure is maintained from the engine's own output events, so it can never drift from the book.
#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "meridian/order_book.hpp"
#include "meridian/types.hpp"

namespace meridian {

inline constexpr std::int64_t kUnlimited = std::numeric_limits<std::int64_t>::max();

struct RiskLimits {
  Qty max_order_qty = kUnlimited;
  std::int64_t max_order_notional = kUnlimited;  // price ticks * lots
  Qty max_long = kUnlimited;
  Qty max_short = kUnlimited;
  std::int64_t max_open_orders = kUnlimited;
  Price price_collar = kUnlimited;  // max |price - reference| in ticks
  std::int64_t msgs_per_sec = 0;    // 0 disables throttling
  std::int64_t burst = 0;
};

class RiskEngine {
 public:
  explicit RiskEngine(std::size_t max_participants = 1024, std::size_t max_symbols = 16)
      : participants_(max_participants), exposure_(max_participants * max_symbols), max_symbols_(max_symbols) {}

  void register_participant(OwnerId id, const RiskLimits& limits) {
    if (id < 0 || static_cast<std::size_t>(id) >= participants_.size())
      throw std::out_of_range("RiskEngine: participant id out of range");
    if (limits.msgs_per_sec < 0 || limits.burst < 0 || limits.burst > 1'000'000 ||
        (limits.msgs_per_sec > 0 && limits.burst < 1))
      throw std::invalid_argument("RiskEngine: bad throttle");
    Participant& p = participants_[static_cast<std::size_t>(id)];
    p.limits = limits;
    p.registered = true;
    p.tokens = limits.burst * kNanosPerSec;
  }

  void set_kill(OwnerId id, bool on) noexcept {
    if (Participant* p = participant(id)) p->killed = on;
  }
  void set_global_halt(bool on) noexcept { halted_ = on; }

  // May rewrite a market order into a collared limit order.
  RejectReason check_new(NewOrder& c, const OrderBook& book) noexcept {
    Participant* p = participant(c.owner);
    if (p == nullptr) return RejectReason::RiskUnknownParticipant;
    if (halted_ || p->killed) return RejectReason::RiskKillSwitch;
    if (!take_token(*p, c.ts)) return RejectReason::RiskThrottled;
    const RiskLimits& lim = p->limits;
    if (c.qty > lim.max_order_qty) return RejectReason::RiskMaxOrderQuantity;
    if (c.symbol >= max_symbols_) return RejectReason::UnknownSymbol;

    const Price ref = book.reference_price();
    if (c.type == OrdType::Market) {
      if (lim.price_collar != kUnlimited && ref > 0) {
        c.type = OrdType::Limit;
        c.price = std::clamp(c.side == Side::Buy ? ref + lim.price_collar : ref - lim.price_collar,
                             book.min_price(), book.max_price());
      }
    } else if (lim.price_collar != kUnlimited && ref > 0) {
      const Price dist = c.price > ref ? c.price - ref : ref - c.price;
      if (dist > lim.price_collar) return RejectReason::RiskPriceCollar;
    }
    if (c.type == OrdType::Limit && lim.max_order_notional != kUnlimited) {
      const __int128 notional = static_cast<__int128>(c.price) * c.qty;
      if (notional > lim.max_order_notional) return RejectReason::RiskMaxNotional;
    }
    if (p->open_orders >= lim.max_open_orders) return RejectReason::RiskMaxOpenOrders;
    if (!position_ok(exposure(c.owner, c.symbol), c.side, c.qty, lim)) return RejectReason::RiskPositionLimit;
    return RejectReason::None;
  }

  RejectReason check_replace(const ReplaceOrder& c, const OrderView& v, const OrderBook& book) noexcept {
    Participant* p = participant(c.owner);
    if (p == nullptr) return RejectReason::RiskUnknownParticipant;
    if (halted_ || p->killed) return RejectReason::RiskKillSwitch;
    if (!take_token(*p, c.ts)) return RejectReason::RiskThrottled;
    const RiskLimits& lim = p->limits;
    if (c.new_qty > lim.max_order_qty) return RejectReason::RiskMaxOrderQuantity;
    const Price ref = book.reference_price();
    if (lim.price_collar != kUnlimited && ref > 0) {
      const Price dist = c.new_price > ref ? c.new_price - ref : ref - c.new_price;
      if (dist > lim.price_collar) return RejectReason::RiskPriceCollar;
    }
    if (lim.max_order_notional != kUnlimited &&
        static_cast<__int128>(c.new_price) * c.new_qty > lim.max_order_notional)
      return RejectReason::RiskMaxNotional;
    if (c.new_qty > v.leaves && c.symbol < max_symbols_ &&
        !position_ok(exposure(c.owner, c.symbol), v.side, c.new_qty - v.leaves, lim))
      return RejectReason::RiskPositionLimit;
    return RejectReason::None;
  }

  void on_event(const Event& e) noexcept {
    switch (e.type) {
      case EventType::Accepted:
        if (Exposure* x = exposure_for(e.owner, e.symbol)) {
          ++participants_[static_cast<std::size_t>(e.owner)].open_orders;
          open(*x, e.side) += e.qty;
        }
        break;
      case EventType::Trade:
        apply_fill(e.owner, e.symbol, e.side, e.qty, e.leaves == 0);
        apply_fill(e.contra_owner, e.symbol, opposite(e.side), e.qty, e.contra_leaves == 0);
        break;
      case EventType::Cancelled:
        if (Exposure* x = exposure_for(e.owner, e.symbol)) {
          --participants_[static_cast<std::size_t>(e.owner)].open_orders;
          open(*x, e.side) -= e.qty;
        }
        break;
      case EventType::Replaced:
        if (Exposure* x = exposure_for(e.owner, e.symbol)) open(*x, e.side) += e.qty - e.contra_leaves;
        break;
      default:
        break;
    }
  }

  Qty position(OwnerId id, SymbolId s) const noexcept { return exposure_at(id, s).position; }
  Qty open_qty(OwnerId id, SymbolId s, Side side) const noexcept {
    const Exposure& x = exposure_at(id, s);
    return side == Side::Buy ? x.open_buy : x.open_sell;
  }
  std::int64_t open_orders(OwnerId id) const noexcept {
    return valid(id) ? participants_[static_cast<std::size_t>(id)].open_orders : 0;
  }

 private:
  static constexpr std::int64_t kNanosPerSec = 1'000'000'000;

  struct Participant {
    RiskLimits limits;
    bool registered = false;
    bool killed = false;
    std::int64_t open_orders = 0;
    std::int64_t tokens = 0;  // nano-tokens: one message costs kNanosPerSec
    Timestamp last_refill = 0;
  };
  struct Exposure {
    Qty position = 0;
    Qty open_buy = 0;
    Qty open_sell = 0;
  };

  bool valid(OwnerId id) const noexcept { return id >= 0 && static_cast<std::size_t>(id) < participants_.size(); }

  Participant* participant(OwnerId id) noexcept {
    if (!valid(id)) return nullptr;
    Participant& p = participants_[static_cast<std::size_t>(id)];
    return p.registered ? &p : nullptr;
  }

  Exposure& exposure(OwnerId id, SymbolId s) noexcept {
    return exposure_[static_cast<std::size_t>(id) * max_symbols_ + s];
  }
  const Exposure& exposure_at(OwnerId id, SymbolId s) const noexcept {
    static const Exposure kEmpty{};
    if (!valid(id) || s >= max_symbols_) return kEmpty;
    return exposure_[static_cast<std::size_t>(id) * max_symbols_ + s];
  }
  Exposure* exposure_for(OwnerId id, SymbolId s) noexcept {
    if (participant(id) == nullptr || s >= max_symbols_) return nullptr;
    return &exposure(id, s);
  }

  static Qty& open(Exposure& x, Side side) noexcept { return side == Side::Buy ? x.open_buy : x.open_sell; }

  static bool position_ok(const Exposure& x, Side side, Qty qty, const RiskLimits& lim) noexcept {
    if (side == Side::Buy) {
      return lim.max_long == kUnlimited ||
             static_cast<__int128>(x.position) + x.open_buy + qty <= lim.max_long;
    }
    return lim.max_short == kUnlimited ||
           -(static_cast<__int128>(x.position) - x.open_sell - qty) <= lim.max_short;
  }

  void apply_fill(OwnerId id, SymbolId s, Side side, Qty qty, bool done) noexcept {
    Exposure* x = exposure_for(id, s);
    if (x == nullptr) return;
    x->position += side == Side::Buy ? qty : -qty;
    open(*x, side) -= qty;
    if (done) --participants_[static_cast<std::size_t>(id)].open_orders;
  }

  // Token bucket in integer nano-tokens, refilled from sequencer timestamps (deterministic).
  static bool take_token(Participant& p, Timestamp now) noexcept {
    const RiskLimits& lim = p.limits;
    if (lim.msgs_per_sec <= 0) return true;
    const std::int64_t cap = lim.burst * kNanosPerSec;
    const std::int64_t elapsed = std::max<std::int64_t>(0, now - p.last_refill);
    p.last_refill = std::max(p.last_refill, now);
    if (elapsed >= cap / lim.msgs_per_sec) p.tokens = cap;
    else p.tokens = std::min(cap, p.tokens + elapsed * lim.msgs_per_sec);
    if (p.tokens < kNanosPerSec) return false;
    p.tokens -= kNanosPerSec;
    return true;
  }

  std::vector<Participant> participants_;
  std::vector<Exposure> exposure_;
  std::size_t max_symbols_;
  bool halted_ = false;
};

}  // namespace meridian
