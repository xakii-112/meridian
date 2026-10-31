// Single-instrument limit order book with price-time priority.
//
// Layout
//   * One price ladder (array indexed by tick offset) shared by both sides. A resting book is never
//     crossed, so every level holds either bids or asks, never both.
//   * Each level is an intrusive doubly-linked FIFO of order slots (O(1) append / unlink).
//   * Orders live in a preallocated arena; the free list is threaded through the same `next` field.
//     Order ids encode (generation << 32 | slot), so lookup is one array index plus a generation
//     check that rejects stale ids after slot reuse - no hash map on the hot path.
//   * A HierarchicalBitset marks non-empty levels; it finds the next best price in O(log64 N).
//
// Nothing here allocates after construction (the event buffer only grows on a pathological sweep).
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "meridian/bitset.hpp"
#include "meridian/event_buffer.hpp"
#include "meridian/types.hpp"

namespace meridian {

struct InstrumentSpec {
  SymbolId id = 0;
  std::string name;
  Price min_price = 1;
  Price max_price = Price{1} << 20;
  std::size_t order_capacity = std::size_t{1} << 20;
  Price reference_price = 0;  // used by risk collars until the first trade prints
};

struct DepthLevel {
  Price price = 0;
  Qty qty = 0;
  std::int32_t orders = 0;
};

struct OrderView {
  OrderId id = 0;
  ClOrdId cl_ord_id = 0;
  OwnerId owner = 0;
  Side side = Side::Buy;
  Price price = 0;
  Qty leaves = 0;
  std::uint8_t flags = 0;
  Stp stp = Stp::None;
};

class OrderBook {
 public:
  static constexpr std::int64_t kMaxLevels = std::int64_t{1} << 26;

  OrderBook(InstrumentSpec spec, EventBuffer& events)
      : spec_(std::move(spec)),
        events_(events),
        min_price_(spec_.min_price),
        levels_(checked_levels(spec_)),
        occupied_(levels_.size()),
        nodes_(spec_.order_capacity) {
    for (std::size_t i = 0; i < nodes_.size(); ++i)
      nodes_[i].next = i + 1 < nodes_.size() ? static_cast<std::int32_t>(i + 1) : -1;
    free_head_ = 0;
  }

  OrderBook(const OrderBook&) = delete;
  OrderBook& operator=(const OrderBook&) = delete;

  // ---- commands ---------------------------------------------------------------------------

  void add(const NewOrder& c) {
    ts_ = c.ts;
    std::int32_t limit = 0;
    if (const RejectReason r = validate(c, limit); r != RejectReason::None) {
      reject(c, r);
      return;
    }
    if ((c.flags & kPostOnly) && crosses(c.side, limit)) {
      reject(c, RejectReason::PostOnlyWouldCross);
      return;
    }
    const std::int32_t slot = allocate();
    if (slot < 0) [[unlikely]] {
      reject(c, RejectReason::OrderCapacityExceeded);
      return;
    }
    Node& n = nodes_[static_cast<std::size_t>(slot)];
    n.cl_ord_id = c.cl_ord_id;
    n.owner = c.owner;
    n.side = c.side;
    n.flags = c.flags;
    n.stp = c.stp;
    n.leaves = c.qty;
    n.level = limit;
    n.price = min_price_ + limit;
    const OrderId id = make_id(slot, n.generation);

    Event& a = events_.emit(EventType::Accepted, ts_, spec_.id);
    a.order_id = id;
    a.cl_ord_id = c.cl_ord_id;
    a.owner = c.owner;
    a.side = c.side;
    a.price = c.type == OrdType::Market ? 0 : n.price;
    a.qty = c.qty;
    a.leaves = c.qty;

    if (c.tif == Tif::Fok && !can_fill(n, limit)) {
      finish(slot, id, CancelReason::FillOrKill);
      return;
    }
    const bool killed = match(slot, id, limit);
    if (n.leaves == 0) {
      release(slot);
      return;
    }
    if (killed) {
      finish(slot, id, CancelReason::SelfTradePrevention);
      return;
    }
    if (c.tif != Tif::Gtc) {
      finish(slot, id, CancelReason::ImmediateOrCancel);
      return;
    }
    rest(slot);
  }

  void cancel(const CancelOrder& c) {
    ts_ = c.ts;
    std::int32_t slot = -1;
    if (!resolve(c.order_id, slot)) {
      op_reject(EventType::CancelRejected, c.order_id, c.owner, RejectReason::UnknownOrder);
      return;
    }
    const Node& n = nodes_[static_cast<std::size_t>(slot)];
    if (n.owner != c.owner) {
      op_reject(EventType::CancelRejected, c.order_id, c.owner, RejectReason::NotOwner);
      return;
    }
    const std::int32_t level = n.level;
    const Side side = n.side;
    cancel_resting(slot, CancelReason::Requested);
    emit_level(level, side);
  }

  void replace(const ReplaceOrder& c) {
    ts_ = c.ts;
    std::int32_t slot = -1;
    if (!resolve(c.order_id, slot)) {
      op_reject(EventType::ReplaceRejected, c.order_id, c.owner, RejectReason::UnknownOrder);
      return;
    }
    Node& n = nodes_[static_cast<std::size_t>(slot)];
    if (n.owner != c.owner) {
      op_reject(EventType::ReplaceRejected, c.order_id, c.owner, RejectReason::NotOwner);
      return;
    }
    if (c.new_qty <= 0) {
      op_reject(EventType::ReplaceRejected, c.order_id, c.owner, RejectReason::InvalidQuantity);
      return;
    }
    if (c.new_price < spec_.min_price || c.new_price > spec_.max_price) {
      op_reject(EventType::ReplaceRejected, c.order_id, c.owner, RejectReason::PriceOutOfRange);
      return;
    }
    const auto idx = static_cast<std::int32_t>(c.new_price - min_price_);
    const Qty prev_leaves = n.leaves;
    const std::int32_t old_idx = n.level;

    if (idx == old_idx && c.new_qty <= n.leaves) {  // size-down in place: keeps queue priority
      levels_[static_cast<std::size_t>(old_idx)].qty -= n.leaves - c.new_qty;
      n.leaves = c.new_qty;
      emit_replaced(slot, c.order_id, prev_leaves);
      emit_level(old_idx, n.side);
      return;
    }
    if ((n.flags & kPostOnly) && crosses(n.side, idx)) {
      op_reject(EventType::ReplaceRejected, c.order_id, c.owner, RejectReason::PostOnlyWouldCross);
      return;
    }
    unlink(slot);
    n.level = idx;
    n.price = c.new_price;
    n.leaves = c.new_qty;
    emit_replaced(slot, c.order_id, prev_leaves);
    emit_level(old_idx, n.side);
    const bool killed = match(slot, c.order_id, idx);
    if (n.leaves == 0) {
      release(slot);
      return;
    }
    if (killed) {
      finish(slot, c.order_id, CancelReason::SelfTradePrevention);
      return;
    }
    rest(slot);
  }

  // Cancels every resting order of `owner` (all owners if owner < 0). Rare path: linear in the
  // arena high-water mark.
  std::size_t cancel_all(OwnerId owner, Timestamp ts, CancelReason reason) {
    ts_ = ts;
    std::size_t cancelled = 0;
    for (std::int32_t s = 0; s < high_water_; ++s) {
      const Node& o = nodes_[static_cast<std::size_t>(s)];
      if (!o.active || (owner >= 0 && o.owner != owner)) continue;
      const std::int32_t level = o.level;
      const Side side = o.side;
      cancel_resting(s, reason);
      emit_level(level, side);
      ++cancelled;
    }
    return cancelled;
  }

  // ---- queries ----------------------------------------------------------------------------

  const InstrumentSpec& spec() const noexcept { return spec_; }
  Price min_price() const noexcept { return spec_.min_price; }
  Price max_price() const noexcept { return spec_.max_price; }
  bool has_bid() const noexcept { return best_bid_ >= 0; }
  bool has_ask() const noexcept { return best_ask_ >= 0; }
  Price best_bid() const noexcept { return min_price_ + best_bid_; }  // requires has_bid()
  Price best_ask() const noexcept { return min_price_ + best_ask_; }  // requires has_ask()
  Qty best_bid_qty() const noexcept { return has_bid() ? levels_[static_cast<std::size_t>(best_bid_)].qty : 0; }
  Qty best_ask_qty() const noexcept { return has_ask() ? levels_[static_cast<std::size_t>(best_ask_)].qty : 0; }
  std::size_t active_orders() const noexcept { return active_; }
  Price last_trade_price() const noexcept { return last_trade_; }
  Price reference_price() const noexcept { return last_trade_ != 0 ? last_trade_ : spec_.reference_price; }

  std::size_t depth(Side side, std::span<DepthLevel> out) const noexcept {
    std::size_t n = 0;
    if (side == Side::Buy) {
      for (std::int64_t i = best_bid_; i >= 0 && n < out.size(); i = occupied_.prev(i - 1)) out[n++] = level_view(i);
    } else {
      for (std::int64_t i = best_ask_; i >= 0 && n < out.size(); i = occupied_.next(i + 1)) out[n++] = level_view(i);
    }
    return n;
  }

  bool find(OrderId id, OrderView& out) const noexcept {
    std::int32_t slot = -1;
    if (!resolve(id, slot)) return false;
    const Node& n = nodes_[static_cast<std::size_t>(slot)];
    out = OrderView{id, n.cl_ord_id, n.owner, n.side, n.price, n.leaves, n.flags, n.stp};
    return true;
  }

  // Full structural audit (test/debug helper): returns an empty string when the book is sound.
  std::string check_invariants() const {
    std::size_t resting = 0;
    std::int64_t max_bid = -1;
    std::int64_t min_ask = -1;
    for (std::size_t i = 0; i < levels_.size(); ++i) {
      const Level& lv = levels_[i];
      Qty qty = 0;
      std::int32_t count = 0;
      std::int32_t prev = -1;
      int side = -1;
      for (std::int32_t s = lv.head; s != -1; s = nodes_[static_cast<std::size_t>(s)].next) {
        const Node& n = nodes_[static_cast<std::size_t>(s)];
        if (!n.active) return "inactive order linked at level " + std::to_string(i);
        if (n.prev != prev) return "broken prev link at level " + std::to_string(i);
        if (n.level != static_cast<std::int32_t>(i)) return "order level mismatch at " + std::to_string(i);
        if (n.leaves <= 0) return "non-positive leaves at level " + std::to_string(i);
        if (side == -1) side = static_cast<int>(n.side);
        if (side != static_cast<int>(n.side)) return "mixed sides at level " + std::to_string(i);
        qty += n.leaves;
        ++count;
        prev = s;
        if (count > static_cast<std::int32_t>(nodes_.size())) return "cycle at level " + std::to_string(i);
      }
      if (lv.tail != prev) return "tail mismatch at level " + std::to_string(i);
      if (qty != lv.qty) return "qty mismatch at level " + std::to_string(i);
      if (count != lv.count) return "count mismatch at level " + std::to_string(i);
      if (occupied_.test(i) != (count > 0)) return "bitset mismatch at level " + std::to_string(i);
      if (count > 0) {
        if (side == static_cast<int>(Side::Buy)) max_bid = static_cast<std::int64_t>(i);
        else if (min_ask < 0) min_ask = static_cast<std::int64_t>(i);
      }
      resting += static_cast<std::size_t>(count);
    }
    if (max_bid != best_bid_) return "best bid stale";
    if (min_ask != best_ask_) return "best ask stale";
    if (best_bid_ >= 0 && best_ask_ >= 0 && best_bid_ >= best_ask_) return "book crossed";
    if (resting != active_) return "active order count mismatch";
    return {};
  }

 private:
  struct Level {
    std::int32_t head = -1;
    std::int32_t tail = -1;
    Qty qty = 0;
    std::int32_t count = 0;
  };

  struct Node {  // 48 bytes
    ClOrdId cl_ord_id = 0;
    Price price = 0;
    Qty leaves = 0;
    OwnerId owner = 0;
    std::int32_t next = -1;
    std::int32_t prev = -1;
    std::int32_t level = 0;
    std::uint32_t generation = 0;
    Side side = Side::Buy;
    std::uint8_t flags = 0;
    Stp stp = Stp::None;
    bool active = false;
  };

  static std::size_t checked_levels(const InstrumentSpec& s) {
    if (s.max_price < s.min_price || s.max_price - s.min_price + 1 > kMaxLevels)
      throw std::invalid_argument("OrderBook: invalid price range");
    if (s.order_capacity == 0 || s.order_capacity > (std::size_t{1} << 30))
      throw std::invalid_argument("OrderBook: invalid order capacity");
    return static_cast<std::size_t>(s.max_price - s.min_price + 1);
  }

  static OrderId make_id(std::int32_t slot, std::uint32_t generation) noexcept {
    return (static_cast<OrderId>(generation) << 32) | static_cast<OrderId>(static_cast<std::uint32_t>(slot));
  }

  bool resolve(OrderId id, std::int32_t& slot) const noexcept {
    if (id <= 0) return false;
    const auto s = static_cast<std::uint64_t>(id) & 0xFFFF'FFFFu;
    const auto gen = static_cast<std::uint32_t>(static_cast<std::uint64_t>(id) >> 32);
    if (s >= nodes_.size()) return false;
    const Node& n = nodes_[s];
    if (!n.active || n.generation != gen) return false;
    slot = static_cast<std::int32_t>(s);
    return true;
  }

  std::int32_t allocate() noexcept {
    const std::int32_t slot = free_head_;
    if (slot < 0) return -1;
    Node& n = nodes_[static_cast<std::size_t>(slot)];
    free_head_ = n.next;
    n.generation = (n.generation + 1) & 0x7FFF'FFFFu;  // keeps ids positive
    if (n.generation == 0) n.generation = 1;
    n.active = true;
    n.next = n.prev = -1;
    ++active_;
    high_water_ = std::max(high_water_, slot + 1);
    return slot;
  }

  void release(std::int32_t slot) noexcept {
    Node& n = nodes_[static_cast<std::size_t>(slot)];
    n.active = false;
    n.next = free_head_;
    free_head_ = slot;
    --active_;
  }

  RejectReason validate(const NewOrder& c, std::int32_t& limit) const noexcept {
    if (c.qty <= 0) return RejectReason::InvalidQuantity;
    if (c.side != Side::Buy && c.side != Side::Sell) return RejectReason::InvalidSide;
    if ((c.flags & ~kPostOnly) != 0 || c.stp > Stp::CancelBoth) return RejectReason::InvalidFlags;
    if (c.tif > Tif::Fok) return RejectReason::InvalidTimeInForce;
    if (c.type == OrdType::Market) {
      if (c.tif == Tif::Gtc) return RejectReason::InvalidTimeInForce;
      if (c.flags & kPostOnly) return RejectReason::InvalidFlags;
      limit = c.side == Side::Buy ? static_cast<std::int32_t>(levels_.size() - 1) : 0;
      return RejectReason::None;
    }
    if (c.type != OrdType::Limit) return RejectReason::InvalidOrderType;
    if (c.price < spec_.min_price || c.price > spec_.max_price) return RejectReason::PriceOutOfRange;
    if ((c.flags & kPostOnly) && c.tif != Tif::Gtc) return RejectReason::InvalidTimeInForce;
    limit = static_cast<std::int32_t>(c.price - min_price_);
    return RejectReason::None;
  }

  bool crosses(Side side, std::int32_t idx) const noexcept {
    return side == Side::Buy ? (best_ask_ >= 0 && idx >= best_ask_) : (best_bid_ >= 0 && idx <= best_bid_);
  }

  // FOK pre-check: can the aggressor be filled completely without self-trade prevention killing it?
  bool can_fill(const Node& t, std::int32_t limit) const noexcept {
    Qty need = t.leaves;
    const bool stp = t.stp != Stp::None;
    if (t.side == Side::Buy) {
      for (std::int64_t i = best_ask_; i >= 0 && i <= limit; i = occupied_.next(i + 1)) {
        if (!drain_level(t, i, stp, need)) return false;
        if (need <= 0) return true;
      }
    } else {
      for (std::int64_t i = best_bid_; i >= 0 && i >= limit; i = occupied_.prev(i - 1)) {
        if (!drain_level(t, i, stp, need)) return false;
        if (need <= 0) return true;
      }
    }
    return false;
  }

  bool drain_level(const Node& t, std::int64_t i, bool stp, Qty& need) const noexcept {
    const Level& lv = levels_[static_cast<std::size_t>(i)];
    if (!stp) {
      need -= lv.qty;
      return true;
    }
    for (std::int32_t s = lv.head; s != -1; s = nodes_[static_cast<std::size_t>(s)].next) {
      const Node& m = nodes_[static_cast<std::size_t>(s)];
      if (m.owner == t.owner) {
        if (t.stp != Stp::CancelResting) return false;
        continue;  // would be cancelled, contributes nothing
      }
      need -= m.leaves;
      if (need <= 0) return true;
    }
    return true;
  }

  // Returns true if self-trade prevention killed the aggressor.
  bool match(std::int32_t slot, OrderId taker_id, std::int32_t limit) {
    Node& t = nodes_[static_cast<std::size_t>(slot)];
    if (t.side == Side::Buy) {
      while (t.leaves > 0 && best_ask_ >= 0 && best_ask_ <= limit)
        if (match_level(t, taker_id, best_ask_)) return true;
    } else {
      while (t.leaves > 0 && best_bid_ >= 0 && best_bid_ >= limit)
        if (match_level(t, taker_id, best_bid_)) return true;
    }
    return false;
  }

  bool match_level(Node& t, OrderId taker_id, std::int32_t i) {
    Level& lv = levels_[static_cast<std::size_t>(i)];
    const Side maker_side = opposite(t.side);
    bool killed = false;
    bool touched = false;
    std::int32_t s = lv.head;
    while (s != -1 && t.leaves > 0) {
      Node& m = nodes_[static_cast<std::size_t>(s)];
      const std::int32_t next = m.next;
      if (t.stp != Stp::None && m.owner == t.owner) [[unlikely]] {
        if (t.stp == Stp::CancelAggressor) {
          killed = true;
          break;
        }
        cancel_resting(s, CancelReason::SelfTradePrevention);
        touched = true;
        if (t.stp == Stp::CancelBoth) {
          killed = true;
          break;
        }
        s = next;
        continue;
      }
      const Qty q = std::min(t.leaves, m.leaves);
      m.leaves -= q;
      t.leaves -= q;
      lv.qty -= q;
      touched = true;
      emit_trade(s, m, t, taker_id, q);
      if (m.leaves == 0) {
        unlink(s);
        release(s);
      }
      s = next;
    }
    if (touched) emit_level(i, maker_side);
    return killed;
  }

  void rest(std::int32_t slot) {
    Node& n = nodes_[static_cast<std::size_t>(slot)];
    const std::int32_t i = n.level;
    Level& lv = levels_[static_cast<std::size_t>(i)];
    n.prev = lv.tail;
    n.next = -1;
    if (lv.tail != -1) nodes_[static_cast<std::size_t>(lv.tail)].next = slot;
    else lv.head = slot;
    lv.tail = slot;
    lv.qty += n.leaves;
    if (lv.count++ == 0) occupied_.set(static_cast<std::size_t>(i));
    if (n.side == Side::Buy) {
      if (i > best_bid_) best_bid_ = i;
    } else if (best_ask_ < 0 || i < best_ask_) {
      best_ask_ = i;
    }
    emit_level(i, n.side);
  }

  void unlink(std::int32_t slot) noexcept {
    Node& n = nodes_[static_cast<std::size_t>(slot)];
    const std::int32_t i = n.level;
    Level& lv = levels_[static_cast<std::size_t>(i)];
    if (n.prev != -1) nodes_[static_cast<std::size_t>(n.prev)].next = n.next;
    else lv.head = n.next;
    if (n.next != -1) nodes_[static_cast<std::size_t>(n.next)].prev = n.prev;
    else lv.tail = n.prev;
    n.next = n.prev = -1;
    lv.qty -= n.leaves;
    if (--lv.count == 0) {
      occupied_.reset(static_cast<std::size_t>(i));
      // All set bits below the best bid are bids and all set bits above the best ask are asks.
      if (n.side == Side::Buy) {
        if (i == best_bid_) best_bid_ = static_cast<std::int32_t>(occupied_.prev(i - 1));
      } else if (i == best_ask_) {
        best_ask_ = static_cast<std::int32_t>(occupied_.next(i + 1));
      }
    }
  }

  void cancel_resting(std::int32_t slot, CancelReason reason) {
    const Node& n = nodes_[static_cast<std::size_t>(slot)];
    emit_cancel(slot, make_id(slot, n.generation), reason);
    unlink(slot);
    release(slot);
  }

  void finish(std::int32_t slot, OrderId id, CancelReason reason) {
    emit_cancel(slot, id, reason);
    release(slot);
  }

  DepthLevel level_view(std::int64_t i) const noexcept {
    const Level& lv = levels_[static_cast<std::size_t>(i)];
    return DepthLevel{min_price_ + i, lv.qty, lv.count};
  }

  // ---- event emission ----------------------------------------------------------------------

  void reject(const NewOrder& c, RejectReason r) {
    Event& e = events_.emit(EventType::Rejected, ts_, spec_.id);
    e.cl_ord_id = c.cl_ord_id;
    e.owner = c.owner;
    e.side = c.side;
    e.price = c.price;
    e.qty = c.qty;
    e.reason = static_cast<std::uint8_t>(r);
  }

  void op_reject(EventType type, OrderId id, OwnerId owner, RejectReason r) {
    Event& e = events_.emit(type, ts_, spec_.id);
    e.order_id = id;
    e.owner = owner;
    e.reason = static_cast<std::uint8_t>(r);
  }

  void emit_trade(std::int32_t maker_slot, const Node& m, const Node& t, OrderId taker_id, Qty q) {
    Event& e = events_.emit(EventType::Trade, ts_, spec_.id);
    e.order_id = make_id(maker_slot, m.generation);
    e.cl_ord_id = m.cl_ord_id;
    e.owner = m.owner;
    e.side = m.side;
    e.price = m.price;
    e.qty = q;
    e.leaves = m.leaves;
    e.contra_order_id = taker_id;
    e.contra_cl_ord_id = t.cl_ord_id;
    e.contra_owner = t.owner;
    e.contra_leaves = t.leaves;
    last_trade_ = m.price;
  }

  void emit_cancel(std::int32_t slot, OrderId id, CancelReason reason) {
    const Node& n = nodes_[static_cast<std::size_t>(slot)];
    Event& e = events_.emit(EventType::Cancelled, ts_, spec_.id);
    e.order_id = id;
    e.cl_ord_id = n.cl_ord_id;
    e.owner = n.owner;
    e.side = n.side;
    e.price = n.price;
    e.qty = n.leaves;
    e.leaves = 0;
    e.reason = static_cast<std::uint8_t>(reason);
  }

  void emit_replaced(std::int32_t slot, OrderId id, Qty prev_leaves) {
    const Node& n = nodes_[static_cast<std::size_t>(slot)];
    Event& e = events_.emit(EventType::Replaced, ts_, spec_.id);
    e.order_id = id;
    e.cl_ord_id = n.cl_ord_id;
    e.owner = n.owner;
    e.side = n.side;
    e.price = n.price;
    e.qty = n.leaves;
    e.leaves = n.leaves;
    e.contra_leaves = prev_leaves;
  }

  void emit_level(std::int32_t i, Side side) {
    const Level& lv = levels_[static_cast<std::size_t>(i)];
    Event& e = events_.emit(EventType::BookLevel, ts_, spec_.id);
    e.side = side;
    e.price = min_price_ + i;
    e.qty = lv.qty;
    e.leaves = lv.count;
  }

  InstrumentSpec spec_;
  EventBuffer& events_;
  Price min_price_;
  std::vector<Level> levels_;
  HierarchicalBitset occupied_;
  std::vector<Node> nodes_;
  std::int32_t free_head_ = -1;
  std::int32_t high_water_ = 0;
  std::int32_t best_bid_ = -1;
  std::int32_t best_ask_ = -1;
  std::size_t active_ = 0;
  Price last_trade_ = 0;
  Timestamp ts_ = 0;
};

}  // namespace meridian
