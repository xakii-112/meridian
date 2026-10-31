// Naive reference model of the matching rules: a flat vector scanned linearly on every step.
// It is deliberately written in the most obvious way possible and used as a differential-testing
// oracle for the optimized OrderBook (same inputs must yield the same normalized event stream).
#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

#include "meridian/order_book.hpp"
#include "meridian/types.hpp"

namespace meridian::testing {

// Engine events reduced to the fields both implementations can agree on (engine order ids are
// slot-based, so orders are identified by cl_ord_id instead).
struct NormEvent {
  EventType type{};
  ClOrdId cl = 0;
  ClOrdId contra_cl = 0;
  Price price = 0;
  Qty qty = 0;
  std::uint8_t reason = 0;

  friend bool operator==(const NormEvent&, const NormEvent&) = default;
};

inline std::vector<NormEvent> normalize(std::span<const Event> events) {
  std::vector<NormEvent> out;
  for (const Event& e : events) {
    switch (e.type) {
      case EventType::BookLevel:
        break;
      case EventType::Trade:
        out.push_back({e.type, e.cl_ord_id, e.contra_cl_ord_id, e.price, e.qty, 0});
        break;
      case EventType::Accepted:
        out.push_back({e.type, e.cl_ord_id, 0, 0, e.qty, 0});
        break;
      case EventType::Rejected:
        out.push_back({e.type, e.cl_ord_id, 0, 0, 0, e.reason});
        break;
      case EventType::Cancelled:
        out.push_back({e.type, e.cl_ord_id, 0, 0, e.qty, e.reason});
        break;
      case EventType::Replaced:
        out.push_back({e.type, e.cl_ord_id, 0, e.price, e.qty, 0});
        break;
      case EventType::CancelRejected:
      case EventType::ReplaceRejected:
        out.push_back({e.type, 0, 0, 0, 0, e.reason});
        break;
    }
  }
  return out;
}

class ReferenceBook {
 public:
  struct Order {
    ClOrdId cl;
    OwnerId owner;
    Side side;
    Price price;
    Qty leaves;
    std::uint8_t flags;
    Stp stp;
    std::int64_t prio;
  };

  ReferenceBook(Price min_price, Price max_price) : min_(min_price), max_(max_price) {}

  std::vector<NormEvent> add(const NewOrder& c) {
    out_.clear();
    if (const RejectReason r = validate(c); r != RejectReason::None) {
      out_.push_back({EventType::Rejected, c.cl_ord_id, 0, 0, 0, static_cast<std::uint8_t>(r)});
      return out_;
    }
    Order o{c.cl_ord_id, c.owner, c.side, c.type == OrdType::Market ? (c.side == Side::Buy ? max_ : min_) : c.price,
            c.qty, c.flags, c.stp, 0};
    if ((c.flags & kPostOnly) && crosses(o.side, o.price)) {
      out_.push_back({EventType::Rejected, c.cl_ord_id, 0, 0, 0, static_cast<std::uint8_t>(RejectReason::PostOnlyWouldCross)});
      return out_;
    }
    out_.push_back({EventType::Accepted, c.cl_ord_id, 0, 0, c.qty, 0});
    if (c.tif == Tif::Fok && !can_fill(o)) {
      cancelled(o.cl, o.leaves, CancelReason::FillOrKill);
      return out_;
    }
    const bool killed = match(o);
    if (o.leaves == 0) return out_;
    if (killed) cancelled(o.cl, o.leaves, CancelReason::SelfTradePrevention);
    else if (c.tif != Tif::Gtc) cancelled(o.cl, o.leaves, CancelReason::ImmediateOrCancel);
    else rest(o);
    return out_;
  }

  std::vector<NormEvent> cancel(ClOrdId cl, OwnerId owner) {
    out_.clear();
    auto it = find(cl);
    if (it == book_.end()) return op_reject(EventType::CancelRejected, RejectReason::UnknownOrder);
    if (it->owner != owner) return op_reject(EventType::CancelRejected, RejectReason::NotOwner);
    cancelled(it->cl, it->leaves, CancelReason::Requested);
    book_.erase(it);
    return out_;
  }

  std::vector<NormEvent> replace(ClOrdId cl, OwnerId owner, Price price, Qty qty) {
    out_.clear();
    auto it = find(cl);
    if (it == book_.end()) return op_reject(EventType::ReplaceRejected, RejectReason::UnknownOrder);
    if (it->owner != owner) return op_reject(EventType::ReplaceRejected, RejectReason::NotOwner);
    if (qty <= 0) return op_reject(EventType::ReplaceRejected, RejectReason::InvalidQuantity);
    if (price < min_ || price > max_) return op_reject(EventType::ReplaceRejected, RejectReason::PriceOutOfRange);
    if (price == it->price && qty <= it->leaves) {
      it->leaves = qty;
      out_.push_back({EventType::Replaced, cl, 0, price, qty, 0});
      return out_;
    }
    if ((it->flags & kPostOnly) && crosses(it->side, price))
      return op_reject(EventType::ReplaceRejected, RejectReason::PostOnlyWouldCross);
    Order o = *it;
    book_.erase(it);
    o.price = price;
    o.leaves = qty;
    out_.push_back({EventType::Replaced, cl, 0, price, qty, 0});
    const bool killed = match(o);
    if (o.leaves == 0) return out_;
    if (killed) cancelled(o.cl, o.leaves, CancelReason::SelfTradePrevention);
    else rest(o);
    return out_;
  }

  std::vector<DepthLevel> depth(Side side) const {
    std::vector<DepthLevel> levels;
    std::vector<Order> sorted;
    for (const Order& o : book_)
      if (o.side == side) sorted.push_back(o);
    std::sort(sorted.begin(), sorted.end(), [&](const Order& a, const Order& b) {
      return side == Side::Buy ? a.price > b.price : a.price < b.price;
    });
    for (const Order& o : sorted) {
      if (levels.empty() || levels.back().price != o.price) levels.push_back({o.price, 0, 0});
      levels.back().qty += o.leaves;
      ++levels.back().orders;
    }
    return levels;
  }

  std::size_t size() const { return book_.size(); }

 private:
  RejectReason validate(const NewOrder& c) const {
    if (c.qty <= 0) return RejectReason::InvalidQuantity;
    if ((c.flags & ~kPostOnly) != 0) return RejectReason::InvalidFlags;
    if (c.type == OrdType::Market) {
      if (c.tif == Tif::Gtc) return RejectReason::InvalidTimeInForce;
      if (c.flags & kPostOnly) return RejectReason::InvalidFlags;
      return RejectReason::None;
    }
    if (c.price < min_ || c.price > max_) return RejectReason::PriceOutOfRange;
    if ((c.flags & kPostOnly) && c.tif != Tif::Gtc) return RejectReason::InvalidTimeInForce;
    return RejectReason::None;
  }

  std::vector<Order>::iterator find(ClOrdId cl) {
    return std::find_if(book_.begin(), book_.end(), [&](const Order& o) { return o.cl == cl; });
  }

  // Best resting order the aggressor could trade with (price priority, then time priority).
  std::optional<std::size_t> best_contra(const Order& t) const {
    std::optional<std::size_t> best;
    for (std::size_t i = 0; i < book_.size(); ++i) {
      const Order& m = book_[i];
      if (m.side == t.side) continue;
      const bool acceptable = t.side == Side::Buy ? m.price <= t.price : m.price >= t.price;
      if (!acceptable) continue;
      if (!best) {
        best = i;
        continue;
      }
      const Order& b = book_[*best];
      const bool better = t.side == Side::Buy ? std::tie(m.price, m.prio) < std::tie(b.price, b.prio)
                                              : (m.price > b.price || (m.price == b.price && m.prio < b.prio));
      if (better) best = i;
    }
    return best;
  }

  std::vector<Order> contra_in_priority(const Order& t) const {
    std::vector<Order> v;
    for (const Order& m : book_) {
      if (m.side == t.side) continue;
      if (t.side == Side::Buy ? m.price <= t.price : m.price >= t.price) v.push_back(m);
    }
    std::sort(v.begin(), v.end(), [&](const Order& a, const Order& b) {
      if (a.price != b.price) return t.side == Side::Buy ? a.price < b.price : a.price > b.price;
      return a.prio < b.prio;
    });
    return v;
  }

  bool can_fill(const Order& t) const {
    Qty need = t.leaves;
    for (const Order& m : contra_in_priority(t)) {
      if (t.stp != Stp::None && m.owner == t.owner) {
        if (t.stp != Stp::CancelResting) return false;
        continue;
      }
      need -= m.leaves;
      if (need <= 0) return true;
    }
    return false;
  }

  bool match(Order& t) {
    while (t.leaves > 0) {
      const auto bi = best_contra(t);
      if (!bi) return false;
      Order& m = book_[*bi];
      if (t.stp != Stp::None && m.owner == t.owner) {
        if (t.stp == Stp::CancelAggressor) return true;
        cancelled(m.cl, m.leaves, CancelReason::SelfTradePrevention);
        book_.erase(book_.begin() + static_cast<std::ptrdiff_t>(*bi));
        if (t.stp == Stp::CancelBoth) return true;
        continue;
      }
      const Qty q = std::min(t.leaves, m.leaves);
      m.leaves -= q;
      t.leaves -= q;
      out_.push_back({EventType::Trade, m.cl, t.cl, m.price, q, 0});
      if (m.leaves == 0) book_.erase(book_.begin() + static_cast<std::ptrdiff_t>(*bi));
    }
    return false;
  }

  bool crosses(Side side, Price price) const {
    for (const Order& m : book_) {
      if (m.side == side) continue;
      if (side == Side::Buy ? m.price <= price : m.price >= price) return true;
    }
    return false;
  }

  void rest(Order o) {
    o.prio = next_prio_++;
    book_.push_back(o);
  }

  void cancelled(ClOrdId cl, Qty qty, CancelReason r) {
    out_.push_back({EventType::Cancelled, cl, 0, 0, qty, static_cast<std::uint8_t>(r)});
  }

  std::vector<NormEvent> op_reject(EventType type, RejectReason r) {
    out_.push_back({type, 0, 0, 0, 0, static_cast<std::uint8_t>(r)});
    return out_;
  }

  Price min_;
  Price max_;
  std::vector<Order> book_;
  std::vector<NormEvent> out_;
  std::int64_t next_prio_ = 0;
};

}  // namespace meridian::testing
