// Order-entry gateway: decodes MWP frames, maps participant-scoped client order ids to engine
// order ids, drives the engine, and (optionally) encodes the outbound execution-report stream.
#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "meridian/matching_engine.hpp"
#include "meridian/types.hpp"
#include "meridian/wire.hpp"

namespace meridian {

// Open-addressing (linear probing, backward-shift deletion) map (owner, cl_ord_id) -> order id.
// Flat storage: no per-entry allocation, no tombstones.
class OrderIdMap {
 public:
  explicit OrderIdMap(std::size_t expected = 1 << 16) { rebuild(std::bit_ceil(std::max<std::size_t>(16, expected * 2))); }

  bool insert(OwnerId owner, ClOrdId cl, OrderId value) {
    if ((size_ + 1) * 2 > slots_.size()) rebuild(slots_.size() * 2);
    for (std::size_t i = hash(owner, cl) & mask_;; i = (i + 1) & mask_) {
      Slot& s = slots_[i];
      if (!s.used) {
        s = Slot{cl, value, owner, true};
        ++size_;
        return true;
      }
      if (s.cl == cl && s.owner == owner) return false;
    }
  }

  const OrderId* find(OwnerId owner, ClOrdId cl) const noexcept {
    for (std::size_t i = hash(owner, cl) & mask_;; i = (i + 1) & mask_) {
      const Slot& s = slots_[i];
      if (!s.used) return nullptr;
      if (s.cl == cl && s.owner == owner) return &s.value;
    }
  }

  bool erase(OwnerId owner, ClOrdId cl) noexcept {
    std::size_t i = hash(owner, cl) & mask_;
    for (;; i = (i + 1) & mask_) {
      if (!slots_[i].used) return false;
      if (slots_[i].cl == cl && slots_[i].owner == owner) break;
    }
    // Backward-shift: pull later entries of the probe run into the hole if their home slot allows.
    for (std::size_t j = i;;) {
      j = (j + 1) & mask_;
      if (!slots_[j].used) break;
      const std::size_t k = hash(slots_[j].owner, slots_[j].cl) & mask_;
      const bool stays = i <= j ? (i < k && k <= j) : (i < k || k <= j);
      if (stays) continue;
      slots_[i] = slots_[j];
      i = j;
    }
    slots_[i].used = false;
    --size_;
    return true;
  }

  std::size_t size() const noexcept { return size_; }

 private:
  struct Slot {
    ClOrdId cl = 0;
    OrderId value = 0;
    OwnerId owner = 0;
    bool used = false;
  };

  static std::size_t hash(OwnerId owner, ClOrdId cl) noexcept {
    std::uint64_t x = static_cast<std::uint64_t>(cl) ^ (static_cast<std::uint64_t>(static_cast<std::uint32_t>(owner)) << 40);
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdull;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ull;
    x ^= x >> 33;
    return static_cast<std::size_t>(x);
  }

  void rebuild(std::size_t capacity) {
    std::vector<Slot> old = std::move(slots_);
    slots_.assign(capacity, Slot{});
    mask_ = capacity - 1;
    size_ = 0;
    for (const Slot& s : old)
      if (s.used) insert(s.owner, s.cl, s.value);
  }

  std::vector<Slot> slots_;
  std::size_t mask_ = 0;
  std::size_t size_ = 0;
};

class Gateway {
 public:
  struct Stats {
    std::int64_t frames = 0;
    std::int64_t malformed = 0;
    std::int64_t unknown_refs = 0;
    std::int64_t duplicate_ids = 0;
  };

  explicit Gateway(MatchingEngine& engine, std::size_t expected_live = 1 << 16) : engine_(engine), live_(expected_live) {}

  // When set, every non-BookLevel engine event is encoded as a 64-byte outbound frame.
  void set_outbound(std::vector<std::byte>* sink) noexcept { outbound_ = sink; }

  // Returns false if the frame was malformed, duplicated a live id, or referenced an unknown order.
  bool on_frame(std::span<const std::byte> frame) {
    ++stats_.frames;
    wire::Inbound m;
    if (!wire::decode(frame, m)) {
      ++stats_.malformed;
      return false;
    }
    switch (m.type) {
      case wire::MsgType::EnterOrder: {
        if (live_.find(m.owner, m.cl_ord_id) != nullptr) {
          ++stats_.duplicate_ids;
          return false;
        }
        NewOrder o;
        o.ts = m.ts;
        o.cl_ord_id = m.cl_ord_id;
        o.price = m.price;
        o.qty = m.qty;
        o.owner = m.owner;
        o.symbol = m.symbol;
        o.side = m.side;
        o.type = m.ord_type;
        o.tif = m.tif;
        o.flags = m.flags;
        o.stp = m.stp;
        engine_.submit(o);
        break;
      }
      case wire::MsgType::Cancel: {
        const OrderId* id = live_.find(m.owner, m.cl_ord_id);
        if (id == nullptr) {
          ++stats_.unknown_refs;
          return false;
        }
        engine_.cancel(CancelOrder{m.ts, *id, m.owner, m.symbol});
        break;
      }
      case wire::MsgType::Replace: {
        const OrderId* id = live_.find(m.owner, m.cl_ord_id);
        if (id == nullptr) {
          ++stats_.unknown_refs;
          return false;
        }
        engine_.replace(ReplaceOrder{m.ts, *id, m.price, m.qty, m.owner, m.symbol});
        break;
      }
      default:
        ++stats_.malformed;
        return false;
    }
    track(engine_.events());
    return true;
  }

  const Stats& stats() const noexcept { return stats_; }
  std::size_t live_orders() const noexcept { return live_.size(); }

 private:
  void track(std::span<const Event> events) {
    for (const Event& e : events) {
      switch (e.type) {
        case EventType::Accepted:
          live_.insert(e.owner, e.cl_ord_id, e.order_id);
          break;
        case EventType::Trade:
          if (e.leaves == 0) live_.erase(e.owner, e.cl_ord_id);
          if (e.contra_leaves == 0) live_.erase(e.contra_owner, e.contra_cl_ord_id);
          break;
        case EventType::Cancelled:
          live_.erase(e.owner, e.cl_ord_id);
          break;
        default:
          break;
      }
      if (outbound_ != nullptr && e.type != EventType::BookLevel) {
        const std::size_t at = outbound_->size();
        outbound_->resize(at + wire::kOutboundFrame);
        wire::encode(std::span<std::byte, wire::kOutboundFrame>(outbound_->data() + at, wire::kOutboundFrame), e);
      }
    }
  }

  MatchingEngine& engine_;
  OrderIdMap live_;
  std::vector<std::byte>* outbound_ = nullptr;
  Stats stats_;
};

}  // namespace meridian
