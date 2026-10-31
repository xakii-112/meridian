// Output tape for one inbound command. The book appends events here; the caller drains them
// before issuing the next command. Storage is reused, so steady-state processing never allocates.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "meridian/types.hpp"

namespace meridian {

class EventBuffer {
 public:
  explicit EventBuffer(std::size_t capacity = std::size_t{1} << 12) : events_(capacity == 0 ? 1 : capacity) {}

  void clear() noexcept { size_ = 0; }
  std::span<const Event> view() const noexcept { return {events_.data(), size_}; }
  std::size_t size() const noexcept { return size_; }
  std::int64_t sequence() const noexcept { return seq_; }

  // The returned reference is valid until the next emit().
  Event& emit(EventType type, Timestamp ts, SymbolId symbol) {
    if (size_ == events_.size()) [[unlikely]] events_.resize(events_.size() * 2);
    Event& e = events_[size_++];
    e = Event{};
    e.seq = ++seq_;
    e.ts = ts;
    e.symbol = symbol;
    e.type = type;
    return e;
  }

 private:
  std::vector<Event> events_;
  std::size_t size_ = 0;
  std::int64_t seq_ = 0;
};

}  // namespace meridian
