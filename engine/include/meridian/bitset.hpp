// 64-ary summary tree over a flat bit array.
//
// Bit i of layer L+1 is set iff word i of layer L is non-zero, so "next set bit at or after x"
// and "previous set bit at or before x" touch O(log64 N) words. For a 2^20-tick price ladder that
// is at most four words, whatever the gap between occupied price levels. The order book uses it
// to step from an exhausted level to the next non-empty one without scanning empty ticks.
#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace meridian {

class HierarchicalBitset {
 public:
  static constexpr std::size_t kMaxCapacity = std::size_t{1} << 30;

  explicit HierarchicalBitset(std::size_t capacity) : capacity_(capacity) {
    if (capacity == 0 || capacity > kMaxCapacity) throw std::invalid_argument("HierarchicalBitset: bad capacity");
    std::size_t bits = capacity;
    std::size_t total = 0;
    do {
      const std::size_t words = (bits + 63) / 64;
      offset_[layers_] = total;
      words_in_[layers_] = words;
      total += words;
      ++layers_;
      bits = words;
    } while (bits > 1);
    words_.assign(total, 0);
  }

  std::size_t capacity() const noexcept { return capacity_; }

  bool test(std::size_t i) const noexcept { return (word(0, i >> 6) >> (i & 63)) & 1u; }

  void set(std::size_t i) noexcept {
    for (int l = 0; l < layers_; ++l) {
      std::uint64_t& w = word(l, i >> 6);
      const bool was_empty = w == 0;
      w |= std::uint64_t{1} << (i & 63);
      if (!was_empty) return;  // parent bit already set
      i >>= 6;
    }
  }

  void reset(std::size_t i) noexcept {
    for (int l = 0; l < layers_; ++l) {
      std::uint64_t& w = word(l, i >> 6);
      w &= ~(std::uint64_t{1} << (i & 63));
      if (w != 0) return;  // word still non-empty: parent bit stays
      i >>= 6;
    }
  }

  // Smallest set index >= from, or -1.
  std::int64_t next(std::int64_t from) const noexcept {
    if (from < 0) from = 0;
    if (static_cast<std::size_t>(from) >= capacity_) return -1;
    return next_in(0, static_cast<std::size_t>(from));
  }

  // Largest set index <= from, or -1.
  std::int64_t prev(std::int64_t from) const noexcept {
    if (from < 0) return -1;
    if (static_cast<std::size_t>(from) >= capacity_) from = static_cast<std::int64_t>(capacity_) - 1;
    return prev_in(0, static_cast<std::size_t>(from));
  }

 private:
  std::uint64_t& word(int l, std::size_t w) noexcept { return words_[offset_[l] + w]; }
  const std::uint64_t& word(int l, std::size_t w) const noexcept { return words_[offset_[l] + w]; }

  std::int64_t next_in(int l, std::size_t from) const noexcept {
    const std::size_t w = from >> 6;
    if (w >= words_in_[l]) return -1;
    const std::uint64_t m = word(l, w) & (~std::uint64_t{0} << (from & 63));
    if (m != 0) return static_cast<std::int64_t>((w << 6) | static_cast<std::size_t>(std::countr_zero(m)));
    if (l + 1 == layers_) return -1;  // top layer is a single word
    const std::int64_t nw = next_in(l + 1, w + 1);
    if (nw < 0) return -1;
    return (nw << 6) | std::countr_zero(word(l, static_cast<std::size_t>(nw)));
  }

  std::int64_t prev_in(int l, std::size_t from) const noexcept {
    const std::size_t w = from >> 6;
    const std::uint64_t m = word(l, w) & (~std::uint64_t{0} >> (63 - (from & 63)));
    if (m != 0) return static_cast<std::int64_t>((w << 6) | static_cast<std::size_t>(63 - std::countl_zero(m)));
    if (l + 1 == layers_ || w == 0) return -1;
    const std::int64_t pw = prev_in(l + 1, w - 1);
    if (pw < 0) return -1;
    return (pw << 6) | (63 - std::countl_zero(word(l, static_cast<std::size_t>(pw))));
  }

  std::vector<std::uint64_t> words_;
  std::array<std::size_t, 8> offset_{};
  std::array<std::size_t, 8> words_in_{};
  int layers_ = 0;
  std::size_t capacity_;
};

}  // namespace meridian
