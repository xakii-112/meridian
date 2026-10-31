#include <gtest/gtest.h>

#include <random>
#include <set>

#include "meridian/bitset.hpp"

using meridian::HierarchicalBitset;

namespace {

std::int64_t ref_next(const std::set<std::int64_t>& s, std::int64_t from) {
  auto it = s.lower_bound(std::max<std::int64_t>(from, 0));
  return it == s.end() ? -1 : *it;
}

std::int64_t ref_prev(const std::set<std::int64_t>& s, std::int64_t from) {
  if (from < 0) return -1;
  auto it = s.upper_bound(from);
  if (it == s.begin()) return -1;
  return *--it;
}

}  // namespace

TEST(HierarchicalBitset, EmptyHasNoBits) {
  HierarchicalBitset b(1000);
  EXPECT_EQ(b.next(0), -1);
  EXPECT_EQ(b.prev(999), -1);
  EXPECT_EQ(b.prev(-1), -1);
  EXPECT_EQ(b.next(1000), -1);
}

TEST(HierarchicalBitset, BoundaryBits) {
  HierarchicalBitset b(64 * 64 * 64 + 1);  // forces four layers
  const std::size_t last = 64 * 64 * 64;
  b.set(0);
  b.set(last);
  EXPECT_EQ(b.next(1), static_cast<std::int64_t>(last));
  EXPECT_EQ(b.prev(static_cast<std::int64_t>(last) - 1), 0);
  b.reset(0);
  EXPECT_EQ(b.prev(static_cast<std::int64_t>(last) - 1), -1);
  EXPECT_EQ(b.next(0), static_cast<std::int64_t>(last));
  b.reset(last);
  EXPECT_EQ(b.next(0), -1);
}

TEST(HierarchicalBitset, RejectsBadCapacity) {
  EXPECT_THROW(HierarchicalBitset(0), std::invalid_argument);
}

class BitsetFuzz : public ::testing::TestWithParam<std::size_t> {};

TEST_P(BitsetFuzz, AgreesWithOrderedSet) {
  const std::size_t n = GetParam();
  HierarchicalBitset b(n);
  std::set<std::int64_t> ref;
  std::mt19937_64 rng(n * 7919 + 1);
  std::uniform_int_distribution<std::int64_t> idx(0, static_cast<std::int64_t>(n) - 1);
  std::uniform_int_distribution<std::int64_t> query(-2, static_cast<std::int64_t>(n) + 2);
  for (int step = 0; step < 20000; ++step) {
    const std::int64_t i = idx(rng);
    if (rng() % 3 != 0) {
      b.set(static_cast<std::size_t>(i));
      ref.insert(i);
    } else {
      b.reset(static_cast<std::size_t>(i));
      ref.erase(i);
    }
    const std::int64_t q = query(rng);
    ASSERT_EQ(b.next(q), ref_next(ref, q)) << "n=" << n << " q=" << q;
    ASSERT_EQ(b.prev(q), ref_prev(ref, std::min<std::int64_t>(q, static_cast<std::int64_t>(n) - 1)))
        << "n=" << n << " q=" << q;
    if (q >= 0 && q < static_cast<std::int64_t>(n)) ASSERT_EQ(b.test(static_cast<std::size_t>(q)), ref.count(q) == 1);
  }
}

INSTANTIATE_TEST_SUITE_P(Sizes, BitsetFuzz, ::testing::Values(1, 2, 63, 64, 65, 4095, 4096, 4097, 300'001));
