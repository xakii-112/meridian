// Differential fuzzing: the optimized OrderBook and the naive ReferenceBook receive the same random
// command stream (all order types, TIFs, post-only, self-trade prevention, cancels, replaces and
// deliberately invalid requests). After every command we require:
//   1. identical normalized event streams,
//   2. identical aggregated depth on both sides,
//   3. a structurally sound book (check_invariants),
//   4. BookLevel market-data events that agree with the book they describe.
#include <gtest/gtest.h>

#include <map>
#include <random>
#include <sstream>
#include <vector>

#include "meridian/matching_engine.hpp"
#include "reference_book.hpp"

using namespace meridian;
using meridian::testing::normalize;
using meridian::testing::NormEvent;
using meridian::testing::ReferenceBook;

namespace {

constexpr Price kMin = 100;
constexpr Price kMax = 140;

std::string describe(const std::vector<NormEvent>& v) {
  std::ostringstream s;
  for (const NormEvent& e : v)
    s << to_string(e.type) << "(cl=" << e.cl << " contra=" << e.contra_cl << " px=" << e.price << " qty=" << e.qty
      << " r=" << int(e.reason) << ") ";
  return s.str();
}

class Fuzzer {
 public:
  explicit Fuzzer(std::uint64_t seed) : rng_(seed), ref_(kMin, kMax) {
    book_ = &engine_.add_instrument({0, "FUZZ", kMin, kMax, 4096, 0});
  }

  void step(int i) {
    const int action = pick(100);
    std::vector<NormEvent> expected;
    if (action < 55 || live_.empty()) {
      const NewOrder o = random_order();
      expected = ref_.add(o);
      engine_.submit(o);
    } else if (action < 80) {
      auto [cl, id, owner] = random_live();
      if (pick(10) == 0) owner = owner % 3 + 1;  // wrong owner
      if (pick(20) == 0) {                        // stale / unknown id
        cl = -1;
        id = 0x0000'7FFF'0000'0001LL;
      }
      expected = ref_.cancel(cl, owner);
      engine_.cancel({i, id, owner, 0});
    } else {
      auto [cl, id, owner] = random_live();
      const Price px = range(kMin - 2, kMax + 2);
      const Qty qty = range(0, 12);
      expected = ref_.replace(cl, owner, px, qty);
      engine_.replace({i, id, px, qty, owner, 0});
    }
    const std::vector<NormEvent> actual = normalize(engine_.events());
    ASSERT_EQ(actual, expected) << "step " << i << "\n engine: " << describe(actual) << "\n ref:    " << describe(expected);
    track();
    ASSERT_EQ(book_->check_invariants(), "") << "step " << i;
    check_depth(i);
  }

 private:
  struct Live {
    ClOrdId cl;
    OrderId id;
    OwnerId owner;
  };

  int pick(int n) { return static_cast<int>(rng_() % static_cast<std::uint64_t>(n)); }
  std::int64_t range(std::int64_t lo, std::int64_t hi) {
    return lo + static_cast<std::int64_t>(rng_() % static_cast<std::uint64_t>(hi - lo + 1));
  }

  NewOrder random_order() {
    NewOrder o;
    o.cl_ord_id = next_cl_++;
    o.owner = static_cast<OwnerId>(range(1, 3));
    o.side = pick(2) ? Side::Buy : Side::Sell;
    o.qty = pick(40) == 0 ? 0 : range(1, 15);
    const int t = pick(100);
    o.type = t < 92 ? OrdType::Limit : OrdType::Market;
    const int tif = pick(100);
    o.tif = tif < 70 ? Tif::Gtc : (tif < 88 ? Tif::Ioc : Tif::Fok);
    // Prices cluster around the middle so the book crosses often; a few fall outside the band.
    o.price = pick(30) == 0 ? range(kMin - 3, kMax + 3) : range(115, 125);
    if (pick(7) == 0) o.flags = kPostOnly;
    if (pick(4) == 0) o.stp = static_cast<Stp>(range(1, 3));
    return o;
  }

  Live random_live() {
    auto it = live_.begin();
    std::advance(it, pick(static_cast<int>(live_.size())));
    return it->second;
  }

 public:
  struct Coverage {
    int trades = 0;
    int stp_cancels = 0;
    int fok_kills = 0;
    int replaces = 0;
    int rejects = 0;
  };
  const Coverage& coverage() const { return cov_; }

 private:
  void track() {
    for (const Event& e : engine_.events()) {
      cov_.trades += e.type == EventType::Trade;
      cov_.replaces += e.type == EventType::Replaced;
      cov_.rejects += e.type == EventType::Rejected;
      if (e.type == EventType::Cancelled) {
        cov_.stp_cancels += e.reason == static_cast<std::uint8_t>(CancelReason::SelfTradePrevention);
        cov_.fok_kills += e.reason == static_cast<std::uint8_t>(CancelReason::FillOrKill);
      }
      switch (e.type) {
        case EventType::Accepted:
          live_[e.cl_ord_id] = Live{e.cl_ord_id, e.order_id, e.owner};
          break;
        case EventType::Trade:
          if (e.leaves == 0) live_.erase(e.cl_ord_id);
          if (e.contra_leaves == 0) live_.erase(e.contra_cl_ord_id);
          break;
        case EventType::Cancelled:
          live_.erase(e.cl_ord_id);
          break;
        case EventType::BookLevel:
          levels_[e.price] = {e.side, e.qty};
          break;
        default:
          break;
      }
    }
  }

  void check_depth(int i) {
    DepthLevel buf[64];
    std::map<Price, std::pair<Side, Qty>> actual;
    for (Side side : {Side::Buy, Side::Sell}) {
      const std::size_t n = book_->depth(side, buf);
      const std::vector<DepthLevel> want = ref_.depth(side);
      ASSERT_EQ(n, want.size()) << "step " << i;
      for (std::size_t k = 0; k < n; ++k) {
        ASSERT_EQ(buf[k].price, want[k].price) << "step " << i;
        ASSERT_EQ(buf[k].qty, want[k].qty) << "step " << i;
        ASSERT_EQ(buf[k].orders, want[k].orders) << "step " << i;
        actual[buf[k].price] = {side, buf[k].qty};
      }
    }
    // Market-data consistency: replaying the BookLevel stream reproduces the book exactly.
    // (A price belongs to one side at a time, so the latest update per price is authoritative.)
    for (const auto& [price, lv] : levels_) {
      const auto it = actual.find(price);
      if (lv.second == 0) {
        ASSERT_TRUE(it == actual.end()) << "step " << i << " px " << price << " should be empty";
      } else {
        ASSERT_TRUE(it != actual.end()) << "step " << i << " px " << price << " missing";
        ASSERT_EQ(it->second.first, lv.first) << "step " << i << " px " << price;
        ASSERT_EQ(it->second.second, lv.second) << "step " << i << " px " << price;
      }
    }
    for (const auto& [price, side_qty] : actual) ASSERT_TRUE(levels_.count(price)) << "step " << i;
  }

  std::mt19937_64 rng_;
  MatchingEngine engine_;
  OrderBook* book_;
  ReferenceBook ref_;
  std::map<ClOrdId, Live> live_;
  std::map<Price, std::pair<Side, Qty>> levels_;
  ClOrdId next_cl_ = 1;
  Coverage cov_;
};

}  // namespace

class DifferentialFuzz : public ::testing::TestWithParam<std::uint64_t> {};

TEST_P(DifferentialFuzz, OptimizedBookMatchesReferenceModel) {
  Fuzzer f(GetParam());
  for (int i = 0; i < 4000; ++i) {
    f.step(i);
    if (::testing::Test::HasFatalFailure()) return;
  }
  // Guard against a vacuous fuzzer: every interesting path must actually be exercised.
  EXPECT_GT(f.coverage().trades, 500);
  EXPECT_GT(f.coverage().stp_cancels, 5);
  EXPECT_GT(f.coverage().fok_kills, 5);
  EXPECT_GT(f.coverage().replaces, 100);
  EXPECT_GT(f.coverage().rejects, 50);
}

INSTANTIATE_TEST_SUITE_P(Seeds, DifferentialFuzz, ::testing::Range<std::uint64_t>(1, 41));
