#include <gtest/gtest.h>

#include <vector>

#include "meridian/matching_engine.hpp"

using namespace meridian;

namespace {

class BookTest : public ::testing::Test {
 protected:
  explicit BookTest(std::size_t capacity = 1024) : book_(&engine_.add_instrument({0, "TEST", 1, 1000, capacity, 100})) {}

  NewOrder order(Side side, Price price, Qty qty, OwnerId owner = 1) {
    NewOrder o;
    o.ts = ++ts_;
    o.cl_ord_id = next_cl_++;
    o.side = side;
    o.price = price;
    o.qty = qty;
    o.owner = owner;
    return o;
  }

  // Submits and returns the engine order id (0 if not accepted).
  OrderId submit(const NewOrder& o) {
    engine_.submit(o);
    for (const Event& e : engine_.events())
      if (e.type == EventType::Accepted) return e.order_id;
    return 0;
  }

  OrderId limit(Side side, Price price, Qty qty, OwnerId owner = 1) { return submit(order(side, price, qty, owner)); }

  std::vector<Event> of(EventType type) const {
    std::vector<Event> v;
    for (const Event& e : engine_.events())
      if (e.type == type) v.push_back(e);
    return v;
  }

  void expect_sound() const { EXPECT_EQ(book_->check_invariants(), ""); }

  MatchingEngine engine_;
  OrderBook* book_;
  ClOrdId next_cl_ = 1;
  Timestamp ts_ = 0;
};

class TinyBookTest : public BookTest {
 protected:
  TinyBookTest() : BookTest(2) {}
};

}  // namespace

TEST_F(BookTest, RestsNonCrossingOrders) {
  limit(Side::Buy, 99, 10);
  limit(Side::Sell, 101, 5);
  ASSERT_TRUE(book_->has_bid());
  ASSERT_TRUE(book_->has_ask());
  EXPECT_EQ(book_->best_bid(), 99);
  EXPECT_EQ(book_->best_bid_qty(), 10);
  EXPECT_EQ(book_->best_ask(), 101);
  EXPECT_EQ(book_->best_ask_qty(), 5);
  EXPECT_EQ(engine_.trade_count(), 0);
  expect_sound();
}

TEST_F(BookTest, FillsAtRestingPrice) {
  limit(Side::Sell, 100, 10, 2);
  limit(Side::Buy, 105, 10, 1);
  const auto trades = of(EventType::Trade);
  ASSERT_EQ(trades.size(), 1u);
  EXPECT_EQ(trades[0].price, 100);  // maker price, not the aggressor's limit
  EXPECT_EQ(trades[0].qty, 10);
  EXPECT_EQ(trades[0].side, Side::Sell);
  EXPECT_EQ(trades[0].owner, 2);
  EXPECT_EQ(trades[0].contra_owner, 1);
  EXPECT_FALSE(book_->has_bid());
  EXPECT_FALSE(book_->has_ask());
  EXPECT_EQ(book_->active_orders(), 0u);
  expect_sound();
}

TEST_F(BookTest, PartialFillRestsRemainder) {
  limit(Side::Sell, 100, 4, 2);
  limit(Side::Buy, 100, 10);
  ASSERT_EQ(of(EventType::Trade).size(), 1u);
  EXPECT_EQ(book_->best_bid(), 100);
  EXPECT_EQ(book_->best_bid_qty(), 6);
  EXPECT_FALSE(book_->has_ask());
  expect_sound();
}

TEST_F(BookTest, SweepsLevelsInPriceOrder) {
  limit(Side::Sell, 103, 4, 2);
  limit(Side::Sell, 101, 2, 2);
  limit(Side::Sell, 102, 3, 2);
  limit(Side::Buy, 102, 10);
  const auto trades = of(EventType::Trade);
  ASSERT_EQ(trades.size(), 2u);
  EXPECT_EQ(trades[0].price, 101);
  EXPECT_EQ(trades[0].qty, 2);
  EXPECT_EQ(trades[1].price, 102);
  EXPECT_EQ(trades[1].qty, 3);
  EXPECT_EQ(book_->best_bid(), 102);
  EXPECT_EQ(book_->best_bid_qty(), 5);
  EXPECT_EQ(book_->best_ask(), 103);
  expect_sound();
}

TEST_F(BookTest, TimePriorityWithinLevel) {
  const ClOrdId first = next_cl_;
  limit(Side::Sell, 100, 2, 2);
  limit(Side::Sell, 100, 2, 3);
  limit(Side::Sell, 100, 2, 4);
  limit(Side::Buy, 100, 5);
  const auto trades = of(EventType::Trade);
  ASSERT_EQ(trades.size(), 3u);
  EXPECT_EQ(trades[0].cl_ord_id, first);
  EXPECT_EQ(trades[1].cl_ord_id, first + 1);
  EXPECT_EQ(trades[2].cl_ord_id, first + 2);
  EXPECT_EQ(trades[2].qty, 1);
  EXPECT_EQ(trades[2].leaves, 1);
  EXPECT_EQ(book_->best_ask_qty(), 1);
  expect_sound();
}

TEST_F(BookTest, IocCancelsRemainder) {
  limit(Side::Sell, 100, 3, 2);
  NewOrder o = order(Side::Buy, 100, 5);
  o.tif = Tif::Ioc;
  submit(o);
  ASSERT_EQ(of(EventType::Trade).size(), 1u);
  const auto cancels = of(EventType::Cancelled);
  ASSERT_EQ(cancels.size(), 1u);
  EXPECT_EQ(cancels[0].qty, 2);
  EXPECT_EQ(cancels[0].reason, static_cast<std::uint8_t>(CancelReason::ImmediateOrCancel));
  EXPECT_FALSE(book_->has_bid());
  expect_sound();
}

TEST_F(BookTest, FokKillsWhenLiquidityInsufficient) {
  limit(Side::Sell, 100, 3, 2);
  limit(Side::Sell, 101, 3, 2);
  NewOrder o = order(Side::Buy, 101, 7);
  o.tif = Tif::Fok;
  submit(o);
  EXPECT_TRUE(of(EventType::Trade).empty());
  const auto cancels = of(EventType::Cancelled);
  ASSERT_EQ(cancels.size(), 1u);
  EXPECT_EQ(cancels[0].reason, static_cast<std::uint8_t>(CancelReason::FillOrKill));
  EXPECT_EQ(book_->best_ask_qty(), 3);  // untouched
  expect_sound();
}

TEST_F(BookTest, FokFillsAcrossLevels) {
  limit(Side::Sell, 100, 3, 2);
  limit(Side::Sell, 101, 3, 2);
  NewOrder o = order(Side::Buy, 101, 6);
  o.tif = Tif::Fok;
  submit(o);
  EXPECT_EQ(of(EventType::Trade).size(), 2u);
  EXPECT_TRUE(of(EventType::Cancelled).empty());
  EXPECT_FALSE(book_->has_ask());
  expect_sound();
}

TEST_F(BookTest, MarketOrderSweepsThenCancelsRemainder) {
  limit(Side::Buy, 98, 2, 2);
  limit(Side::Buy, 97, 2, 2);
  NewOrder o = order(Side::Sell, 0, 10);
  o.type = OrdType::Market;
  o.tif = Tif::Ioc;
  submit(o);
  EXPECT_EQ(of(EventType::Trade).size(), 2u);
  ASSERT_EQ(of(EventType::Cancelled).size(), 1u);
  EXPECT_EQ(of(EventType::Cancelled)[0].qty, 6);
  EXPECT_FALSE(book_->has_bid());
  expect_sound();
}

TEST_F(BookTest, MarketGtcIsRejected) {
  NewOrder o = order(Side::Buy, 0, 1);
  o.type = OrdType::Market;
  submit(o);
  ASSERT_EQ(of(EventType::Rejected).size(), 1u);
  EXPECT_EQ(of(EventType::Rejected)[0].reason, static_cast<std::uint8_t>(RejectReason::InvalidTimeInForce));
}

TEST_F(BookTest, PostOnlyRejectsWhenItWouldTake) {
  limit(Side::Sell, 100, 5, 2);
  NewOrder o = order(Side::Buy, 100, 1);
  o.flags = kPostOnly;
  EXPECT_EQ(submit(o), 0);
  EXPECT_EQ(of(EventType::Rejected)[0].reason, static_cast<std::uint8_t>(RejectReason::PostOnlyWouldCross));
  o = order(Side::Buy, 99, 1);
  o.flags = kPostOnly;
  EXPECT_NE(submit(o), 0);
  EXPECT_EQ(book_->best_bid(), 99);
  expect_sound();
}

TEST_F(BookTest, RejectsInvalidRequests) {
  EXPECT_EQ(limit(Side::Buy, 100, 0), 0);
  EXPECT_EQ(of(EventType::Rejected)[0].reason, static_cast<std::uint8_t>(RejectReason::InvalidQuantity));
  EXPECT_EQ(limit(Side::Buy, 1001, 1), 0);
  EXPECT_EQ(of(EventType::Rejected)[0].reason, static_cast<std::uint8_t>(RejectReason::PriceOutOfRange));
  NewOrder o = order(Side::Buy, 100, 1);
  o.symbol = 7;
  submit(o);
  EXPECT_EQ(of(EventType::Rejected)[0].reason, static_cast<std::uint8_t>(RejectReason::UnknownSymbol));
}

TEST_F(BookTest, CancelRemovesOrderAndUpdatesBest) {
  const OrderId a = limit(Side::Buy, 99, 10);
  limit(Side::Buy, 98, 5);
  engine_.cancel({++ts_, a, 1, 0});
  ASSERT_EQ(of(EventType::Cancelled).size(), 1u);
  EXPECT_EQ(of(EventType::Cancelled)[0].qty, 10);
  EXPECT_EQ(book_->best_bid(), 98);
  expect_sound();
}

TEST_F(BookTest, CancelRejectsUnknownStaleAndForeignIds) {
  const OrderId a = limit(Side::Buy, 99, 10);
  engine_.cancel({++ts_, a, 2, 0});
  EXPECT_EQ(of(EventType::CancelRejected)[0].reason, static_cast<std::uint8_t>(RejectReason::NotOwner));
  engine_.cancel({++ts_, a, 1, 0});
  EXPECT_EQ(of(EventType::Cancelled).size(), 1u);
  const OrderId b = limit(Side::Buy, 99, 10);  // reuses the slot with a new generation
  EXPECT_NE(a, b);
  engine_.cancel({++ts_, a, 1, 0});
  EXPECT_EQ(of(EventType::CancelRejected)[0].reason, static_cast<std::uint8_t>(RejectReason::UnknownOrder));
  engine_.cancel({++ts_, 123456789, 1, 0});
  EXPECT_EQ(of(EventType::CancelRejected)[0].reason, static_cast<std::uint8_t>(RejectReason::UnknownOrder));
  EXPECT_EQ(book_->best_bid_qty(), 10);
  expect_sound();
}

TEST_F(BookTest, ReplaceSizeDownKeepsPriority) {
  const ClOrdId first = next_cl_;
  const OrderId a = limit(Side::Buy, 99, 10, 1);
  limit(Side::Buy, 99, 10, 3);
  engine_.replace({++ts_, a, 99, 4, 1, 0});
  ASSERT_EQ(of(EventType::Replaced).size(), 1u);
  EXPECT_EQ(of(EventType::Replaced)[0].contra_leaves, 10);
  EXPECT_EQ(book_->best_bid_qty(), 14);
  limit(Side::Sell, 99, 1, 2);
  EXPECT_EQ(of(EventType::Trade)[0].cl_ord_id, first);
  expect_sound();
}

TEST_F(BookTest, ReplaceSizeUpLosesPriority) {
  const OrderId a = limit(Side::Buy, 99, 10, 1);
  const ClOrdId second = next_cl_;
  limit(Side::Buy, 99, 10, 3);
  engine_.replace({++ts_, a, 99, 12, 1, 0});
  limit(Side::Sell, 99, 1, 2);
  EXPECT_EQ(of(EventType::Trade)[0].cl_ord_id, second);
  expect_sound();
}

TEST_F(BookTest, ReplaceToCrossingPriceExecutes) {
  const OrderId bid = limit(Side::Buy, 99, 5, 1);
  limit(Side::Sell, 101, 3, 2);
  engine_.replace({++ts_, bid, 101, 5, 1, 0});
  const auto trades = of(EventType::Trade);
  ASSERT_EQ(trades.size(), 1u);
  EXPECT_EQ(trades[0].price, 101);
  EXPECT_EQ(trades[0].qty, 3);
  EXPECT_EQ(book_->best_bid(), 101);
  EXPECT_EQ(book_->best_bid_qty(), 2);
  EXPECT_FALSE(book_->has_ask());
  expect_sound();
}

TEST_F(BookTest, SelfTradePreventionCancelResting) {
  limit(Side::Sell, 100, 5, 1);
  limit(Side::Sell, 100, 5, 2);
  NewOrder o = order(Side::Buy, 100, 5, 1);
  o.stp = Stp::CancelResting;
  submit(o);
  const auto cancels = of(EventType::Cancelled);
  ASSERT_EQ(cancels.size(), 1u);
  EXPECT_EQ(cancels[0].owner, 1);
  EXPECT_EQ(cancels[0].reason, static_cast<std::uint8_t>(CancelReason::SelfTradePrevention));
  ASSERT_EQ(of(EventType::Trade).size(), 1u);
  EXPECT_EQ(of(EventType::Trade)[0].owner, 2);
  EXPECT_FALSE(book_->has_ask());
  expect_sound();
}

TEST_F(BookTest, SelfTradePreventionCancelAggressor) {
  limit(Side::Sell, 100, 5, 2);
  limit(Side::Sell, 101, 5, 1);
  NewOrder o = order(Side::Buy, 101, 8, 1);
  o.stp = Stp::CancelAggressor;
  submit(o);
  ASSERT_EQ(of(EventType::Trade).size(), 1u);
  const auto cancels = of(EventType::Cancelled);
  ASSERT_EQ(cancels.size(), 1u);
  EXPECT_EQ(cancels[0].qty, 3);
  EXPECT_EQ(cancels[0].reason, static_cast<std::uint8_t>(CancelReason::SelfTradePrevention));
  EXPECT_EQ(book_->best_ask(), 101);  // own resting order untouched
  EXPECT_FALSE(book_->has_bid());
  expect_sound();
}

TEST_F(BookTest, SelfTradePreventionCancelBoth) {
  limit(Side::Sell, 100, 5, 1);
  NewOrder o = order(Side::Buy, 100, 8, 1);
  o.stp = Stp::CancelBoth;
  submit(o);
  EXPECT_TRUE(of(EventType::Trade).empty());
  EXPECT_EQ(of(EventType::Cancelled).size(), 2u);
  EXPECT_FALSE(book_->has_ask());
  EXPECT_FALSE(book_->has_bid());
  expect_sound();
}

TEST_F(BookTest, DepthAggregatesLevelsBestFirst) {
  limit(Side::Buy, 98, 1);
  limit(Side::Buy, 99, 2);
  limit(Side::Buy, 99, 3);
  limit(Side::Buy, 95, 4);
  DepthLevel out[8];
  const std::size_t n = book_->depth(Side::Buy, out);
  ASSERT_EQ(n, 3u);
  EXPECT_EQ(out[0].price, 99);
  EXPECT_EQ(out[0].qty, 5);
  EXPECT_EQ(out[0].orders, 2);
  EXPECT_EQ(out[1].price, 98);
  EXPECT_EQ(out[2].price, 95);
}

TEST_F(BookTest, BookLevelEventsCarryAggregates) {
  limit(Side::Sell, 100, 3, 2);
  limit(Side::Sell, 100, 4, 2);
  const auto lv = of(EventType::BookLevel);
  ASSERT_EQ(lv.size(), 1u);
  EXPECT_EQ(lv[0].qty, 7);
  EXPECT_EQ(lv[0].leaves, 2);
  limit(Side::Buy, 100, 7);
  const auto lv2 = of(EventType::BookLevel);
  ASSERT_EQ(lv2.size(), 1u);
  EXPECT_EQ(lv2[0].qty, 0);
  EXPECT_EQ(lv2[0].side, Side::Sell);
}

TEST_F(BookTest, MassCancelByOwner) {
  limit(Side::Buy, 99, 1, 1);
  limit(Side::Sell, 101, 1, 1);
  limit(Side::Buy, 98, 1, 2);
  EXPECT_EQ(engine_.mass_cancel(1, ++ts_), 2u);
  EXPECT_EQ(book_->best_bid(), 98);
  EXPECT_FALSE(book_->has_ask());
  expect_sound();
}

TEST_F(BookTest, LargeGapsBetweenLevels) {
  limit(Side::Sell, 1000, 1, 2);
  limit(Side::Sell, 2, 1, 2);
  limit(Side::Buy, 1, 1, 3);
  limit(Side::Buy, 1000, 2, 4);  // sweeps both asks across the whole ladder
  EXPECT_EQ(of(EventType::Trade).size(), 2u);
  EXPECT_FALSE(book_->has_ask());
  EXPECT_EQ(book_->best_bid(), 1);
  expect_sound();
}

TEST_F(TinyBookTest, RejectsWhenArenaIsFull) {
  const OrderId a = limit(Side::Buy, 99, 1);
  limit(Side::Buy, 98, 1);
  EXPECT_EQ(limit(Side::Buy, 97, 1), 0);
  EXPECT_EQ(of(EventType::Rejected)[0].reason, static_cast<std::uint8_t>(RejectReason::OrderCapacityExceeded));
  engine_.cancel({++ts_, a, 1, 0});
  EXPECT_NE(limit(Side::Buy, 97, 1), 0);
  expect_sound();
}

TEST(TapeHash, IsDeterministicAndSensitive) {
  auto run = [](Qty last_qty) {
    MatchingEngine e;
    e.add_instrument({0, "H", 1, 1000, 64, 0});
    NewOrder s{1, 1, 100, 5, 1, 0, Side::Sell};
    NewOrder b{2, 2, 100, last_qty, 2, 0, Side::Buy};
    e.submit(s);
    e.submit(b);
    return e.tape_hash();
  };
  EXPECT_EQ(run(3), run(3));
  EXPECT_NE(run(3), run(4));
  EXPECT_NE(run(3), kTapeHashSeed);
}
