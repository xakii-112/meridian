#include <gtest/gtest.h>

#include "meridian/matching_engine.hpp"

using namespace meridian;

namespace {

class RiskTest : public ::testing::Test {
 protected:
  RiskTest() : engine_(&risk_) {
    engine_.add_instrument({0, "RISK", 1, 100000, 1024, 10000});
    RiskLimits open;
    risk_.register_participant(kMaker, open);
  }

  RejectReason submit(OwnerId owner, Side side, Price price, Qty qty, Timestamp ts = 0) {
    NewOrder o;
    o.ts = ts;
    o.cl_ord_id = next_cl_++;
    o.owner = owner;
    o.side = side;
    o.price = price;
    o.qty = qty;
    return submit(o);
  }

  RejectReason submit(const NewOrder& o) {
    engine_.submit(o);
    for (const Event& e : engine_.events())
      if (e.type == EventType::Rejected) return static_cast<RejectReason>(e.reason);
    return RejectReason::None;
  }

  OrderId last_accepted() const {
    for (const Event& e : engine_.events())
      if (e.type == EventType::Accepted) return e.order_id;
    return 0;
  }

  static constexpr OwnerId kTrader = 1;
  static constexpr OwnerId kMaker = 2;
  RiskEngine risk_{64, 4};
  MatchingEngine engine_;
  ClOrdId next_cl_ = 1;
};

}  // namespace

TEST_F(RiskTest, UnknownParticipantIsRejected) {
  EXPECT_EQ(submit(7, Side::Buy, 10000, 1), RejectReason::RiskUnknownParticipant);
}

TEST_F(RiskTest, MaxOrderQuantity) {
  RiskLimits lim;
  lim.max_order_qty = 100;
  risk_.register_participant(kTrader, lim);
  EXPECT_EQ(submit(kTrader, Side::Buy, 9990, 101), RejectReason::RiskMaxOrderQuantity);
  EXPECT_EQ(submit(kTrader, Side::Buy, 9990, 100), RejectReason::None);
}

TEST_F(RiskTest, PriceCollarFollowsLastTrade) {
  RiskLimits lim;
  lim.price_collar = 50;
  risk_.register_participant(kTrader, lim);
  EXPECT_EQ(submit(kTrader, Side::Buy, 9949, 1), RejectReason::RiskPriceCollar);
  EXPECT_EQ(submit(kTrader, Side::Buy, 9950, 1), RejectReason::None);
  // Print a trade at 10200: the collar re-centres on the last trade.
  submit(kMaker, Side::Sell, 10200, 1);
  submit(kMaker, Side::Buy, 10200, 1);
  EXPECT_EQ(engine_.book(0)->last_trade_price(), 10200);
  EXPECT_EQ(submit(kTrader, Side::Sell, 10150, 1), RejectReason::None);
  EXPECT_EQ(submit(kTrader, Side::Sell, 10149, 1), RejectReason::RiskPriceCollar);
}

TEST_F(RiskTest, MarketOrdersAreProtectedByTheCollar) {
  RiskLimits lim;
  lim.price_collar = 20;
  risk_.register_participant(kTrader, lim);
  submit(kMaker, Side::Sell, 10010, 5);
  submit(kMaker, Side::Sell, 10030, 5);  // beyond reference + collar
  NewOrder m;
  m.cl_ord_id = next_cl_++;
  m.owner = kTrader;
  m.side = Side::Buy;
  m.type = OrdType::Market;
  m.tif = Tif::Ioc;
  m.qty = 10;
  EXPECT_EQ(submit(m), RejectReason::None);
  std::int64_t filled = 0;
  for (const Event& e : engine_.events())
    if (e.type == EventType::Trade) filled += e.qty;
  EXPECT_EQ(filled, 5);  // stopped at 10020, never touched 10030
  EXPECT_EQ(engine_.book(0)->best_ask(), 10030);
}

TEST_F(RiskTest, MaxNotional) {
  RiskLimits lim;
  lim.max_order_notional = 10000 * 10;
  risk_.register_participant(kTrader, lim);
  EXPECT_EQ(submit(kTrader, Side::Buy, 10000, 11), RejectReason::RiskMaxNotional);
  EXPECT_EQ(submit(kTrader, Side::Buy, 10000, 10), RejectReason::None);
}

TEST_F(RiskTest, PositionLimitCountsRestingExposure) {
  RiskLimits lim;
  lim.max_long = 10;
  lim.max_short = 4;
  risk_.register_participant(kTrader, lim);
  EXPECT_EQ(submit(kTrader, Side::Buy, 9990, 6), RejectReason::None);
  EXPECT_EQ(risk_.open_qty(kTrader, 0, Side::Buy), 6);
  EXPECT_EQ(submit(kTrader, Side::Buy, 9980, 5), RejectReason::RiskPositionLimit);
  EXPECT_EQ(submit(kTrader, Side::Buy, 9980, 4), RejectReason::None);
  EXPECT_EQ(submit(kTrader, Side::Sell, 10010, 5), RejectReason::RiskPositionLimit);
  EXPECT_EQ(submit(kTrader, Side::Sell, 10010, 4), RejectReason::None);
}

TEST_F(RiskTest, FillsMovePositionsAndOpenQuantity) {
  risk_.register_participant(kTrader, RiskLimits{});
  submit(kMaker, Side::Sell, 10000, 8);
  submit(kTrader, Side::Buy, 10000, 5);
  EXPECT_EQ(risk_.position(kTrader, 0), 5);
  EXPECT_EQ(risk_.position(kMaker, 0), -5);
  EXPECT_EQ(risk_.open_qty(kMaker, 0, Side::Sell), 3);
  EXPECT_EQ(risk_.open_qty(kTrader, 0, Side::Buy), 0);
  EXPECT_EQ(risk_.open_orders(kTrader), 0);
  EXPECT_EQ(risk_.open_orders(kMaker), 1);
}

TEST_F(RiskTest, ReplaceAdjustsOpenQuantityAndIsChecked) {
  RiskLimits lim;
  lim.max_long = 10;
  risk_.register_participant(kTrader, lim);
  submit(kTrader, Side::Buy, 9990, 4);
  const OrderId id = last_accepted();
  engine_.replace({0, id, 9990, 2, kTrader, 0});
  EXPECT_EQ(risk_.open_qty(kTrader, 0, Side::Buy), 2);
  engine_.replace({0, id, 9990, 11, kTrader, 0});
  ASSERT_EQ(engine_.events().size(), 1u);
  EXPECT_EQ(engine_.events()[0].type, EventType::ReplaceRejected);
  EXPECT_EQ(static_cast<RejectReason>(engine_.events()[0].reason), RejectReason::RiskPositionLimit);
  EXPECT_EQ(risk_.open_qty(kTrader, 0, Side::Buy), 2);
}

TEST_F(RiskTest, MaxOpenOrders) {
  RiskLimits lim;
  lim.max_open_orders = 2;
  risk_.register_participant(kTrader, lim);
  EXPECT_EQ(submit(kTrader, Side::Buy, 9990, 1), RejectReason::None);
  EXPECT_EQ(submit(kTrader, Side::Buy, 9991, 1), RejectReason::None);
  EXPECT_EQ(submit(kTrader, Side::Buy, 9992, 1), RejectReason::RiskMaxOpenOrders);
}

TEST_F(RiskTest, ThrottleRefillsOnSequencerTime) {
  RiskLimits lim;
  lim.msgs_per_sec = 10;
  lim.burst = 2;
  risk_.register_participant(kTrader, lim);
  const Timestamp t0 = 1'000'000'000;
  EXPECT_EQ(submit(kTrader, Side::Buy, 9990, 1, t0), RejectReason::None);
  EXPECT_EQ(submit(kTrader, Side::Buy, 9990, 1, t0), RejectReason::None);
  EXPECT_EQ(submit(kTrader, Side::Buy, 9990, 1, t0), RejectReason::RiskThrottled);
  EXPECT_EQ(submit(kTrader, Side::Buy, 9990, 1, t0 + 50'000'000), RejectReason::RiskThrottled);
  EXPECT_EQ(submit(kTrader, Side::Buy, 9990, 1, t0 + 100'000'000), RejectReason::None);
}

TEST_F(RiskTest, KillSwitchPullsQuotesAndBlocksOrders) {
  risk_.register_participant(kTrader, RiskLimits{});
  submit(kTrader, Side::Buy, 9990, 3);
  submit(kTrader, Side::Sell, 10010, 3);
  EXPECT_EQ(engine_.set_kill_switch(kTrader, true, 5), 2u);
  for (const Event& e : engine_.events())
    if (e.type == EventType::Cancelled) EXPECT_EQ(e.reason, static_cast<std::uint8_t>(CancelReason::KillSwitch));
  EXPECT_FALSE(engine_.book(0)->has_bid());
  EXPECT_EQ(risk_.open_orders(kTrader), 0);
  EXPECT_EQ(submit(kTrader, Side::Buy, 9990, 1), RejectReason::RiskKillSwitch);
  engine_.set_kill_switch(kTrader, false, 6);
  EXPECT_EQ(submit(kTrader, Side::Buy, 9990, 1), RejectReason::None);
}

TEST_F(RiskTest, GlobalHalt) {
  risk_.set_global_halt(true);
  EXPECT_EQ(submit(kMaker, Side::Buy, 9990, 1), RejectReason::RiskKillSwitch);
  risk_.set_global_halt(false);
  EXPECT_EQ(submit(kMaker, Side::Buy, 9990, 1), RejectReason::None);
}

TEST(RiskConfig, RejectsInvalidThrottle) {
  RiskEngine r(4, 1);
  RiskLimits bad;
  bad.msgs_per_sec = 10;
  bad.burst = 0;
  EXPECT_THROW(r.register_participant(0, bad), std::invalid_argument);
  EXPECT_THROW(r.register_participant(9, RiskLimits{}), std::out_of_range);
}
