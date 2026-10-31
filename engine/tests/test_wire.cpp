#include <gtest/gtest.h>

#include <array>
#include <map>
#include <random>
#include <vector>

#include "meridian/gateway.hpp"
#include "meridian/wire.hpp"

using namespace meridian;

namespace {

using InFrame = std::array<std::byte, wire::kInboundFrame>;

InFrame frame(const wire::Inbound& m) {
  InFrame f{};
  wire::encode(std::span<std::byte, wire::kInboundFrame>(f), m);
  return f;
}

wire::Inbound enter(OwnerId owner, ClOrdId cl, Side side, Price price, Qty qty, Tif tif = Tif::Gtc) {
  wire::Inbound m;
  m.type = wire::MsgType::EnterOrder;
  m.owner = owner;
  m.cl_ord_id = cl;
  m.side = side;
  m.price = price;
  m.qty = qty;
  m.tif = tif;
  return m;
}

wire::Inbound cancel(OwnerId owner, ClOrdId cl) {
  wire::Inbound m;
  m.type = wire::MsgType::Cancel;
  m.owner = owner;
  m.cl_ord_id = cl;
  return m;
}

}  // namespace

TEST(Wire, EnterOrderRoundTrip) {
  wire::Inbound m = enter(42, 0x1122334455667788LL, Side::Sell, 10'123, 77, Tif::Fok);
  m.ord_type = OrdType::Limit;
  m.flags = kPostOnly;
  m.stp = Stp::CancelBoth;
  m.symbol = 513;
  m.ts = -5;
  const InFrame f = frame(m);
  EXPECT_EQ(static_cast<char>(f[0]), 'O');
  wire::Inbound d;
  ASSERT_TRUE(wire::decode(f, d));
  EXPECT_EQ(d.owner, 42);
  EXPECT_EQ(d.cl_ord_id, 0x1122334455667788LL);
  EXPECT_EQ(d.side, Side::Sell);
  EXPECT_EQ(d.price, 10'123);
  EXPECT_EQ(d.qty, 77);
  EXPECT_EQ(d.tif, Tif::Fok);
  EXPECT_EQ(d.flags, kPostOnly);
  EXPECT_EQ(d.stp, Stp::CancelBoth);
  EXPECT_EQ(d.symbol, 513);
  EXPECT_EQ(d.ts, -5);
}

TEST(Wire, RejectsMalformedFrames) {
  InFrame f = frame(enter(1, 1, Side::Buy, 100, 1));
  wire::Inbound d;
  EXPECT_FALSE(wire::decode(std::span<const std::byte>(f.data(), 47), d));
  f[1] = std::byte{2};  // invalid side
  EXPECT_FALSE(wire::decode(f, d));
  f = frame(enter(1, 1, Side::Buy, 100, 1));
  f[0] = std::byte{'Z'};
  EXPECT_FALSE(wire::decode(f, d));
}

TEST(Wire, OutboundRoundTrip) {
  Event e;
  e.type = EventType::Trade;
  e.side = Side::Sell;
  e.reason = 3;
  e.symbol = 9;
  e.seq = 1234567;
  e.cl_ord_id = 55;
  e.order_id = (7LL << 32) | 3;
  e.price = 10'050;
  e.qty = 12;
  e.leaves = 4;
  e.ts = 987654321;
  std::array<std::byte, wire::kOutboundFrame> f{};
  wire::encode(std::span<std::byte, wire::kOutboundFrame>(f), e);
  const wire::Outbound o = wire::decode_outbound(f);
  EXPECT_EQ(o.type, wire::MsgType::Executed);
  EXPECT_EQ(o.side, Side::Sell);
  EXPECT_EQ(o.reason, 3);
  EXPECT_EQ(o.symbol, 9);
  EXPECT_EQ(o.seq, 1234567);
  EXPECT_EQ(o.cl_ord_id, 55);
  EXPECT_EQ(o.order_id, e.order_id);
  EXPECT_EQ(o.price, 10'050);
  EXPECT_EQ(o.qty, 12);
  EXPECT_EQ(o.leaves, 4);
  EXPECT_EQ(o.ts, 987654321);
}

TEST(Wire, TapeHeaderRoundTrip) {
  std::array<std::byte, wire::kTapeHeader> f{};
  wire::TapeHeader h;
  h.symbol = 3;
  h.frame_count = 1'000'000;
  h.min_price = 1;
  h.max_price = 1 << 20;
  wire::encode_header(std::span<std::byte, wire::kTapeHeader>(f), h);
  wire::TapeHeader d;
  ASSERT_TRUE(wire::decode_header(f, d));
  EXPECT_EQ(d.symbol, 3);
  EXPECT_EQ(d.frame_count, 1'000'000u);
  EXPECT_EQ(d.max_price, 1 << 20);
  f[0] = std::byte{'X'};
  EXPECT_FALSE(wire::decode_header(f, d));
}

TEST(Gateway, MapsClientIdsThroughTheOrderLifecycle) {
  MatchingEngine engine;
  engine.add_instrument({0, "GW", 1, 1000, 256, 0});
  Gateway gw(engine);
  std::vector<std::byte> out;
  gw.set_outbound(&out);

  EXPECT_TRUE(gw.on_frame(frame(enter(1, 10, Side::Buy, 99, 5))));
  EXPECT_TRUE(gw.on_frame(frame(enter(2, 10, Side::Buy, 98, 5))));  // same cl id, other owner: fine
  EXPECT_FALSE(gw.on_frame(frame(enter(1, 10, Side::Buy, 97, 5))));  // duplicate live id
  EXPECT_EQ(gw.stats().duplicate_ids, 1);
  EXPECT_EQ(gw.live_orders(), 2u);

  wire::Inbound rep;
  rep.type = wire::MsgType::Replace;
  rep.owner = 1;
  rep.cl_ord_id = 10;
  rep.price = 99;
  rep.qty = 2;
  EXPECT_TRUE(gw.on_frame(frame(rep)));
  EXPECT_EQ(engine.book(0)->best_bid_qty(), 2);

  EXPECT_TRUE(gw.on_frame(frame(enter(3, 1, Side::Sell, 99, 2))));  // fully fills owner 1
  EXPECT_EQ(gw.live_orders(), 1u);
  EXPECT_FALSE(gw.on_frame(frame(cancel(1, 10))));  // already filled
  EXPECT_EQ(gw.stats().unknown_refs, 1);
  EXPECT_TRUE(gw.on_frame(frame(cancel(2, 10))));
  EXPECT_EQ(gw.live_orders(), 0u);

  ASSERT_EQ(out.size() % wire::kOutboundFrame, 0u);
  std::vector<wire::MsgType> types;
  for (std::size_t at = 0; at < out.size(); at += wire::kOutboundFrame)
    types.push_back(wire::decode_outbound(std::span<const std::byte, wire::kOutboundFrame>(out.data() + at, wire::kOutboundFrame)).type);
  const std::vector<wire::MsgType> expected{wire::MsgType::Accepted, wire::MsgType::Accepted, wire::MsgType::Replaced,
                                            wire::MsgType::Accepted, wire::MsgType::Executed, wire::MsgType::Canceled};
  EXPECT_EQ(types, expected);
}

TEST(OrderIdMap, AgreesWithStdMapUnderChurn) {
  OrderIdMap m(8);  // small on purpose: exercises growth and long probe runs
  std::map<std::pair<OwnerId, ClOrdId>, OrderId> ref;
  std::mt19937_64 rng(99);
  for (int i = 0; i < 200'000; ++i) {
    const OwnerId owner = static_cast<OwnerId>(rng() % 4);
    const ClOrdId cl = static_cast<ClOrdId>(rng() % 2000);
    const auto key = std::make_pair(owner, cl);
    switch (rng() % 3) {
      case 0: {
        const OrderId v = static_cast<OrderId>(rng() >> 1);
        ASSERT_EQ(m.insert(owner, cl, v), ref.emplace(key, v).second);
        break;
      }
      case 1:
        ASSERT_EQ(m.erase(owner, cl), ref.erase(key) == 1);
        break;
      default: {
        const OrderId* got = m.find(owner, cl);
        const auto it = ref.find(key);
        ASSERT_EQ(got != nullptr, it != ref.end());
        if (got) ASSERT_EQ(*got, it->second);
      }
    }
    ASSERT_EQ(m.size(), ref.size());
  }
}
