// Core vocabulary types for the Meridian matching engine.
//
// Prices are integer ticks and quantities are integer lots: floating point never touches the
// matching path. Timestamps are supplied by the sequencer in front of the engine; the engine
// never reads a clock, which keeps it a pure, deterministic state machine (same input tape ->
// same output tape, bit for bit).
#pragma once

#include <cstdint>
#include <string_view>

namespace meridian {

using Price = std::int64_t;
using Qty = std::int64_t;
using OrderId = std::int64_t;
using ClOrdId = std::int64_t;
using OwnerId = std::int32_t;
using SymbolId = std::uint16_t;
using Timestamp = std::int64_t;  // nanoseconds

enum class Side : std::uint8_t { Buy = 0, Sell = 1 };
constexpr Side opposite(Side s) noexcept { return s == Side::Buy ? Side::Sell : Side::Buy; }

enum class OrdType : std::uint8_t { Limit = 0, Market = 1 };
enum class Tif : std::uint8_t { Gtc = 0, Ioc = 1, Fok = 2 };

// Order flags (bit set).
inline constexpr std::uint8_t kPostOnly = 1;

// Self-trade prevention instruction carried by the aggressing order.
enum class Stp : std::uint8_t { None = 0, CancelResting = 1, CancelAggressor = 2, CancelBoth = 3 };

enum class RejectReason : std::uint8_t {
  None = 0,
  InvalidQuantity,
  InvalidSide,
  PriceOutOfRange,
  InvalidTimeInForce,
  InvalidOrderType,
  InvalidFlags,
  PostOnlyWouldCross,
  OrderCapacityExceeded,
  UnknownSymbol,
  UnknownOrder,
  NotOwner,
  RiskUnknownParticipant,
  RiskKillSwitch,
  RiskThrottled,
  RiskMaxOrderQuantity,
  RiskPriceCollar,
  RiskMaxNotional,
  RiskMaxOpenOrders,
  RiskPositionLimit,
};

enum class CancelReason : std::uint8_t {
  Requested = 0,
  ImmediateOrCancel,
  FillOrKill,
  SelfTradePrevention,
  MassCancel,
  KillSwitch,
};

enum class EventType : std::uint8_t {
  Accepted = 0,
  Rejected,
  Trade,
  Cancelled,
  Replaced,
  CancelRejected,
  ReplaceRejected,
  BookLevel,
};

struct NewOrder {
  Timestamp ts = 0;
  ClOrdId cl_ord_id = 0;
  Price price = 0;  // ignored for market orders
  Qty qty = 0;
  OwnerId owner = 0;
  SymbolId symbol = 0;
  Side side = Side::Buy;
  OrdType type = OrdType::Limit;
  Tif tif = Tif::Gtc;
  std::uint8_t flags = 0;
  Stp stp = Stp::None;
};

struct CancelOrder {
  Timestamp ts = 0;
  OrderId order_id = 0;
  OwnerId owner = 0;
  SymbolId symbol = 0;
};

// Replace sets a new limit price and a new *leaves* quantity. Reducing quantity at the same price
// keeps queue priority; any other change re-queues the order (and may make it marketable).
struct ReplaceOrder {
  Timestamp ts = 0;
  OrderId order_id = 0;
  Price new_price = 0;
  Qty new_qty = 0;
  OwnerId owner = 0;
  SymbolId symbol = 0;
};

// One fixed-size record type for every engine output, so the output tape is a flat array.
//
//   Accepted        order_id/cl_ord_id/owner/side, price (0 for market), qty = leaves = order qty
//   Rejected        cl_ord_id/owner/side/price/qty of the request, reason = RejectReason
//   Trade           order_* = resting (maker) order, contra_* = aggressor (taker) order,
//                   side = maker side, price = maker price, qty = fill, leaves/contra_leaves after fill
//   Cancelled       order_*, qty = leaves that were cancelled, reason = CancelReason
//   Replaced        order_*, price/qty/leaves = new values, contra_leaves = leaves before replace
//   Cancel/ReplaceRejected  order_id/owner, reason = RejectReason
//   BookLevel       side/price, qty = aggregate resting qty at that price, leaves = order count
struct Event {
  std::int64_t seq = 0;
  Timestamp ts = 0;
  OrderId order_id = 0;
  ClOrdId cl_ord_id = 0;
  Price price = 0;
  Qty qty = 0;
  Qty leaves = 0;
  OrderId contra_order_id = 0;
  ClOrdId contra_cl_ord_id = 0;
  Qty contra_leaves = 0;
  OwnerId owner = 0;
  OwnerId contra_owner = 0;
  SymbolId symbol = 0;
  EventType type = EventType::Accepted;
  Side side = Side::Buy;
  std::uint8_t reason = 0;
};

constexpr std::string_view to_string(EventType t) noexcept {
  switch (t) {
    case EventType::Accepted: return "Accepted";
    case EventType::Rejected: return "Rejected";
    case EventType::Trade: return "Trade";
    case EventType::Cancelled: return "Cancelled";
    case EventType::Replaced: return "Replaced";
    case EventType::CancelRejected: return "CancelRejected";
    case EventType::ReplaceRejected: return "ReplaceRejected";
    case EventType::BookLevel: return "BookLevel";
  }
  return "?";
}

constexpr std::string_view to_string(RejectReason r) noexcept {
  switch (r) {
    case RejectReason::None: return "None";
    case RejectReason::InvalidQuantity: return "InvalidQuantity";
    case RejectReason::InvalidSide: return "InvalidSide";
    case RejectReason::PriceOutOfRange: return "PriceOutOfRange";
    case RejectReason::InvalidTimeInForce: return "InvalidTimeInForce";
    case RejectReason::InvalidOrderType: return "InvalidOrderType";
    case RejectReason::InvalidFlags: return "InvalidFlags";
    case RejectReason::PostOnlyWouldCross: return "PostOnlyWouldCross";
    case RejectReason::OrderCapacityExceeded: return "OrderCapacityExceeded";
    case RejectReason::UnknownSymbol: return "UnknownSymbol";
    case RejectReason::UnknownOrder: return "UnknownOrder";
    case RejectReason::NotOwner: return "NotOwner";
    case RejectReason::RiskUnknownParticipant: return "RiskUnknownParticipant";
    case RejectReason::RiskKillSwitch: return "RiskKillSwitch";
    case RejectReason::RiskThrottled: return "RiskThrottled";
    case RejectReason::RiskMaxOrderQuantity: return "RiskMaxOrderQuantity";
    case RejectReason::RiskPriceCollar: return "RiskPriceCollar";
    case RejectReason::RiskMaxNotional: return "RiskMaxNotional";
    case RejectReason::RiskMaxOpenOrders: return "RiskMaxOpenOrders";
    case RejectReason::RiskPositionLimit: return "RiskPositionLimit";
  }
  return "?";
}

}  // namespace meridian
