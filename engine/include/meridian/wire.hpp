// Meridian Wire Protocol (MWP): fixed-size little-endian binary frames, OUCH/ITCH style.
//
// Inbound frames are 48 bytes, outbound frames 64 bytes. Fixed sizes mean a session reader never
// parses a length prefix and every field sits at a constant offset (single mov per field).
//
// Inbound (48 bytes)
//   off  size  EnterOrder 'O'        Cancel 'X'         Replace 'U'
//    0    1    type                  type               type
//    1    1    side                  -                  -
//    2    1    ord type              -                  -
//    3    1    tif                   -                  -
//    4    1    flags                 -                  -
//    5    1    stp                   -                  -
//    6    2    symbol                symbol             symbol
//    8    4    owner                 owner              owner
//   12    4    reserved              reserved           reserved
//   16    8    cl_ord_id             cl_ord_id          cl_ord_id (of the order to change)
//   24    8    price                 -                  new price
//   32    8    qty                   -                  new qty (leaves)
//   40    8    timestamp             timestamp          timestamp
//
// Outbound (64 bytes): type, side, reason, pad, symbol, pad, seq, cl_ord_id, order_id, price,
// qty, leaves, timestamp. Contra-party identity is deliberately not on the wire.
//
// Tape files (.mwp) are a 40-byte header followed by inbound frames:
//   "MWP1", u16 version, u16 symbol, u32 frame size, u32 reserved, u64 frame count,
//   i64 min price, i64 max price.
#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>

#include "meridian/types.hpp"

namespace meridian::wire {

static_assert(std::endian::native == std::endian::little, "MWP assumes a little-endian host");

inline constexpr std::size_t kInboundFrame = 48;
inline constexpr std::size_t kOutboundFrame = 64;
inline constexpr std::size_t kTapeHeader = 40;

enum class MsgType : std::uint8_t {
  EnterOrder = 'O',
  Cancel = 'X',
  Replace = 'U',
  Accepted = 'A',
  Executed = 'E',
  Canceled = 'C',
  Rejected = 'J',
  Replaced = 'R',
  CancelReject = 'K',
  ReplaceReject = 'Q',
  BookLevel = 'L',
};

struct Inbound {
  MsgType type = MsgType::EnterOrder;
  Side side = Side::Buy;
  OrdType ord_type = OrdType::Limit;
  Tif tif = Tif::Gtc;
  std::uint8_t flags = 0;
  Stp stp = Stp::None;
  SymbolId symbol = 0;
  OwnerId owner = 0;
  ClOrdId cl_ord_id = 0;
  Price price = 0;
  Qty qty = 0;
  Timestamp ts = 0;
};

struct TapeHeader {
  std::uint16_t version = 1;
  SymbolId symbol = 0;
  std::uint32_t frame_size = kInboundFrame;
  std::uint64_t frame_count = 0;
  Price min_price = 0;
  Price max_price = 0;
};

namespace detail {
template <class T>
inline void put(std::byte* p, T v) noexcept {
  std::memcpy(p, &v, sizeof(T));
}
template <class T>
inline T get(const std::byte* p) noexcept {
  T v;
  std::memcpy(&v, p, sizeof(T));
  return v;
}
}  // namespace detail

inline void encode(std::span<std::byte, kInboundFrame> f, const Inbound& m) noexcept {
  using detail::put;
  std::byte* p = f.data();
  std::memset(p, 0, kInboundFrame);
  put<std::uint8_t>(p + 0, static_cast<std::uint8_t>(m.type));
  put<std::uint8_t>(p + 1, static_cast<std::uint8_t>(m.side));
  put<std::uint8_t>(p + 2, static_cast<std::uint8_t>(m.ord_type));
  put<std::uint8_t>(p + 3, static_cast<std::uint8_t>(m.tif));
  put<std::uint8_t>(p + 4, m.flags);
  put<std::uint8_t>(p + 5, static_cast<std::uint8_t>(m.stp));
  put<std::uint16_t>(p + 6, m.symbol);
  put<std::int32_t>(p + 8, m.owner);
  put<std::int64_t>(p + 16, m.cl_ord_id);
  put<std::int64_t>(p + 24, m.price);
  put<std::int64_t>(p + 32, m.qty);
  put<std::int64_t>(p + 40, m.ts);
}

// Returns false for unknown message types or out-of-range enum fields.
inline bool decode(std::span<const std::byte> f, Inbound& m) noexcept {
  using detail::get;
  if (f.size() < kInboundFrame) return false;
  const std::byte* p = f.data();
  const auto type = get<std::uint8_t>(p + 0);
  if (type != 'O' && type != 'X' && type != 'U') return false;
  m.type = static_cast<MsgType>(type);
  m.symbol = get<std::uint16_t>(p + 6);
  m.owner = get<std::int32_t>(p + 8);
  m.cl_ord_id = get<std::int64_t>(p + 16);
  m.price = get<std::int64_t>(p + 24);
  m.qty = get<std::int64_t>(p + 32);
  m.ts = get<std::int64_t>(p + 40);
  if (m.type != MsgType::EnterOrder) return true;
  const auto side = get<std::uint8_t>(p + 1);
  const auto ord_type = get<std::uint8_t>(p + 2);
  const auto tif = get<std::uint8_t>(p + 3);
  const auto stp = get<std::uint8_t>(p + 5);
  if (side > 1 || ord_type > 1 || tif > 2 || stp > 3) return false;
  m.side = static_cast<Side>(side);
  m.ord_type = static_cast<OrdType>(ord_type);
  m.tif = static_cast<Tif>(tif);
  m.flags = get<std::uint8_t>(p + 4);
  m.stp = static_cast<Stp>(stp);
  return true;
}

constexpr MsgType outbound_type(EventType t) noexcept {
  switch (t) {
    case EventType::Accepted: return MsgType::Accepted;
    case EventType::Rejected: return MsgType::Rejected;
    case EventType::Trade: return MsgType::Executed;
    case EventType::Cancelled: return MsgType::Canceled;
    case EventType::Replaced: return MsgType::Replaced;
    case EventType::CancelRejected: return MsgType::CancelReject;
    case EventType::ReplaceRejected: return MsgType::ReplaceReject;
    case EventType::BookLevel: return MsgType::BookLevel;
  }
  return MsgType::Rejected;
}

inline void encode(std::span<std::byte, kOutboundFrame> f, const Event& e) noexcept {
  using detail::put;
  std::byte* p = f.data();
  std::memset(p, 0, kOutboundFrame);
  put<std::uint8_t>(p + 0, static_cast<std::uint8_t>(outbound_type(e.type)));
  put<std::uint8_t>(p + 1, static_cast<std::uint8_t>(e.side));
  put<std::uint8_t>(p + 2, e.reason);
  put<std::uint16_t>(p + 4, e.symbol);
  put<std::int64_t>(p + 8, e.seq);
  put<std::int64_t>(p + 16, e.cl_ord_id);
  put<std::int64_t>(p + 24, e.order_id);
  put<std::int64_t>(p + 32, e.price);
  put<std::int64_t>(p + 40, e.qty);
  put<std::int64_t>(p + 48, e.leaves);
  put<std::int64_t>(p + 56, e.ts);
}

struct Outbound {
  MsgType type;
  Side side;
  std::uint8_t reason;
  SymbolId symbol;
  std::int64_t seq;
  ClOrdId cl_ord_id;
  OrderId order_id;
  Price price;
  Qty qty;
  Qty leaves;
  Timestamp ts;
};

inline Outbound decode_outbound(std::span<const std::byte, kOutboundFrame> f) noexcept {
  using detail::get;
  const std::byte* p = f.data();
  return Outbound{static_cast<MsgType>(get<std::uint8_t>(p + 0)), static_cast<Side>(get<std::uint8_t>(p + 1)),
                  get<std::uint8_t>(p + 2), get<std::uint16_t>(p + 4), get<std::int64_t>(p + 8),
                  get<std::int64_t>(p + 16), get<std::int64_t>(p + 24), get<std::int64_t>(p + 32),
                  get<std::int64_t>(p + 40), get<std::int64_t>(p + 48), get<std::int64_t>(p + 56)};
}

inline void encode_header(std::span<std::byte, kTapeHeader> f, const TapeHeader& h) noexcept {
  using detail::put;
  std::byte* p = f.data();
  std::memset(p, 0, kTapeHeader);
  std::memcpy(p, "MWP1", 4);
  put<std::uint16_t>(p + 4, h.version);
  put<std::uint16_t>(p + 6, h.symbol);
  put<std::uint32_t>(p + 8, h.frame_size);
  put<std::uint64_t>(p + 16, h.frame_count);
  put<std::int64_t>(p + 24, h.min_price);
  put<std::int64_t>(p + 32, h.max_price);
}

inline bool decode_header(std::span<const std::byte> f, TapeHeader& h) noexcept {
  using detail::get;
  if (f.size() < kTapeHeader || std::memcmp(f.data(), "MWP1", 4) != 0) return false;
  const std::byte* p = f.data();
  h.version = get<std::uint16_t>(p + 4);
  h.symbol = get<std::uint16_t>(p + 6);
  h.frame_size = get<std::uint32_t>(p + 8);
  h.frame_count = get<std::uint64_t>(p + 16);
  h.min_price = get<std::int64_t>(p + 24);
  h.max_price = get<std::int64_t>(p + 32);
  return h.version == 1 && h.frame_size == kInboundFrame && h.min_price <= h.max_price;
}

}  // namespace meridian::wire
