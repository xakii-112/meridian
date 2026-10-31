// meridian_replay: replays an MWP tape (.mwp) through the gateway + matching engine and prints a
// JSON summary, including the tape hash used for cross-implementation determinism checks.
//
//   meridian_replay <tape.mwp> [--expect-hash 0x...] [--outbound out.bin]
//
// Exit status: 0 on success, 1 on I/O or format errors, 2 if --expect-hash does not match.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "meridian/gateway.hpp"
#include "meridian/matching_engine.hpp"
#include "meridian/wire.hpp"

using namespace meridian;

namespace {

std::string json_escape(const char* s) {
  std::string out;
  for (; *s; ++s) {
    if (*s == '"' || *s == '\\') out += '\\';
    out += *s;
  }
  return out;
}

bool read_file(const char* path, std::vector<std::byte>& out) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return false;
  in.seekg(0, std::ios::end);
  const auto size = static_cast<std::size_t>(in.tellg());
  in.seekg(0);
  out.resize(size);
  in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(size));
  return static_cast<bool>(in);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <tape.mwp> [--expect-hash 0x...] [--outbound out.bin]\n", argv[0]);
    return 1;
  }
  const char* path = argv[1];
  const char* outbound_path = nullptr;
  bool check_hash = false;
  std::uint64_t expected = 0;
  for (int i = 2; i + 1 < argc; i += 2) {
    if (std::strcmp(argv[i], "--expect-hash") == 0) {
      check_hash = true;
      expected = std::strtoull(argv[i + 1], nullptr, 0);
    } else if (std::strcmp(argv[i], "--outbound") == 0) {
      outbound_path = argv[i + 1];
    }
  }

  std::vector<std::byte> tape;
  if (!read_file(path, tape)) {
    std::fprintf(stderr, "cannot read %s\n", path);
    return 1;
  }
  wire::TapeHeader header;
  if (!wire::decode_header(tape, header)) {
    std::fprintf(stderr, "%s: not an MWP1 tape\n", path);
    return 1;
  }
  const std::size_t body = tape.size() - wire::kTapeHeader;
  if (body % wire::kInboundFrame != 0 || body / wire::kInboundFrame != header.frame_count) {
    std::fprintf(stderr, "%s: frame count mismatch\n", path);
    return 1;
  }

  MatchingEngine engine;
  InstrumentSpec spec;
  spec.id = header.symbol;
  spec.name = "TAPE";
  spec.min_price = header.min_price;
  spec.max_price = header.max_price;
  spec.order_capacity = std::size_t{1} << 20;
  OrderBook& book = engine.add_instrument(spec);
  Gateway gateway(engine, std::size_t{1} << 16);
  std::vector<std::byte> outbound;
  if (outbound_path != nullptr) {
    outbound.reserve(header.frame_count * 2 * wire::kOutboundFrame);
    gateway.set_outbound(&outbound);
  }

  std::int64_t accepted = 0, rejected = 0, cancelled = 0, replaced = 0, op_rejects = 0;
  const auto t0 = std::chrono::steady_clock::now();
  for (std::size_t k = 0; k < header.frame_count; ++k) {
    gateway.on_frame(std::span<const std::byte>(tape.data() + wire::kTapeHeader + k * wire::kInboundFrame, wire::kInboundFrame));
    for (const Event& e : engine.events()) {
      accepted += e.type == EventType::Accepted;
      rejected += e.type == EventType::Rejected;
      cancelled += e.type == EventType::Cancelled;
      replaced += e.type == EventType::Replaced;
      op_rejects += e.type == EventType::CancelRejected || e.type == EventType::ReplaceRejected;
    }
  }
  const auto t1 = std::chrono::steady_clock::now();
  const double secs = std::chrono::duration<double>(t1 - t0).count();

  if (outbound_path != nullptr) {
    std::ofstream out(outbound_path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(outbound.data()), static_cast<std::streamsize>(outbound.size()));
  }

  const bool ok = !check_hash || engine.tape_hash() == expected;
  std::printf("{\n");
  std::printf("  \"tape\": \"%s\",\n", json_escape(path).c_str());
  std::printf("  \"frames\": %llu,\n", static_cast<unsigned long long>(header.frame_count));
  std::printf("  \"accepted\": %lld,\n  \"rejected\": %lld,\n  \"cancelled\": %lld,\n  \"replaced\": %lld,\n",
              static_cast<long long>(accepted), static_cast<long long>(rejected), static_cast<long long>(cancelled),
              static_cast<long long>(replaced));
  std::printf("  \"op_rejects\": %lld,\n", static_cast<long long>(op_rejects));
  std::printf("  \"gateway_unknown_refs\": %lld,\n", static_cast<long long>(gateway.stats().unknown_refs));
  std::printf("  \"trades\": %lld,\n  \"volume\": %lld,\n", static_cast<long long>(engine.trade_count()),
              static_cast<long long>(engine.traded_volume()));
  std::printf("  \"resting_orders\": %zu,\n", book.active_orders());
  if (book.has_bid()) std::printf("  \"best_bid\": %lld,\n", static_cast<long long>(book.best_bid()));
  if (book.has_ask()) std::printf("  \"best_ask\": %lld,\n", static_cast<long long>(book.best_ask()));
  std::printf("  \"tape_hash\": \"0x%016llx\",\n", static_cast<unsigned long long>(engine.tape_hash()));
  if (check_hash) std::printf("  \"expected_hash\": \"0x%016llx\",\n  \"hash_match\": %s,\n",
                              static_cast<unsigned long long>(expected), ok ? "true" : "false");
  std::printf("  \"elapsed_ms\": %.3f,\n", secs * 1e3);
  std::printf("  \"frames_per_sec\": %.0f\n", static_cast<double>(header.frame_count) / secs);
  std::printf("}\n");
  return ok ? 0 : 2;
}
