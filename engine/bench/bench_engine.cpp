// meridian_bench: tick-to-trade style latency and throughput benchmark for the matching engine.
//
// Method
//   * A closed-loop flow generator produces a realistic message mix around a drifting mid:
//     passive adds (depth ~ geometric), cancels, cancel/replaces and marketable IOC orders,
//     keeping ~10-20k orders resting. Only the engine call is timed.
//   * Each call is bracketed with serialized TSC reads (lfence; rdtsc ... rdtscp); the fixed timer
//     overhead (min of empty brackets) is subtracted. Samples go into a log-linear histogram with
//     64 sub-buckets per octave (<1.6% relative error), reported per message type.
//   * A global operator new counts heap allocations inside the timed region (expected: zero).
//   * Throughput: the recorded command tape is replayed into fresh engines in a tight loop; all
//     replays must reproduce the same tape hash (determinism check).
//
//   meridian_bench [--ops N] [--json out.json]
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include <cpuid.h>
#include <x86intrin.h>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

#include "meridian/matching_engine.hpp"

// ---- allocation accounting ---------------------------------------------------------------------
namespace {
std::atomic<bool> g_counting{false};
std::atomic<std::int64_t> g_allocations{0};
}  // namespace

void* operator new(std::size_t n) {
  if (g_counting.load(std::memory_order_relaxed)) g_allocations.fetch_add(1, std::memory_order_relaxed);
  if (void* p = std::malloc(n ? n : 1)) return p;
  throw std::bad_alloc();
}
void* operator new[](std::size_t n) { return ::operator new(n); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

using namespace meridian;

namespace {

// ---- utilities -------------------------------------------------------------------------------

class Rng {  // xoshiro256**
 public:
  explicit Rng(std::uint64_t seed) {
    for (auto& s : s_) {
      seed += 0x9e3779b97f4a7c15ull;
      std::uint64_t z = seed;
      z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
      z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
      s = z ^ (z >> 31);
    }
  }
  std::uint64_t next() noexcept {
    const std::uint64_t r = std::rotl(s_[1] * 5, 7) * 9;
    const std::uint64_t t = s_[1] << 17;
    s_[2] ^= s_[0];
    s_[3] ^= s_[1];
    s_[1] ^= s_[2];
    s_[0] ^= s_[3];
    s_[2] ^= t;
    s_[3] = std::rotl(s_[3], 45);
    return r;
  }
  std::uint64_t below(std::uint64_t n) noexcept {
    return static_cast<std::uint64_t>((static_cast<unsigned __int128>(next()) * n) >> 64);
  }
  // Geometric number of failures before success, p = success probability.
  std::int64_t geometric(double p) noexcept {
    const double u = (static_cast<double>(next() >> 11) + 0.5) * 0x1.0p-53;
    return static_cast<std::int64_t>(std::floor(std::log(u) / std::log1p(-p)));
  }

 private:
  std::uint64_t s_[4];
};

inline std::uint64_t tsc_begin() noexcept {
  _mm_lfence();
  return __rdtsc();
}
inline std::uint64_t tsc_end() noexcept {
  unsigned aux;
  const std::uint64_t t = __rdtscp(&aux);
  _mm_lfence();
  return t;
}

double calibrate_tsc_ghz() {
  using clock = std::chrono::steady_clock;
  const auto c0 = clock::now();
  const std::uint64_t t0 = tsc_begin();
  while (clock::now() - c0 < std::chrono::milliseconds(300)) {
  }
  const std::uint64_t t1 = tsc_end();
  const double ns = std::chrono::duration<double, std::nano>(clock::now() - c0).count();
  return static_cast<double>(t1 - t0) / ns;
}

std::string cpu_brand() {
  unsigned regs[12] = {};
  if (__get_cpuid_max(0x80000000, nullptr) < 0x80000004) return "unknown";
  for (unsigned i = 0; i < 3; ++i) __get_cpuid(0x80000002 + i, &regs[i * 4], &regs[i * 4 + 1], &regs[i * 4 + 2], &regs[i * 4 + 3]);
  std::string s(reinterpret_cast<const char*>(regs), 48);
  s.erase(s.find_last_not_of(std::string(" \0", 2)) + 1);
  s.erase(0, s.find_first_not_of(' '));
  return s;
}

// Log-linear histogram: exact below 64, then 64 sub-buckets per power of two.
class Histogram {
 public:
  static constexpr int kBits = 6;
  static constexpr std::uint64_t kSub = 1u << kBits;

  void record(std::uint64_t v) noexcept {
    ++counts_[index(v)];
    ++total_;
    max_ = std::max(max_, v);
    sum_ += static_cast<double>(v);
  }
  std::uint64_t count() const noexcept { return total_; }
  std::uint64_t max() const noexcept { return max_; }
  double mean() const noexcept { return total_ ? sum_ / static_cast<double>(total_) : 0.0; }
  double percentile(double p) const noexcept {
    if (total_ == 0) return 0;
    const auto target = static_cast<std::uint64_t>(std::ceil(p / 100.0 * static_cast<double>(total_)));
    std::uint64_t seen = 0;
    for (std::size_t i = 0; i < counts_.size(); ++i) {
      seen += counts_[i];
      if (seen >= std::max<std::uint64_t>(target, 1)) return midpoint(i);
    }
    return static_cast<double>(max_);
  }

 private:
  static std::size_t index(std::uint64_t v) noexcept {
    if (v < kSub) return static_cast<std::size_t>(v);
    const int msb = 63 - std::countl_zero(v);
    const std::uint64_t top = v >> (msb - kBits);
    return static_cast<std::size_t>(msb - kBits + 1) * kSub + static_cast<std::size_t>(top - kSub);
  }
  static double midpoint(std::size_t i) noexcept {
    if (i < kSub) return static_cast<double>(i);
    const std::size_t octave = i / kSub;
    const std::size_t sub = i % kSub;
    const double low = std::ldexp(static_cast<double>(kSub + sub), static_cast<int>(octave) - 1);
    return low + std::ldexp(1.0, static_cast<int>(octave) - 1) / 2.0;
  }
  std::array<std::uint64_t, 64 * kSub> counts_{};
  std::uint64_t total_ = 0;
  std::uint64_t max_ = 0;
  double sum_ = 0;
};

// ---- flow --------------------------------------------------------------------------------------

using Command = std::variant<NewOrder, CancelOrder, ReplaceOrder>;
enum Kind : int { kPassive = 0, kAggressive, kCancel, kReplace, kKinds };
constexpr const char* kKindNames[kKinds] = {"add_passive", "add_marketable", "cancel", "replace"};

class Flow {
 public:
  Flow(std::uint64_t seed, Price mid) : rng_(seed), mid_(mid) {
    live_.reserve(1 << 16);
    pos_.reserve(1 << 17);
  }

  Command next(const OrderBook& book, Kind& kind) {
    ts_ += 250;
    if (++steps_ % 64 == 0) mid_ += static_cast<Price>(rng_.below(3)) - 1;
    // Exchange-like mix: ~50% adds, ~35% cancels, ~10% replaces, ~5% marketable.
    const int r = static_cast<int>(rng_.below(100));
    const bool crowded = live_.size() > 30'000;
    const bool sparse = live_.size() < 10'000;
    if (!live_.empty() && !sparse && (r < 35 || crowded)) {
      kind = kCancel;
      const OrderId id = live_[rng_.below(live_.size())];
      return CancelOrder{ts_, id, owner_of(id), 0};
    }
    if (!live_.empty() && r < 45) {
      const OrderId id = live_[rng_.below(live_.size())];
      OrderView v;
      if (book.find(id, v)) {
        kind = kReplace;
        const Price shift = static_cast<Price>(rng_.below(5)) - 2;
        const Qty qty = 1 + static_cast<Qty>(rng_.below(100));
        return ReplaceOrder{ts_, id, v.price + shift, qty, v.owner, 0};
      }
    }
    NewOrder o;
    o.ts = ts_;
    o.cl_ord_id = cl_++;
    o.owner = 1 + static_cast<OwnerId>(rng_.below(8));
    o.side = rng_.below(2) ? Side::Buy : Side::Sell;
    if (r < 50) {
      kind = kAggressive;
      o.tif = Tif::Ioc;
      const Price through = static_cast<Price>(rng_.below(4));
      o.price = o.side == Side::Buy ? mid_ + through : mid_ - through;
      o.qty = 1 + std::min<Qty>(rng_.geometric(0.02), 400);
    } else {
      kind = kPassive;
      const Price depth = 1 + std::min<Price>(rng_.geometric(0.3), 50);
      o.price = o.side == Side::Buy ? mid_ - depth : mid_ + depth;
      o.qty = 1 + static_cast<Qty>(rng_.below(100));
      if (rng_.below(20) == 0) o.flags = kPostOnly;
    }
    return o;
  }

  void observe(std::span<const Event> events) {
    for (const Event& e : events) {
      switch (e.type) {
        case EventType::Accepted:
          add(e.order_id, e.owner);
          break;
        case EventType::Trade:
          if (e.leaves == 0) remove(e.order_id);
          if (e.contra_leaves == 0) remove(e.contra_order_id);
          break;
        case EventType::Cancelled:
          remove(e.order_id);
          break;
        default:
          break;
      }
    }
  }

  std::size_t live() const noexcept { return live_.size(); }

 private:
  struct Slot {
    std::size_t index;
    OwnerId owner;
  };
  OwnerId owner_of(OrderId id) const { return pos_.at(id).owner; }
  void add(OrderId id, OwnerId owner) {
    pos_[id] = Slot{live_.size(), owner};
    live_.push_back(id);
  }
  void remove(OrderId id) {
    const auto it = pos_.find(id);
    if (it == pos_.end()) return;
    const std::size_t i = it->second.index;
    const OrderId last = live_.back();
    live_[i] = last;
    pos_[last].index = i;
    live_.pop_back();
    pos_.erase(id);
  }

  Rng rng_;
  Price mid_;
  ClOrdId cl_ = 1;
  Timestamp ts_ = 0;
  std::int64_t steps_ = 0;
  std::vector<OrderId> live_;
  std::unordered_map<OrderId, Slot> pos_;
};

inline void apply(MatchingEngine& e, const Command& c) {
  switch (c.index()) {
    case 0: e.submit(*std::get_if<NewOrder>(&c)); break;
    case 1: e.cancel(*std::get_if<CancelOrder>(&c)); break;
    default: e.replace(*std::get_if<ReplaceOrder>(&c)); break;
  }
}

InstrumentSpec bench_spec() {
  InstrumentSpec s;
  s.id = 0;
  s.name = "BENCH";
  s.min_price = 1;
  s.max_price = Price{1} << 20;
  s.order_capacity = std::size_t{1} << 20;
  s.reference_price = 500'000;
  return s;
}

void pin_and_boost() {
#ifdef _WIN32
  SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
  SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
  SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR{1} << 2);
#endif
}

struct LatencyRun {
  std::array<Histogram, kKinds> by_kind{};
  Histogram all{};
  std::int64_t allocations = 0;
  std::uint64_t hash = 0;
  std::int64_t trades = 0;
  double avg_live = 0;
};

LatencyRun run_latency(bool with_risk, std::int64_t warmup, std::int64_t ops, std::uint64_t overhead,
                       std::vector<Command>* tape) {
  RiskEngine risk(64, 1);
  RiskLimits lim;
  lim.max_order_qty = 1'000'000;
  lim.price_collar = 10'000;
  lim.max_long = lim.max_short = 1'000'000'000;
  for (OwnerId o = 1; o <= 8; ++o) risk.register_participant(o, lim);
  MatchingEngine engine(with_risk ? &risk : nullptr, 1 << 14);
  const OrderBook& book = engine.add_instrument(bench_spec());
  Flow flow(42, 500'000);
  LatencyRun run;
  double live_sum = 0;
  for (std::int64_t i = 0; i < warmup + ops; ++i) {
    Kind kind;
    const Command c = flow.next(book, kind);
    if (tape) tape->push_back(c);
    const bool measure = i >= warmup;
    g_counting.store(measure, std::memory_order_relaxed);
    const std::uint64_t t0 = tsc_begin();
    apply(engine, c);
    const std::uint64_t t1 = tsc_end();
    g_counting.store(false, std::memory_order_relaxed);
    if (measure) {
      const std::uint64_t d = t1 - t0 > overhead ? t1 - t0 - overhead : 0;
      run.by_kind[kind].record(d);
      run.all.record(d);
      live_sum += static_cast<double>(flow.live());
    }
    flow.observe(engine.events());
  }
  run.allocations = g_allocations.exchange(0);
  run.hash = engine.tape_hash();
  run.trades = engine.trade_count();
  run.avg_live = live_sum / static_cast<double>(ops);
  return run;
}

void print_hist(std::FILE* f, const char* name, const Histogram& h, double ghz, bool last) {
  auto ns = [&](double cycles) { return cycles / ghz; };
  std::fprintf(f,
               "      \"%s\": {\"count\": %llu, \"mean_ns\": %.1f, \"p50_ns\": %.1f, \"p90_ns\": %.1f, \"p99_ns\": %.1f, "
               "\"p99_9_ns\": %.1f, \"p99_99_ns\": %.1f, \"max_ns\": %.1f}%s\n",
               name, static_cast<unsigned long long>(h.count()), ns(h.mean()), ns(h.percentile(50)), ns(h.percentile(90)),
               ns(h.percentile(99)), ns(h.percentile(99.9)), ns(h.percentile(99.99)), ns(static_cast<double>(h.max())),
               last ? "" : ",");
}

void print_run(std::FILE* f, const char* name, const LatencyRun& r, double ghz, bool last) {
  std::fprintf(f, "    \"%s\": {\n", name);
  std::fprintf(f, "      \"heap_allocations_in_timed_region\": %lld,\n", static_cast<long long>(r.allocations));
  std::fprintf(f, "      \"avg_resting_orders\": %.0f,\n", r.avg_live);
  std::fprintf(f, "      \"trades\": %lld,\n", static_cast<long long>(r.trades));
  print_hist(f, "all", r.all, ghz, false);
  for (int k = 0; k < kKinds; ++k) print_hist(f, kKindNames[k], r.by_kind[k], ghz, k + 1 == kKinds);
  std::fprintf(f, "    }%s\n", last ? "" : ",");
}

}  // namespace

int main(int argc, char** argv) {
  std::int64_t ops = 3'000'000;
  const char* json_path = nullptr;
  for (int i = 1; i + 1 < argc; i += 2) {
    if (std::strcmp(argv[i], "--ops") == 0) ops = std::atoll(argv[i + 1]);
    else if (std::strcmp(argv[i], "--json") == 0) json_path = argv[i + 1];
  }
  const std::int64_t warmup = 500'000;
  pin_and_boost();
  const double ghz = calibrate_tsc_ghz();

  std::uint64_t overhead = ~std::uint64_t{0};
  for (int i = 0; i < 100'000; ++i) {
    const std::uint64_t t0 = tsc_begin();
    const std::uint64_t t1 = tsc_end();
    overhead = std::min(overhead, t1 - t0);
  }

  std::vector<Command> tape;
  tape.reserve(static_cast<std::size_t>(warmup + ops));
  std::fprintf(stderr, "latency run (no risk), %lld ops...\n", static_cast<long long>(ops));
  const LatencyRun plain = run_latency(false, warmup, ops, overhead, &tape);
  std::fprintf(stderr, "latency run (risk gate on)...\n");
  const LatencyRun risky = run_latency(true, warmup, ops, overhead, nullptr);

  std::fprintf(stderr, "throughput replay of %zu commands...\n", tape.size());
  double best_mps = 0;
  bool deterministic = true;
  for (int rep = 0; rep < 3; ++rep) {
    MatchingEngine engine(nullptr, 1 << 14);
    engine.add_instrument(bench_spec());
    const auto t0 = std::chrono::steady_clock::now();
    for (const Command& c : tape) apply(engine, c);
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    best_mps = std::max(best_mps, static_cast<double>(tape.size()) / secs);
    deterministic = deterministic && engine.tape_hash() == plain.hash;
  }

  std::FILE* f = json_path ? std::fopen(json_path, "w") : stdout;
  if (f == nullptr) {
    std::fprintf(stderr, "cannot open %s\n", json_path);
    return 1;
  }
  std::fprintf(f, "{\n");
  std::fprintf(f, "  \"cpu\": \"%s\",\n", cpu_brand().c_str());
  std::fprintf(f, "  \"tsc_ghz\": %.4f,\n", ghz);
  std::fprintf(f, "  \"timer_overhead_cycles\": %llu,\n", static_cast<unsigned long long>(overhead));
  std::fprintf(f, "  \"compiler\": \"clang %s\",\n", __clang_version__);
  std::fprintf(f, "  \"warmup_ops\": %lld,\n  \"measured_ops\": %lld,\n", static_cast<long long>(warmup),
               static_cast<long long>(ops));
  std::fprintf(f, "  \"latency\": {\n");
  print_run(f, "engine", plain, ghz, false);
  print_run(f, "engine_with_risk", risky, ghz, true);
  std::fprintf(f, "  },\n");
  std::fprintf(f, "  \"throughput\": {\"commands\": %zu, \"best_msgs_per_sec\": %.0f, \"deterministic_replay\": %s},\n",
               tape.size(), best_mps, deterministic ? "true" : "false");
  std::fprintf(f, "  \"tape_hash\": \"0x%016llx\"\n", static_cast<unsigned long long>(plain.hash));
  std::fprintf(f, "}\n");
  if (f != stdout) std::fclose(f);
  return deterministic && plain.allocations == 0 ? 0 : 3;
}
