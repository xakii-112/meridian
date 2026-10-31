"""Builds dashboard/index.html from the template and the measured results.

    python dashboard/build.py [--skip-tests]

Inputs: results/bench.json (meridian_bench), results/study.json (meridian_research.study).
Test counts are not typed in: the C++ suites (plain and, if built, ASan/UBSan) and the Python
suite are run here and their machine-readable reports are parsed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _exe(build_dir: str) -> Path | None:
    for name in ("meridian_tests.exe", "meridian_tests"):
        p = ROOT / "engine" / build_dir / name
        if p.exists():
            return p
    return None


def _toolchain_env() -> dict:
    """Sanitizer builds load the compiler's runtime DLLs, so put the toolchain's bin dir on PATH."""
    env = dict(os.environ)
    dirs = [Path(p).parent for p in (shutil.which("clang++"),) if p]
    local = os.environ.get("LOCALAPPDATA")
    if local:
        dirs += sorted(Path(local, "Microsoft", "WinGet", "Packages").glob("*LLVM-MinGW*/*/bin"))
    env["PATH"] = os.pathsep.join([str(d) for d in dirs] + [env.get("PATH", "")])
    return env


def gtest_counts(exe: Path | None) -> dict | None:
    if exe is None:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        report = Path(tmp) / "gtest.json"
        subprocess.run([str(exe), "--gtest_brief=1", f"--gtest_output=json:{report}"], capture_output=True,
                       timeout=900, env=_toolchain_env())
        if not report.exists():
            return None
        data = json.loads(report.read_text())
    return {"total": data["tests"], "passed": data["tests"] - data["failures"] - data.get("errors", 0)}


def pytest_counts() -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        report = Path(tmp) / "pytest.xml"
        subprocess.run([sys.executable, "-m", "pytest", "-q", f"--junitxml={report}"], cwd=ROOT / "research",
                       capture_output=True, timeout=900)
        suite = ET.parse(report).getroot()
        suite = suite if suite.tag == "testsuite" else suite.find("testsuite")
    total = int(suite.get("tests"))
    bad = int(suite.get("failures")) + int(suite.get("errors")) + int(suite.get("skipped"))
    return {"total": total, "passed": total - bad}


def fuzz_shape() -> tuple[int, int]:
    src = (ROOT / "engine" / "tests" / "test_fuzz.cpp").read_text()
    lo, hi = map(int, re.search(r"Range<std::uint64_t>\((\d+),\s*(\d+)\)", src).groups())
    steps = int(re.search(r"for \(int i = 0; i < (\d+); \+\+i\)", src).group(1))
    return hi - lo, steps


def compact_study(study: dict) -> dict:
    keep = ("generated_at", "runtime_s", "sessions", "seeds", "market", "maker", "calibration", "glft", "tuning",
            "latency", "replay")
    out = {k: study[k] for k in keep}
    out["comparison"] = [{k: v for k, v in c.items() if k != "inventory_hist"} for c in study["comparison"]]
    out["trace"] = {k: study["trace"][k] for k in ("zoom", "zoom_fills", "jump_ticks")}
    return out


def compact_real(real: dict) -> dict:
    keep = ("generated_at", "session", "recording", "validation", "split_s", "stats", "hawkes", "lead_lag",
            "lead_weight", "realized_spreads", "feed_delays", "fill_intensity", "model", "comparison", "champion", "latency",
            "fees_bps")
    out = {k: real[k] for k in keep}
    out["trace"] = {k: real["trace"][k] for k in ("zoom", "zoom_fills", "zoom_move_usd")}
    return out


def _finite(x):
    """JSON has no NaN/Infinity (JSON.parse rejects them): map them to null."""
    if isinstance(x, float) and (x != x or x in (float("inf"), float("-inf"))):
        return None
    if isinstance(x, dict):
        return {k: _finite(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_finite(v) for v in x]
    return x


def paper_sessions() -> list[dict]:
    """Live paper sessions, each paired with the offline replay of its own recording."""
    folder = ROOT / "results" / "paper"
    runs = [json.loads(p.read_text()) for p in sorted(folder.glob("paper_*.json"))] if folder.exists() else []
    key = lambda r: Path(r["recording"]).name if r.get("recording") else None
    replays = {key(r): r for r in runs if r["mode"] == "replay" and r.get("recording")}
    out = []
    for r in (r for r in runs if r["mode"] == "live"):
        rep = replays.get(key(r))
        reproduced = rep is not None and rep["fills_arr"] == r["fills_arr"] and rep["gross_pnl_usd"] == r["gross_pnl_usd"]
        out.append({"label": r["label"], "minutes": r["hours"] * 60, "fills": r["fills"], "volume_btc": r["volume_btc"],
                    "notional_usd": r["notional_usd"], "gross_usd": r["gross_pnl_usd"], "net10_usd": r["net_pnl_usd"]["10.0"],
                    "edge_per_btc": r["gross_edge_per_btc"], "markout_1s": r["markout_usd_per_btc"]["1.0"],
                    "latency_s": r["config"]["latency_s"], "msgs_per_sec": r["msgs_per_sec"],
                    "replay_reproduces_live": reproduced,
                    "replay_matches_backtest": bool(rep and rep.get("verify") and rep["verify"]["identical"])})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-tests", action="store_true", help="reuse counts from the previous build")
    ap.add_argument("--real", type=Path, default=ROOT / "results" / "real" / "real_study.json")
    args = ap.parse_args()
    bench = json.loads((ROOT / "results" / "bench.json").read_text())
    study = json.loads((ROOT / "results" / "study.json").read_text())
    cache = ROOT / "results" / "tests.json"
    if args.skip_tests and cache.exists():
        tests = json.loads(cache.read_text())
    else:
        seeds, steps = fuzz_shape()
        tests = {"cpp": gtest_counts(_exe("build")), "asan": gtest_counts(_exe("build-asan")), "python": pytest_counts(),
                 "fuzz_seeds": seeds, "fuzz_steps": steps}
        cache.write_text(json.dumps(tests, indent=1))
    real = compact_real(json.loads(args.real.read_text())) if args.real.exists() else None
    data = json.dumps(_finite({"bench": bench, "study": compact_study(study), "tests": tests, "real": real,
                               "paper": paper_sessions()}), separators=(",", ":"), allow_nan=False)
    data = data.replace("</", "<\\/")
    html = (ROOT / "dashboard" / "template.html").read_text(encoding="utf-8").replace("__MERIDIAN_DATA__", data)
    out = ROOT / "dashboard" / "index.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({len(html) / 1024:.0f} KiB); tests: {json.dumps(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
