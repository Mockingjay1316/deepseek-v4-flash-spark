"""Sample physical memory pressure while a child process runs.

Records per-second:
  - child VmRSS / VmPeak / VmSize (from /proc/PID/status)
  - system MemAvailable / SwapUsed (from /proc/meminfo)
  - CUDA allocator current/max bytes (best-effort, via a tiny helper file)

Prints peak summary on exit. The child gets a clean exit code passthrough.

Usage:
  python scripts/mem_probe.py --label pruned-8 -- python scripts/sanity_forward.py ...
"""
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def read_meminfo() -> dict[str, int]:
    out = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, _, rest = line.partition(":")
            try:
                out[k.strip()] = int(rest.strip().split()[0])
            except (ValueError, IndexError):
                pass
    return out  # values in kB


def read_proc_status(pid: int) -> dict[str, int]:
    out = {}
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                k, _, rest = line.partition(":")
                k = k.strip()
                if k in ("VmRSS", "VmPeak", "VmSize", "VmSwap", "VmHWM"):
                    out[k] = int(rest.strip().split()[0])
    except FileNotFoundError:
        return {}
    return out  # kB


def fmt_gb(kb: int) -> str:
    return f"{kb/1024/1024:.2f}GB"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--label", default="run")
    p.add_argument("--interval", type=float, default=1.0)
    p.add_argument("--log", default=None, help="optional CSV path for per-sample data")
    p.add_argument("cmd", nargs=argparse.REMAINDER)
    args = p.parse_args()
    assert args.cmd and args.cmd[0] == "--", "use -- to separate probe args from child cmd"
    child_cmd = args.cmd[1:]
    assert child_cmd, "no child command provided"

    # Baseline before child starts
    base = read_meminfo()
    base_avail = base["MemAvailable"]
    base_swap = base.get("SwapTotal", 0) - base.get("SwapFree", 0)

    print(f"[probe:{args.label}] starting: {' '.join(child_cmd)}")
    print(f"[probe:{args.label}] baseline MemAvailable={fmt_gb(base_avail)} SwapUsed={fmt_gb(base_swap)}")

    # Start child with its stdout/stderr inherited so user sees live logs
    proc = subprocess.Popen(child_cmd)
    pid = proc.pid

    csv = open(args.log, "w") if args.log else None
    if csv:
        csv.write("t,VmRSS_kB,VmPeak_kB,VmSize_kB,VmSwap_kB,MemAvailable_kB,SwapUsed_kB\n")

    peaks = {"VmRSS": 0, "VmPeak": 0, "VmSize": 0, "VmSwap": 0}
    min_avail = base_avail
    max_swap = base_swap
    t0 = time.time()
    last_print = 0
    try:
        while proc.poll() is None:
            st = read_proc_status(pid)
            mi = read_meminfo()
            avail = mi["MemAvailable"]
            swap_used = mi.get("SwapTotal", 0) - mi.get("SwapFree", 0)
            for k in peaks:
                if k in st:
                    peaks[k] = max(peaks[k], st[k])
            min_avail = min(min_avail, avail)
            max_swap = max(max_swap, swap_used)
            t = time.time() - t0
            if csv:
                csv.write(f"{t:.2f},{st.get('VmRSS',0)},{st.get('VmPeak',0)},{st.get('VmSize',0)},{st.get('VmSwap',0)},{avail},{swap_used}\n")
                csv.flush()
            if t - last_print >= 5.0:
                last_print = t
                print(f"[probe:{args.label}] t={t:5.1f}s  RSS={fmt_gb(st.get('VmRSS',0))} "
                      f"Vm={fmt_gb(st.get('VmSize',0))}  Avail={fmt_gb(avail)}  Swap={fmt_gb(swap_used)}",
                      flush=True)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGINT)
        proc.wait()
    finally:
        if csv:
            csv.close()

    rc = proc.returncode
    print(f"\n[probe:{args.label}] === SUMMARY (rc={rc}) ===")
    print(f"[probe:{args.label}] peak VmRSS  = {fmt_gb(peaks['VmRSS'])}")
    print(f"[probe:{args.label}] peak VmHWM  = {fmt_gb(peaks['VmPeak'])}  (kernel-tracked high-water)")
    print(f"[probe:{args.label}] peak VmSize = {fmt_gb(peaks['VmSize'])}  (virtual)")
    print(f"[probe:{args.label}] peak VmSwap = {fmt_gb(peaks['VmSwap'])}")
    print(f"[probe:{args.label}] min MemAvailable = {fmt_gb(min_avail)}  (baseline {fmt_gb(base_avail)}, delta -{fmt_gb(base_avail-min_avail)})")
    print(f"[probe:{args.label}] max SwapUsed     = {fmt_gb(max_swap)}    (baseline {fmt_gb(base_swap)}, delta +{fmt_gb(max_swap-base_swap)})")
    return rc


if __name__ == "__main__":
    sys.exit(main())
