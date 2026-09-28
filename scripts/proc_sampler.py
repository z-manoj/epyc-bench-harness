"""Sample CPU and memory of one server (all processes in session --sid) plus its CPUs and NUMA node, as CSV.

Columns: t (epoch s), proc_cores (CPU time of the session / wall time, in cores), rss_gb and anon_gb (RssAnon, i.e.
excluding file-backed pages such as mmapped weights; both summed over the session), nproc, nthreads, cpus_busy_pct
(mean busy % of --cpus from /proc/stat), node_used_gb (MemUsed of --node). Runs until SIGTERM.

Only lock-free /proc files are read. smaps/smaps_rollup take the target's mmap lock for a full page-table walk
(~1.5 s on a 140 GB process) and slowed a faulting vLLM worker ~11x, so PSS is not sampled.
"""
import argparse
import os
import signal
import time

TCK = os.sysconf("SC_CLK_TCK")
PAGE = os.sysconf("SC_PAGE_SIZE")


def parse_cpus(spec: str) -> set[int]:
    out = set()
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return out


def session_procs(sid: int):
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat") as f:
                s = f.read()
        except OSError:
            continue
        f = s[s.rindex(")") + 2:].split()
        # after "(comm) ": state ppid pgrp session ...; utime/stime are fields 14/15, num_threads 20, rss 24
        if int(f[3]) == sid:
            yield int(d), int(f[11]) + int(f[12]), int(f[17]), int(f[21])


def anon_kb(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("RssAnon:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return 0


def cpu_times(cpus: set[int]):
    busy = total = 0
    with open("/proc/stat") as f:
        for line in f:
            if not line.startswith("cpu") or line.startswith("cpu "):
                continue
            p = line.split()
            if int(p[0][3:]) in cpus:
                v = list(map(int, p[1:9]))
                idle = v[3] + v[4]
                busy += sum(v) - idle
                total += sum(v)
    return busy, total


def node_used_gb(node: int) -> float:
    with open(f"/sys/devices/system/node/node{node}/meminfo") as f:
        for line in f:
            if "MemUsed:" in line:
                return int(line.split()[-2]) / 2**20
    return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sid", type=int, required=True)
    ap.add_argument("--cpus", required=True)
    ap.add_argument("--node", type=int, default=0)
    ap.add_argument("--interval", type=float, default=0.25)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cpus = parse_cpus(a.cpus)
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))

    prev_ticks = {}
    prev_cpu = cpu_times(cpus)
    prev_t = time.time()
    nxt = prev_t
    with open(a.out, "w", buffering=1) as out:
        out.write("t,proc_cores,rss_gb,anon_gb,nproc,nthreads,cpus_busy_pct,node_used_gb\n")
        while not stop:
            nxt += a.interval
            time.sleep(max(0.0, nxt - time.time()))
            t = time.time()
            procs = list(session_procs(a.sid))
            # CPU of processes that exited between samples is lost; only matters during startup/shutdown
            dticks = sum(ticks - prev_ticks.get(pid, ticks) for pid, ticks, _, _ in procs)
            prev_ticks = {pid: ticks for pid, ticks, _, _ in procs}
            busy, total = cpu_times(cpus)
            db, dt_ = busy - prev_cpu[0], total - prev_cpu[1]
            prev_cpu = (busy, total)
            out.write(f"{t:.3f},{dticks / TCK / (t - prev_t):.2f},"
                      f"{sum(r for *_, r in procs) * PAGE / 2**30:.3f},"
                      f"{sum(anon_kb(pid) for pid, *_ in procs) / 2**20:.3f},"
                      f"{len(procs)},{sum(n for _, _, n, _ in procs)},"
                      f"{100 * db / dt_ if dt_ else 0:.1f},{node_used_gb(a.node):.2f}\n")
            prev_t = t


if __name__ == "__main__":
    main()
