# Load Test Results

## Methodology

- **Tool**: [Locust](https://locust.io/) 2.45.0
- **Target**: `app.main:app`, v1.0.0 model, served via `uvicorn`
- **Load profile**: 1,000 concurrent simulated users, ramped at 10 users/second, sustained for 180 seconds total
- **Traffic mix**: `POST /predict` weighted 10:1 against `/health` and `/version`, using randomized-but-schema-valid payloads (see `locustfile.py`) — not one fixed request repeated forever
- **Run command**:
  ```
  locust -f load_tests/locustfile.py --host http://localhost:PORT --headless -u 1000 -r 10 -t 180s
  ```

This ended up producing three genuinely different, real results across three environments — which turned out to be far more informative than a single clean run would have been.

## Run 1 — Windows native (`uvicorn app.main:app --reload`, no Docker)

**Result: crashed.** RPS climbed normally to ~20-25 req/s, then latency spiked sharply (p50/p95 both racing toward 15,000-21,000ms) before RPS collapsed to 0 and failures spiked to 66%, with `ConnectionRefusedError: [WinError 10061]` on all subsequent requests.

**Root cause, confirmed via the server's own traceback:**
```
ValueError: too many file descriptors in select()
```
`uvicorn` on Windows runs on `asyncio`'s `SelectorEventLoop`, which relies on the `select()` syscall. Windows' `select()` implementation has a hard, fixed ceiling of **512 sockets** (`FD_SETSIZE`, defined in `winsock2.h`) — not a configurable limit, not Windows "throttling" as a policy, but a genuine architectural constraint of that specific syscall on that specific OS. Once concurrent connections passed that ceiling (first failure appeared roughly 74 seconds into a 10-users/sec ramp, i.e. around ~740 total spawned users), `select()` itself raised, which killed the event loop, which took the entire single-process server down — explaining why `/health` stopped responding *entirely* rather than just slowing down.

This is a documented, known limitation specific to running `asyncio`-based Python servers natively on Windows — multiple independent sources confirm the same code does not hit this ceiling when run inside a Linux container, since Linux uses `epoll` instead of `select()`, with no equivalent hardcoded limit.

## Run 2 — Docker container, Linux, single worker (GitHub Codespaces, 2 vCPU)

Same 1,000-user, 180-second test, run against the identical codebase inside the already-built Docker image (`python:3.13-slim`), single `uvicorn` process, no `--workers` flag.

| Metric | Value |
|---|---|
| Total requests | 3,002 |
| Failed requests | **0 (0.00%)** |
| `/predict` requests/sec | 23.80 |
| `/predict` median (p50) latency | 12,000 ms |
| `/predict` average latency | 12,819 ms |
| `/predict` max latency | 27,621 ms |

**The crash did not reproduce.** Zero failures at the exact same 1,000-concurrent-user load that took the Windows-native process down entirely — direct, measured confirmation that containerizing on Linux eliminates the `select()` ceiling. This alone is a concrete, evidence-backed reason to containerize, not just a generic best practice being cited.

**But a different, real bottleneck showed up: severe latency under load** (12-second median, vs. ~150ms measured in an earlier lightly-loaded 50-user run). Root cause: `predict_price()` in `main.py` is a **synchronous** (`def`, not `async def`) endpoint doing real CPU work (pandas transforms + XGBoost inference). FastAPI dispatches sync endpoints to a background thread pool with a limited default size (~40 threads). At 1,000 concurrent in-flight requests, the great majority were simply queued, waiting for a thread to free up — the 12-second median is mostly *queue wait time*, not actual inference time.

**Net effect of containerizing at this load level: a hard crash became graceful (if slow) degradation.** That's a genuine reliability improvement, and it correctly surfaced the *next* real bottleneck instead of hiding it.

## Run 3 — Docker container, Linux, 2 workers (`--workers 2`, matching the Codespace's 2 vCPUs)

Same test again, same container image, this time launched with `uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2` — deliberately sized to match `nproc`'s reported 2 cores, since over-provisioning workers beyond available cores adds context-switching overhead rather than real parallelism for CPU-bound work like this.

| Metric | 1 worker | 2 workers | Change |
|---|---|---|---|
| Total requests (180s) | 3,002 | 3,717 | **+23.8%** |
| Failed requests | 0 | 0 | — |
| `/predict` requests/sec | 23.80 | 26.60 | +11.8% |
| `/predict` median (p50) latency | 12,000 ms | 11,000 ms | -8.3% |
| `/predict` average latency | 12,819 ms | 10,567 ms | -17.6% |
| `/predict` max latency | 27,621 ms | 23,189 ms | -16.0% |

**A real but modest improvement — exactly what correctly-sized scaling on constrained hardware should look like.** Not a dramatic collapse to milliseconds (that would have been suspicious on a 2-core machine), but a consistent, measurable gain across every metric, with throughput improving proportionally more (+23.8% total work done) than any single latency figure — consistent with the theory that the bottleneck is genuinely thread-pool/CPU-bound, and scales with real available parallelism rather than some other unrelated factor.

## Summary — what this three-run comparison actually demonstrates

1. **Windows-native**: hard crash at ~740 concurrent users (OS-level `select()` ceiling — a genuine architectural limit, not a config issue)
2. **Linux container, 1 worker**: survives the same load with 0 failures, but severe latency (12s median) from thread-pool queuing
3. **Linux container, 2 workers**: real, proportional improvement (+24% throughput) matching the available 2-core hardware

## Honest next steps this points to (not yet done)

- **Convert `/predict` to `async def`** and offload the CPU-bound inference to a properly-sized `ThreadPoolExecutor` or `ProcessPoolExecutor` via `run_in_executor`, rather than relying on FastAPI's default thread pool sizing — this is the most likely lever to meaningfully cut the 10+ second latencies, more so than adding workers alone.
- **Re-run this same comparison on a machine with more cores** (Codespaces machine types go up to 8 or 16 cores) to see whether the workers-vs-throughput relationship continues to scale linearly, or whether the ceiling shifts to some other bottleneck (e.g. Python's GIL, model load memory, or downstream I/O).
- **Test behind `gunicorn` with `uvicorn` worker classes**, which is the more common production pattern for running multiple `uvicorn` workers with proper process management, versus `uvicorn --workers` directly.

## How to reproduce this yourself

```bash
# Install load testing dependencies
pip install -r requirements-load.txt

# Single worker (Linux/Docker recommended -- see Run 1 above for why native Windows will hit a hard ceiling well before 1,000 users)
docker build -t house-price-api .
docker run -d --name house-price-api -p 8000:8000 house-price-api
locust -f load_tests/locustfile.py --host http://localhost:8000 --headless -u 1000 -r 10 -t 180s

# Multi-worker comparison (set --workers to match your actual core count via `nproc`)
docker run -d --name house-price-api-workers -p 8001:8000 house-price-api \
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
locust -f load_tests/locustfile.py --host http://localhost:8001 --headless -u 1000 -r 10 -t 180s
```