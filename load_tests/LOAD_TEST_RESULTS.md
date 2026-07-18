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

## Run 4 — `async def /predict` with an explicit (but misconfigured) thread pool

Runs 2 and 3 pointed at FastAPI's hidden default sync-endpoint thread pool as the likely source of the 12-second median latency. The natural next step: convert `/predict` to `async def` and explicitly offload the CPU-bound inference (pandas + XGBoost) to a dedicated `ThreadPoolExecutor` via `run_in_executor`, with the pool size controlled by a new `PREDICT_THREAD_POOL_SIZE` setting instead of an invisible framework default.

**First attempt used a bad default: `cpu_count() * 5`.** On this 2-core Codespace, that computed to a **10-thread pool**.

| Metric | Run 2 (implicit default, ~40 threads) | Run 4 (explicit pool, 10 threads) | Change |
|---|---|---|---|
| `/predict` median | 12,000 ms | 29,000 ms | **~2.4x worse** |
| `/predict` p95 | — | 35,000 ms | — |

**This was a real regression, and the cause was straightforward once measured**: FastAPI/Starlette's actual hidden default isn't an arbitrary mystery number — it's `anyio`'s `CapacityLimiter`, defaulting to **40** concurrent threads, independent of CPU count. The "fix" replaced a 40-thread pool with a 10-thread one. Cutting available concurrency by 4x on a queuing-bound workload produced almost exactly the latency increase queuing theory would predict. The async conversion itself wasn't the problem — the specific default I chose for it was actively worse than what it replaced.

**Fix**: changed the default to `max(40, cpu_count() * 5)` — a floor matching the concurrency level already known to work reasonably (from Run 2), with room to scale up on genuinely multi-core hardware, and still fully overridable via `PREDICT_THREAD_POOL_SIZE` for further tuning.

## Run 5 — `async def /predict` with the corrected pool size (40 threads)

Same 1,000-user, 180-second test, rerun after the fix, confirmed to be running in isolation (`docker ps` showed only this one container active — earlier containers from Runs 2 and 3 were confirmed stopped first, ruling out cross-container CPU contention as a factor).

| Metric | Run 2 (implicit, 40 threads) | Run 5 (explicit, 40 threads) | Change |
|---|---|---|---|
| `/predict` median | 12,000 ms | 19,000 ms | worse |
| `/predict` total requests (180s) | 3,002 | 4,752 | **+58% more work done** |
| `/predict` requests/sec | 23.80 | 26.4 | better |
| Failures | 0 | 0 | same |

**This result is genuinely ambiguous, and I'm stating that plainly rather than forcing a clean story onto it.** Median latency looks worse, but total throughput and requests/sec both improved — an inconsistent combination if pool size were the only variable at play. The more likely explanation: **Locust itself was running on the same 2-core Codespace as the server under test.** At 1,000 simulated users, Locust's own process is a real, non-trivial CPU consumer — meaning the load generator and the system under test were directly competing for the same two physical cores. That's a genuine methodological limitation of this specific setup, not a property of the API's threading design, and it's consistent with `/health` and `/version` — endpoints that do almost no real work — also showing multi-second medians and p99s past 80,000ms in this run, which shouldn't happen if the bottleneck were purely about `/predict`'s thread pool.

**I'm not claiming Run 5 proves the async conversion helped or hurt.** What it does show, with more confidence:
- The explicit, sized thread pool is not obviously worse than the implicit default once sized correctly (unlike Run 4, which clearly was)
- Load testing on a 2-core machine where both the client (Locust) and server share those same 2 cores is a confounded setup — a real limitation worth naming rather than a result worth overclaiming

## Summary — what this full comparison actually demonstrates

1. **Windows-native**: hard crash at ~740 concurrent users (OS-level `select()` ceiling — a genuine architectural limit, not a config issue)
2. **Linux container, 1 worker, implicit thread pool**: survives the same load with 0 failures, but severe latency (12s median) from thread-pool queuing
3. **Linux container, 2 processes (`--workers 2`)**: real, proportional improvement (+24% throughput) matching the available 2-core hardware
4. **Linux container, explicit thread pool, misconfigured (10 threads)**: a real, measured regression (~2.4x worse median) — caused by replacing Starlette's 40-thread default with a smaller one, not by the async conversion itself
5. **Linux container, explicit thread pool, corrected (40 threads)**: an ambiguous result, most likely confounded by running the load generator on the same constrained hardware as the server under test — not a clean signal either way

The most valuable outcome of Runs 4-5 isn't a throughput number — it's that the thread pool size is now a **deliberate, visible, tunable setting** (`PREDICT_THREAD_POOL_SIZE` in `app/config.py`) instead of an invisible framework default nobody was looking at. That's real value independent of whether this specific constrained test environment could cleanly prove a throughput win.

## Honest next steps this points to (not yet done)

- **Run Locust from a separate machine than the server** — the standard, correct practice for load testing, and the fix for the Run 5 confound. Testing client and server on the same 2-core box was a real limitation of this setup, not of the API.
- **Re-run the full comparison on hardware with more cores** to see whether the workers-vs-throughput and thread-pool-size-vs-latency relationships hold, or whether a different bottleneck emerges (e.g. Python's GIL under genuinely high parallelism, or downstream I/O).
- **Test behind `gunicorn` with `uvicorn` worker classes** — the more common production pattern for running multiple `uvicorn` workers with proper process management, versus `uvicorn --workers` directly.
- **Systematically sweep `PREDICT_THREAD_POOL_SIZE`** (e.g. 20, 40, 80, 160) with the client-server confound removed, to find where diminishing returns actually begin on real hardware — right now the value is a reasoned floor, not an empirically optimal number.

## How to reproduce this yourself

```bash
# Install load testing dependencies
pip install -r requirements-load.txt

# Single worker, implicit default pool (Linux/Docker recommended -- see Run 1 above for why native Windows will hit a hard ceiling well before 1,000 users)
docker build -t house-price-api .
docker run -d --name house-price-api -p 8000:8000 house-price-api
locust -f load_tests/locustfile.py --host http://localhost:8000 --headless -u 1000 -r 10 -t 180s

# Multi-worker comparison (set --workers to match your actual core count via `nproc`)
docker run -d --name house-price-api-workers -p 8001:8000 house-price-api \
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
locust -f load_tests/locustfile.py --host http://localhost:8001 --headless -u 1000 -r 10 -t 180s

# Explicit thread pool size override, for tuning experiments
docker run -d --name house-price-api-tuned -p 8003:8000 \
    -e PREDICT_THREAD_POOL_SIZE=80 house-price-api
locust -f load_tests/locustfile.py --host http://localhost:8003 --headless -u 1000 -r 10 -t 180s

# IMPORTANT: for a clean result, run locust from a DIFFERENT machine
# than the one running the container, to avoid the client/server CPU
# contention confound identified in Run 5 above.
```