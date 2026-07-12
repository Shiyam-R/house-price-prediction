# Load Test Results

## Methodology

- **Tool**: [Locust](https://locust.io/) 2.45.0
- **Target**: `app.main:app` served via `uvicorn` (single worker, no `--reload`), v1.0.0 model
- **Load profile**: 50 concurrent simulated users, ramped up at 10 users/second, sustained for 60 seconds
- **Traffic mix**: `POST /predict` weighted 10:1 against `/health` and `/version`, using randomized-but-schema-valid payloads (see `load_tests/locustfile.py`) — not one fixed request repeated forever
- **Run command**:
  ```
  locust -f load_tests/locustfile.py --host http://localhost:8000 --headless -u 50 -r 10 -t 60s
  ```

## ⚠️ Environment caveat — read before quoting these numbers

These results were measured in a constrained, shared sandbox container — **not** production-equivalent hardware, and **not** the target Docker image (they were run directly against `uvicorn`, single worker, without the container/network overhead the real Dockerfile setup adds). Absolute numbers here will not match what you'd see running this on your own machine, in Docker Desktop, or on real cloud infrastructure — likely by a significant margin in either direction.

**What these numbers ARE genuinely useful for:**
- Proving the load test methodology itself works and the API holds up correctly under concurrent load (0% failures once the test script itself was correctly matched to the API's schema)
- Giving you a real, reproducible baseline you can re-run and compare against after any future change (e.g. "did this optimization actually help?")
- A concrete artifact and talking point for interviews: *"I load-tested it, here's exactly how, here's what I measured, and here's why I don't treat the absolute numbers as production-representative."* That last part — knowing the difference — is itself the signal a reviewer is looking for.

**What they are NOT:** a claim about what this API would do in a real deployment. Don't state "my API handles 34 req/s" as a general fact — state "under this specific test methodology, in this specific environment, I measured this."

## Results — `POST /predict` (the primary, expensive endpoint)

| Metric | Value |
|---|---|
| Total requests | 1,703 |
| Failed requests | 0 (0.00%) |
| Requests/sec | 28.4 |
| Median (p50) latency | 150 ms |
| p90 latency | 430 ms |
| p95 latency | 610 ms |
| p99 latency | 960 ms |
| Max latency | 1,200 ms |

## Results — aggregated across all endpoints

| Metric | Value |
|---|---|
| Total requests | 2,068 |
| Failed requests | 0 (0.00%) |
| Requests/sec | 34.5 |
| Median (p50) latency | 120 ms |
| p95 latency | 560 ms |

## Observations

- **Zero failures at 50 concurrent users** — the API and its Pydantic validation held up correctly under sustained concurrent load with varied inputs.
- **`/predict` is the clear latency bottleneck relative to `/health`/`/version`** (p50 150ms vs ~15ms) — expected, since it's the only endpoint doing real feature engineering + model inference rather than returning a static/near-static response.
- **The gap between p50 (150ms) and p99 (960ms) is wide.** This is worth investigating further rather than treating as acceptable by default — likely candidates: single-worker `uvicorn` serializing requests rather than truly parallelizing CPU-bound inference, and/or XGBoost inference itself not being the bottleneck so much as Python-level request handling contention. A natural next step (not yet done) would be re-running this against `uvicorn --workers 4` or behind `gunicorn` with multiple worker processes, to see whether p99 improves — that comparison would be a stronger interview talking point than a single run.
- **First-caught bug**: the initial version of this exact load test script had incorrect literal values for `GarageCond`, `BsmtFinType1`, and `MSZoning` (mismatched against `app/schemas.py`'s actual accepted values), which produced a 35% "failure" rate that was actually the API *correctly* rejecting badly-formed test input, not a real API problem. Fixed by cross-checking the literal values directly against `schemas.py` before re-running — a good reminder that a load test's own bugs can easily masquerade as the system-under-test's bugs if you don't verify which side actually failed.

## How to reproduce this yourself

```bash
# Install load testing dependencies
pip install -r requirements-load.txt

# Terminal 1 — start the API
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Terminal 2 — run the load test (headless, prints summary and exits)
locust -f load_tests/locustfile.py --host http://localhost:8000 \
    --headless -u 50 -r 10 -t 60s

# Or run interactively with the web UI instead:
locust -f load_tests/locustfile.py --host http://localhost:8000
# then open http://localhost:8089
```
