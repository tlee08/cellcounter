# Production-grade function tracing logging

**Session ID:** ses_12bb64cf9ffeXIP2uao6RLzZxj
**Created:** 6/17/2026, 4:34:20 PM
**Updated:** 6/17/2026, 4:37:37 PM

---

## User

Look at contxt7, consult tavily, use caveman skills and karpathy guidelines. How can I make the logging better (so i can see/trace what functions are being run)? I want it to be production grade (like orchestrator for just the logging part... existing frameworks may be suitable).

---

## Current State: Broken

**49 log calls. Zero errors logged. Zero function traceability.** You cannot answer "what ran, when, how long" from current logs.

Key issues:

- `setup_logging()` runs at `import cellcounter` — side effect, no config control
- No per-step timing, no run-id, no correlation across 20+ pipeline steps
- `print()` in library code (dask_utils, template_utils) — invisible in file logs
- Zero `logger.error()` / `logger.exception()` — exceptions swallowed by bare `print(f"Error: {e}")`
- I/O functions silent. Visual check functions silent. Elastix bypasses Python logging.
- `warnings.filterwarnings("ignore")` globally suppresses signals

---

## Framework Choice: **Loguru**

| Need                                  | Structlog                   | Loguru winner?                              |
| ------------------------------------- | --------------------------- | ------------------------------------------- |
| Function entry/exit tracing decorator | DIY                         | Built-in recipe ✅                          |
| Structured JSON output                | Processor chain             | `serialize=True` ✅                         |
| Context binding (run_id, step)        | `bind_contextvars`          | `bind()` + `contextualize()` ✅             |
| Exception with traceback              | `format_exc_info` processor | `@logger.catch` ✅                          |
| API simplicity (team adoption)        | Complex processor chain     | `logger.info("msg")` identical to stdlib ✅ |
| Rotating file + compression           | Need external handler       | Built-in `rotation` + `compression` ✅      |
| Filter by module/bind                 | Handler filter              | `filter=` on add ✅                         |

Structlog is better if you need OpenTelemetry trace propagation through microservices. Cellcounter is a single-process pipeline — loguru's simplicity wins.

---

## Plan

### Phase 1: Replace `logging_utils.py` with loguru config

```python
# src/cellcounter/utils/logging_utils.py → rewrite
import sys
from pathlib import Path
from loguru import logger

def configure_logging(
    level: str = "INFO",
    log_file: Path | str | None = None,
    json_output: bool = False,
) -> None:
    logger.remove()  # Clear defaults

    # Console: human-readable format
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{extra[run_id]}</cyan> | <level>{message}</level>",
    )

    # File: structured (JSON if json_output, else detailed)
    log_file = log_file or CACHE_DIR / "cellcounter.log"
    logger.add(
        log_file,
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        compression="gz",
        serialize=json_output,
        format="{time} | {level} | {name}:{function}:{line} | {extra[run_id]} | {message}",
    )
```

### Phase 2: `@trace` decorator

```python
# src/cellcounter/utils/logging_utils.py
import time
from loguru import logger

def trace(func):
    """Log function entry, exit, duration. Exception → full traceback."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug("→ {}.{}() called", func.__module__, func.__qualname__)
        t0 = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - t0
            logger.debug("← {}.{}() done ({:.2f}s)", func.__module__, func.__qualname__, elapsed)
            return result
        except Exception:
            elapsed = time.perf_counter() - t0
            logger.exception(
                "✗ {}.{}() FAILED after {:.2f}s",
                func.__module__, func.__qualname__, elapsed,)
            raise
    return wrapper
```

### Phase 3: Pipeline step orchestration logging

In `pipeline.py:run_pipeline_steps`, wrap each step with timing + step counter:

```python
from loguru import logger

def run_pipeline_steps(self, in_fp, *, steps=None, overwrite=False):
    steps = steps or [...]
    run_id = uuid.uuid4().hex[:8]

    with logger.contextualize(run_id=run_id):
        logger.info("Pipeline start — {} steps, overwrite={}", len(steps), overwrite)
        t_total = time.perf_counter()

        for i, step in enumerate(steps, 1):
            t0 = time.perf_counter()
            logger.info("[{}/{}] Step: {}", i, len(steps), step)

            try:
                if step == "tiff2zarr":
                    self.tiff2zarr(in_fp, overwrite=overwrite)
                else:
                    getattr(self, step)(overwrite=overwrite)
            except Exception:
                logger.error("[{}/{}] Step {} FAILED", i, len(steps), step)
                raise

            elapsed = time.perf_counter() - t0
            logger.info("[{}/{}] Step {} done ({:.1f}s)", i, len(steps), step, elapsed)

        logger.info(
            "Pipeline complete — {} steps, total {:.1f}s",
            len(steps), time.perf_counter() - t_total,
        )
```

### Phase 4: Clean up noise

- Remove `import cellcounter` → `setup_logging()` side effect. Make user call `configure_logging()` explicitly.
- Replace `print()` in `dask_utils.py:63` with `logger.info()`.
- Replace `print(f"Error: {e}")` in `scripts/init.py:34` with `logger.exception()`.
- Remove `warnings.filterwarnings("ignore")` from `__init__.py` (or scope to specific known warnings).
- Add `@logger.catch` / `logger.exception()` in all bare `except` blocks.
- Send ITK-Elastix console output through loguru by wrapping `SetLogToConsole`.

### Phase 5: Dependency

`pyproject.toml` add: `"loguru>=0.7.3"` (1 dependency, pure Python, no native deps)

---

## What Logs Will Look Like After

**Console (human-readable):**

```
14:32:01 | INFO     | a3f1b2d8 | Pipeline start — 25 steps, overwrite=False
14:32:01 | INFO     | a3f1b2d8 | [1/25] Step: tiff2zarr
14:32:59 | INFO     | a3f1b2d8 | [1/25] Step tiff2zarr done (58.2s)
14:32:59 | INFO     | a3f1b2d8 | [2/25] Step: reg_ref_prepare
14:33:12 | INFO     | a3f1b2d8 | [2/25] Step reg_ref_prepare done (12.8s)
...
15:47:03 | INFO     | a3f1b2d8 | Pipeline complete — 25 steps, total 4492.1s
```

**File (JSON mode, queryable):**

```json
{
  "time": "2026-06-17T14:32:01Z",
  "level": "INFO",
  "message": "[1/25] Step: tiff2zarr",
  "extra": { "run_id": "a3f1b2d8" }
}
```

**Error with full traceback** (instead of silent `print(e)`):

```
14:35:22 | ERROR    | a3f1b2d8 | ✗ cellcounter.pipeline.pipeline.tiff2zarr() FAILED after 3.2s
Traceback (most recent call last):
  File ...
```

---

## Tradeoffs Considered

| Decision                                                                                 | Why                                                                                                                                          |
| ---------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| Loguru over structlog                                                                    | Single-process pipeline, no need for processor-chain complexity. Decorator recipe is exactly what user asked for                             |
| Keep stdlib `logging.getLogger(__name__)` pattern or switch? → Switch to `loguru.logger` | loguru has zero-config module names via `{name}` format field. No need for `getLogger(__name__)` per module                                  |
| JSON by default? → Opt-in via env var                                                    | Not everyone has log ingestion pipeline. Human-readable default, JSON when `CELLCOUNTER_LOG_JSON=1`                                          |
| Decorator vs context manager for steps? → Context manager (`logger.contextualize`)       | Step-level context (run_id, step_name) is per-invocation, not per-function. Decorator for low-level trace, context manager for orchestration |

---

Shall I implement this? Or prefer different approach (structlog, different trace format, etc)?

---
