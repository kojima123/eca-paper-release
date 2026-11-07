"""Cross-provider LLM token / call accounting for runtime LLM hooks.

Captures usage from each `_run_*_structured` call site on the active arcgym
path (stable analyzer, causal_pass, lock_hypothesis, task_hypothesis_slate)
so a single run produces a `llm_token_usage.json` artifact giving:

- per-provider call counts
- per-provider input/output/total token totals (when reported by the provider)
- per-model totals
- per-call_site breakdown (so causal_pass vs stable_analyzer vs
  lock_hypothesis can be compared)
- prompt / response character totals (always present, even when the
  provider doesn't report token counts)
- wall-clock seconds spent in LLM calls

Providers that don't surface usage in their structured response (codex CLI)
still contribute call counts, char totals, and wall time; their token totals
remain None / zero rather than a fabricated estimate. The
``prompt_tokens_known_calls`` counter lets a reader compute a coverage ratio
(e.g. "40 of 60 calls had token telemetry").

State is keyed by run *workdir* (resolved absolute path). Each `record(...)`
call appends to the workdir's in-memory state and persists the snapshot to
``<workdir>/llm_token_usage.json`` on every call so the artifact survives
early termination (timeout, ctrl-c).
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

_LOCK = threading.Lock()
_PER_WORKDIR_STATE: dict[str, dict[str, Any]] = {}


def _empty_bucket() -> dict[str, Any]:
    return {
        "calls": 0,
        "failures": 0,
        "prompt_chars": 0,
        "response_chars": 0,
        "wall_seconds": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "prompt_tokens_known_calls": 0,
        "completion_tokens_known_calls": 0,
    }


def _empty_state() -> dict[str, Any]:
    return {
        "totals": _empty_bucket(),
        "by_provider": {},
        "by_call_site": {},
        "by_model": {},
        "calls": [],
    }


def _accumulate(
    bucket: dict[str, Any],
    *,
    prompt_chars: int,
    response_chars: int,
    wall_seconds: float,
    prompt_tokens: Optional[int],
    completion_tokens: Optional[int],
    total_tokens: Optional[int],
    status: str,
) -> None:
    bucket["calls"] += 1
    if status != "ok":
        bucket["failures"] += 1
    bucket["prompt_chars"] += int(prompt_chars or 0)
    bucket["response_chars"] += int(response_chars or 0)
    bucket["wall_seconds"] = round(
        float(bucket.get("wall_seconds", 0.0)) + float(wall_seconds or 0.0), 4
    )
    if prompt_tokens is not None:
        bucket["prompt_tokens"] += int(prompt_tokens)
        bucket["prompt_tokens_known_calls"] += 1
    if completion_tokens is not None:
        bucket["completion_tokens"] += int(completion_tokens)
        bucket["completion_tokens_known_calls"] += 1
    if total_tokens is not None:
        bucket["total_tokens"] += int(total_tokens)
    elif prompt_tokens is not None and completion_tokens is not None:
        bucket["total_tokens"] += int(prompt_tokens) + int(completion_tokens)


def _state_payload(state: dict[str, Any], *, include_call_log: bool = True) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "totals": dict(state["totals"]),
        "by_provider": {k: dict(v) for k, v in state["by_provider"].items()},
        "by_call_site": {k: dict(v) for k, v in state["by_call_site"].items()},
        "by_model": {k: dict(v) for k, v in state["by_model"].items()},
    }
    if include_call_log:
        payload["calls"] = list(state["calls"])
    return payload


def record(
    workdir: Path | str,
    *,
    provider: str,
    model: Optional[str],
    call_site: str,
    prompt_chars: int,
    response_chars: int,
    wall_seconds: float,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    total_tokens: Optional[int] = None,
    status: str = "ok",
) -> None:
    """Record one LLM call against the run rooted at *workdir*.

    Aggregates into in-memory state and persists to
    ``<workdir>/llm_token_usage.json`` on every call. Failures inside the
    persist step are swallowed (logged at WARNING) so token accounting can
    never break the agent loop.
    """
    if not provider:
        provider = "unknown"
    if not call_site:
        call_site = "unknown"
    model_key = (model or "").strip() or "unspecified"
    try:
        key = str(Path(workdir).resolve())
    except Exception:
        key = str(workdir)
    with _LOCK:
        state = _PER_WORKDIR_STATE.get(key)
        if state is None:
            state = _empty_state()
            _PER_WORKDIR_STATE[key] = state
        for bucket in (
            state["totals"],
            state["by_provider"].setdefault(provider, _empty_bucket()),
            state["by_call_site"].setdefault(call_site, _empty_bucket()),
            state["by_model"].setdefault(model_key, _empty_bucket()),
        ):
            _accumulate(
                bucket,
                prompt_chars=prompt_chars,
                response_chars=response_chars,
                wall_seconds=wall_seconds,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                status=status,
            )
        state["calls"].append(
            {
                "provider": provider,
                "model": model_key,
                "call_site": call_site,
                "prompt_chars": int(prompt_chars or 0),
                "response_chars": int(response_chars or 0),
                "wall_seconds": round(float(wall_seconds or 0.0), 4),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "status": status,
            }
        )
        snapshot = _state_payload(state)
    try:
        target = Path(workdir) / "llm_token_usage.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except Exception as exc:
        log.warning("[token_tracker] persist failed for %s: %s", workdir, exc)


def get_state(workdir: Path | str) -> dict[str, Any]:
    """Return a snapshot of the recorded state for *workdir* (empty if none)."""
    try:
        key = str(Path(workdir).resolve())
    except Exception:
        key = str(workdir)
    with _LOCK:
        state = _PER_WORKDIR_STATE.get(key)
        if state is None:
            return _state_payload(_empty_state())
        return _state_payload(state)


def reset(workdir: Optional[Path | str] = None) -> None:
    """Reset state. If *workdir* is None, clears everything; otherwise just
    that workdir's state.
    """
    with _LOCK:
        if workdir is None:
            _PER_WORKDIR_STATE.clear()
            return
        try:
            key = str(Path(workdir).resolve())
        except Exception:
            key = str(workdir)
        _PER_WORKDIR_STATE.pop(key, None)
