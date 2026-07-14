"""Opt-in candidate tracing and retrieval diagnostics.

The search path deliberately does not attach this data to normal results.  A
``SearchTrace`` is supplied explicitly by benchmarks, or created by the server
when debug logging is enabled.  Keeping the collector out of the return value
also makes traced/untraced result equivalence straightforward to verify.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from statistics import fmean, pstdev
from typing import Any, ClassVar

TRACE_SCHEMA_VERSION = "1.0"


def _score(value: Any) -> float | None:
    """Return a JSON-safe numeric score when possible."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _matches_expected(path: str, expected_files: list[str]) -> bool:
    normalized = path.replace("\\", "/").casefold()
    return any(
        expected.replace("\\", "/").casefold() in normalized
        for expected in expected_files
    )


@dataclass
class SearchTrace:
    """Structured trace for one project-local ``hybrid_search`` call."""

    project_id: int | str | None = None
    project_name: str | None = None
    schema_version: ClassVar[str] = TRACE_SCHEMA_VERSION
    original_query: str = ""
    query: str = ""
    limit: int = 0
    config: dict[str, Any] = field(default_factory=dict)
    decisions: dict[str, Any] = field(default_factory=dict)
    channels: dict[str, dict[str, Any]] = field(default_factory=dict)
    candidates: dict[str, dict[str, Any]] = field(default_factory=dict)
    stages: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    errors: list[dict[str, str]] = field(default_factory=list)
    final: list[str] = field(default_factory=list)

    def start(
        self,
        *,
        original_query: str,
        query: str,
        limit: int,
        config: dict[str, Any],
    ) -> None:
        """Reset and initialize the collector for a single search call."""
        self.original_query = original_query
        self.query = query
        self.limit = limit
        self.config = dict(config)
        self.decisions.clear()
        self.channels.clear()
        self.candidates.clear()
        self.stages.clear()
        self.errors.clear()
        self.final.clear()

    def candidate_key(self, chunk_id: Any) -> str:
        project = self.project_id if self.project_id is not None else self.project_name
        return f"{project}:{chunk_id}" if project is not None else str(chunk_id)

    def decision(self, name: str, **details: Any) -> None:
        self.decisions[name] = details

    def channel_status(
        self,
        label: str,
        status: str,
        *,
        reason: str | None = None,
        requested_limit: int | None = None,
        score_kind: str | None = None,
    ) -> None:
        channel = self.channels.setdefault(label, {"candidates": []})
        channel["status"] = status
        if reason is not None:
            channel["reason"] = reason
        if requested_limit is not None:
            channel["requested_limit"] = requested_limit
        if score_kind is not None:
            channel["score_kind"] = score_kind

    def _candidate(self, chunk_id: Any, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        key = self.candidate_key(chunk_id)
        candidate = self.candidates.get(key)
        if candidate is None:
            candidate = {
                "key": key,
                "chunk_id": chunk_id,
                "project_id": self.project_id,
                "project_name": self.project_name,
                "file_id": None,
                "file_path": "",
                "source_type": None,
                "channels": {},
                "stage_ranks": {},
                "events": [],
                "final_rank": None,
            }
            self.candidates[key] = candidate
        if metadata:
            for name in ("file_id", "file_path", "source_type"):
                value = metadata.get(name)
                if value not in (None, ""):
                    candidate[name] = value
        return candidate

    def record_channel(
        self,
        label: str,
        items: list[dict[str, Any]],
        *,
        score_kind: str,
        metadata: dict[Any, dict[str, Any]] | None = None,
        normalized_scores: dict[Any, float] | None = None,
        requested_limit: int | None = None,
    ) -> None:
        """Record one source channel before fusion or post-filtering."""
        status = "used" if items else "empty"
        self.channel_status(
            label,
            status,
            requested_limit=requested_limit,
            score_kind=score_kind,
        )
        channel = self.channels[label]
        keys: list[str] = []
        for rank, item in enumerate(items, start=1):
            chunk_id = item["id"]
            candidate = self._candidate(chunk_id, (metadata or {}).get(chunk_id))
            occurrence: dict[str, Any] = {
                "rank": rank,
                "score": _score(item.get("score")),
                "score_kind": score_kind,
                "eligible_for_fusion": True,
            }
            if normalized_scores and chunk_id in normalized_scores:
                occurrence["normalized_score"] = _score(normalized_scores[chunk_id])
            candidate["channels"][label] = occurrence
            candidate["events"].append({
                "stage": label,
                "eligible": True,
                "reason": "retrieved",
                "rank": rank,
            })
            keys.append(candidate["key"])
        channel["candidates"] = keys
        channel["candidate_count"] = len(keys)

    def reject(
        self,
        chunk_id: Any,
        *,
        stage: str,
        reason: str,
        channel: str | None = None,
        **details: Any,
    ) -> None:
        candidate = self._candidate(chunk_id)
        if channel and channel in candidate["channels"]:
            candidate["channels"][channel]["eligible_for_fusion"] = False
        event = {"stage": stage, "eligible": False, "reason": reason}
        event.update(details)
        candidate["events"].append(event)

    def record_stage(
        self,
        name: str,
        items: list[dict[str, Any]],
        *,
        score_field: str | None = None,
        channel: str | None = None,
    ) -> None:
        entries: list[dict[str, Any]] = []
        for rank, item in enumerate(items, start=1):
            candidate = self._candidate(item["id"])
            candidate["stage_ranks"][name] = rank
            if channel and channel in candidate["channels"]:
                candidate["channels"][channel]["fusion_rank"] = rank
            entry = {"key": candidate["key"], "rank": rank}
            if score_field:
                entry["score"] = _score(item.get(score_field))
            entries.append(entry)
        self.stages[name] = entries

    def record_fusion(
        self,
        items: list[dict[str, Any]],
        *,
        labels: list[str],
        weights: list[float],
        rrf_k: int = 60,
    ) -> None:
        self.record_stage("union", items, score_field="rrf_score")
        weight_by_label = dict(zip(labels, weights, strict=False))
        for item in items:
            candidate = self._candidate(item["id"])
            contributions: dict[str, float] = {}
            for label, occurrence in candidate["channels"].items():
                if label not in weight_by_label or not occurrence.get("eligible_for_fusion", True):
                    continue
                rank = occurrence.get("fusion_rank", occurrence["rank"])
                contributions[label] = weight_by_label[label] / (rrf_k + rank)
            candidate["fusion"] = {
                "rank": candidate["stage_ranks"]["union"],
                "rrf_score": _score(item.get("rrf_score")),
                "contributions": contributions,
            }

    def record_rescore(self, name: str, items: list[dict[str, Any]]) -> None:
        self.record_stage(name, items, score_field="rrf_score")
        for item in items:
            candidate = self._candidate(item["id"])
            candidate.setdefault("rescores", {})[name] = _score(item.get("rrf_score"))

    def record_dedup(
        self,
        *,
        stage: str,
        before: list[dict[str, Any]],
        after: list[dict[str, Any]],
    ) -> None:
        self.record_stage(stage, after, score_field="rrf_score")
        kept_keys = {self.candidate_key(item["id"]) for item in after}
        keeper_by_path = {
            self._candidate(item["id"]).get("file_path", ""): self.candidate_key(item["id"])
            for item in after
            if self._candidate(item["id"]).get("file_path", "")
        }
        for item in before:
            key = self.candidate_key(item["id"])
            if key in kept_keys:
                continue
            candidate = self._candidate(item["id"])
            path = candidate.get("file_path", "")
            self.reject(
                item["id"],
                stage=stage,
                reason="duplicate_file",
                duplicate_of=keeper_by_path.get(path),
                file_path=path,
            )

    def error(self, stage: str, error: BaseException) -> None:
        self.errors.append({
            "stage": stage,
            "type": type(error).__name__,
            "message": str(error),
        })

    def finalize(self, results: list[dict[str, Any]]) -> None:
        self.final = []
        for rank, result in enumerate(results, start=1):
            candidate = self._candidate(result["id"], result)
            candidate["final_rank"] = rank
            candidate["final_score"] = _score(result.get("score"))
            candidate["events"].append({
                "stage": "final",
                "eligible": True,
                "reason": "returned",
                "rank": rank,
            })
            self.final.append(candidate["key"])

        for candidate in self.candidates.values():
            if candidate["final_rank"] is not None:
                continue
            if "union" in candidate["stage_ranks"]:
                if not any(
                    event.get("reason") == "duplicate_file"
                    for event in candidate["events"]
                ):
                    candidate["events"].append({
                        "stage": "final",
                        "eligible": False,
                        "reason": "result_limit",
                        "limit": self.limit,
                    })
            elif any(not event["eligible"] for event in candidate["events"]):
                continue
            else:
                candidate["events"].append({
                    "stage": "fusion",
                    "eligible": False,
                    "reason": "not_in_fusion_input",
                })

    def candidate_keys_for_channel(self, label: str, *, eligible_only: bool = True) -> list[str]:
        keys = self.channels.get(label, {}).get("candidates", [])
        if not eligible_only:
            return list(keys)
        return [
            key for key in keys
            if self.candidates[key]["channels"][label].get("eligible_for_fusion", True)
        ]

    def validate_complete_attribution(self) -> list[str]:
        """Return invariant violations rather than raising inside production search."""
        problems: list[str] = []
        for label, channel in self.channels.items():
            for key in channel.get("candidates", []):
                if key not in self.candidates:
                    problems.append(f"{label}: missing candidate {key}")
                elif label not in self.candidates[key]["channels"]:
                    problems.append(f"{label}: candidate {key} lacks channel attribution")
        for key in self.final:
            candidate = self.candidates.get(key)
            if not candidate or candidate.get("final_rank") is None:
                problems.append(f"final candidate {key} lacks final rank")
        for key, candidate in self.candidates.items():
            if candidate.get("final_rank") is None and not any(
                event.get("eligible") is False for event in candidate.get("events", [])
            ):
                problems.append(f"candidate {key} lacks an exclusion reason")
        return problems

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scope": {"project_id": self.project_id, "project_name": self.project_name},
            "query": {"original": self.original_query, "effective": self.query},
            "limit": self.limit,
            "config": self.config,
            "decisions": self.decisions,
            "channels": self.channels,
            "stages": self.stages,
            "candidates": list(self.candidates.values()),
            "final": self.final,
            "errors": self.errors,
            "attribution_complete": not self.validate_complete_attribution(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


@dataclass
class FederatedSearchTrace:
    """Trace project-local candidates through global fusion and reranking."""

    query: str
    limit: int
    schema_version: ClassVar[str] = TRACE_SCHEMA_VERSION
    projects: list[SearchTrace] = field(default_factory=list)
    stages: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    reranker: dict[str, Any] = field(default_factory=dict)
    errors: list[dict[str, str]] = field(default_factory=list)

    @staticmethod
    def result_key(result: dict[str, Any]) -> str:
        project = result.get("project_id", result.get("project_name"))
        return f"{project}:{result.get('id')}"

    def add_project(self, trace: SearchTrace) -> None:
        self.projects.append(trace)

    def _record_results(self, name: str, results: list[dict[str, Any]]) -> None:
        self.stages[name] = [
            {
                "key": self.result_key(result),
                "project_id": result.get("project_id"),
                "project_name": result.get("project_name"),
                "chunk_id": result.get("id"),
                "file_path": result.get("file_path", ""),
                "rank": rank,
                "score": _score(result.get("score")),
            }
            for rank, result in enumerate(results, start=1)
        ]

    def record_project_union(self, results: list[dict[str, Any]]) -> None:
        self._record_results("project_union", results)

    def record_rerank_pool(self, results: list[dict[str, Any]]) -> None:
        self._record_results("rerank_pool", results)
        self.reranker.update({
            "status": "running",
            "pool_size": len(results),
            "pool_keys": [self.result_key(result) for result in results],
        })

    def record_rerank_result(
        self,
        pool: list[dict[str, Any]],
        reranked: list[tuple[int, float]],
        results: list[dict[str, Any]],
    ) -> None:
        self.reranker.update({
            "status": "used",
            "results": [
                {
                    "key": self.result_key(pool[index]),
                    "input_rank": index + 1,
                    "output_rank": rank,
                    "reranker_score": _score(score),
                }
                for rank, (index, score) in enumerate(reranked, start=1)
            ],
        })
        self._record_results("reranked", results)

    def record_reranker_skipped(self, reason: str) -> None:
        self.reranker = {"status": "skipped", "reason": reason}

    def record_reranker_error(self, error: BaseException) -> None:
        self.reranker.update({
            "status": "error",
            "type": type(error).__name__,
            "message": str(error),
        })

    def record_final(self, results: list[dict[str, Any]]) -> None:
        self._record_results("final", results)

    def error(self, stage: str, error: BaseException) -> None:
        self.errors.append({
            "stage": stage,
            "type": type(error).__name__,
            "message": str(error),
        })

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "query": self.query,
            "limit": self.limit,
            "projects": [trace.to_dict() for trace in self.projects],
            "stages": self.stages,
            "reranker": self.reranker,
            "cross_project_diagnostics": cross_project_diagnostics(self),
            "errors": self.errors,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def _candidate_set_metrics(
    keys: list[str],
    candidates: dict[str, dict[str, Any]],
    expected_files: list[str],
    cutoffs: tuple[int, ...],
) -> dict[str, Any]:
    paths = [str(candidates[key].get("file_path", "")) for key in keys if key in candidates]
    known_paths = [path for path in paths if path]
    unique_files = len(set(known_paths))
    duplicate_count = len(known_paths) - unique_files
    recall = {
        str(k): float(any(_matches_expected(path, expected_files) for path in paths[:k]))
        for k in cutoffs
    }
    return {
        "candidate_count": len(keys),
        "known_file_count": len(known_paths),
        "unique_file_count": unique_files,
        "duplicate_file_count": duplicate_count,
        "duplicate_file_rate": round(duplicate_count / len(known_paths), 6) if known_paths else 0.0,
        "file_diversity": round(unique_files / len(known_paths), 6) if known_paths else 0.0,
        "recall_at_k": recall,
    }


def candidate_retrieval_diagnostics(
    trace: SearchTrace,
    expected_files: list[str],
    *,
    cutoffs: tuple[int, ...] = (10, 20, 50),
) -> dict[str, Any]:
    """Compute per-channel and fused candidate diagnostics for one query."""
    channels = {
        label: _candidate_set_metrics(
            trace.candidate_keys_for_channel(label),
            trace.candidates,
            expected_files,
            cutoffs,
        )
        for label in trace.channels
        if trace.channels[label].get("status") in {"used", "empty"}
    }
    union_keys = [entry["key"] for entry in trace.stages.get("union", [])]
    return {
        "channels": channels,
        "union": _candidate_set_metrics(union_keys, trace.candidates, expected_files, cutoffs),
        "attribution_complete": not trace.validate_complete_attribution(),
    }


def aggregate_candidate_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Macro-average query-level candidate diagnostics by named candidate set."""
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        diagnostics = row.get("candidate_diagnostics")
        if not diagnostics:
            continue
        for label, values in diagnostics.get("channels", {}).items():
            buckets.setdefault(f"channel/{label}", []).append(values)
        buckets.setdefault("union", []).append(diagnostics["union"])

    output: dict[str, Any] = {}
    for label, values in sorted(buckets.items()):
        cutoff_names = sorted({cutoff for value in values for cutoff in value["recall_at_k"]}, key=int)
        output[label] = {
            "queries": len(values),
            "mean_candidate_count": round(fmean(value["candidate_count"] for value in values), 6),
            "mean_duplicate_file_rate": round(fmean(value["duplicate_file_rate"] for value in values), 6),
            "mean_file_diversity": round(fmean(value["file_diversity"] for value in values), 6),
            "recall_at_k": {
                cutoff: round(fmean(value["recall_at_k"].get(cutoff, 0.0) for value in values), 6)
                for cutoff in cutoff_names
            },
        }
    return output


def federated_candidate_diagnostics(
    trace: FederatedSearchTrace,
    expected_files: list[str],
    *,
    cutoffs: tuple[int, ...] = (10, 20, 50),
) -> dict[str, Any]:
    """Report recall/diversity for global union, rerank pool, and final sets."""
    candidate_map = {
        entry["key"]: entry
        for stage in trace.stages.values()
        for entry in stage
    }
    return {
        stage: _candidate_set_metrics(
            [entry["key"] for entry in trace.stages.get(stage, [])],
            candidate_map,
            expected_files,
            cutoffs,
        )
        for stage in ("project_union", "rerank_pool", "final")
        if stage in trace.stages
    }


def cross_project_diagnostics(trace: FederatedSearchTrace) -> dict[str, Any]:
    """Describe project attribution and score-scale comparability.

    RRF and reranker scores are not probabilities.  The output therefore
    reports distribution and concentration diagnostics instead of claiming
    probabilistic calibration.
    """
    entries = trace.stages.get("project_union", [])
    by_project: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        label = str(entry.get("project_name") or entry.get("project_id"))
        by_project.setdefault(label, []).append(entry)

    projects: dict[str, Any] = {}
    total = len(entries)
    for label, project_entries in sorted(by_project.items()):
        scores = [entry["score"] for entry in project_entries if entry.get("score") is not None]
        projects[label] = {
            "candidate_count": len(project_entries),
            "candidate_share": round(len(project_entries) / total, 6) if total else 0.0,
            "best_global_rank": min(entry["rank"] for entry in project_entries),
            "score_min": min(scores) if scores else None,
            "score_max": max(scores) if scores else None,
            "score_mean": fmean(scores) if scores else None,
            "score_stddev": pstdev(scores) if len(scores) > 1 else 0.0 if scores else None,
        }

    labels = sorted(projects)
    overlaps = []
    for index, left in enumerate(labels):
        for right in labels[index + 1:]:
            left_min, left_max = projects[left]["score_min"], projects[left]["score_max"]
            right_min, right_max = projects[right]["score_min"], projects[right]["score_max"]
            overlaps.append({
                "projects": [left, right],
                "score_ranges_overlap": bool(
                    left_min is not None
                    and right_min is not None
                    and max(left_min, right_min) <= min(left_max, right_max)
                ),
            })

    shares = [project["candidate_share"] for project in projects.values()]
    return {
        "scores_are_probabilities": False,
        "comparison": "raw project-local scores sorted globally",
        "project_count": len(projects),
        "source_concentration_hhi": round(sum(share * share for share in shares), 6),
        "projects": projects,
        "pairwise_score_range_overlap": overlaps,
        "final_project_sequence": [
            entry.get("project_name") or entry.get("project_id")
            for entry in trace.stages.get("final", [])
        ],
    }
