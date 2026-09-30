#!/usr/bin/env python3

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from decision_client import DecisionError, decide, decision_payload, request_json, resolve_target, validate_answer


def percentile(values, fraction):
    values = sorted(values)
    return values[max(0, math.ceil(len(values) * fraction) - 1)] if values else None


def verify(base_url, model, timeout, repeats):
    fixtures = [
        ("color-red", "The object is red.", "What color is the object?", ["red", "blue"], None, "red"),
        ("color-blue", "The object is blue.", "What color is the object?", ["red", "blue"], None, "blue"),
        ("positive", "I love this. It is excellent.", "How positive is this sentiment?", None, ["negative", "neutral", "positive"], "high"),
        ("negative", "I hate this. It is terrible.", "How positive is this sentiment?", None, ["negative", "neutral", "positive"], "low"),
    ]
    receipt = {"started_at": datetime.now(timezone.utc).isoformat(), "base_url": base_url,
               "model": model, "status": "failed", "samples": [], "errors": [],
               "probabilities": "normalized label probabilities; calibration not measured",
               "cold_start": "not measured; first request may already have a warm server cache",
               "scope": "text only; DecisionText JSON is not a vision tensor path"}
    for repeat in range(repeats):
        for name, text, question, labels, levels, expected in fixtures:
            start = time.perf_counter()
            try:
                result = decide(base_url, decision_payload(model, text, question, labels, levels), timeout)
                answer = result["answers"]["decision"]
                if labels and answer["choice"] != expected:
                    raise DecisionError(f"fixture {name}: expected {expected}, got {answer['choice']}")
                if levels and ((expected == "high" and answer["score"] <= 1) or (expected == "low" and answer["score"] >= 1)):
                    raise DecisionError(f"fixture {name}: sentiment score did not separate around neutral")
                receipt["samples"].append({"fixture": name, "repeat": repeat,
                    "endpoint": "/v1/decisions", "elapsed_ms": (time.perf_counter() - start) * 1000,
                    "answer": answer, "usage": result["usage"]})
            except DecisionError as exc:
                receipt["errors"].append(str(exc))
                return receipt
    payload = {"model": model, "state": "The object is red.", "questions": {
        "color": {"type": "choice", "instructions": "What color is the object?", "criteria": {"red": None, "blue": None}},
        "sentiment": {"type": "score", "instructions": "How positive is the sentiment?", "criteria": ["negative", "neutral", "positive"]},
        "is_red": {"type": "noul", "instructions": "Is the object red?",
                   "criteria": {"true": "The object is red", "false": "The object is not red"}}}}
    start = time.perf_counter()
    try:
        result = request_json(base_url, "/v1/systemone", payload, timeout)
        if result.get("model") != model or not isinstance(result.get("usage"), dict):
            raise DecisionError("systemone returned wrong model or missing usage")
        if type(result["usage"].get("output_tokens")) is not int or result["usage"]["output_tokens"] != 0:
            raise DecisionError("native systemone must report zero output tokens")
        answers = result.get("answers")
        if not isinstance(answers, dict) or set(answers) != {"color", "sentiment", "is_red"}:
            raise DecisionError("systemone answer ids do not match")
        validate_answer(answers["color"], "choice", ["red", "blue"], "x_label_mass")
        validate_answer(answers["sentiment"], "score", ["0", "1", "2"], "x_label_mass")
        for answer in (answers["color"], answers["sentiment"]):
            confidence = answer.get("confidence")
            if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise DecisionError("systemone confidence must be finite and in [0, 1]")
        if answers["color"]["choice"] != "red" or answers["sentiment"].get("legend") != {"0": "negative", "1": "neutral", "2": "positive"}:
            raise DecisionError("systemone choice or legend mismatch")
        noul = answers["is_red"]
        if not isinstance(noul, dict) or noul.get("type") != "noul":
            raise DecisionError("systemone noul answer type mismatch")
        for key in ("noul", "x_label_mass"):
            value = noul.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1.00001:
                raise DecisionError(f"systemone {key} must be finite and in [0, 1]")
        if noul["noul"] <= .5:
            raise DecisionError("systemone noul did not recognize the red object")
        receipt["samples"].append({"fixture": "systemone", "endpoint": "/v1/systemone",
                                  "elapsed_ms": (time.perf_counter() - start) * 1000, "answers": answers,
                                  "usage": result["usage"]})
        wrong = decision_payload(model + "-deliberately-invalid", "red", "Color?", ["red", "blue"])
        try:
            request_json(base_url, "/v1/decisions", wrong, timeout)
        except DecisionError as exc:
            if not str(exc).startswith(("HTTP 400:", "HTTP 404:")):
                raise DecisionError(f"wrong-model rejection was not HTTP 400/404: {exc}") from exc
            receipt["wrong_model_rejection"] = str(exc)
        else:
            raise DecisionError("server accepted an invalid model identity")
    except DecisionError as exc:
        receipt["errors"].append(str(exc))
        return receipt
    decision_samples = [s for s in receipt["samples"] if s["endpoint"] == "/v1/decisions"]
    warm = [s["elapsed_ms"] for s in decision_samples if s["repeat"] >= 1]
    receipt["latency_ms"] = {"first_request": decision_samples[0]["elapsed_ms"],
                             "first_pass": {s["fixture"]: s["elapsed_ms"] for s in decision_samples if s["repeat"] == 0},
                             "warm_p50": percentile(warm, .5), "warm_p95": percentile(warm, .95),
                             "warm_samples": len(warm)}
    receipt["status"] = "passed"
    return receipt


def main():
    parser = argparse.ArgumentParser(description='Run bounded live native decision checks and write a machine receipt.')
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "classifier.json")
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--repeats", type=int, choices=range(1, 6), default=3)
    args = parser.parse_args()
    try:
        base_url, model = resolve_target(args.base_url, args.model, args.config)
    except DecisionError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    receipt = verify(base_url, model, args.timeout, args.repeats)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "receipt": str(args.receipt),
                      "errors": receipt["errors"], "latency_ms": receipt.get("latency_ms")}))
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
