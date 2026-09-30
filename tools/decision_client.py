#!/usr/bin/env python3

import argparse
import json
import math
import sys
import unicodedata
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


class DecisionError(ValueError):
    pass


def resolve_target(base_url, model, config):
    if base_url is not None and model is not None:
        return base_url, model
    try:
        settings = json.loads(Path(config).read_text(encoding="utf-8"))
        if model is None:
            model = settings["model"]
        if base_url is None:
            host, port = settings["host"], settings["port"]
            if not isinstance(host, str) or not host.strip() or type(port) is not int or not 1 <= port <= 65535:
                raise ValueError("invalid host or port")
            base_url = f"http://{host}:{port}"
        if not isinstance(model, str) or not model.strip():
            raise ValueError("invalid model")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise DecisionError(f"cannot load routing defaults from {config}: {exc}") from exc
    return base_url, model


def decision_payload(model, input_value, question, labels=None, levels=None):
    if not model.strip() or not question.strip():
        raise DecisionError("model and question must be nonblank")
    if not isinstance(input_value, (str, dict, list)) or not input_value:
        raise DecisionError("input must be nonblank text, an object, or an array")
    if isinstance(input_value, str) and not input_value.strip():
        raise DecisionError("input must be nonblank")
    if (labels is None) == (levels is None):
        raise DecisionError("provide either labels or levels")
    values = labels if labels is not None else levels
    limit = 26 if labels is not None else 10
    if not 2 <= len(values) <= limit or any(not v.strip() for v in values):
        raise DecisionError(f"provide 2 to {limit} nonblank values")
    item = {"id": "decision", "question": question}
    if labels is not None:
        keys = [v.strip().casefold() for v in labels]
        if len(set(keys)) != len(keys) or any(
            unicodedata.category(c) in ("Cc", "Zl", "Zp") for v in labels for c in v
        ):
            raise DecisionError("labels must be distinct and contain no control characters")
        item.update(type="choice", options=[{"name": v} for v in labels])
    else:
        item.update(type="score", levels=levels)
    return {"model": model, "input": input_value, "questions": [item]}


def request_json(base_url, path, payload, timeout):
    parsed = urlsplit(base_url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.query or parsed.fragment:
        raise DecisionError("base-url must be an HTTP(S) service URL")
    if not math.isfinite(timeout) or timeout <= 0:
        raise DecisionError("timeout must be positive and finite")
    request = Request(base_url.rstrip("/") + path,
                      data=json.dumps(payload, allow_nan=False).encode(),
                      headers={"Content-Type": "application/json", "Accept": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            if response.headers.get_content_type() != "application/json":
                raise DecisionError("server returned non-JSON content type")
            result = json.load(response)
    except HTTPError as exc:
        detail = exc.read(4096).decode(errors="replace")
        raise DecisionError(f"HTTP {exc.code}: {detail}") from exc
    except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise DecisionError(f"request failed: {exc}") from exc
    if not isinstance(result, dict):
        raise DecisionError("server response must be a JSON object")
    return result


def validate_answer(answer, kind, names, mass_key="label_mass"):
    if not isinstance(answer, dict) or answer.get("type") != kind:
        raise DecisionError("answer type does not match request")
    probabilities = answer.get("probabilities")
    if not isinstance(probabilities, dict) or set(probabilities) != set(names):
        raise DecisionError("probability labels do not match request")
    values = list(probabilities.values())
    if any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in values):
        raise DecisionError("probabilities must be finite and in [0, 1]")
    if not math.isclose(math.fsum(values), 1, rel_tol=1e-5, abs_tol=1e-5):
        raise DecisionError("probabilities must sum to one")
    mass = answer.get(mass_key)
    if type(mass) not in (int, float) or not math.isfinite(mass) or not 0 <= mass <= 1.00001:
        raise DecisionError("label mass must be finite and in [0, 1]")
    if kind == "choice":
        choice = answer.get("choice")
        if choice not in names or probabilities[choice] != max(values):
            raise DecisionError("choice must be a maximum-probability label")
    else:
        score = answer.get("score")
        expected = math.fsum(int(name) * probabilities[name] for name in names)
        if type(score) not in (int, float) or not math.isfinite(score) or not math.isclose(score, expected, abs_tol=1e-5):
            raise DecisionError("score must equal the expected level index")


def decide(base_url, payload, timeout=30):
    result = request_json(base_url, "/v1/decisions", payload, timeout)
    if result.get("object") != "decisions" or result.get("model") != payload["model"]:
        raise DecisionError("response object or model does not match request")
    if type(result.get("prompt_format_version")) is not int or not isinstance(result.get("usage"), dict):
        raise DecisionError("missing native decision response metadata")
    if type(result["usage"].get("completion_tokens")) is not int or result["usage"]["completion_tokens"] != 0:
        raise DecisionError("native decisions must report zero completion tokens")
    if not isinstance(result.get("answers"), dict) or set(result["answers"]) != {q["id"] for q in payload["questions"]}:
        raise DecisionError("answer ids do not match request")
    for item in payload["questions"]:
        names = [v["name"] for v in item["options"]] if item["type"] == "choice" else [str(i) for i in range(len(item["levels"]))]
        validate_answer(result["answers"][item["id"]], item["type"], names)
    return result


def main():
    parser = argparse.ArgumentParser(description="Call SGLang's native decision endpoint using only the Python standard library.")
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "classifier.json")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input")
    inputs.add_argument("--input-file", type=Path, help="UTF-8 text, or JSON object/array in a .json file")
    parser.add_argument("--question", required=True)
    choices = parser.add_mutually_exclusive_group(required=True)
    choices.add_argument("--labels", nargs="+")
    choices.add_argument("--levels", nargs="+")
    parser.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args()
    try:
        base_url, model = resolve_target(args.base_url, args.model, args.config)
        value = args.input
        if args.input_file:
            value = args.input_file.read_text(encoding="utf-8")
            if args.input_file.suffix.lower() == ".json":
                value = json.loads(value)
        payload = decision_payload(model, value, args.question, args.labels, args.levels)
        print(json.dumps(decide(base_url, payload, args.timeout), allow_nan=False))
    except (DecisionError, OSError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
