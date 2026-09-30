import contextlib
import copy
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from decision_client import DecisionError, decide, decision_payload, resolve_target
from verify_classifier import verify


@contextlib.contextmanager
def server(response=None, status=200, content_type="application/json"):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, payload))
            body = response(payload, self.path) if callable(response) else response
            code = 404 if payload["model"].endswith("-deliberately-invalid") else status
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.end_headers()
            self.wfile.write((body if isinstance(body, str) else json.dumps(body)).encode())

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}", requests
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join()


def native_response(payload, path="/v1/decisions"):
    if path == "/v1/systemone":
        return {"model": payload["model"], "usage": {"input_tokens": 20, "output_tokens": 0},
                "answers": {"color": {"type": "choice", "choice": "red", "confidence": .8,
                                       "probabilities": {"red": .9, "blue": .1}, "x_label_mass": .4},
                            "is_red": {"type": "noul", "noul": .9, "x_label_mass": .4},
                            "sentiment": {"type": "score", "score": 1., "confidence": .1,
                                          "probabilities": {"0": .2, "1": .6, "2": .2},
                                          "legend": {"0": "negative", "1": "neutral", "2": "positive"}, "x_label_mass": .3}}}
    item = payload["questions"][0]
    if item["type"] == "choice":
        chosen = "blue" if "blue" in str(payload["input"]) else "red"
        probabilities = {o["name"]: (.9 if o["name"] == chosen else .1) for o in item["options"]}
        answer = {"type": "choice", "choice": chosen, "probabilities": probabilities, "label_mass": .4}
    else:
        probabilities = {"0": .1, "1": .2, "2": .7} if "love" in payload["input"] else {"0": .7, "1": .2, "2": .1}
        answer = {"type": "score", "score": sum(int(k) * p for k, p in probabilities.items()),
                  "probabilities": probabilities, "label_mass": .3}
    return {"object": "decisions", "model": payload["model"], "prompt_format_version": 1,
            "answers": {item["id"]: answer}, "usage": {"prompt_tokens": 20, "completion_tokens": 0, "total_tokens": 20}}


class DecisionClientTests(unittest.TestCase):
    def setUp(self):
        self.payload = decision_payload("local-qwen", "red", "Color?", ["red", "blue"])

    def test_exact_native_payload_and_transport(self):
        with server(native_response) as (url, requests):
            answer = decide(url, self.payload)
        self.assertEqual(requests, [("/v1/decisions", {"model": "local-qwen", "input": "red",
            "questions": [{"id": "decision", "type": "choice", "question": "Color?",
                           "options": [{"name": "red"}, {"name": "blue"}]}]})])
        self.assertEqual(answer["answers"]["decision"]["choice"], "red")

    def test_score_uses_level_indices(self):
        payload = decision_payload("local-qwen", "love", "Sentiment?", levels=["negative", "neutral", "positive"])
        with server(native_response) as (url, requests):
            result = decide(url, payload)
        self.assertEqual(requests[0][1]["questions"][0]["levels"], ["negative", "neutral", "positive"])
        self.assertAlmostEqual(result["answers"]["decision"]["score"], 1.6)

    def test_invalid_cli_shapes(self):
        for labels in (["red"], ["red", " RED "], ["red\n", "blue"], [" ", "blue"]):
            with self.subTest(labels=labels), self.assertRaises(DecisionError):
                decision_payload("qwen", "text", "question", labels)
        with self.assertRaises(DecisionError):
            decision_payload("qwen", "text", "question", ["red", "blue"], ["a", "b"])

    def test_response_errors_rejected(self):
        good = native_response(self.payload)
        mutations = [lambda r: r.update(model="wrong"), lambda r: r.update(object="chat.completion"),
                     lambda r: r["answers"]["decision"]["probabilities"].update(red=float("nan")),
                     lambda r: r["answers"]["decision"]["probabilities"].update(red=.5),
                     lambda r: r["answers"]["decision"].update(choice="blue"),
                     lambda r: r["answers"]["decision"].update(label_mass=-.1),
                     lambda r: r.update(answers={}),
                     lambda r: r["usage"].update(completion_tokens=1)]
        for mutate in mutations:
            result = copy.deepcopy(good)
            mutate(result)
            with self.subTest(result=result), server(result) as (url, _), self.assertRaises(DecisionError):
                decide(url, self.payload)

    def test_http_content_and_json_errors(self):
        for body, status, content_type in [("bad", 503, "text/plain"), ("<html>", 200, "text/html"),
                                          ("bad", 200, "application/json")]:
            with self.subTest(status=status, content_type=content_type), server(body, status, content_type) as (url, _):
                with self.assertRaises(DecisionError):
                    decide(url, self.payload)

    def test_cli_round_trip(self):
        with server(native_response) as (url, _):
            result = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "tools/decision_client.py"),
                "--base-url", url, "--model", "local-qwen", "--input", "red", "--question", "Color?",
                "--labels", "red", "blue"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["model"], "local-qwen")

    def test_routing_defaults_and_explicit_override(self):
        scratch = Path(__file__).resolve().parents[1] / ".scratch"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as temporary:
            config = Path(temporary) / "config.json"
            config.write_text(json.dumps({"host": "spark-example", "port": 30009, "model": "configured-qwen"}))
            self.assertEqual(resolve_target(None, None, config), ("http://spark-example:30009", "configured-qwen"))
            self.assertEqual(resolve_target("http://explicit:123", "explicit-qwen", config), ("http://explicit:123", "explicit-qwen"))
            self.assertEqual(resolve_target("http://explicit:123", None, config), ("http://explicit:123", "configured-qwen"))
            config.unlink()
            self.assertEqual(resolve_target("http://explicit:123", "explicit-qwen", config), ("http://explicit:123", "explicit-qwen"))
            with self.assertRaisesRegex(DecisionError, "cannot load routing defaults"):
                resolve_target(None, None, config)

    def test_verifier_all_native_paths_and_bounded_measurements(self):
        with server(native_response) as (url, requests):
            receipt = verify(url, "local-qwen", 5, 2)
        self.assertEqual(receipt["status"], "passed", receipt["errors"])
        self.assertEqual(len(requests), 10)
        self.assertEqual(receipt["latency_ms"]["warm_samples"], 4)
        self.assertEqual(len(receipt["latency_ms"]["first_pass"]), 4)
        self.assertEqual(receipt["samples"][0]["usage"]["completion_tokens"], 0)
        self.assertEqual(receipt["samples"][-1]["usage"]["output_tokens"], 0)
        self.assertGreater(receipt["latency_ms"]["warm_p95"], 0)

    def test_verifier_single_pass_has_no_warm_samples(self):
        with server(native_response) as (url, _):
            receipt = verify(url, "local-qwen", 5, 1)
        self.assertEqual(receipt["status"], "passed", receipt["errors"])
        self.assertEqual(receipt["latency_ms"]["warm_samples"], 0)
        self.assertIsNone(receipt["latency_ms"]["warm_p50"])

    def test_verifier_refuses_systemone_generation(self):
        def generating(payload, path):
            result = native_response(payload, path)
            if path == "/v1/systemone":
                result["usage"]["output_tokens"] = 1
            return result
        with server(generating) as (url, _):
            receipt = verify(url, "local-qwen", 5, 1)
        self.assertEqual(receipt["status"], "failed")
        self.assertIn("zero output tokens", receipt["errors"][0])

    def test_verifier_fails_semantic_mismatch(self):
        def wrong(payload, path):
            result = native_response(payload, path)
            result["answers"]["decision"].update(choice="red", probabilities={"red": .9, "blue": .1})
            return result
        with server(wrong) as (url, _):
            receipt = verify(url, "local-qwen", 5, 1)
        self.assertEqual(receipt["status"], "failed")
        self.assertIn("expected blue", receipt["errors"][0])


if __name__ == "__main__":
    unittest.main()
