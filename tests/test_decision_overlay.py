import ast
import asyncio
import importlib.util
import json
import math
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
(ROOT / ".scratch").mkdir(exist_ok=True)
SPEC = importlib.util.spec_from_file_location(
    "overlay_builder", ROOT / "tools/build_decision_overlay.py"
)
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


class CandidateScoringTest(unittest.IsolatedAsyncioTestCase):
    def scorer(self, logprobs, completion_tokens=0):
        calls = []

        class Manager:
            async def generate_request(self, batch, request):
                calls.append(batch)
                yield [{"meta_info": {
                    "prompt_tokens": 7, "completion_tokens": completion_tokens,
                    "output_token_ids_logprobs": [
                        [(value, token, "") for token, value in row.items()]
                    ],
                }} for row in logprobs]

            def _extract_logprobs_for_tokens(self, values, labels):
                return {token: value for value, token, _ in values if token in labels}

        async def encode_all(adapted):
            return [[1, 2] for _ in logprobs], adapted

        source = "class Scorer:\n" + BUILDER.SCORING_METHOD
        namespace = {"math": math, "GenerateReqInput": SimpleNamespace,
                     "SimpleNamespace": SimpleNamespace, "_encode_all": encode_all}
        exec(compile(source, "fixture_score", "exec"), namespace)
        instance = namespace["Scorer"]()
        instance.tokenizer_manager = Manager()
        return instance, calls

    async def test_candidate_distribution_preserves_raw_mass_and_no_generation(self):
        scorer, calls = self.scorer([{11: math.log(.15), 22: math.log(.05)},
                                    {33: math.log(.02), 44: math.log(.08)}])
        prompts, labels, result = await scorer._score([[11, 22], [33, 44]], None)
        self.assertEqual(calls[0].sampling_params, {"max_new_tokens": 0})
        self.assertEqual(calls[0].logprob_start_len, -1)
        self.assertEqual(calls[0].token_ids_logprob, [[11, 22], [33, 44]])
        self.assertEqual(result.prompt_tokens, 14)
        self.assertAlmostEqual(result.scores[0][0], .75)
        self.assertAlmostEqual(result.scores[1][0], .2)
        self.assertAlmostEqual(sum(map(math.exp, result.token_logprobs[0])), .2)

    async def test_temperature_matches_candidate_softmax_not_full_vocab_mass(self):
        scorer, _ = self.scorer([{11: math.log(.15), 22: math.log(.05)}])
        _, _, result = await scorer._score([[11, 22]], None, temperature=2)
        expected = math.sqrt(3) / (math.sqrt(3) + 1)
        self.assertAlmostEqual(result.scores[0][0], expected)
        self.assertAlmostEqual(sum(map(math.exp, result.token_logprobs[0])), .2)

    async def test_zero_probability_and_underflow_are_handled(self):
        scorer, _ = self.scorer([{11: -10000., 22: -10001., 33: -math.inf}])
        _, _, result = await scorer._score([[11, 22, 33]], None)
        self.assertAlmostEqual(result.scores[0][0], 1 / (1 + math.exp(-1)))
        self.assertEqual(result.scores[0][2], 0)

    async def test_missing_labels_fail_instead_of_guessing(self):
        scorer, _ = self.scorer([{11: math.log(.15)}])
        with self.assertRaisesRegex(RuntimeError, "omitted"):
            await scorer._score([[11, 22]], None)

    async def test_generated_tokens_fail(self):
        scorer, _ = self.scorer([{11: 0.}], completion_tokens=1)
        with self.assertRaisesRegex(RuntimeError, "generated"):
            await scorer._score([[11]], None)

    async def test_nonfinite_candidates_fail(self):
        for row in ({11: -math.inf}, {11: math.nan}, {11: math.inf}):
            scorer, _ = self.scorer([row])
            with self.assertRaises(RuntimeError):
                await scorer._score([[11]], None)


class OverlayBuilderTest(unittest.TestCase):
    def test_model_identity_guard_rejects_unknown_names_on_both_routes(self):
        base = ROOT.parent / "installed-sglang"
        upstream = ROOT.parent / "upstream-sglang/python/sglang"
        if not base.exists() or not upstream.exists():
            self.skipTest("exported runtime source needed for handler validation")
        with tempfile.TemporaryDirectory(dir=ROOT / ".scratch") as directory:
            BUILDER.build_overlay(base, upstream, directory)
            tree = ast.parse((Path(directory) / "sglang/srt/entrypoints/openai/serving_decisions.py").read_text())
        cls = next(node for node in tree.body
                   if isinstance(node, ast.ClassDef) and node.name == "OpenAIServingDecisions")
        method = next(node for node in cls.body
                      if isinstance(node, ast.FunctionDef) and node.name == "_validate_server")
        namespace = {"Optional": __import__("typing").Optional,
                     "get_exec": lambda: SimpleNamespace(
                         features=SimpleNamespace(enable_mis=False),
                         dllm=SimpleNamespace(dllm_algorithm=None))}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "model_guard", "exec"), namespace)
        class Handler:
            _validate_server = namespace["_validate_server"]
            _parse_model_parameter = staticmethod(lambda model: (model, None))
        handler = Handler()
        handler.tokenizer_manager = SimpleNamespace(
            served_model_name="qwen-classifier", server_args=SimpleNamespace(model_path="/models/qwen"),
            is_generation=True, tokenizer=object())
        handler.template_manager = SimpleNamespace(chat_template_name=None)
        handler.chat_encoding_spec = None
        handler.prompt_text_is_lossy = False
        for route in ("/v1/decisions", "/v1/systemone"):
            handler.route = route
            self.assertIsNone(handler._validate_server("qwen-classifier"))
            self.assertIsNone(handler._validate_server("/models/qwen"))
            self.assertIn("is not served", handler._validate_server("unknown-model"))
            self.assertIn("is not served", handler._validate_server("default"))

    def test_unreadable_template_is_not_declared_lossless(self):
        class Tokenizer:
            def apply_chat_template(self, *args, **kwargs):
                raise ValueError("required template kwarg missing")

        namespace = {}
        exec("class Probe:\n" + BUILDER.ROUND_TRIP_METHOD, namespace)
        probe = namespace["Probe"]()
        probe.tokenizer_manager = SimpleNamespace(tokenizer=Tokenizer())
        self.assertTrue(probe._probe_prompt_text_round_trip())

    def test_anchor_drift_fails_closed(self):
        with self.assertRaises(ValueError):
            BUILDER.replace_once("other version", "expected anchor", "new")

    def test_build_is_deterministic_and_does_not_replace_gpu_runtime(self):
        base = ROOT.parent / "installed-sglang"
        upstream = ROOT.parent / "upstream-sglang/python/sglang"
        if not base.exists() or not upstream.exists():
            self.skipTest("exported runtime source needed for integration builder check")
        with tempfile.TemporaryDirectory(dir=ROOT / ".scratch") as directory:
            first, second = Path(directory) / "first", Path(directory) / "second"
            manifest = BUILDER.build_overlay(base, upstream, first)
            self.assertEqual(manifest, BUILDER.build_overlay(base, upstream, second))
            self.assertEqual(len(manifest["output_sha256"]), 6)
            self.assertEqual(manifest["generation_tokens"], 0)
            server = ast.parse((first / "sglang/srt/entrypoints/http_server.py").read_text())
            exception_handlers = [node for node in server.body
                                  if isinstance(node, ast.AsyncFunctionDef)
                                  and node.name == "validation_exception_handler"]
            self.assertEqual(len(exception_handlers), 2)
            self.assertNotIn('request.url.path == "/v1/systemone"',
                             ast.unparse(exception_handlers[0]))
            self.assertIn("/v1/systemone", ast.unparse(exception_handlers[1]))
            self.assertTrue(all("entrypoints/" in name
                                for name in manifest["output_sha256"]))
            for relative in manifest["output_sha256"]:
                self.assertEqual((first / relative).read_bytes(), (second / relative).read_bytes())
                ast.parse((first / relative).read_text())


@unittest.skipUnless(os.environ.get("DECISION_OVERLAY_IMAGE"), "optional Docker ASGI measurement")
class DockerRouteTest(unittest.TestCase):
    def test_actual_base_and_overlay_fastapi_validation(self):
        probe = '''import asyncio,httpx,json
from types import SimpleNamespace
from sglang.srt.entrypoints.http_server import app
async def check():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url="http://test") as client:
        results={}
        for path in ("/v1/decisions","/v1/systemone"):
            response=await client.post(path,json={})
            results[path]={"status":response.status_code,"keys":list(response.json())}
        if results["/v1/decisions"]["status"] != 404:
            from sglang.srt.entrypoints.openai.serving_decisions import OpenAIServingDecisions
            from sglang.srt.entrypoints.systemone.serving import SystemOneServing
            manager=SimpleNamespace(served_model_name="qwen-classifier",server_args=SimpleNamespace(model_path="/models/qwen"))
            for handler_type,attribute in ((OpenAIServingDecisions,"openai_serving_decisions"),(SystemOneServing,"systemone_serving")):
                handler=handler_type.__new__(handler_type)
                handler.tokenizer_manager=manager
                setattr(app.state,attribute,handler)
            payloads={"/v1/decisions":{"model":"unknown-model","input":"hello","questions":[{"id":"c","type":"choice","question":"classify","options":[{"name":"a"},{"name":"b"}]}]},"/v1/systemone":{"model":"unknown-model","state":"hello","questions":{"c":{"type":"choice","criteria":{"a":None,"b":None}}}}}
            for path,payload in payloads.items():
                response=await client.post(path,json=payload)
                results[path]["wrong_model_status"]=response.status_code
                results[path]["wrong_model_message"]=response.json().get("message")
        print(json.dumps(results,sort_keys=True))
asyncio.run(check())
'''
        script = f'''import subprocess,sys,pathlib,shutil,sglang,json
probe={probe!r}
def run():
    completed=subprocess.run([sys.executable,"-c",probe],capture_output=True,text=True)
    if completed.returncode:
        raise RuntimeError(completed.stderr)
    return json.loads(completed.stdout.strip().splitlines()[-1])
before=run()
shutil.copytree("/decision-overlay/sglang",pathlib.Path(sglang.__file__).parent,dirs_exist_ok=True)
after=run()
print(json.dumps({{"instrument_run_id":"decision-overlay-20260930-fastapi01","before":before,"after":after}},sort_keys=True))
'''
        with tempfile.TemporaryDirectory(dir=ROOT / ".scratch") as directory:
            BUILDER.build_overlay(ROOT.parent / "installed-sglang",
                                  ROOT.parent / "upstream-sglang/python/sglang", directory)
            completed = subprocess.run([
                "docker", "run", "--rm", "-v", f"{directory}:/decision-overlay:ro",
                "--entrypoint", "python", os.environ["DECISION_OVERLAY_IMAGE"], "-c", script,
            ], capture_output=True, text=True, timeout=180)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        receipt = json.loads(completed.stdout.strip().splitlines()[-1])
        self.assertEqual(receipt["before"]["/v1/decisions"]["status"], 404)
        self.assertEqual(receipt["before"]["/v1/systemone"]["status"], 404)
        self.assertEqual(receipt["after"]["/v1/decisions"]["status"], 400)
        self.assertEqual(receipt["after"]["/v1/systemone"]["status"], 422)
        self.assertEqual(receipt["after"]["/v1/systemone"]["keys"], ["detail"])
        for path in ("/v1/decisions", "/v1/systemone"):
            self.assertEqual(receipt["after"][path]["wrong_model_status"], 400)
            self.assertIn("is not served", receipt["after"][path]["wrong_model_message"])
        print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
