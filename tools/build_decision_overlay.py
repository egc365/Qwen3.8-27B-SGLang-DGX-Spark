#!/usr/bin/env python3

import argparse
import ast
import hashlib
import json
from pathlib import Path


SCORING_METHOD = '''    async def _score(self, adapted_request, raw_request, temperature=1.0):
        prompts, label_token_ids = await _encode_all(adapted_request)
        batch = GenerateReqInput(
            input_ids=prompts,
            token_ids_logprob=label_token_ids,
            return_logprob=True,
            logprob_start_len=-1,
            stream=False,
            sampling_params={"max_new_tokens": 0},
        )
        results = await self.tokenizer_manager.generate_request(
            batch, raw_request
        ).__anext__()
        if not isinstance(results, list) or len(results) != len(prompts):
            raise RuntimeError("decision scoring returned the wrong batch size")
        scores, token_logprobs, prompt_tokens = [], [], 0
        for result, labels in zip(results, label_token_ids):
            meta = result["meta_info"]
            if meta.get("completion_tokens", 0) != 0:
                raise RuntimeError("decision scoring unexpectedly generated tokens")
            positions = meta.get("output_token_ids_logprobs", [])
            if not positions:
                raise RuntimeError("decision scoring returned no candidate logprobs")
            by_id = self.tokenizer_manager._extract_logprobs_for_tokens(
                positions[0], labels
            )
            if any(token not in by_id for token in labels):
                raise RuntimeError("decision scoring omitted a candidate token")
            row = [by_id[token] for token in labels]
            if any(math.isnan(value) or value == math.inf for value in row):
                raise RuntimeError("decision scoring returned invalid candidate logprobs")
            maximum = max(row)
            if maximum == -math.inf:
                raise RuntimeError("decision scoring assigned zero mass to all candidates")
            weights = [math.exp((value - maximum) / temperature) for value in row]
            total = math.fsum(weights)
            scores.append([weight / total for weight in weights])
            token_logprobs.append(row)
            prompt_tokens += meta.get("prompt_tokens", 0)
        return prompts, label_token_ids, SimpleNamespace(
            scores=scores, token_logprobs=token_logprobs, prompt_tokens=prompt_tokens
        )
'''


ROUND_TRIP_METHOD = '''    def _probe_prompt_text_round_trip(self):
        tokenizer = self.tokenizer_manager.tokenizer
        if tokenizer is None:
            return False
        messages = [{"role": "user", "content": "x"}]
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, return_dict=False
            )
            text_ids = tokenizer.encode(prompt, add_special_tokens=False)
            direct_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=False
            )
            return list(text_ids) != list(direct_ids)
        except Exception:
            return True
'''


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"expected one source anchor: {old[:100]!r}")
    return source.replace(old, new, 1)


def replace_method(source, class_name, method_name, replacement):
    cls = next(node for node in ast.parse(source).body
               if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in cls.body
                  if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                  and node.name == method_name)
    lines = source.splitlines(keepends=True)
    lines[method.lineno - 1:method.end_lineno] = [replacement]
    return "".join(lines)


def build_overlay(base, upstream, output):
    base, upstream, output = Path(base), Path(upstream), Path(output)
    source_hashes = {}

    def read(root, relative):
        text = (root / relative).read_text()
        source_hashes[("base/" if root == base else "upstream/") + relative] = (
            hashlib.sha256(text.encode()).hexdigest()
        )
        return text

    protocol_path = "srt/entrypoints/openai/protocol.py"
    protocol = read(base, protocol_path)
    if "class DecisionRequest(" in protocol:
        raise ValueError("base already contains decisions; do not overlay it")
    upstream_protocol = read(upstream, protocol_path)
    start = upstream_protocol.index("def is_blank_decision_text(")
    end = upstream_protocol.index("\n\nclass V1RerankReqInput(", start)
    protocol += "\n\nimport unicodedata\n\n" + upstream_protocol[start:end] + "\n"

    handler_path = "srt/entrypoints/openai/serving_decisions.py"
    handler = read(upstream, handler_path)
    handler = replace_once(handler, "import string\n", "import string\nfrom types import SimpleNamespace\n")
    handler = replace_once(handler, "import msgspec\n", "import msgspec\nfrom sglang.srt.managers.io_struct import GenerateReqInput\n")
    handler = replace_once(handler,
        "self.prompt_text_is_lossy = chat_serving._prompt_text_round_trip_is_lossy",
        "self.prompt_text_is_lossy = self._probe_prompt_text_round_trip()")
    handler = replace_once(handler,
        "or self.template_manager.suggested_reasoning_parser",
        'or getattr(self.template_manager, "suggested_reasoning_parser", None)')
    handler = replace_once(handler, "        route = self.route\n",
        '        route = self.route\n'
        '        served_names = {self.tokenizer_manager.served_model_name,\n'
        '                        self.tokenizer_manager.server_args.model_path}\n'
        '        if model not in served_names:\n'
        '            return f"model {model!r} is not served by {route}"\n')
    handler = handler.replace("config = self.template_manager.reasoning_config",
                              'config = getattr(self.template_manager, "reasoning_config", None)')
    handler = replace_method(handler, "OpenAIServingDecisions", "_score", SCORING_METHOD)
    handler = replace_once(handler, "    def _request_id_prefix(self)",
                            ROUND_TRIP_METHOD + "\n    def _request_id_prefix(self)")

    http_path = "srt/entrypoints/http_server.py"
    http = read(base, http_path)
    http = replace_once(http, "import orjson\n", "import json\nimport orjson\n")
    imports = '''from sglang.srt.entrypoints.openai.protocol import DecisionRequest
from sglang.srt.entrypoints.openai.serving_decisions import OpenAIServingDecisions
from sglang.srt.entrypoints.systemone.protocol import SystemOneRequest
from sglang.srt.entrypoints.systemone.serving import SystemOneServing
'''
    http = replace_once(http, "from fastapi.exceptions import RequestValidationError\n",
                        "from fastapi.exceptions import RequestValidationError\n" + imports)
    anchor = "    fast_api_app.state.openai_serving_embedding = OpenAIServingEmbedding("
    initialize = '''    fast_api_app.state.openai_serving_decisions = OpenAIServingDecisions(
        fast_api_app.state.openai_serving_chat
    )
    fast_api_app.state.systemone_serving = SystemOneServing(
        fast_api_app.state.openai_serving_chat
    )
'''
    http = replace_once(http, anchor, initialize + anchor)
    route_anchor = '@app.post("/v1/chat/completions", dependencies=[Depends(validate_json_request)])'
    routes = '''@app.post("/v1/decisions", dependencies=[Depends(validate_json_request)])
async def v1_decisions_request(request: DecisionRequest, raw_request: Request):
    return await raw_request.app.state.openai_serving_decisions.handle_request(
        request, raw_request
    )


@app.post("/v1/systemone", dependencies=[Depends(validate_json_request)])
async def systemone_decisions(request: SystemOneRequest, raw_request: Request):
    return await raw_request.app.state.systemone_serving.handle_request(
        request, raw_request
    )


'''
    http = replace_once(http, route_anchor, routes + route_anchor)
    error_anchor = '    if request.url.path.startswith("/v1/messages"):\n'
    validation = next(node for node in ast.parse(http).body
                      if isinstance(node, ast.AsyncFunctionDef)
                      and node.name == "validation_exception_handler"
                      and any(isinstance(decorator, ast.Call)
                              and any(isinstance(arg, ast.Name)
                                      and arg.id == "RequestValidationError"
                                      for arg in decorator.args)
                              for decorator in node.decorator_list))
    lines = http.splitlines(keepends=True)
    validation_source = "".join(lines[validation.lineno - 1:validation.end_lineno])
    validation_source = replace_once(validation_source, error_anchor,
        '    if request.url.path == "/v1/systemone":\n'
        '        return ORJSONResponse(status_code=422, content={"detail": '
        'json.loads(json.dumps(exc.errors(), default=str))})\n' + error_anchor)
    lines[validation.lineno - 1:validation.end_lineno] = [validation_source]
    http = "".join(lines)

    files = {protocol_path: protocol, handler_path: handler, http_path: http}
    for relative in ("srt/entrypoints/systemone/__init__.py",
                     "srt/entrypoints/systemone/protocol.py",
                     "srt/entrypoints/systemone/serving.py"):
        files[relative] = read(upstream, relative)
    for relative, text in sorted(files.items()):
        ast.parse(text)
        destination = output / "sglang" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text)
    manifest = {
        "source_sha256": dict(sorted(source_hashes.items())),
        "output_sha256": {"sglang/" + path: hashlib.sha256(text.encode()).hexdigest()
                          for path, text in sorted(files.items())},
        "compatibility": "native upstream rendering/schemas; direct zero-token candidate logprob batch",
        "generation_tokens": 0,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description='Build the upstream native decision handlers on the installed SGLang runtime.')
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = build_overlay(args.base, args.upstream, args.output)
    print(json.dumps({"files": len(manifest["output_sha256"]), "generation_tokens": 0}))


if __name__ == "__main__":
    main()
