#!/usr/bin/env bash
set -euo pipefail

MODEL="qwen3.8-27b-sglang"

for base_url in \
  http://127.0.0.1:8888/v1 \
  http://127.0.0.1:4000/v1; do
  echo "Testing ${base_url}"
  curl -fsS "${base_url}/models" | jq -e --arg model "${MODEL}" \
    '.data | any(.id == $model)' >/dev/null

  math_response="$(curl -fsS "${base_url}/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$(jq -nc --arg model "${MODEL}" '{
      model: $model,
      messages: [{role: "user", content: "Return only the integer result of 17 multiplied by 23."}],
      max_tokens: 64,
      temperature: 0,
      chat_template_kwargs: {enable_thinking: false}
    }')")"
  jq -e '.choices[0].message.content | contains("391")' \
    <<<"${math_response}" >/dev/null

  tool_response="$(curl -fsS "${base_url}/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$(jq -nc --arg model "${MODEL}" '{
      model: $model,
      messages: [{role: "user", content: "Call record_result with result 391. Do not answer in prose."}],
      tools: [{
        type: "function",
        function: {
          name: "record_result",
          description: "Record an integer calculation result.",
          parameters: {
            type: "object",
            properties: {result: {type: "integer"}},
            required: ["result"]
          }
        }
      }],
      tool_choice: "required",
      max_tokens: 128,
      temperature: 0,
      chat_template_kwargs: {enable_thinking: false}
    }')")"
  jq -e '
    .choices[0].message.tool_calls[0]
    | .function.name == "record_result"
      and ((.function.arguments | fromjson).result == 391)
  ' <<<"${tool_response}" >/dev/null
done

echo "Direct SGLang and Switchyard math/tool smoke tests passed."
