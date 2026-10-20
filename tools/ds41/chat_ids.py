"""tools/ds41/chat_ids.py - real chat prompts as token ids, rendered and tokenized as serve/server.py does it
(serve/deepseek.py: the official encoder and the DeepSeek tokenizer), for ds41_generate --ids.

    python tools/ds41/chat_ids.py --tokenizer PACK_DIR --out DIR

Writes DIR/<name>.ids (comma separated) for a few kinds of chat: a Chinese question, an English explanation, a coding
task, and an agent turn with a tool. Thinking is on, as the server renders a request without an effort.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from serve.deepseek import DeepSeekTemplate, DeepSeekTokenizer  # noqa: E402

TOOL = {"name": "read_file", "description": "Read a file of the repository",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}
CHATS = {
    "zh_chat": [{"role": "user", "content": "我下个月要去日本旅行一周，预算一万五人民币，帮我规划一下行程和预算分配。"}],
    "en_explain": [{"role": "user", "content": "Explain how a transformer's attention works, with a small worked "
                                                "example, for someone who knows linear algebra."}],
    "code": [{"role": "user", "content": "Write a Python class for an LRU cache with get and put in O(1), with type "
                                          "hints and a few unit tests."}],
    "agent": [{"role": "system", "content": "You are a coding agent. Use the tools to inspect the repository."},
              {"role": "user", "content": "Find out why the server returns 500 on /v1/messages when the request has "
                                          "an image, and propose a fix."}],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tok, tpl = DeepSeekTokenizer(a.tokenizer), DeepSeekTemplate()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, messages in CHATS.items():
        text = tpl.render(messages, tools=[TOOL] if name == "agent" else None)
        ids = tok.encode(text, parse_special=True)
        (out / f"{name}.ids").write_text(",".join(map(str, ids)))
        print(f"{name}: {len(ids)} tokens")
    return 0


if __name__ == "__main__":
    sys.exit(main())
