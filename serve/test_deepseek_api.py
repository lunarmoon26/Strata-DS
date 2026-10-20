"""HTTP adapters with the real tokenizer and a scripted CPU engine; never a GPU."""
import json
import os
from pathlib import Path
import queue
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

from serve.deepseek import DeepSeekTemplate, DeepSeekTokenizer
from serve.server import MockEngine, Service, serve

TOKENIZER = os.environ.get("DS41_TOKENIZER_DIR")
MODEL = "deepseek-v4.1-flash-1017"
TOOL = {"name": "lookup", "description": "Find a city", "parameters": {
    "type": "object", "properties": {"city": {"type": "string"}}}}
CALL = ('\n\n<｜DSML｜ calls><｜DSML｜ invoke name="lookup">\n'
        '<｜DSML｜ parameter name="city" string="true">台北</｜DSML｜ parameter>\n'
        '</｜DSML｜ invoke></｜DSML｜ calls>')


@unittest.skipUnless(TOKENIZER, "set DS41_TOKENIZER_DIR; tokenizer files only")
class HttpAdapters(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = DeepSeekTokenizer(Path(TOKENIZER))

    def server(self, script="323", context=524288):
        engine = MockEngine(self.tok, script, max_context=context)
        svc = Service(engine, self.tok, DeepSeekTemplate(), model_name=MODEL,
                      format="deepseek_v41", fit_max_tokens=False)
        svc.answer_room = False
        httpd = serve(svc, port=0)
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        return f"http://127.0.0.1:{httpd.server_address[1]}"

    def request(self, base, path, body):
        req = urllib.request.Request(base + path, json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        return urllib.request.urlopen(req, timeout=10)

    def chat(self, **kwargs):
        return {"model": MODEL, "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 128, "chat_template_kwargs": {"enable_thinking": False}, **kwargs}

    def test_openai_nonstream_and_stream(self):
        base = self.server()
        with self.request(base, "/v1/chat/completions", self.chat()) as response:
            result = json.load(response)
        self.assertEqual(result["choices"][0]["message"]["content"], "323")
        with self.request(base, "/v1/chat/completions", self.chat(stream=True)) as response:
            data = response.read().decode()
        chunks = [json.loads(line[6:]) for line in data.splitlines()
                  if line.startswith("data: ") and line != "data: [DONE]"]
        self.assertEqual("".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"]), "323")
        self.assertIn("data: [DONE]", data)

    def test_responses_and_anthropic(self):
        base = self.server()
        with self.request(base, "/v1/responses", {"model": MODEL, "input": "hi", "max_output_tokens": 128,
                                                 "reasoning": {"effort": "none"}}) as response:
            result = json.load(response)
        self.assertEqual("".join(c.get("text", "") for item in result["output"]
                                 for c in item.get("content", [])), "323")
        with self.request(base, "/v1/messages", self.chat(thinking={"type": "disabled"})) as response:
            result = json.load(response)
        self.assertEqual("".join(c.get("text", "") for c in result["content"]), "323")

    def test_dsml_tool_call_reaches_all_protocols(self):
        base = self.server(CALL)
        with self.request(base, "/v1/chat/completions", self.chat(
                tools=[{"type": "function", "function": TOOL}])) as response:
            result = json.load(response)
        call = result["choices"][0]["message"]["tool_calls"][0]["function"]
        self.assertEqual((call["name"], json.loads(call["arguments"])), ("lookup", {"city": "台北"}))
        with self.request(base, "/v1/messages", self.chat(
                thinking={"type": "disabled"}, tools=[{"name": "lookup", "input_schema": TOOL["parameters"]}])) as response:
            result = json.load(response)
        calls = [c for c in result["content"] if c["type"] == "tool_use"]
        self.assertEqual(calls[0]["input"], {"city": "台北"})
        with self.request(base, "/v1/responses", {"model": MODEL, "input": "hi", "max_output_tokens": 128,
                "reasoning": {"effort": "none"}, "tools": [{"type": "function", **TOOL}]}) as response:
            result = json.load(response)
        calls = [c for c in result["output"] if c["type"] == "function_call"]
        self.assertEqual(json.loads(calls[0]["arguments"]), {"city": "台北"})

    def test_overflow_and_images_fail_before_generation(self):
        base = self.server(context=256)
        for body in (self.chat(messages=[{"role": "user", "content": "long prompt " * 1000}]),
                     self.chat(messages=[{"role": "user", "content": [
                         {"type": "image_url", "image_url": {"url": "https://invalid.example/image.png"}}]}])):
            with self.assertRaises(urllib.error.HTTPError) as caught:
                self.request(base, "/v1/chat/completions", body)
            self.assertEqual(caught.exception.code, 400)
            caught.exception.close()

    def test_native_main_loads_profile_through_a_cpu_protocol_standin(self):
        # Exercise main's tokenizer/config/engine-class selection as well as the
        # HTTP handlers. The executable is a tiny Python pipe peer, not ds41_serve.
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        exe = directory / "fake-ds41"
        tokens = self.tok.encode("323") + [self.tok.eos_id]
        exe.write_text(f"#!{sys.executable}\nimport sys\n"
                       "print('INFO engine=cpu-test batch_slots=0', flush=True)\n"
                       "print('READY 524288 stop', flush=True)\n"
                       "for line in sys.stdin:\n"
                       " if line.startswith('QUIT'): break\n"
                       " if line.startswith('GEN '):\n"
                       "  print('RESUME 0', flush=True)\n"
                       f"  for token in {tokens!r}: print('T', token, flush=True)\n"
                       f"  print('DONE {len(tokens)} 1 1.0 1.0 stop', flush=True)\n")
        exe.chmod(0o700)
        config = directory / "profile.json"
        config.write_text(json.dumps({"format": "deepseek_v41", "model_name": MODEL,
                                     "tokenizer": TOKENIZER, "exe": str(exe), "args": [],
                                     "host": "127.0.0.1", "sampling": {"temperature": 0}}))
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        process = subprocess.Popen([sys.executable, "-u", "-m", "serve.server", "--engine", "strata",
                                    "--config", str(config), "--port", str(port)],
                                   cwd=Path(__file__).resolve().parents[1],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        def cleanup():
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            process.stdout.close()
        self.addCleanup(cleanup)
        lines = queue.Queue()
        def reader():
            for line in process.stdout:
                lines.put(line)
            lines.put(None)
        threading.Thread(target=reader, daemon=True).start()
        captured = []
        while True:
            line = lines.get(timeout=15)
            self.assertIsNotNone(line, "".join(captured))
            captured.append(line)
            if line.startswith("ready: "):
                break
        try:
            with self.request(f"http://127.0.0.1:{port}", "/v1/chat/completions", self.chat()) as response:
                result = json.load(response)
        except urllib.error.HTTPError as e:
            with e:
                self.fail(e.read().decode())
        self.assertEqual(result["choices"][0]["message"]["content"], "323")


if __name__ == "__main__":
    unittest.main()
