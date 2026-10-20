"""DeepSeek format tests. Golden ids come from HF tokenizers, not runtime code."""
import copy
import json
import os
from pathlib import Path
import unittest

from ds41.proto.ref import encoding as ref

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_DIR = os.environ.get('DS41_TOKENIZER_DIR')
TOOL = {'name': 'lookup', 'description': 'Find a city.', 'parameters': {
    'type': 'object', 'properties': {'city': {'type': 'string'}, 'count': {'type': 'integer'}}}}
MESSAGES = [
    {'role': 'system', 'content': 'Be precise. Literal </think>.'},
    {'role': 'user', 'content': 'Find 台北.'},
    {'role': 'assistant', 'content': 'Checking.', 'reasoning_content': 'Use the tool.',
     'tool_calls': [{'function': {'name': 'lookup', 'arguments': {'city': '台北', 'count': 2}}}]},
    {'role': 'tool', 'content': 'Found two.'},
    {'role': 'system', 'content': 'Keep it short.'},
    {'role': 'user', 'content': 'Now answer.'}]


def reference_prompt(messages, tools=None, thinking=True, effort=None):
    messages = copy.deepcopy(messages)
    if tools:
        if not messages or messages[0]['role'] != 'system':
            messages.insert(0, {'role': 'system', 'content': ''})
        messages[0]['tools'] = [{'type': 'function', 'function': t} for t in tools]
    return ref.encode_messages(messages, thinking_mode='thinking' if thinking else 'chat',
                               reasoning_effort=effort)


@unittest.skipUnless(TOKENIZER_DIR, 'set DS41_TOKENIZER_DIR to the model tokenizer directory')
class TokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from serve.deepseek import DeepSeekTokenizer
        cls.tok = DeepSeekTokenizer(TOKENIZER_DIR)

    def test_hf_golden_and_roundtrip(self):
        from serve.server import Detokenizer
        golden = json.loads((ROOT / 'serve/deepseek_golden.json').read_text())
        for case in golden['cases']:
            with self.subTest(name=case['name']):
                ids = self.tok.encode(case['text'])
                self.assertEqual(ids, case['ids'])
                self.assertEqual(self.tok.decode(ids), case['text'])
                detok = Detokenizer(self.tok)
                self.assertEqual(''.join(detok.push(i) for i in ids), case['text'])
        self.assertEqual(self.tok.eos_id, golden['eos_id'])


class RendererTests(unittest.TestCase):
    def test_reference_prompts(self):
        from serve.deepseek import DeepSeekTemplate
        template = DeepSeekTemplate()
        for messages in (MESSAGES, [{'role': 'user', 'content': 'hi'}], MESSAGES[:-1]):
            for tools in (None, [TOOL]):
                for thinking in (True, False):
                    for effort, mapped in ((None, None), ('low', 50), ('medium', 75), ('xhigh', 100)):
                        with self.subTest(thinking=thinking, effort=effort, tools=bool(tools)):
                            self.assertEqual(template.render(messages, tools=tools, enable_thinking=thinking,
                                                             reasoning_effort=effort),
                                             reference_prompt(messages, tools, thinking, mapped))

    def test_normalizers_deepseek_and_tool_ids(self):
        from serve.frontend import openai_to_messages, anthropic_to_messages
        from serve.responses import input_messages
        messages = copy.deepcopy(MESSAGES)
        messages[2]['tool_calls'][0]['id'] = 'call_a'
        messages[3]['tool_call_id'] = 'call_a'
        normalized, _, _ = openai_to_messages({'messages': messages}, deepseek=True)
        self.assertEqual(normalized[4]['role'], 'system')
        self.assertEqual(normalized[2]['tool_calls'][0]['id'], 'call_a')
        self.assertEqual(normalized[3]['tool_call_id'], 'call_a')
        normalized, _, _ = anthropic_to_messages({'messages': [
            {'role': 'user', 'content': 'hi'},
            {'role': 'assistant', 'content': [{'type': 'thinking', 'thinking': 'Think.'},
             {'type': 'tool_use', 'id': 'a', 'name': 'lookup', 'input': {'city': '台北'}}]},
            {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'a', 'content': 'ok'}]},
            {'role': 'system', 'content': 'late'}]}, deepseek=True)
        self.assertEqual(normalized[-1]['role'], 'system')
        self.assertEqual(normalized[1]['tool_calls'][0]['id'], 'a')
        self.assertEqual(normalized[2]['tool_call_id'], 'a')
        self.assertEqual(normalized[1]['reasoning_content'], 'Think.')
        self.assertEqual(input_messages({'input': MESSAGES[:2] + [MESSAGES[4]]},
                                        deepseek=True)[-1]['role'], 'system')


def completion(thinking=True, calls=True):
    text = ('Need a lookup.\n</think>' if thinking else '') + 'Checking 台北.\n'
    if calls:
        body = ref.tool_call_template.format(dsml_token=ref.dsml_token,
            tool_call_tag_name=ref.tool_call_tag_name, name='lookup',
            arguments=ref.encode_arguments_to_dsml({'arguments': {
                'city': '台北 "true"\n🦊', 'literal': '123', 'count': 2,
                'active': True, 'extra': {'a': [None, 1]}}}))
        text += '\n\n' + ref.tool_calls_template.format(dsml_token=ref.dsml_token,
            tc_block_name=ref.tool_calls_block_name, tool_calls=body)
    return text + ref.eos_token


def collect_parser(parts, thinking=True):
    from serve.deepseek import DeepSeekOutputParser
    parser = DeepSeekOutputParser(thinking=thinking, stream_tools=True)
    events = []
    for part in parts:
        events.extend(parser.feed(part))
    events.extend(parser.finish())
    out = {'reasoning': '', 'content': '', 'calls': [], 'starts': [], 'args': []}
    ids = {}
    for ev in events:
        if ev.kind in ('reasoning', 'content'):
            out[ev.kind] += ev.text
        elif ev.kind == 'tool_start':
            ids[ev.call.id] = len(out['starts'])
            out['starts'].append(ev.call.name)
            out['args'].append('')
        elif ev.kind == 'tool_args':
            out['args'][ids[ev.call.id]] += ev.text
        elif ev.kind == 'tool_call':
            out['calls'].append((ev.call.name, ev.call.arguments))
    return out


class ParserTests(unittest.TestCase):
    def test_every_character_boundary(self):
        for thinking in (True, False):
            for calls in (True, False):
                text = completion(thinking, calls)
                expected = ref.parse_message_from_completion_text(text, 'thinking' if thinking else 'chat')
                whole = collect_parser([text], thinking)
                self.assertEqual(whole['reasoning'], expected['reasoning_content'])
                self.assertEqual(whole['content'], expected['content'])
                self.assertEqual(whole['calls'], [(c['function']['name'], json.loads(c['function']['arguments']))
                                                  for c in expected['tool_calls']])
                self.assertEqual([json.loads(s) for s in whole['args']], [c[1] for c in whole['calls']])
                for i in range(len(text) + 1):
                    self.assertEqual(collect_parser([text[:i], text[i:]], thinking), whole, i)
                self.assertEqual(collect_parser(list(text), thinking), whole)

    def test_partial_tags_and_eos(self):
        from serve.deepseek import DeepSeekOutputParser
        p = DeepSeekOutputParser(thinking=False, stream_tools=True)
        self.assertEqual(p.feed('Hello\n\n<｜DSM')[0].text, 'Hello')
        self.assertEqual(p.finish(), [])
        self.assertEqual(collect_parser(['hello' + ref.eos_token + 'ignored'], False)['content'], 'hello')

    @unittest.skipUnless(TOKENIZER_DIR, 'set DS41_TOKENIZER_DIR')
    def test_every_token_boundary(self):
        from serve.deepseek import DeepSeekTokenizer
        from serve.server import Detokenizer
        tok = DeepSeekTokenizer(TOKENIZER_DIR)
        for thinking in (True, False):
            text = completion(thinking)
            ids = tok.encode(text)
            d = Detokenizer(tok)
            parts = [d.push(i) for i in ids]
            whole = collect_parser([text], thinking)
            self.assertEqual(collect_parser(parts, thinking), whole)
            for i in range(len(parts) + 1):
                self.assertEqual(collect_parser([''.join(parts[:i]), ''.join(parts[i:])], thinking), whole)


class ConfigTests(unittest.TestCase):
    def test_monitor_gpu_ids_do_not_change_engine_visibility(self):
        from unittest.mock import patch
        from serve.server import telemetry_gpu_list, child_env
        cfg = {'format': 'deepseek_v41', 'telemetry_gpus': [0, 1], 'args': ['--devices', '0,1']}
        self.assertEqual(telemetry_gpu_list(cfg), [0, 1])
        self.assertEqual(telemetry_gpu_list({'gpu': [1, 0]}), [1, 0])
        self.assertEqual(telemetry_gpu_list({}), [])
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'GPU-example-first,GPU-example-second'}):
            self.assertEqual(child_env(cfg)['CUDA_VISIBLE_DEVICES'], 'GPU-example-first,GPU-example-second')
        for ids in ([0, 0], [-1, 0], ['not-a-device']):
            with self.assertRaises(ValueError):
                telemetry_gpu_list({'telemetry_gpus': ids})

    def test_engine_args(self):
        from serve.server import engine_args, effort_end_args
        args = ['--pack', '/pack', '--max-context', '8192', '--expert-profile', '/profile',
                '--ram-budget-gib', '32', '--threads', '4']
        cfg = {'format': 'deepseek_v41', 'args': args, 'parallel': 4, 'gpu': [0, 1],
               'layer_split': 'bad', 'vram_elastic': True, 'expert_profile_save': '/learned',
               'effort_position': 'end'}
        # the config's own args, nothing added (no batch, layer split, VRAM or profile flags); ds41_serve checks them
        self.assertEqual(engine_args(cfg), args)
        self.assertIsNone(effort_end_args(cfg, 'missing', None))
        tuned = args + ['--vram-slots', '64', '--prefill-chunk', '32768']
        self.assertEqual(engine_args({'format': 'deepseek_v41', 'args': tuned}), tuned)

    @unittest.skipUnless(TOKENIZER_DIR, 'set DS41_TOKENIZER_DIR')
    def test_service_constants_and_prompt(self):
        from serve.deepseek import DeepSeekTemplate, DeepSeekTokenizer
        from serve.server import Service, MockEngine
        tok = DeepSeekTokenizer(TOKENIZER_DIR)
        svc = Service(MockEngine(tok, ''), tok, DeepSeekTemplate(), format='deepseek_v41')
        self.assertEqual(svc.stop_ids, {tok.eos_id})
        self.assertEqual(svc.reasoning_wrap_up, '</think>')
        svc.effort_end = True
        want = reference_prompt(MESSAGES, [TOOL], True, 50)
        self.assertEqual(tok.decode(svc.encode_prompt(MESSAGES, [TOOL], {'reasoning_effort': 'low'})), want)
        with self.assertRaisesRegex(ValueError, 'DeepSeek.*images'):
            svc.prepare([{'role': 'user', 'content': [{'type': 'image', 'source': 'x'}]}], None, {})
        with self.assertRaisesRegex(ValueError, 'DeepSeek.*VRAM'):
            svc.vram(None)


class ParserEdgeTests(unittest.TestCase):
    def test_multiple_calls_and_empty_arguments(self):
        text = 'ok\n\n<｜DSML｜ calls>\n' + '\n'.join(
            ref.tool_call_template.format(dsml_token=ref.dsml_token, tool_call_tag_name=ref.tool_call_tag_name,
                                         name=name, arguments=ref.encode_arguments_to_dsml({'arguments': args}))
            for name, args in [('lookup', {}), ('files.read', {'path': 'a\r\n\tb', 'n': None})])
        text += '</｜DSML｜ calls>' + ref.eos_token
        # The reference requires a newline between the final invoke and calls tags.
        text = text.replace('</｜DSML｜ invoke></｜DSML｜ calls>', '</｜DSML｜ invoke>\n</｜DSML｜ calls>')
        expected = ref.parse_message_from_completion_text(text, 'chat')
        whole = collect_parser([text], False)
        self.assertEqual(whole['calls'], [
            ('.'.join(filter(None, (c.get('namespace'), c['function']['name']))),
             json.loads(c['function']['arguments'])) for c in expected['tool_calls']])
        for i in range(len(text) + 1):
            self.assertEqual(collect_parser([text[:i], text[i:]], False), whole)

    def test_server_parser_interface(self):
        # Service.run calls finish(reason) and reads pending / rescued / refused, as for OutputParser (#1058)
        from serve.deepseek import DeepSeekOutputParser
        for reason in (None, 'stop', 'length', 'cancel'):
            parser = DeepSeekOutputParser(thinking=False)
            self.assertEqual([(e.kind, e.text) for e in parser.feed('hi') + parser.finish(reason)], [('content', 'hi')])
            self.assertEqual((parser.pending, parser.rescued, parser.refused), ([], 0, 0))

    def test_length_limit_preserves_plain_newlines(self):
        self.assertEqual(collect_parser(['hello\n'], False)['content'], 'hello\n')
        self.assertEqual(collect_parser(['hello\n\n'], False)['content'], 'hello\n\n')


@unittest.skipUnless(TOKENIZER_DIR, 'set DS41_TOKENIZER_DIR')
class BudgetTests(unittest.TestCase):
    def test_mock_eos_and_budget_resume(self):
        import threading
        from serve.deepseek import DeepSeekTemplate, DeepSeekTokenizer
        from serve.server import MockEngine, Service
        tok = DeepSeekTokenizer(TOKENIZER_DIR)
        engine = MockEngine(tok, ['Thinking more and more. ' * 10, 'Answer.\n'])
        self.assertEqual(engine.scripts[0][-1], tok.eos_id)
        svc = Service(engine, tok, DeepSeekTemplate(), format='deepseek_v41')
        svc.reasoning_budget_tokens = 3
        ids, thinking, maximum = svc.prepare([{'role': 'user', 'content': 'hi'}], None, {}, 100)
        result = list(svc.run(ids, thinking, None, maximum, {}, threading.Event()))
        self.assertEqual(engine.turns, 2)
        self.assertTrue(tok.decode(engine.last_prompt).endswith('</think>'))
        self.assertNotIn('<|im_', tok.decode(engine.last_prompt))
        self.assertEqual(''.join(ev.text for kind, ev in result if kind == 'event' and ev.kind == 'content'), 'Answer.\n')
        self.assertEqual(result[-1][1]['finish'], 'stop')


class ToolChoiceTests(unittest.TestCase):
    def test_forced_call_is_not_written_for_deepseek(self):
        # forced_call writes the Qwen call opening (<tool_call> / <function=); DeepSeek calls are DSML, so a forced
        # tool_choice acts as "auto" there instead of putting a foreign format into the reply
        from serve.frontend import forced_call
        tools = [TOOL]
        self.assertIsNotNone(forced_call('required', tools))
        for choice in ('required', {'type': 'function', 'function': {'name': 'lookup'}}, {'type': 'any'},
                       {'type': 'tool', 'name': 'lookup'}):
            with self.subTest(choice=choice):
                self.assertIsNone(forced_call(choice, tools, deepseek=True))


class ImageTests(unittest.TestCase):
    def test_normalizers_reject_images_before_text_conversion(self):
        from serve.frontend import openai_to_messages, anthropic_to_messages
        for role in ('system', 'user', 'assistant', 'tool'):
            with self.subTest(role=role), self.assertRaisesRegex(ValueError, 'DeepSeek.*images'):
                openai_to_messages({'messages': [{'role': role, 'content': [
                    {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AA=='}}]}]},
                    deepseek=True)
        for req in ({'system': [{'type': 'image', 'source': {'type': 'url', 'url': 'x'}}]},
                    {'messages': [{'role': 'user', 'content': [{'type': 'tool_result', 'content': [
                        {'type': 'image', 'source': {'type': 'url', 'url': 'x'}}]}]}]}):
            with self.assertRaisesRegex(ValueError, 'DeepSeek.*images'):
                anthropic_to_messages(req, deepseek=True)
