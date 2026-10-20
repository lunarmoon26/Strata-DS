"""DeepSeek V4.1 adapters. The reference encoder stays in the repository checkout."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import regex

from ds41.proto.ref import encoding as reference
from serve.frontend import Event, OutputParser, ToolCall, images_of
from tools.strata_tokenizer import BYTE_TO_UNICODE, Tokenizer


class DeepSeekTokenizer(Tokenizer):
    """Read the model's byte BPE and ordered Split pre-tokenizers without HF at runtime."""

    def __init__(self, directory):
        directory = Path(directory)
        data = json.loads((directory / 'tokenizer.json').read_text(encoding='utf-8'))
        model = data['model']
        if model['type'] != 'BPE' or data.get('normalizer') not in (None, {'type': 'Sequence', 'normalizers': []}):
            raise ValueError('DeepSeek requires byte BPE without a normalizer')
        vocab = dict(model['vocab'])
        added = data['added_tokens']
        for token in added:
            if any(token.get(k) for k in ('single_word', 'lstrip', 'rstrip')):
                raise ValueError('unsupported DeepSeek added-token matching options')
            vocab[token['content']] = token['id']
        tokens = [None] * (max(vocab.values()) + 1)
        for text, i in vocab.items():
            tokens[i] = text
        merges = [' '.join(m) if isinstance(m, list) else m for m in model['merges']]
        super().__init__(tokens, merges, pre='deepseek_v41')
        self.special_tokens = {t['content']: t['id'] for t in added}
        self._special_re = self._alt(list(self.special_tokens))
        self._always_re = self._special_re
        self._added_bytes = {t['id']: t['content'].encode('utf-8') for t in added}
        pre = data['pre_tokenizer']
        if pre['type'] != 'Sequence':
            raise ValueError('DeepSeek requires a Sequence pre-tokenizer')
        parts = pre['pretokenizers']
        self.splits = []
        for part in parts[:-1]:
            if part['type'] != 'Split' or part['behavior'] != 'Isolated' or part['invert']:
                raise ValueError('unsupported DeepSeek pre-tokenizer split')
            self.splits.append(regex.compile(part['pattern']['Regex']))
        if parts[-1]['type'] != 'ByteLevel' or parts[-1]['add_prefix_space'] or parts[-1]['use_regex']:
            raise ValueError('unsupported DeepSeek byte-level pre-tokenizer')
        cfg = json.loads((directory / 'tokenizer_config.json').read_text(encoding='utf-8'))
        eos = cfg['eos_token']
        eos = eos['content'] if isinstance(eos, dict) else eos
        self.eos_id = self.special_tokens[eos]

    def _encode_plain(self, text):
        pieces = [text]
        for pattern in self.splits:
            split = []
            for piece in pieces:
                pos = 0
                for match in pattern.finditer(piece):
                    if match.start() > pos:
                        split.append(piece[pos:match.start()])
                    if match.end() > match.start():
                        split.append(match.group())
                    pos = match.end()
                if pos < len(piece):
                    split.append(piece[pos:])
            pieces = split
        out = []
        for piece in pieces:
            mapped = ''.join(BYTE_TO_UNICODE[b] for b in piece.encode('utf-8'))
            out.extend(self.ids[token] for token in self._bpe(mapped))
        return out

    def token_bytes(self, i):
        if i in self._added_bytes:
            return self._added_bytes[i]
        return super().token_bytes(i)


class DeepSeekTemplate:
    """Use the official encoder in place. setup starts the server from this checkout."""

    # Rendering capabilities of the reference encoder, not model quality claims.
    caps = {"supports_tools": True, "supports_tool_calls": True,
            "supports_system_role": True, "supports_parallel_tool_calls": True,
            "supports_preserve_reasoning": False}
    source = "DeepSeek reference encoder: ds41/proto/ref/encoding.py (not a Jinja template)"

    def render(self, messages, tools=None, add_generation_prompt=True, **kwargs):
        if images_of(messages):
            raise ValueError('DeepSeek V4.1 does not support images in this server')
        messages = copy.deepcopy(messages)
        if tools:
            if not messages or messages[0]['role'] != 'system':
                messages.insert(0, {'role': 'system', 'content': ''})
            messages[0]['tools'] = [{'type': 'function', 'function': t} for t in tools]
        # Normalized low -> 50, medium -> 75, high/xhigh/max -> 100.
        # No effort keeps the reference default (75). Disabled thinking uses chat mode.
        effort = kwargs.get('reasoning_effort')
        effort = {None: None, 'low': 50, 'medium': 75, 'high': 100, 'xhigh': 100}.get(effort, effort)
        return reference.encode_messages(messages, thinking_mode='chat' if kwargs.get('enable_thinking') is False else 'thinking',
                               reasoning_effort=effort)


class DeepSeekOutputParser:
    """Stream text and tool headers. Hold each argument until its closing parameter tag."""

    CALLS = '\n\n<｜DSML｜ calls>'
    CALLS_END = '</｜DSML｜ calls>'
    INVOKE_END = '</｜DSML｜ invoke>'
    PARAM_END = '</｜DSML｜ parameter>'
    HEADER = regex.compile(r'<｜DSML｜ invoke name="(.*?)">\n', regex.DOTALL)
    PARAM = regex.compile(r'<｜DSML｜ parameter name="(.*?)" string="(true|false)">', regex.DOTALL)
    _hold = OutputParser._hold

    def __init__(self, thinking=True, tools=None, stream_tools=False):
        self.state = 'reasoning' if thinking else 'content'
        self.buf = ''
        self.stream_tools = stream_tools
        self.call = None
        self.parameter = None
        self.first = True
        # OutputParser's #1058 counters, which Service.run reads: DeepSeek calls are never taken from the reasoning
        self.pending = []
        self.rescued = 0
        self.refused = 0

    def feed(self, delta):
        if self.state == 'done':
            return []
        self.buf += delta
        out = []
        while self.buf:
            if self.state in ('reasoning', 'content'):
                tags = (reference.thinking_end_token, reference.eos_token) if self.state == 'reasoning' else (
                    self.CALLS, reference.eos_token)
                found = [(self.buf.find(tag), tag) for tag in tags if tag in self.buf]
                if not found:
                    safe = len(self.buf) - self._hold(self.buf, tags)
                    if safe:
                        out.append(Event(self.state, self.buf[:safe]))
                        self.buf = self.buf[safe:]
                    break
                at, tag = min(found)
                if at:
                    out.append(Event(self.state, self.buf[:at]))
                self.buf = self.buf[at + len(tag):]
                if tag == reference.eos_token:
                    self.state, self.buf = 'done', ''
                else:
                    self.state = 'content' if tag == reference.thinking_end_token else 'calls'
            elif self.state == 'calls':
                self.buf = self.buf.lstrip('\n')
                if self.buf.startswith(self.CALLS_END):
                    self.buf = self.buf[len(self.CALLS_END):]
                    self.state = 'end'
                    continue
                match = self.HEADER.match(self.buf)
                if not match:
                    break
                self.call = ToolCall(match[1], {})
                self.first = True
                self.buf = self.buf[match.end():]
                self.state = 'parameter'
                if self.stream_tools:
                    out.extend([Event('tool_start', call=self.call), Event('tool_args', '{', self.call)])
            elif self.state == 'parameter':
                self.buf = self.buf.lstrip('\n')
                if self.buf.startswith(self.INVOKE_END):
                    self.buf = self.buf[len(self.INVOKE_END):]
                    if self.stream_tools:
                        out.append(Event('tool_args', '}', self.call))
                    out.append(Event('tool_call', call=self.call))
                    self.call = None
                    self.state = 'calls'
                    continue
                match = self.PARAM.match(self.buf)
                if not match:
                    break
                self.parameter = (match[1], match[2])
                self.buf = self.buf[match.end():]
                self.state = 'value'
            elif self.state == 'value':
                at = self.buf.find(self.PARAM_END)
                if at < 0:
                    break
                name, string = self.parameter
                value = self.buf[:at]
                value = value if string == 'true' else json.loads(value)
                if name in self.call.arguments:
                    raise ValueError(f'duplicate DeepSeek parameter: {name}')
                self.call.arguments[name] = value
                if self.stream_tools:
                    text = ('' if self.first else ',') + json.dumps(name) + ':' + json.dumps(value, ensure_ascii=False)
                    out.append(Event('tool_args', text, self.call))
                self.first = False
                self.buf = self.buf[at + len(self.PARAM_END):]
                self.state = 'parameter'
            else:
                if self.buf.startswith(reference.eos_token):
                    self.state, self.buf = 'done', ''
                break
        return out

    def finish(self, reason=None):
        """End of generation; `reason` (how the turn ended) is accepted as OutputParser.finish takes it."""
        out = self.feed('')
        if self.state in ('reasoning', 'content') and self.buf and not self.buf.strip('\n'):
            out.append(Event(self.state, self.buf))
        # A length limit can leave a partial tag or call. Do not expose protocol fragments as text.
        self.buf = ''
        return out
