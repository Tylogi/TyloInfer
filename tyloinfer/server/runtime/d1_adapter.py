"""Modified text-only d1 prompt/readout behind the common System One contract.

Prompt compatibility follows LiquidAI/d1-2B and d1_reference.py under the
LFM Open License v1.0; see NOTICE and LICENSES/LiquidAI-d1-LFM-Open-License.txt.
"""
from __future__ import annotations

import json
import time
from uuid import uuid4

from tyloinfer.runtime.d1_reference import (
    pack_requests,
    question_layout,
    read_answer,
    validate_request,
)
from tyloinfer.server.runtime.backend import OpenAIChatBackend
from tyloinfer.server.runtime.client import BackendError


def invalid(message: str):
    raise BackendError('invalid_request_error', message, status_code=400)


def text(value, name: str, maximum: int, *, nonempty=False) -> str:
    if not isinstance(value, str) or len(value.encode('utf-8')) > maximum or (nonempty and not value):
        invalid(f'invalid decision {name}')
    return value


def prepare_request(body: dict, model: str) -> dict:
    allowed = {'model', 'state', 'questions', 'instructions', 'chat_template_kwargs', 'cache_salt', 'joint'}
    if not isinstance(body, dict) or set(body) - allowed:
        invalid('unsupported decision parameter or capability')
    if text(body.get('model'), 'model', 512, nonempty=True) != model:
        raise BackendError('model_not_found', 'the requested model is not loaded', status_code=404)
    if 'state' not in body or body['state'] is None:
        invalid('this d1 backend requires text or JSON state; media-only decisions are unavailable')
    try:
        state = body['state'] if isinstance(body['state'], str) else json.dumps(body['state'], ensure_ascii=False, allow_nan=False)
    except (ValueError, TypeError):
        invalid('decision state must contain finite JSON values')
    if len(state.encode('utf-8')) > 262144:
        invalid('decision state exceeds the text limit')
    if 'joint' in body and (type(body['joint']) is not bool or body['joint']):
        invalid('joint decisions are unsupported')
    if 'cache_salt' in body:
        text(body['cache_salt'], 'cache_salt', 256)
    kwargs = body.get('chat_template_kwargs', {})
    if not isinstance(kwargs, dict) or kwargs:
        invalid('d1 uses its trained prompt; chat template overrides are unsupported')
    shared = text(body.get('instructions', ''), 'instructions', 65536)
    questions = body.get('questions')
    if not isinstance(questions, dict) or not 1 <= len(questions) <= 64:
        invalid('decisions require 1 to 64 named questions')
    mapped = {}
    for name, question in questions.items():
        text(name, 'question name', 256, nonempty=True)
        if not isinstance(question, dict) or set(question) - {'type', 'instructions', 'criteria'}:
            invalid('d1 supports text noul, single-choice and score questions only')
        kind = question.get('type')
        instructions = text(question.get('instructions', ''), 'question instructions', 65536)
        criteria = question.get('criteria')
        if kind == 'choice':
            if not isinstance(criteria, dict) or not 1 <= len(criteria) <= 255:
                invalid('choice requires 1 to 255 named options')
            for key, description in criteria.items():
                text(key, 'choice label', 512, nonempty=True)
                if description is not None:
                    text(description, 'choice description', 16384)
        elif kind == 'score':
            if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
                invalid('score requires 2 to 10 levels')
            for value in criteria:
                text(value, 'score level', 512, nonempty=True)
        elif kind == 'noul':
            if criteria is not None:
                if not isinstance(criteria, dict) or set(criteria) - {'true', 'false'}:
                    invalid('noul criteria accepts true and false descriptions only')
                for value in criteria.values():
                    if value is not None:
                        text(value, 'noul description', 16384)
                criteria = {key: criteria.get(key) or default for key, default in (('true', 'Yes / true'), ('false', 'No / false'))}
        else:
            invalid('d1 supports text noul, single-choice and score questions only')
        mapped[name] = {'type': kind, 'instructions': shared + ('\n\n' if shared else '') + instructions}
        if criteria is not None:
            mapped[name]['criteria'] = criteria
    value = {'state': body['state'], 'questions': mapped}
    validate_request(value)
    return value


class D1DecisionBackend(OpenAIChatBackend):
    async def preflight(self, **kwargs):
        raise BackendError('generation_unavailable', 'd1 is a decision model; use /v1/systemone', status_code=501)

    async def stream(self, **kwargs):
        raise BackendError('generation_unavailable', 'd1 is a decision model; use /v1/systemone', status_code=501)
        yield


class D1DecisionAdapter:
    def __init__(self, runtime, model: str, context_size: int):
        self.runtime, self.model, self.context_size = runtime, model, context_size
        self.last_frame = {}
        self.forward_calls = 0

    def decide(self, body: dict) -> dict:
        payload = prepare_request(body, self.model)
        state = payload['state'] if isinstance(payload['state'], str) else json.dumps(payload['state'], ensure_ascii=False, indent=2)
        bos = self.runtime.tokenizer.raw.id_to_token(self.runtime.contract.bos_token_id)
        prefix = bos + '<|im_start|>user\n' + self.runtime.tokenizer.plain(state + '\n\n') + '\nQUESTION:\n'
        for question in payload['questions'].values():
            suffix, _ = question_layout(self.runtime.tokenizer, question, self.runtime.contract.codes)
            if len((prefix + suffix).encode('utf-8')) > 262144:
                invalid('decision read exceeds the text limit')
        try:
            rows = pack_requests(self.runtime.tokenizer, self.runtime.contract, [payload])
        except ValueError as error:
            invalid(str(error))
        if any(len(row['ids']) > self.context_size for row in rows) or sum(len(row['ids']) for row in rows) > 1048576:
            invalid('d1 decision exceeds the loaded context or aggregate input-token limit')
        answers, diagnostics, tokens = {}, {}, 0
        for row in rows:
            candidates = sorted({token for slot in row['slots'] for group in slot[4] for token in group})
            frame = self.runtime.evaluate(row, candidates)
            self.forward_calls += 1
            self.last_frame = frame
            tokens += len(row['ids'])
            elapsed = frame.get('gpu_ms', [0.0])[0]
            for slot, logits in zip(row['slots'], frame['logits'], strict=True):
                _, name, question, _, options = slot
                answer = read_answer(question, dict(zip(candidates, logits, strict=True)), options, self.runtime.calibration)
                if question['type'] == 'noul':
                    yes = answer['noul']
                    answer.update(probabilities={'yes': yes, 'no': 1 - yes}, confidence=max(yes, 1 - yes))
                answers[name] = answer
                diagnostics[name] = {'label_mass': None, 'argmax_is_label': None, 'normalization': 'labels',
                                     'prefill_ms': elapsed, 'prefill_tps': len(row['ids']) * 1000 / elapsed if elapsed > 0 else 0.0,
                                     'shared_forward': len(row['slots']) > 1}
        return {'id': 'decision_' + uuid4().hex, 'object': 'structured_decision', 'created': int(time.time()),
                'model': self.model, 'answers': answers,
                'usage': {'input_tokens': tokens, 'output_tokens': 0, 'input_tokens_details': {'cached_tokens': 0}},
                'diagnostics': diagnostics}
