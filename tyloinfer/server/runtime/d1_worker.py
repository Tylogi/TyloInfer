"""Private managed transport for the native d1 decision engine."""
from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Request

from tyloinfer.runtime.d1_native import D1Native
from tyloinfer.server.runtime.client import BackendError
from tyloinfer.server.runtime.d1_adapter import D1DecisionAdapter


class D1Worker:
    def __init__(self, runtime, model: str, context_size: int):
        if not 1 <= context_size <= runtime.contract.text.max_position_embeddings:
            raise ValueError('d1 context exceeds the trained model context')
        self.runtime = runtime
        self.adapter = D1DecisionAdapter(runtime, model, context_size)
        self.pending = None
        self.closed = False

    def health(self):
        if self.closed or self.runtime.process.poll() is not None:
            raise BackendError('model_not_ready', 'd1 native engine is not running', status_code=503)
        return {'status': 'ok', 'model': self.adapter.model, 'model_type': 'lfm3_d1',
                'backend': self.runtime.backend, 'context_size': self.adapter.context_size,
                'decisions_available': True, 'generation_available': False, 'probability_available': False,
                'vision_available': False, 'video_available': False, 'mtp_available': False,
                'model_capabilities': {'architecture_family': 'lfm3_d1', 'source': 'native-d1-text',
                                       'features': {'text': True}}}

    def status(self):
        result = self.health()
        ready = self.runtime.ready
        weights = ready.get('resident_packed_weight_bytes', 0) + ready.get('resident_bf16_weight_bytes', 0) + ready.get('layer_buffer_bytes', 0)
        result.update(runtime_state='busy' if self.pending is not None else 'ready',
                      resident_weight_bytes=weights, kv_cache_bytes=0, kv_cache_contexts=0, prefix_cache_bytes=0,
                      prefix_cache_max_bytes=0, prefix_cache_ram_bytes=0, prefix_cache_sessions=0,
                      prefix_cache_enabled=False, resident_memory_dynamic_budget=0,
                      prefix_cache_dynamic_budget=0, resident_weight_dynamic_budget=0,
                      qsa_kv_offload_supported=0, kv_quantization_supported=0, kv_memory_supported=0,
                      full_matrix_dequant_cache_bytes=ready.get('full_matrix_dequant_cache_bytes', 0),
                      decision_forward_calls=self.adapter.forward_calls)
        result.update(ram_weight_bytes=ready.get('streamed_weight_bytes', 0), ram_embedding_bytes=0,
                      jev_weights_streamed=float(ready.get('weight_placement') == 'stream'))
        result['native_metrics'] = {key: value for key, value in self.adapter.last_frame.items()
                                    if key in {'packed_kernel', 'packed_kernel_dispatch', 'packed_gemm_calls_per_forward',
                                               'bf16_gemm_calls_per_forward', 'temporary_pool_reserved_high_bytes',
                                               'temporary_pool_used_high_bytes'}}
        return result

    async def dispatch(self, operation: str, params: dict):
        if operation == 'health':
            return self.health()
        if operation == 'status':
            return self.status()
        if operation == 'models':
            return {'data': [{'id': self.adapter.model, 'object': 'model'}]}
        if operation == 'decide':
            self.health()
            if self.pending is not None:
                raise BackendError('runtime_busy', 'd1 requires an idle decision runtime', status_code=409, retryable=True)
            task = asyncio.create_task(asyncio.to_thread(self.adapter.decide, params))
            self.pending = task
            def finished(done):
                self.pending = None
                if not done.cancelled():
                    done.exception()
            task.add_done_callback(finished)
            try:
                return await asyncio.shield(task)
            except BackendError:
                raise
            except (ValueError, TypeError) as error:
                raise BackendError('invalid_request_error', str(error), status_code=400) from error
            except (RuntimeError, OSError) as error:
                raise BackendError('native_execution_failed', str(error), status_code=503) from error
        if operation == 'count_tokens':
            prompt = params.get('prompt')
            if not isinstance(prompt, str):
                raise BackendError('invalid_request_error', 'prompt must be text', status_code=400)
            return {'prompt_tokens': len(self.runtime.tokenizer.encode(prompt)), 'context_size': self.adapter.context_size,
                    'tokenization': 'raw-no-special-tokens'}
        if operation in {'cache.clear', 'cache.trim', 'cache.refresh'}:
            if self.pending is not None:
                raise BackendError('runtime_busy', 'd1 decision is still executing', status_code=409)
            return {**self.status(), 'released_bytes': 0}
        if operation == 'reload':
            context = params.get('context_size')
            if self.pending is not None:
                raise BackendError('runtime_busy', 'd1 decision is still executing', status_code=409)
            if type(context) is not int or not 1 <= context <= self.runtime.contract.text.max_position_embeddings:
                raise BackendError('invalid_request_error', 'invalid d1 context size', status_code=400)
            self.adapter.context_size = context
            return self.status()
        if operation == 'shutdown':
            await asyncio.to_thread(self.close)
            return {'shutdown': True}
        raise BackendError('capability_unavailable', 'operation is unavailable for the d1 decision backend', status_code=501)

    def close(self):
        if not self.closed:
            self.closed = True
            self.runtime.close()


def error_json(error: BackendError):
    return {'code': error.code, 'message': str(error), 'status_code': error.status_code or 500, 'retryable': error.retryable}


def create_app(worker: D1Worker):
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            await asyncio.to_thread(worker.close)

    app = FastAPI(lifespan=lifespan)
    operations = {'health': 'health', 'status': 'status', 'models': 'models', 'decide': 'decide',
                  'count_tokens': 'count_tokens', 'reload': 'reload', 'cache/clear': 'cache.clear',
                  'cache/trim': 'cache.trim', 'cache/refresh': 'cache.refresh'}

    @app.api_route('/runtime/{path:path}', methods=['GET', 'POST', 'DELETE'])
    async def control(path: str, request: Request):
        try:
            operation = operations.get(path)
            if operation is None:
                raise BackendError('capability_unavailable', 'operation is unavailable for d1', status_code=501)
            expected = 'GET' if operation in {'health', 'status', 'models'} else 'POST'
            if request.method != expected:
                raise BackendError('method_not_allowed', 'invalid runtime control method', status_code=405)
            params = {} if request.method == 'GET' else await request.json()
            if not isinstance(params, dict):
                raise BackendError('invalid_request_error', 'runtime request must be an object', status_code=400)
            result = await worker.dispatch(operation, params)
            return JSONResponse(result)
        except BackendError as error:
            return JSONResponse({'error': error_json(error)}, status_code=error.status_code or 500)
        except ValueError:
            return JSONResponse({'error': {'code': 'invalid_request_error', 'message': 'invalid JSON request'}}, status_code=400)
    return app


async def serve_stdio(worker: D1Worker):
    loop = asyncio.get_running_loop()
    incoming = asyncio.Queue()
    tasks = set()
    def read():
        for line in sys.stdin:
            loop.call_soon_threadsafe(incoming.put_nowait, line)
        loop.call_soon_threadsafe(incoming.put_nowait, None)
    threading.Thread(target=read, daemon=True).start()
    def send(frame):
        print(json.dumps(frame, allow_nan=False), flush=True)
    async def handle(request):
        identifier = request.get('id')
        try:
            if request.get('v') != 1 or not isinstance(request.get('params', {}), dict):
                raise BackendError('invalid_request_error', 'invalid runtime frame', status_code=400)
            result = await worker.dispatch(request.get('op'), request.get('params', {}))
            if identifier is not None:
                send({'v': 1, 'id': identifier, 'type': 'result', 'data': result})
        except BackendError as error:
            send({'v': 1, 'id': identifier, 'type': 'error', 'error': error_json(error)})
    send({'v': 1, 'type': 'ready'})
    try:
        while (line := await incoming.get()) is not None:
            try:
                request = json.loads(line)
                if not isinstance(request, dict):
                    raise ValueError('runtime frame must be an object')
            except ValueError:
                send({'v': 1, 'type': 'error', 'error': {'code': 'invalid_request_error', 'message': 'invalid JSON frame', 'status_code': 400}})
                continue
            task = asyncio.create_task(handle(request))
            tasks.add(task)
            task.add_done_callback(tasks.discard)
        if tasks:
            await asyncio.gather(*tasks)
    finally:
        worker.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--ctx-size', type=int, required=True)
    parser.add_argument('--jev-weights', choices=('stream', 'vram'), default='stream')
    parser.add_argument('--transport', choices=('http', 'stdio'), required=True)
    parser.add_argument('--host', default='127.0.0.1', choices=('127.0.0.1',))
    parser.add_argument('--port', type=int, default=0)
    args = parser.parse_args()
    with D1Native(args.assets, args.native, weights=args.weights, weight_placement=args.jev_weights) as runtime:
        worker = D1Worker(runtime, args.model_name, args.ctx_size)
        if args.transport == 'stdio':
            def terminate(signum, frame):
                worker.close()
                raise SystemExit(0)
            signal.signal(signal.SIGTERM, terminate)
            asyncio.run(serve_stdio(worker))
        else:
            import uvicorn
            uvicorn.run(create_app(worker), host=args.host, port=args.port, log_level='warning')


if __name__ == '__main__':
    main()
