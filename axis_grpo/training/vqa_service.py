"""Official CLIP-FlanT5 pair scoring, isolated from the T2I model environment."""

import argparse
import importlib.machinery
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def official_class():
    """Return the installed t2v_metrics CLIPT5Model class without importing its video backends.

    Empty namespace modules stand in for the t2v_metrics package `__init__` files, which
    import every backend eagerly. No model or scoring function is replaced, and no installed
    file is edited.
    """
    target = 't2v_metrics.models.vqascore_models.clip_t5_model'
    if target in sys.modules:
        return sys.modules[target].CLIPT5Model
    spec = importlib.util.find_spec('t2v_metrics')
    if spec is None:
        raise ImportError('Install official t2v_metrics in the VQA environment')
    root = Path(spec.origin).parent
    for name, directory in [
        ('t2v_metrics', root),
        ('t2v_metrics.models', root / 'models'),
        ('t2v_metrics.models.vqascore_models', root / 'models/vqascore_models'),
    ]:
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(directory)]
            module.__file__ = str(directory / '__init__.py')
            module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
            sys.modules[name] = module
    from t2v_metrics.models.vqascore_models.clip_t5_model import CLIPT5Model

    return CLIPT5Model


class VQABackend:
    def __init__(self, model, device, cache, batch_size, probe_out=None):
        if model not in {'clip-flant5-xxl', 'clip-flant5-xl'} or batch_size < 1:
            raise ValueError('Unsupported VQA model/batch')
        self.model = official_class()(model_name=model, device=device, cache_dir=cache)
        self.batch_size = batch_size
        self.probe_out = probe_out

    def score(self, pairs):
        if self.probe_out:
            from hpdv2.storage import atomic_json
            import time

            original = self.batch_size
            self.batch_size = 1
            start = time.monotonic()
            single = self._score(pairs)
            seconds_single = time.monotonic() - start
            self.batch_size = original
            start = time.monotonic()
            batched = self._score(pairs)
            delta = max(abs(a - b) for a, b in zip(single, batched))
            atomic_json(
                Path(self.probe_out),
                dict(
                    pairs=len(pairs),
                    batch_size=original,
                    single_seconds=seconds_single,
                    batch_seconds=time.monotonic() - start,
                    max_absolute_difference=delta,
                    tolerance=0.01,
                    passed=delta <= 0.01,
                ),
            )
            if delta > 0.01:
                raise RuntimeError(f'VQAScore batch parity failed: {delta}')
            self.probe_out = None
            return batched
        return self._score(pairs)

    def _score(self, pairs):
        import torch

        result = []
        # The public VQAScore wrapper scores one image at a time; the backend's forward takes
        # matched (image, text) pairs, so each chunk runs as one GPU batch.
        with torch.inference_mode():
            for start in range(0, len(pairs), self.batch_size):
                chunk = pairs[start : start + self.batch_size]
                values = self.model.forward(
                    images=[p[0] for p in chunk], texts=[p[1] for p in chunk]
                )
                result.extend(values.float().cpu().reshape(-1).tolist())
        if len(result) != len(pairs) or any(
            not math.isfinite(v) or not 0 <= v <= 1 for v in result
        ):
            raise RuntimeError('Invalid VQAScore probabilities')
        return result


def make_server(host, port, backend, model, staging):
    staging = Path(staging).resolve()
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, status, value):
            raw = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path != '/health':
                return self.send(404, {'error': 'not found'})
            self.send(200, dict(loaded=True, model=model, staging=str(staging)))

        def do_POST(self):
            if self.path != '/score':
                return self.send(404, {'error': 'not found'})
            try:
                n = int(self.headers.get('Content-Length', '0'))
                if not 0 < n <= 4 * 1024 * 1024:
                    raise ValueError('Invalid body size')
                pairs = json.loads(self.rfile.read(n))['pairs']
                if not isinstance(pairs, list) or not pairs:
                    raise ValueError('Empty pairs')
                for pair in pairs:
                    if (
                        not isinstance(pair, list)
                        or len(pair) != 2
                        or not all(isinstance(v, str) for v in pair)
                    ):
                        raise ValueError('Invalid pair')
                    path = Path(pair[0]).resolve()
                    if staging not in path.parents or not path.is_file():
                        raise ValueError(
                            'Image must be an existing file in the shared staging directory'
                        )
            except (ValueError, TypeError, KeyError) as error:
                return self.send(400, {'error': str(error)})
            try:
                with lock:
                    scores = backend.score(pairs)
            except Exception as error:
                return self.send(503, {'error': f'{type(error).__name__}: {error}'})
            self.send(200, dict(model=model, scores=scores))

    class Server(ThreadingHTTPServer):
        request_queue_size = 256
        daemon_threads = True

    return Server((host, port), Handler)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='clip-flant5-xxl')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--cache', required=True)
    p.add_argument('--staging', required=True)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--probe-out', type=Path)
    a = p.parse_args()
    if not a.execute:
        print(json.dumps(dict(execute=False, model=a.model)))
        return
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_CACHE=a.cache)
    backend = VQABackend(a.model, a.device, a.cache, a.batch_size, a.probe_out)
    server = make_server(a.host, a.port, backend, a.model, a.staging)
    print(json.dumps(dict(ready=True, model=a.model, port=server.server_port)), flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
