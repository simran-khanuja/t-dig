"""Bounded cross-request batching and durable, content-addressed reward results."""

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import queue
import threading
import time
import traceback

from hpdv2.storage import atomic_json, digest, read_json, run_lock
from .metrics import score
from .protocol import (
    PROTOCOL,
    QUALITY_SCORER,
    request_id,
    validate_item,
    validate_metrics,
)
from .config import validate_service


class RewardEngine:
    def __init__(self, backend, config, directory):
        self.backend, self.config = backend, config
        self.directory = Path(directory)
        self.identity = digest(config)
        self.completed = 0

    def score_many(self, items):
        for item in items:
            validate_item(item)
        outputs, missing = [None] * len(items), []
        for i, item in enumerate(items):
            path = self.directory / 'results' / f'{request_id(item)}.json'
            if path.exists():
                value = read_json(path)
                if value['service_hash'] != self.identity or value['item'] != item:
                    raise ValueError('Cached reward identity mismatch')
                outputs[i] = value['response']
            else:
                missing.append(i)
        if not missing:
            return outputs
        live = [items[i] for i in missing]
        start = time.monotonic()
        groups = self.backend.render(live)
        if len(groups) != len(live) or any(len(g) != it['k'] for g, it in zip(groups, live)):
            raise ValueError('Wrong rendered group count')
        alignments = self.backend.alignment(groups, live)
        if len(alignments) != len(live):
            raise ValueError('Wrong alignment group count')
        staged = []
        for i, item, images, alignment in zip(missing, live, groups, alignments):
            vectors = self.backend.embed(images, item['tpips_factor'])
            overall = (
                self.backend.embed(images, 'overall')
                if self.config['reward']['floor'] == 'overall' and item['tpips_factor'] != 'overall'
                else None
            )
            drift = self.backend.drift(item, images) if item.get('task') == 'edit' else None
            # Tracked scores use the base prompt, as fidelity does. They are logged only, except
            # that a reward with a quality term reads the QUALITY_SCORER values.
            tracked = (
                self.backend.tracked(images, item['base_prompt'])
                if self.config.get('track_metrics')
                else None
            )
            quality = {}
            if 'quality_weight' in self.config['reward']:
                if 'quality_thresh' not in item:
                    raise ValueError('The quality term needs quality_thresh on every row')
                quality = dict(
                    quality=tracked[QUALITY_SCORER], quality_threshold=item['quality_thresh']
                )
            metrics = score(
                vectors,
                alignment,
                item['tpips_factor'],
                self.config['reward'],
                overall,
                item.get('align_thresh'),
                item['prompts'],
                drift,
                length_cap=item.get('length_cap'),
                **quality,
            )
            detail = dict(alignment=[float(v) for v in alignment])
            if drift is not None:
                detail['drift'] = drift
            if tracked is not None:
                detail['tracked'] = tracked
                metrics.update({k: float(sum(v) / len(v)) for k, v in detail['tracked'].items()})
                validate_metrics(metrics, self.config['track_metrics'])
            response = dict(
                protocol=PROTOCOL,
                request_id=request_id(item),
                renderer=self.config['renderer'],
                reward_hash=digest(self.config['reward']),
                service_hash=self.identity,
                metrics=metrics,
            )
            staged.append((i, item, images, response, detail))
        # A scoring exception propagates as a service failure. Images are written first, on eight
        # threads (PNG encoding releases the GIL); each result record is then written atomically.
        if self.config.get('save_images', False):
            jobs = []
            for _, item, images, _, _ in staged:
                folder = self.directory / 'images' / request_id(item)
                folder.mkdir(parents=True, exist_ok=True)
                jobs += [(image, folder / f'{j:04d}.png') for j, image in enumerate(images)]
            with ThreadPoolExecutor(8) as pool:
                list(pool.map(lambda job: job[0].save(job[1]), jobs))
        for i, item, images, response, detail in staged:
            rid = request_id(item)
            atomic_json(
                self.directory / 'results' / f'{rid}.json',
                dict(
                    service_hash=self.identity,
                    item=item,
                    response=response,
                    detail=detail,
                    batch_items=len(live),
                    batch_seconds=time.monotonic() - start,
                ),
            )
            outputs[i] = response
            self.completed += 1
        atomic_json(
            self.directory / 'status.json',
            dict(
                completed_new=self.completed,
                last_batch_items=len(live),
                last_batch_seconds=time.monotonic() - start,
            ),
        )
        return outputs


class Batcher:
    def __init__(self, engine, max_items=4, wait_ms=20, capacity=256):
        if max_items < 1 or capacity < 1 or wait_ms < 0:
            raise ValueError('Invalid batching limits')
        self.engine, self.max_items, self.wait = engine, max_items, wait_ms / 1000
        self.queue = queue.Queue(maxsize=capacity)
        self.pending = {}
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._work, daemon=True)
        self.thread.start()

    def submit(self, item):
        rid = request_id(item)
        with self.lock:
            if rid in self.pending:
                return self.pending[rid]
            future = Future()
            self.queue.put_nowait((rid, item, future))
            self.pending[rid] = future
            return future

    def _work(self):
        while True:
            first = self.queue.get()
            if first is None:
                return
            batch = [first]
            deadline = time.monotonic() + self.wait
            while len(batch) < self.max_items:
                try:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    entry = self.queue.get(timeout=remaining)
                    if entry is None:
                        self.queue.put(None)
                        break
                    batch.append(entry)
                except queue.Empty:
                    break
            try:
                results = self.engine.score_many([entry[1] for entry in batch])
                if len(results) != len(batch):
                    raise RuntimeError('Reward batch count mismatch')
                for entry, result in zip(batch, results):
                    entry[2].set_result(result)
            except Exception as error:
                traceback.print_exc()
                for entry in batch:
                    entry[2].set_exception(error)
            finally:
                with self.lock:
                    for rid, _, _ in batch:
                        self.pending.pop(rid, None)

    def close(self):
        self.queue.put(None)
        self.thread.join(timeout=30)


def make_server(host, port, batcher, config):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, status, obj):
            raw = json.dumps(obj, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path != '/health':
                return self.send(404, {'error': 'not found'})
            self.send(
                200,
                dict(
                    protocol=PROTOCOL,
                    loaded=True,
                    renderer=config['renderer'],
                    reward_hash=digest(config['reward']),
                    service_hash=digest(config),
                    track_metrics=list(config.get('track_metrics', [])),
                ),
            )

        def do_POST(self):
            if self.path != '/score':
                return self.send(404, {'error': 'not found'})
            try:
                n = int(self.headers.get('Content-Length', '0'))
                if not 0 < n <= 4 * 1024 * 1024:
                    raise ValueError('Invalid request size')
                value = json.loads(self.rfile.read(n))
                item = validate_item(value['item'])
                if (
                    value['protocol'] != PROTOCOL
                    or value['request_id'] != request_id(item)
                    or value['renderer'] != config['renderer']
                    or value['reward_hash'] != digest(config['reward'])
                ):
                    raise ValueError('Reward request identity mismatch')
            except (KeyError, ValueError, TypeError) as error:
                return self.send(400, {'error': str(error)})
            try:
                result = batcher.submit(item).result(timeout=config.get('timeout', 1800))
            except Exception as error:
                return self.send(503, {'error': f'{type(error).__name__}: {error}'})
            self.send(200, result)

    class Server(ThreadingHTTPServer):
        request_queue_size = 256
        daemon_threads = True

    server = Server((host, port), Handler)
    return server


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    config = validate_service(read_json(a.config))
    if not a.execute:
        print(json.dumps(dict(execute=False, config_hash=digest(config))))
        return
    from .backend import GPUBackend

    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_CACHE=config['cache'])
    with run_lock(a.out):
        identity = dict(config=config, config_hash=digest(config))
        if (a.out / 'manifest.json').exists() and read_json(a.out / 'manifest.json') != identity:
            raise ValueError('Reward service configuration changed on resume')
        atomic_json(a.out / 'manifest.json', identity)
        backend = GPUBackend(config)
        engine = RewardEngine(backend, config, a.out)
        batcher = Batcher(engine, config['batch_items'], config['batch_wait_ms'])
        server = make_server(a.host, a.port, batcher, config)
        print(json.dumps(dict(ready=True, host=a.host, port=server.server_port)), flush=True)
        try:
            server.serve_forever()
        finally:
            server.server_close()
            batcher.close()


if __name__ == '__main__':
    main()
