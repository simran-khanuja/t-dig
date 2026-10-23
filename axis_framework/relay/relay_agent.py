"""Cluster side of axis-relay: dial out to the Cloudflare Worker and serve its requests.

    RELAY_SECRET_FILE=~/.config/axis_relay/secret \\
    python axis_framework/relay/relay_agent.py \\
        --relay wss://example.org/axis/__relay --upstream http://127.0.0.1:19100

For clusters that block cloudflared's port (7844) but allow outbound 443. The agent keeps a
WebSocket open to the Worker (worker.js); each message is one participant's HTTP request, which
it replays against the orchestrator and streams back. Nothing listens on the cluster. The wire
format is defined at the top of worker.js and must match it. RELAY_SECRET_FILE defaults to
~/.config/axis_relay/secret.

Reconnects forever (backoff capped at 30 s) and pings every 20 s, since Cloudflare closes an
idle WebSocket after ~100 s. A reconnect fails in-flight requests (the Worker answers 502);
the orchestrator's sessions are on disk.
"""

import argparse
import asyncio
import base64
import json
import os
import random
import struct
import sys
import time

import aiohttp

HEAD, CHUNK, END, ERROR = 1, 2, 3, 4
CHUNK_BYTES = 256 * 1024
HOP = {
    "connection",
    "keep-alive",
    "transfer-encoding",
    "upgrade",
    "proxy-connection",
    "te",
    "trailer",
    "host",
    "content-length",
    "content-encoding",
}


def log(*a):
    print(time.strftime("%F %T"), "[relay]", *a, flush=True)


def frame(kind, rid, payload=b""):
    return struct.pack(">BI", kind, rid) + payload


async def handle(msg, ws, http, upstream, send_lock, stats):
    rid = msg["id"]
    try:
        headers = {k: v for k, v in (msg.get("headers") or {}).items() if k.lower() not in HOP}
        body = base64.b64decode(msg["body"]) if msg.get("body") else None
        async with http.request(
            msg["method"],
            upstream + msg["path"],
            headers=headers,
            data=body,
            allow_redirects=False,
            timeout=aiohttp.ClientTimeout(total=900),
        ) as r:
            out_headers = {k: v for k, v in r.headers.items() if k.lower() not in HOP}
            async with send_lock:
                await ws.send_bytes(
                    frame(
                        HEAD, rid, json.dumps({"status": r.status, "headers": out_headers}).encode()
                    )
                )
            async for chunk in r.content.iter_chunked(CHUNK_BYTES):
                async with send_lock:
                    await ws.send_bytes(frame(CHUNK, rid, chunk))
        async with send_lock:
            await ws.send_bytes(frame(END, rid))
        stats["ok"] += 1
    except Exception as e:
        stats["err"] += 1
        log(f"request {rid} {msg.get('method')} {msg.get('path')} failed: {type(e).__name__}: {e}")
        try:
            async with send_lock:
                await ws.send_bytes(frame(ERROR, rid, f"{type(e).__name__}: {e}".encode()[:2000]))
        except Exception:
            pass


async def pinger(ws):
    while not ws.closed:
        await asyncio.sleep(20)
        await ws.send_str("ping")


async def run(a, secret):
    backoff = 1.0
    stats = {"ok": 0, "err": 0}
    async with aiohttp.ClientSession() as http:
        while True:
            try:
                async with http.ws_connect(
                    a.relay,
                    headers={"Authorization": f"Bearer {secret}"},
                    max_msg_size=4 * 1024 * 1024,
                    autoping=True,
                ) as ws:
                    log(f"connected to {a.relay} -> {a.upstream}")
                    backoff = 1.0
                    send_lock = asyncio.Lock()
                    ping = asyncio.create_task(pinger(ws))
                    tasks = set()
                    try:
                        async for m in ws:
                            if m.type == aiohttp.WSMsgType.TEXT:
                                if m.data == "pong":
                                    continue
                                t = asyncio.create_task(
                                    handle(
                                        json.loads(m.data), ws, http, a.upstream, send_lock, stats
                                    )
                                )
                                tasks.add(t)
                                t.add_done_callback(tasks.discard)
                            elif m.type in (
                                aiohttp.WSMsgType.CLOSE,
                                aiohttp.WSMsgType.CLOSED,
                                aiohttp.WSMsgType.ERROR,
                            ):
                                break
                    finally:
                        ping.cancel()
                        for t in tasks:
                            t.cancel()
                    log(
                        f"disconnected (close code {ws.close_code}); served ok={stats['ok']} "
                        f"err={stats['err']}"
                    )
                    # 1012: a newer agent took over (job handover). Exit instead of
                    # reconnecting, or the two agents would keep taking the socket back.
                    if ws.close_code == 1012:
                        log("replaced by a newer agent; exiting")
                        return
            except aiohttp.WSServerHandshakeError as e:
                log(
                    f"handshake refused: HTTP {e.status} {e.message}"
                    + (" -- wrong RELAY_SECRET?" if e.status == 403 else "")
                )
            except Exception as e:
                log(f"connection failed: {type(e).__name__}: {e}")
            await asyncio.sleep(backoff + random.random())
            backoff = min(backoff * 2, 30.0)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--relay", default="wss://example.org/axis/__relay")
    ap.add_argument("--upstream", default="http://127.0.0.1:19100")
    a = ap.parse_args()
    path = os.path.expanduser(os.environ.get("RELAY_SECRET_FILE", "~/.config/axis_relay/secret"))
    try:
        secret = open(path).read().strip()
    except OSError as e:
        sys.exit(f"[relay] cannot read the shared secret at {path}: {e}")
    if len(secret) < 32:
        sys.exit(f"[relay] {path} holds a secret shorter than 32 characters")
    asyncio.run(run(a, secret))


if __name__ == "__main__":
    main()
