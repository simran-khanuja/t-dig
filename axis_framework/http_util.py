"""Stdlib HTTP server and client, shared by all four processes.

The processes run in three environments: the policy server in PY_TRAIN (vllm), the render and
suggest servers in PY_REWARD (tpips' transformers pin), and the orchestrator in either. PY_REWARD
has flask but not fastapi and PY_TRAIN the reverse, so every service uses the standard
library's `http.server` (the same pattern as `axis_grpo/training/vqa_service.py`).

Routing is a dict of (method, path) -> handler. Handlers take a parsed dict (POST body or
GET query) and return either a JSON-able object or a `Raw` for bytes (images, the UI).
"""

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Raw:
    """A non-JSON response body -- images and the static UI."""

    def __init__(self, body, content_type="application/octet-stream", status=200, headers=None):
        self.body = body
        self.content_type = content_type
        self.status = status
        self.headers = headers or {}


class HttpError(Exception):
    """Raise in a handler to answer with a status other than 200/500."""

    def __init__(self, status, message):
        super().__init__(message)
        self.status = status
        self.message = message


def serve(routes, host, port, name, ready_msg="", base_path=""):
    """Serve `routes` = {(method, "/path"): fn(payload) -> obj | Raw}; block forever.

    The server is threaded so connections are accepted while a model call holds its lock;
    single-threaded, the accept queue fills during a render and callers see TCP timeouts.

    base_path (e.g. "/axis") also serves the routes under that prefix, for a reverse proxy
    that forwards the path unchanged (a Cloudflare tunnel ingress rule does not strip it).
    Unprefixed paths keep working. "/axis" redirects to "/axis/" because the page uses
    relative URLs.

    Every handler gets payload["_client"]: the Cloudflare Access email, user agent and
    CF-IPCountry header. The IP is not recorded.
    """
    base_path = "/" + base_path.strip("/") if base_path.strip("/") else ""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, status, obj):
            if isinstance(obj, Raw):
                body, ctype, status = obj.body, obj.content_type, obj.status
            else:
                body = json.dumps(obj).encode()
                ctype = "application/json"
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            for k, v in (obj.headers if isinstance(obj, Raw) else {}).items():
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass  # no per-request log lines; errors still print below

        def _dispatch(self, method, payload):
            raw_path, _, query = self.path.partition("?")
            if base_path:
                if raw_path == base_path:
                    loc = base_path + "/" + ("?" + query if query else "")
                    return self._send(301, Raw(b"", "text/plain", 301, {"Location": loc}))
                if raw_path.startswith(base_path + "/"):
                    raw_path = raw_path[len(base_path) :]
            path = "/" + raw_path.strip("/")
            payload["_client"] = {
                "access_email": self.headers.get("Cf-Access-Authenticated-User-Email"),
                "user_agent": (self.headers.get("User-Agent") or "")[:300],
                "country": self.headers.get("CF-IPCountry"),
            }
            fn = routes.get((method, path))
            if fn is None:
                # A route ending in "/*" matches by prefix; the handler gets the trailing
                # segments as payload["_path"].
                for (m, p), f in routes.items():
                    if m == method and p.endswith("/*") and path.startswith(p[:-2] + "/"):
                        payload["_path"] = path[len(p) - 2 :].strip("/").split("/")
                        fn = f
                        break
            if fn is None:
                return self._send(404, {"error": f"no route for {method} {path}"})
            try:
                self._send(200, fn(payload))
            except HttpError as e:
                self._send(e.status, {"error": e.message})
            except Exception as e:
                import traceback

                traceback.print_exc()
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

        def do_GET(self):
            q = urllib.parse.urlparse(self.path).query
            payload = {k: v[0] for k, v in urllib.parse.parse_qs(q).items()}
            self._dispatch("GET", payload)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0) or 0)
            raw = self.rfile.read(n) if n else b"{}"
            try:
                payload = json.loads(raw or b"{}")
            except json.JSONDecodeError as e:
                return self._send(400, {"error": f"bad JSON body: {e}"})
            if not isinstance(payload, dict):
                return self._send(400, {"error": "body must be a JSON object"})
            self._dispatch("POST", payload)

    srv = ThreadingHTTPServer((host, port), Handler)
    srv.request_queue_size = 128
    srv.daemon_threads = True
    print(f"[{name}] serving on {host}:{port} {ready_msg}", flush=True)
    srv.serve_forever()


# --------------------------------------------------------------------------- client


def post_json(url, obj, timeout=1800, retries=0):
    """POST JSON and return the decoded reply, retrying network errors *retries* times.

    The default timeout is 1800 s because a render at 1024px/40 steps takes minutes.

    Raises:
        RuntimeError: an HTTP error status (with the server's error text) or no connection.
    """
    body = json.dumps(obj).encode()
    last = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            # Read the body for the server's error text; urlopen raises before reading it.
            detail = e.read().decode("utf-8", "replace")[:2000]
            raise RuntimeError(f"{url} returned {e.code}: {detail}") from e
        except (urllib.error.URLError, OSError) as e:
            last = e
            if attempt < retries:
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"{url} unreachable: {last}")


def get_json(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def wait_healthy(url, timeout=1800, interval=10, label=""):
    """Block until <url>/health answers with ready, or raise RuntimeError after *timeout* s."""
    t0 = time.time()
    err = None
    while time.time() - t0 < timeout:
        try:
            h = get_json(url.rstrip("/") + "/health", timeout=10)
            if h.get("ready", True):
                return h
        except Exception as e:  # not up yet
            err = e
        time.sleep(interval)
    raise RuntimeError(f"{label or url} never became healthy ({err})")
