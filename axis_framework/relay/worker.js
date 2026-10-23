// axis-relay: serve example.org/axis/* from a process on the cluster that can only dial OUT.
//
//   participant --https--> this Worker --> Relay (Durable Object) ==wss==> relay_agent.py (the cluster)
//                                                                         --> localhost:19100
//
// For clusters that allow outbound 443 but not cloudflared's 7844. relay_agent.py keeps a
// WebSocket open to /axis/__relay (authenticated by a shared secret). One Durable Object holds
// that socket; each participant request is sent down it and answered with what the agent
// streams back.
//
// Wire format (relay_agent.py must match):
//   DO -> agent, text:    {"t":"req","id":N,"method":..,"path":"/axis/..?q","headers":{..},"body":"<base64>"}
//   agent -> DO, binary:  1 byte kind | 4 byte big-endian id | payload
//        kind 1 HEAD   payload = UTF-8 JSON {"status":N,"headers":{..}}
//        kind 2 CHUNK  payload = raw body bytes (the agent keeps these <= 256 KiB; Workers cap a
//                      WebSocket message at 1 MiB, and a render is ~1.6 MB per PNG)
//        kind 3 END    no payload
//        kind 4 ERROR  payload = UTF-8 message
//   either side, text "ping" / "pong" keeps the socket from idling out.

// One relay per study: each URL prefix in STUDIES has its own Durable Object and agent (which
// dials <prefix>/__relay). ALIASES maps another prefix to a study's prefix; the path is
// rewritten before relaying.
const STUDIES = ["/axis"];
const ALIASES = {};
function baseOf(pathname) {
  for (const b of [...STUDIES, ...Object.keys(ALIASES)]) {
    if (pathname === b || pathname.startsWith(b + "/")) return b;
  }
  return null;
}
const HEAD_TIMEOUT_MS = 10 * 60 * 1000;      // a cold policy load can take minutes
const MAX_REQUEST_BODY = 1024 * 512;         // the app's POSTs are small JSON
const HOP = new Set(["connection", "keep-alive", "transfer-encoding", "upgrade",
                     "proxy-connection", "te", "trailer", "host", "content-length"]);

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    let base = baseOf(url.pathname);
    if (!base) return new Response("not found", { status: 404 });
    if (ALIASES[base]) {
      const target = ALIASES[base];
      url.pathname = target + url.pathname.slice(base.length);
      request = new Request(url.toString(), request);
      base = target;
    }
    const stub = env.RELAY.get(env.RELAY.idFromName(base.slice(1)));
    return stub.fetch(request);
  },
};

function offline() {
  const html = `<!doctype html><meta charset="utf-8"><title>Offline</title>
<style>body{font:16px/1.5 system-ui,sans-serif;max-width:32rem;margin:15vh auto;padding:0 1rem;color:#333}</style>
<h2>The study is offline right now</h2>
<p>The image server is restarting or not running. Please try again in a few minutes.</p>`;
  return new Response(html, { status: 503, headers: { "content-type": "text/html; charset=utf-8",
                                                      "retry-after": "60" } });
}

function b64(buf) {
  let s = "";
  const bytes = new Uint8Array(buf);
  for (let i = 0; i < bytes.length; i += 0x8000) {
    s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  }
  return btoa(s);
}

export class Relay {
  constructor(state, env) {
    this.state = state;
    this.env = env;
    this.pending = new Map();   // id -> {resolve, writer, timer, started}
    this.nextId = 1;
    // Answered by the runtime without waking the object, so keep-alives cost nothing.
    this.state.setWebSocketAutoResponse(new WebSocketRequestResponsePair("ping", "pong"));
  }

  agent() {
    const socks = this.state.getWebSockets("agent");
    return socks.length ? socks[socks.length - 1] : null;
  }

  async fetch(request) {
    const url = new URL(request.url);
    if (STUDIES.some((b) => url.pathname === b + "/__relay")) return this.connectAgent(request);

    const ws = this.agent();
    if (!ws) return offline();

    let body = "";
    if (request.method !== "GET" && request.method !== "HEAD") {
      const buf = await request.arrayBuffer();
      if (buf.byteLength > MAX_REQUEST_BODY) return new Response("request too large", { status: 413 });
      body = b64(buf);
    }
    const headers = {};
    for (const [k, v] of request.headers) {
      if (!HOP.has(k.toLowerCase())) headers[k] = v;
    }
    const id = this.nextId++;
    if (this.nextId > 0x7fffffff) this.nextId = 1;

    const { readable, writable } = new TransformStream();
    const writer = writable.getWriter();
    const head = new Promise((resolve) => {
      const timer = setTimeout(() => this.fail(id, 504, "the image server took too long to answer"),
                               HEAD_TIMEOUT_MS);
      this.pending.set(id, { resolve, writer, timer, readable });
    });
    try {
      ws.send(JSON.stringify({ t: "req", id, method: request.method,
                               path: url.pathname + url.search, headers, body }));
    } catch (e) {
      this.fail(id, 502, "relay socket closed");
    }
    return head;
  }

  connectAgent(request) {
    const secret = this.env.RELAY_SECRET || "";
    if (!secret || request.headers.get("authorization") !== `Bearer ${secret}`) {
      return new Response("forbidden", { status: 403 });
    }
    if ((request.headers.get("upgrade") || "").toLowerCase() !== "websocket") {
      return new Response("expected a websocket upgrade", { status: 426 });
    }
    // One agent at a time: a reconnect replaces the old socket rather than splitting traffic.
    for (const old of this.state.getWebSockets("agent")) {
      try { old.close(1012, "replaced by a newer agent"); } catch (_) {}
    }
    this.failAll(502, "the image server reconnected; please retry");
    const pair = new WebSocketPair();
    const [client, server] = Object.values(pair);
    this.state.acceptWebSocket(server, ["agent"]);
    return new Response(null, { status: 101, webSocket: client });
  }

  async webSocketMessage(ws, message) {
    if (typeof message === "string") return;   // "ping" is auto-answered; nothing else is expected
    const view = new DataView(message);
    const kind = view.getUint8(0);
    const id = view.getUint32(1);
    const payload = new Uint8Array(message, 5);
    const p = this.pending.get(id);
    if (!p) return;   // timed out or already failed
    if (kind === 1) {
      clearTimeout(p.timer);
      const meta = JSON.parse(new TextDecoder().decode(payload));
      const h = new Headers();
      for (const [k, v] of Object.entries(meta.headers || {})) {
        if (!HOP.has(k.toLowerCase())) h.set(k, v);
      }
      const nullBody = [101, 204, 205, 304].includes(meta.status);
      p.resolve(new Response(nullBody ? null : p.readable, { status: meta.status, headers: h }));
      if (nullBody) this.finish(id);
    } else if (kind === 2) {
      // Not awaited: a participant who closed the tab stops reading, and an awaited write
      // would then stall this handler. Writes still queue in order.
      p.writer.write(payload.slice()).catch(() => this.finish(id));
    } else if (kind === 3) {
      this.finish(id);
    } else if (kind === 4) {
      this.fail(id, 502, new TextDecoder().decode(payload));
    }
  }

  finish(id) {
    const p = this.pending.get(id);
    if (!p) return;
    this.pending.delete(id);
    clearTimeout(p.timer);
    p.writer.close().catch(() => {});
  }

  fail(id, status, message) {
    const p = this.pending.get(id);
    if (!p) return;
    this.pending.delete(id);
    clearTimeout(p.timer);
    // If the head already went out the response has started; all that is left is to cut it.
    p.resolve(new Response(message, { status, headers: { "content-type": "text/plain" } }));
    p.writer.abort(message).catch(() => {});
  }

  failAll(status, message) {
    for (const id of [...this.pending.keys()]) this.fail(id, status, message);
  }

  async webSocketClose(ws) {
    if (!this.agent()) this.failAll(502, "the image server disconnected; please retry");
  }

  async webSocketError(ws) {
    if (!this.agent()) this.failAll(502, "the image server disconnected; please retry");
  }
}
