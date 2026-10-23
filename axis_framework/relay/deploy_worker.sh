#!/bin/bash
# Deploy worker.js as "axis-relay" on <your domain>/axis* through the Cloudflare API.
#
#   ZONE_NAME=<your domain> CF_API_TOKEN_FILE=~/.config/axis_relay/cf_token bash axis_framework/relay/deploy_worker.sh
#
# The token needs: Account > Workers Scripts > Edit, Zone > Workers Routes > Edit and
# Zone > Zone > Read, for your zone (the "Edit Cloudflare Workers" template has these).
#
# SCRIPT (axis-relay) and ROUTE (<ZONE_NAME>/axis*) override the names. Needs curl and jq.
#
# Idempotent: re-running uploads the current worker.js and keeps the existing route and shared
# secret. The secret is created on the first run in ~/.config/axis_relay/secret (mode 600);
# relay_agent.py presents it.
set -euo pipefail

ZONE_NAME=${ZONE_NAME:?set ZONE_NAME=<your Cloudflare zone, e.g. example.org>}
SCRIPT=${SCRIPT:-axis-relay}
ROUTE=${ROUTE:-$ZONE_NAME/axis*}
HERE=$(cd "$(dirname "$0")" && pwd)
CONF=$HOME/.config/axis_relay
TOKEN_FILE=${CF_API_TOKEN_FILE:-$CONF/cf_token}
API=https://api.cloudflare.com/client/v4

mkdir -p "$CONF"; chmod 700 "$CONF"
[[ -r "$TOKEN_FILE" ]] || { echo "no API token at $TOKEN_FILE" >&2; exit 1; }
TOKEN=$(tr -d ' \n' < "$TOKEN_FILE")
auth=(-H "Authorization: Bearer $TOKEN")

if [[ ! -s "$CONF/secret" ]]; then
  (umask 077; head -c 48 /dev/urandom | base64 | tr -d '\n=+/' > "$CONF/secret")
  echo "created shared secret at $CONF/secret"
fi
SECRET=$(cat "$CONF/secret")

zone=$(curl -sS "${auth[@]}" "$API/zones?name=$ZONE_NAME")
ZONE_ID=$(jq -r '.result[0].id // empty' <<<"$zone")
ACCOUNT_ID=$(jq -r '.result[0].account.id // empty' <<<"$zone")
[[ -n "$ZONE_ID" && -n "$ACCOUNT_ID" ]] || { echo "zone lookup failed: $(jq -c '.errors' <<<"$zone")" >&2; exit 1; }
echo "zone $ZONE_NAME = $ZONE_ID (account $ACCOUNT_ID)"

upload() {   # $1 = include migrations (1/0)
  local meta
  meta=$(jq -n --arg secret "$SECRET" --argjson mig "$1" '{
    main_module: "worker.js",
    compatibility_date: "2025-09-01",
    bindings: [
      {type: "durable_object_namespace", name: "RELAY", class_name: "Relay"},
      {type: "secret_text", name: "RELAY_SECRET", text: $secret}
    ]
  } + (if $mig == 1 then {migrations: {new_tag: "v1", new_sqlite_classes: ["Relay"]}} else {} end)')
  curl -sS -X PUT "${auth[@]}" "$API/accounts/$ACCOUNT_ID/workers/scripts/$SCRIPT" \
    -F "metadata=$meta;type=application/json" \
    -F "worker.js=@$HERE/worker.js;type=application/javascript+module"
}

out=$(upload 1)
if [[ $(jq -r '.success' <<<"$out") != "true" ]]; then
  # Already deployed once: the Relay class exists, and re-sending its creating migration is
  # an error. Upload the new code against the existing class instead.
  if grep -qiE "migration|tag" <<<"$out"; then
    out=$(upload 0)
  fi
fi
[[ $(jq -r '.success' <<<"$out") == "true" ]] || { echo "upload failed: $(jq -c '.errors' <<<"$out")" >&2; exit 1; }
echo "uploaded worker $SCRIPT"

routes=$(curl -sS "${auth[@]}" "$API/zones/$ZONE_ID/workers/routes")
existing=$(jq -r --arg p "$ROUTE" '.result[] | select(.pattern == $p) | .id' <<<"$routes")
if [[ -z "$existing" ]]; then
  r=$(curl -sS -X POST "${auth[@]}" -H "Content-Type: application/json" \
        "$API/zones/$ZONE_ID/workers/routes" -d "$(jq -n --arg p "$ROUTE" --arg s "$SCRIPT" '{pattern: $p, script: $s}')")
  [[ $(jq -r '.success' <<<"$r") == "true" ]] || { echo "route failed: $(jq -c '.errors' <<<"$r")" >&2; exit 1; }
  echo "route $ROUTE -> $SCRIPT created"
else
  echo "route $ROUTE already exists"
fi
echo "done. With no agent connected, https://$ZONE_NAME/axis/ answers 503 (offline page)."
