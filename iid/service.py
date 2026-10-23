"""HTTP client for axis_framework's render service; client and server share a filesystem."""

import json
import urllib.request
from urllib.parse import urlsplit

# From axis_framework.latents: with alpha=1 and n_total=1 the latent seed is the request seed plus
# INDEPENDENT_STRIDE, modulo SEED_MODULUS.
SEED_MODULUS = 2**31 - 1
INDEPENDENT_STRIDE = 10_000_019


class RenderService:
    def __init__(self, url, timeout=600):
        parts = urlsplit(url)
        if (
            parts.scheme not in {'http', 'https'}
            or not parts.netloc
            or parts.username
            or parts.password
            or parts.query
            or parts.fragment
        ):
            raise ValueError('Render URL must be HTTP(S), without credentials/query/fragment')
        self.url, self.timeout = url.rstrip('/'), timeout

    def request(self, route, body=None):
        request = urllib.request.Request(
            self.url + route,
            data=None if body is None else json.dumps(body).encode(),
            headers={'Content-Type': 'application/json'},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except Exception as error:
            # Do not persist server messages that might contain credentials.
            raise RuntimeError('Render service request failed: ' + type(error).__name__) from None
        if not isinstance(result, dict):
            raise ValueError('Render service returned a non-object response')
        return result

    def health(self):
        return self.request('/health')

    def render(self, payload):
        return self.request('/render', payload)
