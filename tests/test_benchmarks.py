from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmarks.cli import digest, load_bundle, make_rows, prepare


class BenchmarkTests(unittest.TestCase):
    def test_prepare_plan_is_inert_then_bundle_identity(self):
        raw = b'{"initial_prompt":"A clock."}\n{"initial_prompt":"A clock."}\n'
        spec = {'sha256': digest(raw), 'count': 2}
        with tempfile.TemporaryDirectory() as directory, patch(
            'benchmarks.cli.registry', return_value={'pag-test': spec}
        ):
            root = Path(directory)
            source = root / 'input.jsonl'
            source.write_bytes(raw)
            out = root / 'bundle'
            with patch('urllib.request.urlopen', side_effect=AssertionError('unexpected network')):
                prepare('pag-test', out, execute=False, allow_download=True)
                self.assertFalse(out.exists())
                prepare('pag-test', out, source, execute=True)
            manifest, rows = load_bundle(out)
            self.assertEqual(manifest['count'], 2)
            self.assertNotEqual(rows[0]['id'], rows[1]['id'])  # Never deduplicate benchmarks.
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                prepare('pag-test', out, source, execute=True)
            (out / 'prompts.jsonl').write_text('{}\n')
            with self.assertRaises(ValueError):
                load_bundle(out)

    def test_changed_source_and_missing_download_opt_in(self):
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            make_rows('pag-lexica', b'{}\n', {'sha256': 'invalid', 'count': 1})
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'allow-download'):
                prepare('pag-lexica', Path(directory) / 'out', execute=True)
