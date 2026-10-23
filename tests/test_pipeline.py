"""Offline contract and integration tests: no API calls, no model weights."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from hpdv2.api import CallError, Reply
from hpdv2.cli import main
from hpdv2.cluster import cluster, select_members
from hpdv2.dataset import SCHEMA, load_input, prepare
from hpdv2.policy import parse_decisions, validate_decisions, system_prompt
from hpdv2.runner import execute, make_manifest, plan
from hpdv2.storage import atomic_json, digest, prompt_id, read_events, read_json, run_lock


def decision(row, stage="pass1", keep=True):
    result = {"id": row["id"], "keep": keep, "reason": "Open variation"}
    if stage == "pass2":
        result["variation_axes"] = (
            [
                {"dimension": "material", "values": ["wood", "metal"]},
                {"dimension": "color", "values": ["red", "blue"]},
                {"dimension": "medium", "values": ["photo", "watercolor"]},
            ]
            if keep
            else []
        )
    return result


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "train.json"
        self.source.write_text(
            json.dumps([{"prompt": p} for p in [" a chair ", "a chair", "a car", "a vase"]])
        )
        self.pool = self.root / "pool.json"
        prepare(self.source, self.pool, "train")
        self.data, self.rows = load_input(self.pool, "pass1")
        self.cfg = dict(
            stage="pass1",
            backend="proxy",
            api_base="https://example.invalid",
            model="fake",
            batch_size=2,
            max_tokens=1000,
            timeout=30,
            cache_system=False,
            allow_partial_input=False,
        )
        self.manifest = make_manifest(self.data, self.rows, self.cfg)
        self.directory = self.root / "run"
        self.calls = 0

    def client(self, batch, system):
        self.calls += 1
        return Reply(
            json.dumps([decision(r) for r in reversed(batch)]), "stop", {"total_tokens": 20}, "fake"
        )

    def run_filter(self, client=None, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return execute(
                self.directory,
                self.rows,
                self.manifest,
                client or self.client,
                max_requests=kwargs.pop("max_requests", 20),
                sleep=lambda _: None,
                **kwargs,
            )

    def test_short_ids_reorder_roundtrip_raw_and_resume(self):
        self.manifest = make_manifest(self.data, self.rows, {**self.cfg, "short_request_ids": True})

        def client(batch, system):
            self.assertEqual([r["id"] for r in batch], [f"p{i}" for i in range(len(batch))])
            return self.client(batch, system)

        result = self.run_filter(client=client)
        self.assertEqual(
            result["kept"], [{**decision(r), "prompt": r["prompt"]} for r in self.rows]
        )
        events = read_events(self.directory / "checkpoint.jsonl")
        self.assertEqual(events[0]["request_id_map"]["p0"], self.rows[0]["id"])
        self.assertIn('"id": "p1"', events[0]["attempts"][0]["raw_response"])
        calls = self.calls
        self.run_filter(client=client)
        self.assertEqual(self.calls, calls)

    def test_short_ids_do_not_guess_unknown_or_duplicate_ids(self):
        self.manifest = make_manifest(self.data, self.rows, {**self.cfg, "short_request_ids": True})

        def client(batch, system):
            records = [decision(r) for r in batch]
            records[0]["id"] = "p999"
            return Reply(json.dumps(records), "stop")

        result = self.run_filter(client=client, attempts=1)
        self.assertEqual(result["metadata"]["n_kept"], 0)
        self.assertEqual(result["metadata"]["n_errors"], len(self.rows))

    def test_prepare_dedup_and_hash(self):
        self.assertEqual([r["prompt"] for r in self.rows], ["a chair", "a car", "a vase"])
        self.assertEqual(self.data["pool_hash"], digest(self.rows))
        self.assertEqual(self.data["metadata"]["n_rows"], 4)
        with self.assertRaises(ValueError):
            prepare(self.source, self.pool, "train")

    def test_pool_tampering_rejected(self):
        self.data["prompts"][0]["prompt"] = "changed"
        atomic_json(self.pool, self.data)
        with self.assertRaises(ValueError):
            load_input(self.pool, "pass1")

    def test_reordered_ids_are_mapped_correctly(self):
        result = parse_decisions(
            json.dumps([decision(r) for r in reversed(self.rows)]), self.rows, "pass1"
        )
        self.assertEqual([r["prompt"] for r in result], [r["prompt"] for r in self.rows])

    def test_missing_middle_not_padded(self):
        with self.assertRaises(ValueError):
            validate_decisions([decision(self.rows[0]), decision(self.rows[2])], self.rows, "pass1")

    def test_invalid_schema_variants(self):
        valid = [decision(r, "pass2") for r in self.rows]
        cases = []
        for field, value in [
            ("keep", "true"),
            ("keep", 1),
            ("variation_axes", None),
            ("variation_axes", ["unknown"]),
            ("reason", ""),
            ("variation_axes", []),
            ("id", "unknown"),
        ]:
            rows = json.loads(json.dumps(valid))
            rows[0][field] = value
            cases.append(rows)
        duplicate = json.loads(json.dumps(valid))
        duplicate[1]["id"] = duplicate[0]["id"]
        cases.append(duplicate)
        extra = json.loads(json.dumps(valid))
        extra[0]["extra"] = True
        cases.append(extra)
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                validate_decisions(case, self.rows, "pass2")

    def test_reject_must_not_have_axes(self):
        record = decision(self.rows[0], "pass2")
        record["keep"] = False
        with self.assertRaises(ValueError):
            validate_decisions([record], self.rows[:1], "pass2")

    def test_three_freeform_axes_required(self):
        row = decision(self.rows[0], "pass2")
        row["variation_axes"][0] = {
            "dimension": "time displayed on the clock",
            "values": ["3:00", "9:00"],
        }
        self.assertEqual(
            validate_decisions([row], self.rows[:1], "pass2")[0]["variation_axes"],
            row["variation_axes"],
        )
        for count in range(3):
            short = {**row, "variation_axes": row["variation_axes"][:count]}
            with self.subTest(count=count), self.assertRaises(ValueError):
                validate_decisions([short], self.rows[:1], "pass2")
        row["variation_axes"].append(
            {"dimension": "wall texture", "values": ["smooth plaster", "rough brick"]}
        )
        self.assertEqual(
            len(validate_decisions([row], self.rows[:1], "pass2")[0]["variation_axes"]), 4
        )

    def test_axis_evidence_validation(self):
        base = decision(self.rows[0], "pass2")
        invalid = [
            {"dimension": " ", "values": ["a", "b"]},
            {"dimension": "test", "values": ["a"]},
            {"dimension": "test", "values": ["a", " A "]},
            {"dimension": "test", "values": ["a", "b", "c"]},
            {"dimension": "test", "values": ["", "b"]},
            {"dimension": "test", "values": [1, "b"]},
            {"dimension": "test", "values": "a/b"},
            {"dimension": "test", "values": ["a", "b"], "extra": True},
            {"dimension": " COLOR ", "values": ["a", "b"]},
        ]
        for axis in invalid:
            row = json.loads(json.dumps(base))
            row["variation_axes"][0] = axis
            with self.subTest(axis=axis), self.assertRaises(ValueError):
                validate_decisions([row], self.rows[:1], "pass2")

    def test_old_taxonomy_schema_rejected(self):
        row = decision(self.rows[0], "pass2")
        del row["variation_axes"]
        row.update(primary_axis="physical_attributes", secondary_axes=["artistic_style"])
        with self.assertRaises(ValueError):
            validate_decisions([row], self.rows[:1], "pass2")
        old_pool = {**self.data, "schema_version": 1}
        atomic_json(self.pool, old_pool)
        with self.assertRaises(ValueError):
            load_input(self.pool, "pass1")

    def test_dry_plan_no_writes(self):
        result = plan(self.directory, self.rows, self.manifest, 0)
        self.assertEqual(result["n_selected"], 0)
        self.assertFalse(self.directory.exists())

    def test_limit_zero_no_calls(self):
        result = self.run_filter(limit=0)
        self.assertEqual(self.calls, 0)
        self.assertEqual(result["metadata"]["n_pending"], 3)

    def test_resume_and_order(self):
        result = self.run_filter(limit=1)
        self.assertFalse(result["metadata"]["complete"])
        self.assertEqual(result["metadata"]["n_kept"], 1)
        result = self.run_filter()
        self.assertTrue(result["metadata"]["complete"])
        self.assertEqual(self.calls, 2)
        self.assertEqual([r["id"] for r in result["kept"]], [r["id"] for r in self.rows])
        self.run_filter()
        self.assertEqual(self.calls, 2)

    def test_error_pending_then_retried(self):
        result = self.run_filter(client=lambda *_: Reply("broken", "stop"), attempts=1)
        self.assertEqual(result["metadata"]["n_errors"], 3)
        self.assertEqual(result["kept"], [])
        with self.assertRaises(ValueError):
            load_input(self.directory / "summary.json", "pass2")
        result = self.run_filter()
        self.assertTrue(result["metadata"]["complete"])
        self.assertEqual(result["errors"], [])

    def test_retry_then_success_usage_preserved(self):
        calls = []

        def client(batch, _):
            calls.append(1)
            if len(calls) == 1:
                return Reply("broken", "stop", {"total_tokens": 8})
            return Reply(json.dumps([decision(r) for r in batch]), "stop", {"total_tokens": 10})

        self.run_filter(client, limit=1)
        event = read_events(self.directory / "checkpoint.jsonl")[0]
        self.assertEqual(len(event["attempts"]), 2)
        self.assertEqual(event["attempts"][0]["usage"]["total_tokens"], 8)

    def test_truncated_response_not_accepted(self):
        def client(batch, _):
            return Reply(json.dumps([decision(r) for r in batch]), "length")

        result = self.run_filter(client, attempts=1)
        self.assertEqual(result["kept"], [])

    def test_budget_caps_retries_and_concurrency(self):
        calls = []
        lock = threading.Lock()

        def client(*_):
            with lock:
                calls.append(1)
            raise CallError("429", retryable=True)

        result = self.run_filter(client, workers=3, max_requests=2, attempts=5)
        self.assertEqual(len(calls), 2)
        self.assertEqual(result["metadata"]["n_pending"], 3)

    def test_fatal_error_stops_serial_run(self):
        calls = []

        def client(*_):
            calls.append(1)
            raise CallError("401", retryable=False)

        self.run_filter(client)
        self.assertEqual(len(calls), 1)

    def test_manifest_changes_rejected_before_calls(self):
        self.run_filter(limit=1)
        for key, value in [("model", "other"), ("batch_size", 1), ("cache_system", True)]:
            changed = make_manifest(self.data, self.rows, {**self.cfg, key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                execute(self.directory, self.rows, changed, self.client, max_requests=1)
        self.assertEqual(self.calls, 1)

    def test_foreign_journal_rejected(self):
        self.run_filter(limit=1)
        path = self.directory / "checkpoint.jsonl"
        event = read_events(path)[0]
        event["run_hash"] = "foreign"
        path.write_text(json.dumps(event) + "\n")
        with self.assertRaises(ValueError):
            self.run_filter()

    def test_incomplete_tail_requires_explicit_repair(self):
        self.run_filter(limit=1)
        path = self.directory / "checkpoint.jsonl"
        with path.open("ab") as stream:
            stream.write(b'{"partial":')
        with self.assertRaises(ValueError):
            self.run_filter()
        result = self.run_filter(repair_tail=True)
        self.assertTrue(result["metadata"]["complete"])
        self.assertEqual(len(list(self.directory.glob("*.bak"))), 1)

    def test_interior_corruption_is_not_repaired(self):
        self.run_filter(limit=1)
        path = self.directory / "checkpoint.jsonl"
        path.write_bytes(b"not-json\n")
        with self.assertRaises(ValueError):
            self.run_filter(repair_tail=True)

    def test_second_writer_refused(self):
        with run_lock(self.directory):
            with self.assertRaises(ValueError):
                self.run_filter()

    def test_end_to_end_pass2(self):
        self.run_filter()
        data, rows = load_input(self.directory / "summary.json", "pass2")
        manifest = make_manifest(data, rows, {**self.cfg, "stage": "pass2"})

        def client(batch, _):
            return Reply(
                json.dumps([decision(r, "pass2", i != 0) for i, r in enumerate(batch)]), "stop"
            )

        with contextlib.redirect_stdout(io.StringIO()):
            result = execute(self.root / "pass2", rows, manifest, client, max_requests=20)
        self.assertTrue(result["metadata"]["complete"])
        self.assertEqual(result["metadata"]["n_kept"], 1)
        self.assertEqual(result["metadata"]["n_rejected"], 2)
        self.assertEqual(result["metadata"]["variation_axis_count_distribution"], {"3": 1})
        load_input(self.root / "pass2/summary.json", "cluster")

    def test_cli_execute_requires_budget(self):
        args = [
            "filter",
            "--stage",
            "pass1",
            "--input",
            str(self.pool),
            "--run-dir",
            str(self.directory),
            "--backend",
            "anthropic",
            "--model",
            "fake",
            "--execute",
        ]
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(args), 1)
        self.assertFalse(self.directory.exists())

    def test_partial_lineage_cannot_be_promoted_as_complete(self):
        self.run_filter(limit=1)
        data, rows = load_input(self.directory / "summary.json", "pass2", allow_partial=True)
        manifest = make_manifest(
            data, rows, {**self.cfg, "stage": "pass2", "allow_partial_input": True}
        )

        def client(batch, _):
            return Reply(json.dumps([decision(r, "pass2") for r in batch]), "stop")

        with contextlib.redirect_stdout(io.StringIO()):
            result = execute(self.root / "pilot", rows, manifest, client, max_requests=2)
        self.assertTrue(result["metadata"]["complete"])
        self.assertFalse(result["metadata"]["full_pipeline_complete"])
        with self.assertRaises(ValueError):
            load_input(self.root / "pilot/summary.json", "cluster")
        load_input(self.root / "pilot/summary.json", "cluster", allow_partial=True)

    def test_cli_proxy_requires_explicit_endpoint(self):
        args = [
            "filter",
            "--stage",
            "pass1",
            "--input",
            str(self.pool),
            "--run-dir",
            str(self.directory),
            "--backend",
            "proxy",
            "--model",
            "fake",
        ]
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(args), 1)

    def test_cli_dry_run_no_key_or_sdk(self):
        args = [
            "filter",
            "--stage",
            "pass1",
            "--input",
            str(self.pool),
            "--run-dir",
            str(self.directory),
            "--backend",
            "anthropic",
            "--model",
            "fake",
        ]
        with contextlib.redirect_stdout(io.StringIO()), patch(
            "hpdv2.cli.LiteLLMClient", side_effect=AssertionError
        ):
            self.assertEqual(main(args), 0)
        self.assertFalse(self.directory.exists())

    def test_show_policy_matches_actual_system_prompt(self):
        for stage in ["pass1", "pass2"]:
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(["show-policy", "--stage", stage]), 0)
            self.assertEqual(output.getvalue(), system_prompt(stage) + "\n")


HAS_CLUSTER = all(importlib.util.find_spec(m) for m in ("numpy", "sklearn"))


@unittest.skipUnless(HAS_CLUSTER, "install .[cluster] for real clustering tests")
class ClusterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "input.json"
        self.rows = []
        for text in ["chair", "seat", "car"]:
            row = {"id": prompt_id(text), "prompt": text}
            self.rows.append({**row, **decision(row, "pass2")})
        self.write_input(self.rows)

    def write_input(self, rows):
        atomic_json(
            self.path,
            {
                "schema_version": SCHEMA,
                "kind": "filter_summary",
                "stage": "pass2",
                "run_hash": "test",
                "metadata": {"complete": True, "full_pipeline_complete": True},
                "kept": rows,
            },
        )

    def embed(self, texts):
        return [{"chair": [1, 0], "seat": [1, 0.01], "car": [0, 1]}[t] for t in texts]

    def test_cap_minimum_and_stable_order(self):
        out = self.root / "out.json"
        result = cluster(
            self.path, out, distance=0.1, cap=1, min_size=2, execute=True, embedder=self.embed
        )
        self.assertEqual(result["n_kept"], 1)
        self.assertEqual(result["n_clusters_before_minimum"], 2)
        first = read_json(out)["kept"]
        self.write_input(list(reversed(self.rows)))
        cluster(
            self.path,
            self.root / "out2.json",
            distance=0.1,
            cap=1,
            min_size=2,
            execute=True,
            embedder=self.embed,
        )
        self.assertEqual(first, read_json(self.root / "out2.json")["kept"])

    def test_empty_and_singleton(self):
        for n in [0, 1]:
            self.write_input(self.rows[:n])
            result = cluster(
                self.path,
                self.root / f"out{n}.json",
                distance=0.1,
                execute=True,
                embedder=self.embed,
            )
            self.assertEqual(result["n_kept"], n)

    def test_memory_guard_precedes_embeddings(self):
        with self.assertRaises(ValueError):
            cluster(
                self.path,
                self.root / "out.json",
                distance=0.1,
                execute=True,
                max_matrix_gib=1e-12,
                embedder=lambda _: self.fail("embedded before guard"),
            )

    def test_dry_run_does_not_embed(self):
        result = cluster(
            self.path,
            self.root / "out.json",
            distance=0.1,
            embedder=lambda _: self.fail("dry run embedded"),
        )
        self.assertFalse(result["execute"])
        self.assertFalse((self.root / "out.json").exists())

    def test_farthest_selection(self):
        import numpy as np

        vecs = np.array([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]])
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        selected, _ = select_members(vecs, np.array([0, 0, 0]), 2, 1)
        self.assertIn(2, selected)
        self.assertEqual(len(set(selected)), 2)

    def test_default_is_one_random_member_per_cluster(self):
        out = self.root / "random.json"
        metadata = cluster(self.path, out, distance=0.1, execute=True, embedder=self.embed)
        self.assertEqual(metadata["n_kept"], 2)
        self.assertEqual(metadata["settings"]["selection"], "random")
        self.assertEqual(metadata["settings"]["seed"], 42)
        for group in read_json(out)["cluster_summary"]:
            self.assertEqual(len(group["selected_ids"]), 1)
            self.assertIn(group["selected_ids"][0], group["member_ids"])
        for row in read_json(out)["kept"]:
            self.assertEqual(len(row["variation_axes"]), 3)
            self.assertNotIn("primary_axis", row)

    def test_random_reproducible_across_row_and_cluster_label_order(self):
        import numpy as np

        vectors = np.eye(4)
        ids = ["a", "b", "c", "d"]
        chosen, _ = select_members(
            vectors, np.array([0, 0, 1, 1]), 1, 1, selection="random", seed=7, ids=ids
        )
        reverse, _ = select_members(
            vectors[::-1], np.array([8, 8, 3, 3]), 1, 1, selection="random", seed=7, ids=ids[::-1]
        )
        self.assertEqual({ids[i] for i in chosen}, {ids[::-1][i] for i in reverse})

    def test_random_can_pick_non_centroid_and_respects_cap(self):
        import numpy as np

        vectors = np.array([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]])
        labels = np.zeros(3, dtype=int)
        ids = ["a", "b", "c"]
        picks = {
            select_members(vectors, labels, 1, 1, selection="random", seed=s, ids=ids)[0][0]
            for s in range(30)
        }
        self.assertEqual(picks, {0, 1, 2})
        selected, _ = select_members(vectors, labels, 20, 1, selection="random", seed=3, ids=ids)
        self.assertEqual(sorted(selected), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
