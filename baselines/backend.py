"""Lazy GPU T2I model and isolated persistent PAG policy process."""

import json
import os
from pathlib import Path
import selectors
import subprocess
import sys


class Backend:
    def __init__(self, assets, cfg, run):
        self.assets, self.cfg, self.run = assets, cfg, run
        self.t2i_model = None
        self.worker = None
        self.log = None

    def read_worker(self):
        with selectors.DefaultSelector() as selector:
            selector.register(self.worker.stdout, selectors.EVENT_READ)
            if not selector.select(300):
                raise TimeoutError("PAG decoder timed out; inspect pag-worker.log")
        line = self.worker.stdout.readline()
        if not line:
            raise ValueError("PAG worker exited; inspect pag-worker.log")
        result = json.loads(line)
        if "error" in result:
            raise ValueError(result["error"])
        return result

    def adapt(self, row):
        if self.cfg["method"] != "pag":
            return [row["prompt"]] * self.cfg["k"]
        if self.worker is None:
            a = self.assets["assets"]
            env = os.environ.copy()
            env["PYTHONPATH"] = a["pag_runtime"]["path"]
            env["HF_HUB_OFFLINE"] = "1"
            env["TRANSFORMERS_OFFLINE"] = "1"
            self.log = (self.run / "pag-worker.log").open("a")
            command = [
                sys.executable,
                str(Path(__file__).with_name("pag_worker.py")),
                "--checkpoint",
                a["pag"]["path"],
                "--tokenizer",
                a["tokenizer"]["path"],
                "--device",
                self.cfg["device"],
            ]
            self.worker = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.log,
                text=True,
                env=env,
                bufsize=1,
            )
            if self.read_worker().get("ready") is not True:
                raise ValueError("PAG worker not ready")
        self.worker.stdin.write(
            json.dumps(
                {
                    "id": row["id"],
                    "prompt": row["prompt"],
                    "k": self.cfg["k"],
                    "config": self.cfg["pag"],
                }
            )
            + "\n"
        )
        self.worker.stdin.flush()
        result = self.read_worker()
        if result["id"] != row["id"]:
            raise ValueError("PAG response identity mismatch")
        return result["prompts"]

    def load(self):
        if self.t2i_model is None:
            if self.cfg["renderer"].startswith("sd35"):
                from .sd3 import SD35

                self.t2i_model = SD35(self.assets, self.cfg)
            else:
                from .native_pool import NativeT2IModel

                self.t2i_model = NativeT2IModel(self.assets, self.cfg)

    def render(self, *args):
        self.load()
        return self.t2i_model.render(*args)

    def group(self, row):
        self.load()
        return self.t2i_model.group(row)

    def close(self):
        if self.worker is not None:
            self.worker.stdin.close()
            try:
                self.worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.worker.kill()
                self.worker.wait()
            self.worker.stdout.close()
        if self.log is not None:
            self.log.close()
