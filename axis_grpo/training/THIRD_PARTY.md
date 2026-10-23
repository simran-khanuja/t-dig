# Upstream training dependencies

`trainer.py` adapts the TransferQueue submission and runner lifecycle from [verl](https://github.com/verl-project/verl), `trainer/ppo/v1/trainer_base.py` and `trainer/main_ppo.py` (Copyright 2024 Bytedance Ltd. and/or its affiliates), under the Apache License 2.0. A copy is in [third_party/verl-LICENSE](third_party/verl-LICENSE). The exact inspected framework files are hashed in `runtime.lock.json`. `scripts/patch_verl.py` adds one guard (`and not self.ref_in_actor`) to the installed `trainer/ppo/v1/trainer_base.py`, which LoRA training with the KL loss needs; the lock holds the hash of the patched file.

Model optimization, GRPO advantage estimation, FSDP2, and vLLM rollout use the installed verl implementation. They are not reimplemented in this repository.

The T2I model calls installed Diffusers pipelines through this repository's native T2I model adapter. TPIPS calls the installed TPIPS embedding model. VQAScore imports and calls the installed official CLIP-FlanT5 backend; its image preprocessing, question template, and probability computation are retained. These packages and their model weights remain subject to their respective licenses. No model weights are copied into the source package.
