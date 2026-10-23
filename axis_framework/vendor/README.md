# vendor/

`qwen_vl_utils` (0.0.14, Apache-2.0, https://github.com/QwenLM/Qwen2-VL/tree/main/qwen-vl-utils), copied verbatim; its license is in `qwen_vl_utils/LICENSE`.

WHY IT IS HERE AND NOT PIP-INSTALLED. `servers/suggest_server.py` needs `qwen_vl_utils.process_vision_info`, and the reward environment -- the env that runs the suggest server, because that is where `Qwen3VLForConditionalGeneration` and the T2I model's transformers live -- does not have it. The other two envs do. Pip-ing into these shared envs has broken things here before (README section 1), so the package is copied instead and reaches the suggest server through `PYTHONPATH`, set for that ONE process in `scripts/launch_pool.sh`.

Putting the *whole* of the training environment's site-packages on `PYTHONPATH` would have been the one-line version and is wrong: `PYTHONPATH` precedes site-packages on `sys.path`, so it would shadow the reward environment's torch and transformers with the vllm env's different versions.

It is pure Python and needs only requests / torch / torchvision / packaging / PIL / numpy, all of which the reward environment already has. `av` is a declared dependency of the distribution but is imported lazily on the video path, which the suggest server never takes.
