"""Persistent PAG decoder, run only in the isolated Transformers 4.46.3 overlay."""

import argparse
import json
import sys


def main():
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    if transformers.__version__ != "4.46.3":
        raise ValueError("PAG requires isolated Transformers 4.46.3")
    model = (
        AutoModelForCausalLM.from_pretrained(args.checkpoint, local_files_only=True)
        .eval()
        .to(args.device)
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print(
        json.dumps({"ready": True, "transformers": transformers.__version__}),
        flush=True,
    )
    for line in sys.stdin:
        try:
            req = json.loads(line)
            k = req["k"]
            cfg = req["config"]
            text = req["prompt"].rstrip() + " Rephrase:\n"  # Exact author dataset.py template.
            encoded = tokenizer(text, return_tensors="pt", truncation=False).to(args.device)
            n = encoded.input_ids.shape[1]
            if n + cfg["max_new_tokens"] > model.config.n_positions:
                raise ValueError("PAG prompt exceeds context")
            kwargs = dict(
                do_sample=False,
                min_new_tokens=cfg["min_new_tokens"],
                max_new_tokens=cfg["max_new_tokens"],
                num_beams=k,
                num_return_sequences=k,
                length_penalty=cfg["length_penalty"],
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            if k > 1:
                kwargs.update(num_beam_groups=k, diversity_penalty=cfg["diversity_penalty"])
            with torch.inference_mode():
                out = model.generate(**encoded, **kwargs)
            prompts = tokenizer.batch_decode(out[:, n:], skip_special_tokens=True)
            if len(prompts) != k or any(not p.strip() for p in prompts):
                raise ValueError("Empty/missing PAG rewrite")
            print(json.dumps({"id": req["id"], "prompts": prompts}), flush=True)
        except Exception as error:
            print(json.dumps({"error": f"{type(error).__name__}: {error}"}), flush=True)


if __name__ == "__main__":
    main()
