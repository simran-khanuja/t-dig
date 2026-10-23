"""Workaround for Transformers 5.3.0 passing `fix_mistral_regex` twice, and tokenizer checks.

TPIPS disables the Mistral regex rewrite for its Qwen tokenizer, and Transformers
5.3.0 forwards that keyword twice. `qwen_tokenizer_compat` strips it at
construction and disables the rewrite while the model loads, then restores both
methods. The installed environment is never edited.
"""

from contextlib import contextmanager
import importlib.metadata


@contextmanager
def qwen_tokenizer_compat():
    if importlib.metadata.version('transformers') != '5.3.0':
        yield
        return
    from transformers.tokenization_utils_tokenizers import TokenizersBackend

    original_init = TokenizersBackend.__init__
    original_patch = TokenizersBackend.__dict__['_patch_mistral_regex']

    def init(self, *args, **kwargs):
        flag = kwargs.pop('fix_mistral_regex', None)
        if flag not in (False, None):
            raise ValueError('TPIPS probe forbids modifying Qwen tokenizer regex')
        return original_init(self, *args, **kwargs)

    def unchanged(cls, tokenizer, *args, **kwargs):
        return tokenizer

    TokenizersBackend.__init__ = init
    TokenizersBackend._patch_mistral_regex = classmethod(unchanged)
    try:
        yield
    finally:
        TokenizersBackend.__init__ = original_init
        TokenizersBackend._patch_mistral_regex = original_patch


# The TPIPS checkpoint of the paper, as repo@revision for baselines.assets.local_checkpoint.
TPIPS_CHECKPOINT = 'sywang/TPIPS-Embed-Qwen3VL-8B@86a1493c60121e656a1c075bc7aed88f25fcd4d4'


def verify_tokenizer(model, checkpoint, texts):
    from pathlib import Path
    from tokenizers import Tokenizer

    raw = Tokenizer.from_file(str(Path(checkpoint) / 'tokenizer.json'))
    for text in texts:
        expected = raw.encode(text, add_special_tokens=False).ids
        actual = model.processor.tokenizer.encode(text, add_special_tokens=False)
        if actual != expected:
            raise ValueError('Loaded tokenizer differs from the checkpoint tokenizer')
    return dict(
        verified_texts=len(texts),
        exact_token_id_match=True,
        method='compare to raw tokenizers.Tokenizer from checkpoint tokenizer.json',
    )
