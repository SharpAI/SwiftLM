#!/usr/bin/env python3
"""Generate tiny synthetic checkpoints that exercise the real model-load path.

Issue #128 observed that every defect in the #108/#110/#112 cycle was caught either
by loading a real checkpoint or by code review, and none by the unit suite — because
the bugs lived in weight-and-config-shape assumptions that only a checkpoint on disk
exercises. The obvious fix, running CI against real models, runs into arithmetic:
gemma-4-e2b is 3.6 GB and GitHub allows 10 GB of cache for the whole repository.

llama.cpp solved the same problem by publishing purpose-built tiny models
(`ggml-org/test-model-stories260K`, 1.2 MB) rather than shrinking real ones. Their
files are GGUF and unusable here, but the technique transfers: a checkpoint with the
same *shape* as a real one — same config fields, same weight keys, random values,
a ~300-token vocabulary instead of 150k — runs the same loading code and weighs a
few hundred kilobytes.

What these fixtures test is the plumbing: config parsing, weight-key resolution,
sanitisation, layer materialisation, quantisation metadata. They say nothing about
whether the arithmetic is correct, because the weights are noise. Real checkpoints
remain the only way to judge output quality.

Usage:  python3 scripts/make-test-fixtures.py [output-dir]
"""
import json
import os
import sys

import numpy as np
from safetensors.numpy import save_file
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors

ROOT = sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures"

VOCAB = 288
SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<pad>"]
CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
    "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def write_tokenizer(out):
    """A byte-level BPE, which is what Qwen/GPT-2 ship.

    Not a WordLevel model: swift-transformers rejects those with "BPETokenizer
    requires merges". Merges must also be spelled in the byte-level alphabet — "Ġ"
    for a space rather than a raw 0x20 — or the tokenizers library refuses to build.
    """
    vocab = {t: i for i, t in enumerate(SPECIALS)}
    for ch in sorted(pre_tokenizers.ByteLevel.alphabet()):
        vocab[ch] = len(vocab)
    merges = [("h", "e"), ("l", "l"), ("he", "ll"), ("t", "e")]
    for a, b in merges:
        vocab.setdefault(a + b, len(vocab))
    while len(vocab) < VOCAB:
        vocab[f"<|unused{len(vocab)}|>"] = len(vocab)

    tok = Tokenizer(models.BPE(vocab=vocab, merges=merges, unk_token=None, fuse_unk=False))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.ByteLevel(trim_offsets=False)
    tok.add_special_tokens(SPECIALS)
    tok.save(os.path.join(out, "tokenizer.json"))

    json.dump(
        {
            "tokenizer_class": "PreTrainedTokenizerFast",
            "bos_token": "<|endoftext|>",
            "eos_token": "<|im_end|>",
            "pad_token": "<pad>",
            "unk_token": "<|endoftext|>",
            "chat_template": CHAT_TEMPLATE,
        },
        open(os.path.join(out, "tokenizer_config.json"), "w"),
        indent=2,
    )


def rand(rng, *shape):
    return (rng.standard_normal(shape) * 0.02).astype(np.float16)


def ones(n):
    return np.ones(n, np.float16)


def build_dense(out, stray_shard=False):
    """A plain Qwen2. With stray_shard, an extra .safetensors sits beside the real one
    without appearing in the weight index — the shape behind #118, where files outside
    the index were being loaded and their keys rejected."""
    H, L, HEADS, KVH, INTER = 64, 2, 4, 2, 128
    HD = H // HEADS
    rng = np.random.default_rng(0)

    json.dump(
        {
            "model_type": "qwen2",
            "architectures": ["Qwen2ForCausalLM"],
            "vocab_size": VOCAB,
            "hidden_size": H,
            "intermediate_size": INTER,
            "num_hidden_layers": L,
            "num_attention_heads": HEADS,
            "num_key_value_heads": KVH,
            "max_position_embeddings": 512,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "tie_word_embeddings": False,
        },
        open(os.path.join(out, "config.json"), "w"),
        indent=2,
    )

    w = {
        "model.embed_tokens.weight": rand(rng, VOCAB, H),
        "model.norm.weight": ones(H),
        "lm_head.weight": rand(rng, VOCAB, H),
    }
    for i in range(L):
        p = f"model.layers.{i}"
        w[f"{p}.self_attn.q_proj.weight"] = rand(rng, HEADS * HD, H)
        w[f"{p}.self_attn.q_proj.bias"] = rand(rng, HEADS * HD)
        w[f"{p}.self_attn.k_proj.weight"] = rand(rng, KVH * HD, H)
        w[f"{p}.self_attn.k_proj.bias"] = rand(rng, KVH * HD)
        w[f"{p}.self_attn.v_proj.weight"] = rand(rng, KVH * HD, H)
        w[f"{p}.self_attn.v_proj.bias"] = rand(rng, KVH * HD)
        w[f"{p}.self_attn.o_proj.weight"] = rand(rng, H, HEADS * HD)
        w[f"{p}.mlp.gate_proj.weight"] = rand(rng, INTER, H)
        w[f"{p}.mlp.up_proj.weight"] = rand(rng, INTER, H)
        w[f"{p}.mlp.down_proj.weight"] = rand(rng, H, INTER)
        w[f"{p}.input_layernorm.weight"] = ones(H)
        w[f"{p}.post_attention_layernorm.weight"] = ones(H)

    save_file(w, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})

    if stray_shard:
        # Deliberately absent from weight_map. A loader that globs *.safetensors instead
        # of reading the index picks this up and fails on the unknown key.
        save_file(
            {"not_a_real_module.weight": rand(rng, 8, 8)},
            os.path.join(out, "extra-not-in-index.safetensors"),
            metadata={"format": "pt"},
        )
        json.dump(
            {
                "metadata": {"total_size": 0},
                "weight_map": {k: "model.safetensors" for k in w},
            },
            open(os.path.join(out, "model.safetensors.index.json"), "w"),
        )
    return len(w)


def build_gemma4_kv_shared(out, vestigial):
    """Gemma 4 text with KV-shared layers.

    Two real checkpoints disagree about what a shared layer ships: gemma-4-e4b omits
    its k/v projections, gemma-4-e2b includes them anyway. #120 was the first case
    failing to load; the b674 regression was the second, after a fix that assumed the
    first was universal. `vestigial=True` is the e2b shape.

    Shapes mirror a real gemma-4-e2b checkpoint rather than being guessed.
    """
    H, L, SHARED, HEADS, KVH, HD = 64, 4, 2, 4, 2, 16
    INTER, PLI, VPLI = 128, 32, 16
    rng = np.random.default_rng(0)

    json.dump(
        {
            "model_type": "gemma4_text",
            "architectures": ["Gemma4ForCausalLM"],
            "hidden_size": H,
            "num_hidden_layers": L,
            "intermediate_size": INTER,
            "num_attention_heads": HEADS,
            "head_dim": HD,
            "global_head_dim": HD,
            "rms_norm_eps": 1e-6,
            "vocab_size": VOCAB,
            "num_key_value_heads": KVH,
            "rope_traditional": False,
            "rope_theta": 10000.0,
            "sliding_window": 128,
            "sliding_window_pattern": 1,
            "max_position_embeddings": 512,
            "num_kv_shared_layers": SHARED,
            "use_double_wide_mlp": False,
            "tie_word_embeddings": True,
            "hidden_size_per_layer_input": PLI,
            "vocab_size_per_layer_input": VPLI,
            "final_logit_softcapping": 30.0,
            "enable_moe_block": False,
            "attention_k_eq_v": False,
        },
        open(os.path.join(out, "config.json"), "w"),
        indent=2,
    )

    w = {
        "model.embed_tokens.weight": rand(rng, VOCAB, H),
        "model.norm.weight": ones(H),
        "model.embed_tokens_per_layer.weight": rand(rng, VPLI, L * PLI),
        "model.per_layer_model_projection.weight": rand(rng, L * PLI, H),
        "model.per_layer_projection_norm.weight": ones(PLI),
    }
    boundary = L - SHARED
    for i in range(L):
        p = f"model.layers.{i}"
        w[f"{p}.self_attn.q_proj.weight"] = rand(rng, HEADS * HD, H)
        w[f"{p}.self_attn.o_proj.weight"] = rand(rng, H, HEADS * HD)
        w[f"{p}.self_attn.q_norm.weight"] = ones(HD)
        w[f"{p}.layer_scalar"] = np.ones(1, np.float16)
        if i < boundary or vestigial:
            w[f"{p}.self_attn.k_proj.weight"] = rand(rng, KVH * HD, H)
            w[f"{p}.self_attn.v_proj.weight"] = rand(rng, KVH * HD, H)
            w[f"{p}.self_attn.k_norm.weight"] = ones(HD)
        w[f"{p}.mlp.gate_proj.weight"] = rand(rng, INTER, H)
        w[f"{p}.mlp.up_proj.weight"] = rand(rng, INTER, H)
        w[f"{p}.mlp.down_proj.weight"] = rand(rng, H, INTER)
        w[f"{p}.input_layernorm.weight"] = ones(H)
        w[f"{p}.post_attention_layernorm.weight"] = ones(H)
        w[f"{p}.pre_feedforward_layernorm.weight"] = ones(H)
        w[f"{p}.post_feedforward_layernorm.weight"] = ones(H)
        w[f"{p}.per_layer_input_gate.weight"] = rand(rng, PLI, H)
        w[f"{p}.per_layer_projection.weight"] = rand(rng, H, PLI)
        w[f"{p}.post_per_layer_input_norm.weight"] = ones(H)

    save_file(w, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
    return len(w)


FIXTURES = {
    "dense": (build_dense, {}),
    "stray-shard": (build_dense, {"stray_shard": True}),
    "kv-shared-absent": (build_gemma4_kv_shared, {"vestigial": False}),
    "kv-shared-present": (build_gemma4_kv_shared, {"vestigial": True}),
}

if __name__ == "__main__":
    os.makedirs(ROOT, exist_ok=True)
    total = 0
    for name, (fn, kwargs) in FIXTURES.items():
        out = os.path.join(ROOT, name)
        os.makedirs(out, exist_ok=True)
        n = fn(out, **kwargs)
        write_tokenizer(out)
        size = sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out))
        total += size
        print(f"  {name:<20} {n:>3} tensors  {size/1024:>7.1f} KB")
    print(f"  {'total':<20} {'':>3}          {total/1024:>7.1f} KB")
