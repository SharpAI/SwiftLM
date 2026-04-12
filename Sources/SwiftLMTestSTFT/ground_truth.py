"""
Complete Layer 0 forward pass in Python (Attention + MLP + Residuals + Layer Scalar).
"""
import mlx.core as mx
import mlx.nn as nn
import json

model_dir = '/Users/simba/.cache/huggingface/hub/models--mlx-community--gemma-4-e4b-it-4bit/snapshots/62b0e4e2d06c2f3baeeb0f8b7b18d7308c7786fc'

with open(f'{model_dir}/config.json') as f:
    config = json.load(f)
tc = config['text_config']

w = mx.load(f'{model_dir}/model.safetensors')
prefix = "language_model.model."

def gelu_approx(x):
    return 0.5 * x * (1 + mx.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * mx.power(x, 3))))

import math

# Embed BOS
embed = nn.QuantizedEmbedding(tc['vocab_size'], tc['hidden_size'], 64, 4)
embed.load_weights([
    ("weight", w[f"{prefix}embed_tokens.weight"]),
    ("scales", w[f"{prefix}embed_tokens.scales"]),
    ("biases", w[f"{prefix}embed_tokens.biases"]),
])
inputs = mx.array([[2]])
h_init = embed(inputs) * (tc['hidden_size'] ** 0.5)

# --- Layer 0 Start ---
# 1. Attn branch
ln1_w = w[f"{prefix}layers.0.input_layernorm.weight"]
h_normed = mx.fast.rms_norm(h_init, ln1_w, tc.get('rms_norm_eps', 1e-6))

# QKV Projections
q_w, q_s, q_b = w[f"{prefix}layers.0.self_attn.q_proj.weight"], w[f"{prefix}layers.0.self_attn.q_proj.scales"], w[f"{prefix}layers.0.self_attn.q_proj.biases"]
k_w, k_s, k_b = w[f"{prefix}layers.0.self_attn.k_proj.weight"], w[f"{prefix}layers.0.self_attn.k_proj.scales"], w[f"{prefix}layers.0.self_attn.k_proj.biases"]
v_w, v_s, v_b = w[f"{prefix}layers.0.self_attn.v_proj.weight"], w[f"{prefix}layers.0.self_attn.v_proj.scales"], w[f"{prefix}layers.0.self_attn.v_proj.biases"]

queries = mx.quantized_matmul(h_normed, q_w, scales=q_s, biases=q_b, transpose=True, group_size=64, bits=4)
keys = mx.quantized_matmul(h_normed, k_w, scales=k_s, biases=k_b, transpose=True, group_size=64, bits=4)
values = mx.quantized_matmul(h_normed, v_w, scales=v_s, biases=v_b, transpose=True, group_size=64, bits=4)

# Reshape
B, L = 1, 1
n_heads, n_kv_heads, head_dim = tc['num_attention_heads'], tc.get('num_key_value_heads', 8), tc['head_dim']
queries = queries.reshape(B, L, n_heads, head_dim)
keys = keys.reshape(B, L, n_kv_heads, head_dim)
values = values.reshape(B, L, n_kv_heads, head_dim)

# Internal Norms
q_norm_w = w[f"{prefix}layers.0.self_attn.q_norm.weight"]
k_norm_w = w[f"{prefix}layers.0.self_attn.k_norm.weight"]
queries = mx.fast.rms_norm(queries, q_norm_w, tc.get('rms_norm_eps', 1e-6))
keys = mx.fast.rms_norm(keys, k_norm_w, tc.get('rms_norm_eps', 1e-6))

v_f32 = values.astype(mx.float32)
v_rms = mx.sqrt(mx.mean(v_f32 * v_f32, axis=-1, keepdims=True) + tc.get('rms_norm_eps', 1e-6))
values = (v_f32 / v_rms).astype(values.dtype)

# Attention Output
scale = head_dim ** -0.5
attn_out_raw = mx.fast.scaled_dot_product_attention(queries.transpose(0, 2, 1, 3), keys.transpose(0, 2, 1, 3), values.transpose(0, 2, 1, 3), scale=scale)
attn_out_raw = attn_out_raw.transpose(0, 2, 1, 3).reshape(B, L, -1)

# O Proj
o_w, o_s, o_b = w[f"{prefix}layers.0.self_attn.o_proj.weight"], w[f"{prefix}layers.0.self_attn.o_proj.scales"], w[f"{prefix}layers.0.self_attn.o_proj.biases"]
attn_out = mx.quantized_matmul(attn_out_raw, o_w, scales=o_s, biases=o_b, transpose=True, group_size=64, bits=4)

# Post Attention Norm
post_attn_w = w[f"{prefix}layers.0.post_attention_layernorm.weight"]
attn_res = mx.fast.rms_norm(attn_out, post_attn_w, tc.get('rms_norm_eps', 1e-6))

# 2. MLP branch
# Update for MLP: h_mlp_in = h_init + attn_res
# Is this correct for Gemma 4? Let's assume standard residual.
# Actually, the Swift code does: preMLPNorm = preFeedforwardLayerNorm(x + attnNorm)
pre_ff_w = w[f"{prefix}layers.0.pre_feedforward_layernorm.weight"]
mlp_in_normed = mx.fast.rms_norm(h_init + attn_res, pre_ff_w, tc.get('rms_norm_eps', 1e-6))

# MLP Projections
gp_w, gp_s, gp_b = w[f"{prefix}layers.0.mlp.gate_proj.weight"], w[f"{prefix}layers.0.mlp.gate_proj.scales"], w[f"{prefix}layers.0.mlp.gate_proj.biases"]
up_w, up_s, up_b = w[f"{prefix}layers.0.mlp.up_proj.weight"], w[f"{prefix}layers.0.mlp.up_proj.scales"], w[f"{prefix}layers.0.mlp.up_proj.biases"]
dp_w, dp_s, dp_b = w[f"{prefix}layers.0.mlp.down_proj.weight"], w[f"{prefix}layers.0.mlp.down_proj.scales"], w[f"{prefix}layers.0.mlp.down_proj.biases"]

gate = mx.quantized_matmul(mlp_in_normed, gp_w, scales=gp_s, biases=gp_b, transpose=True, group_size=64, bits=4)
up = mx.quantized_matmul(mlp_in_normed, up_w, scales=up_s, biases=up_b, transpose=True, group_size=64, bits=4)
# Use geluApproximate
activated = (0.5 * gate * (1 + mx.tanh(math.sqrt(2 / math.pi) * (gate + 0.044715 * mx.power(gate, 3))))) * up
mlp_out = mx.quantized_matmul(activated, dp_w, scales=dp_s, biases=dp_b, transpose=True, group_size=64, bits=4)

# Post MLP Norm
post_ff_w = w[f"{prefix}layers.0.post_feedforward_layernorm.weight"]
mlp_res = mx.fast.rms_norm(mlp_out, post_ff_w, tc.get('rms_norm_eps', 1e-6))

# 3. Final Residual + Layer Scalar
# Swift: residualUpdates = attn_res + mlp_res
# Swift: return (h_init + residualUpdates) * ls
ls = w[f"{prefix}layers.0.layer_scalar"]
h_final = (h_init + attn_res + mlp_res) * ls
mx.eval(h_final)

print(f"BOS h_init norm: {mx.sqrt(mx.sum(h_init*h_init)).item():.6f}")
print(f"Attn res norm: {mx.sqrt(mx.sum(attn_res*attn_res)).item():.6f}")
print(f"MLP res norm: {mx.sqrt(mx.sum(mlp_res*mlp_res)).item():.6f}")
print(f"Layer scalar: {ls.item():.6f}")
print(f"Final Layer 0 output norm: {mx.sqrt(mx.sum(h_final*h_final)).item():.6f}")
print(f"First 5 of h_final: {[h_final[0,0,i].item() for i in range(5)]}")
