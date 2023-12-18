from python.torchscale.retnet import RetNetRelPos, RMSNorm, get_activation_fn
from python.torchscale.config import RetNetConfig
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization import QConfigMapping, default_histogram_observer, default_weight_observer

import numpy as np
import torch
import copy
import time
import sys

torch.manual_seed(1999)

def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')\


def theta_shift(x, sin, cos):
    # print("x: ", x)
    # print("sin: ", sin)
    return (x * cos) + (rotate_every_two(x) * sin)

class MultiScaleRetention(nn.Module):

    def __init__(
        self,
        config,
        embed_dim,
        value_dim,
        num_heads,
        gate_fn="swish",
    ):
        super().__init__()
        self.config = config
        self.embed_dim = embed_dim
        self.value_dim = value_dim
        self.num_heads = num_heads 
        self.head_dim = self.value_dim // num_heads
        self.key_dim = self.embed_dim // num_heads
        # self.scaling = self.key_dim**-0.5
        self.scaling = 1

        self.gate_fn = get_activation_fn(activation=str(gate_fn))

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, value_dim, bias=False)
        # self.g_proj = nn.Linear(embed_dim, value_dim, bias=False)

        self.out_proj = nn.Linear(value_dim, embed_dim, bias=False)

        self.group_norm = RMSNorm(self.head_dim, eps=config.layernorm_eps, elementwise_affine=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=2**-2.5)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=2**-2.5)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=2**-2.5)
        # nn.init.xavier_uniform_(self.g_proj.weight, gain=2**-2.5)
        nn.init.xavier_uniform_(self.out_proj.weight)

    # def parallel_forward(self, qr, kr, v, mask):
    #     bsz, tgt_len, embed_dim = v.size()

    #     vr = v.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)

    #     qk_mat = qr @ kr.transpose(-1, -2)  # bsz * m * tgt_len * tgt_len
    #     qk_mat = qk_mat * mask
    #     # invariant after normalization
    #     qk_mat = qk_mat / qk_mat.detach().sum(dim=-1, keepdim=True).abs().clamp(min=1)
    #     output = torch.matmul(qk_mat, vr)
    #     output = output.transpose(1, 2)
    #     return output

    def recurrent_forward(self, qr, kr, v, decay, incremental_state):
        bsz = v.size(0)

        v = v.view(bsz, self.num_heads, self.head_dim, 1) # (bsz, # of head, head dim, 1)
        # print("v shape before matmul: ", v.shape)
        # kr shape: (bsz, # of head, # of token, key dim)
        # print("kr: ", kr)
        # print("v: ", v)
        kv = kr * v # equivalent to matmul(v, kr)
        # print("kv: ", kv)
        # kv shape: (bsz, # of head, head dim, key dim)
        # print("kv shape: ", kv.shape)

        # if "prev_key_value" in incremental_state:
        #     prev_kv = incremental_state["prev_key_value"]
        #     prev_scale = incremental_state["scale"]
        #     scale = prev_scale * decay + 1
        #     kv = prev_kv * (prev_scale.sqrt() * decay / scale.sqrt()).view(
        #         self.num_heads, 1, 1) + kv / scale.sqrt().view(self.num_heads, 1, 1)
        #     # kv = prev_kv * decay.view(self.num_heads, 1, 1) + kv
        # else:
        #     scale = torch.ones_like(decay)
        prev_kv = incremental_state["prev_key_value"]
        prev_scale = incremental_state["scale"]
        scale = prev_scale * decay + 1
        # kv = prev_kv * (prev_scale.sqrt() * decay / scale.sqrt()).view(
        #     self.num_heads, 1, 1) + kv / scale.sqrt().view(self.num_heads, 1, 1)
        # print(f"decay: {decay}")
        # print(f"kv shape: ", prev_kv.shape)
        kv = prev_kv * decay.view(self.num_heads, 1, 1) + kv
        # print("kv: " , kv)

        incremental_state["prev_key_value"] = kv
        incremental_state["scale"] = scale
        # print(f"state: {kv}")
        
        # print("qr shape: ", qr.shape)
        # print("kv shape: ", kv.shape)
        # print("(qr*kv) shape before sum: ", (qr*kv).shape)
        output = torch.sum(qr * kv, dim=3) # equivalent to matmul(kv, qrT)
        # print(f"output dim: {output.shape}")
        # print(f"output: {output}")
        # print("(qr*kv) shape after sum: ", output.shape)
        return output

    # def chunk_recurrent_forward(self, qr, kr, v, inner_mask):
    #     mask, cross_decay, query_inner_decay, value_inner_decay = inner_mask
    #     bsz, tgt_len, embed_dim = v.size()
    #     chunk_len = mask.size(1)
    #     num_chunks = tgt_len // chunk_len

    #     assert tgt_len % chunk_len == 0

    #     qr = qr.view(bsz, self.num_heads, num_chunks, chunk_len, self.key_dim).transpose(1, 2)
    #     kr = kr.view(bsz, self.num_heads, num_chunks, chunk_len, self.key_dim).transpose(1, 2)
    #     v = v.view(bsz, num_chunks, chunk_len, self.num_heads, self.head_dim).transpose(2, 3)

    #     kr_t = kr.transpose(-1, -2)

    #     qk_mat = qr @ kr_t  # bsz * num_heads * chunk_len * chunk_len
    #     qk_mat = qk_mat * mask
    #     inner_scale = qk_mat.detach().abs().sum(dim=-1, keepdim=True).clamp(min=1)
    #     qk_mat = qk_mat / inner_scale
    #     inner_output = torch.matmul(qk_mat,
    #                                 v)  # bsz * num_heads * num_value_heads * chunk_len * head_dim

    #     # reduce kv in one chunk
    #     kv = kr_t @ (v * value_inner_decay)

    #     kv_recurrent = []
    #     cross_scale = []
    #     kv_state = torch.zeros(bsz, self.num_heads, self.key_dim, self.head_dim).to(v)
    #     kv_scale = torch.ones(bsz, self.num_heads, 1, 1).to(v)

    #     # accumulate kv by loop
    #     for i in range(num_chunks):
    #         kv_recurrent.append(kv_state / kv_scale)
    #         cross_scale.append(kv_scale)
    #         kv_state = kv_state * cross_decay + kv[:, i]
    #         kv_scale = kv_state.detach().abs().sum(dim=-2, keepdim=True).max(
    #             dim=-1, keepdim=True).values.clamp(min=1)

    #     kv_recurrent = torch.stack(kv_recurrent, dim=1)
    #     cross_scale = torch.stack(cross_scale, dim=1)

    #     all_scale = torch.maximum(inner_scale, cross_scale)
    #     align_inner_scale = all_scale / inner_scale
    #     align_cross_scale = all_scale / cross_scale

    #     cross_output = (qr * query_inner_decay) @ kv_recurrent
    #     output = inner_output / align_inner_scale + cross_output / align_cross_scale
    #     # output = inner_output / cross_scale + cross_output / inner_scale

    #     output = output.transpose(2, 3)
    #     return output

    def forward(self, x, rel_pos, chunkwise_recurrent=False, incremental_state=None):
        # print(f"previous state: {incremental_state}")
        bsz, tgt_len, _ = x.size()
        # print("x dim: ", x.size())
        (sin, cos), inner_mask = rel_pos

        # projection with weight matrix
        # print("x: ", x)
        # print("self q weight: ", self.q_proj.weight)
        # print("sin: ", sin)
        # print("cos: ", cos)
        q = self.q_proj(x) # (bsz, # of token, hidden dim)
        k = self.k_proj(x) # (bsz, # of token, hidden dim)
        v = self.v_proj(x) # (bsz, # of token, value dim)
        # g = self.g_proj(x) # (bsz, # of token, value dim)
        # print(f'q: {q}')
        # print("q, k shape: ", q.shape)
        # print("v, g shape: ", v.shape)

        # split head
        k *= self.scaling
        q = q.view(bsz, tgt_len, self.num_heads, self.key_dim).transpose(1, 2)
        k = k.view(bsz, tgt_len, self.num_heads, self.key_dim).transpose(1, 2)
        # print("q, k shape after head split: ", q.shape) # (bsz, # of heads, # of token, key dim)

        # rotary position encoding
        # print("k: ", k)
        qr = theta_shift(q, sin, cos) # shape does not change
        kr = theta_shift(k, sin, cos) # shape does not change
        # print("kr: ", kr)
        # print(f'v: {v}')
        # print("q, k shape after rotation: ", qr.shape)


        output = self.recurrent_forward(qr, kr, v, inner_mask, incremental_state)
        # if incremental_state is not None:
        # elif chunkwise_recurrent:
        #     output = self.chunk_recurrent_forward(qr, kr, v, inner_mask)
        # else:
        #     output = self.parallel_forward(qr, kr, v, inner_mask)
        
        # print("output size after recurrent forward: ", output.shape)
        output = self.group_norm(output).reshape(bsz, tgt_len, self.head_dim * self.num_heads)
        # print("output size after GN: ", output.shape)

        # output = self.gate_fn(g) * output
        # print("output size after gate fn: ", output.shape)

        output = self.out_proj(output)
        # print("output size after output projection: ", output.shape)

        return output

def write_weight(file_path, weight):
    weight = weight.detach().numpy()
    (r, c) = weight.shape
    with open(f'{file_path}.txt', "w") as file: # for hardware use
        for i in range(r):
            file.write("{ ")
            for j in range(c):
                file.write(str(weight[i][j]))
                # np.savetxt(file, weight[i][j], fmt='%f', newline="")
                if j != c-1: file.write(", ")
            # np.savetxt(file, weight[i].flatten(), fmt='%f', newline=", ")
            file.write(" },\n") if i != r-1 else file.write("}\n")
    np.savetxt(file_path, weight) # for load use

device = 'cuda:0'
H     = 2
D_e   = 128
D_v   = 128
D_k   = int(D_e/H)
D_h   = int(D_v/H)
NUM_OF_TOKEN = 2
NUM_OF_LAYER = 1
gamma = torch.tensor(1.0)
config = RetNetConfig(
    decoder_layers=1, 
    decoder_embed_dim=D_e, 
    decoder_value_embed_dim=D_v,
    decoder_retention_heads=H, 
    decoder_ffn_embed_dim=512,
)
layers = [MultiScaleRetention(config, D_e, D_v, H).to(device) for i in range(NUM_OF_LAYER)]
for l in layers:
    l.eval()
retnet_rel_pos = RetNetRelPos(config).to(device)

incremental_state = [
    {
        "prev_key_value": torch.zeros(1, H, D_h, D_k).to(device), 
        "scale": gamma
    }
    for i in range(NUM_OF_LAYER)
]

trainable_params = sum(
    p.numel() for p in layers[0].parameters() if p.requires_grad
)*NUM_OF_LAYER

if len(sys.argv) == 1: NUM_OF_TOKEN = 7500
else: NUM_OF_TOKEN = int(sys.argv[1])
input_tokens = torch.rand(NUM_OF_TOKEN, 1, 1, D_e).to(device) # (# of token, batch, # of token in parallel, embedding)

# for name, parameter in layers[0].named_parameters():
#     if not parameter.requires_grad:
#         continue
#     params = parameter.numel()
#     print(f'name: {name}, params: {params}')



# print(f"--- Running RET with D_e={D_e}, HEAD={H}, # of layers={NUM_OF_LAYER}, # of token={NUM_OF_TOKEN} ---")
# print(f"# of parameters: {trainable_params}")
start_time = time.time()
# retention_rel_pos = retnet_rel_pos(0, True, chunkwise_recurrent=False)
# ((sin, cos), mask) = retention_rel_pos
# rel_pos = ((sin.to(device), cos.to(device)), mask.to(device))
for i in range(NUM_OF_TOKEN):
    retention_rel_pos = retnet_rel_pos(i, True, chunkwise_recurrent=False)
    y = input_tokens[i]

    for l in range(NUM_OF_LAYER):
        y = layers[l](y, retention_rel_pos, incremental_state=incremental_state[l])
print("%s" % (time.time() - start_time))
print(f"{torch.cuda.memory_allocated()}")
#     # print(f"output y: {y}")
# print("execution time: %s seconds" % (time.time() - start_time))


# Pytorch FX
# retention.eval()
# qconfig = torch.ao.quantization.QConfig(
#     activation=default_histogram_observer,
#     weight=default_weight_observer.with_args(dtype=torch.qint8))

# qconfig_mapping = QConfigMapping().set_global(qconfig)
# model_to_quantize = copy.deepcopy(retention)
# prepared_model = prepare_fx(model_to_quantize, qconfig_mapping, example_inputs=input_ids)

# quantized_model = convert_fx(prepared_model)
# # print(quantized_model)

# # print(quantized_model.q_proj.weight().int_repr())
# print(quantized_model.q_proj.scale)
# quantized_model.q_proj.scale = 16
# print(quantized_model.q_proj.scale)
# print(quantized_model.q_proj.weight().int_repr())
# # quantized_model.eval()

# print("------------------")
# y = quantized_model(input, retention_rel_pos, incremental_state=incremental_state)



# parallel inference
# model.chunkwise_recurrent = False
# parallel_state, _ = model(input_ids, features_only=True)

# recurrent inference
# incremental_state = {}
# rnn_state = []
# for i in range(input_ids.shape[1]):
#     # print("input: ", input_ids[:, :i+1])
#     print("input size: ", input_ids[:, :i+1].shape)
#     rnn_h, _ = model(input_ids[:, :i+1], incremental_state=incremental_state, features_only=True)
#     rnn_state.append(rnn_h)
# rnn_state = torch.cat(rnn_state, dim=1)

# print("----------------------------------")
# print("printing quantized model")
# print(model.layers[0].retention.q_proj.weight())
# print(model.layers[0].retention.q_proj.weight().int_repr())

# chunkwise recurrent inference
# model.chunkwise_recurrent = True
# chunk_state, _ = model(input_ids, features_only=True)

# compare
# print('parallel vs recurrent', torch.allclose(parallel_state, rnn_state, atol=1e-5))
# print('parallel vs chunkwise', torch.allclose(parallel_state, chunk_state, atol=1e-5))
