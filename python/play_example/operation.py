import numpy as np
import torch
def rotate_every_two(x):
  x1 = x[:, ::2]
  x2 = x[:, 1::2]
  x = torch.stack((-x2, x1), dim=-1)
  return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')\


def theta_shift(x, sin, cos):
  return (x * cos) + (rotate_every_two(x) * sin)

def rotation(a, sin, cos):
  # sin = 0.6
  # cos = 0.8
  # print(f"a: {a}")
  # print(f"sin: {sin}")
  # print(f"cos: {cos}")
  rotate_a = theta_shift(a, sin, cos)
  # print(f"rotate a: ", rotate_a)
  # print("a: ", a)
  # print("rotated a: ", rotate_a)
  quant_a = torch.quantize_per_tensor(a, scale=1/32, zero_point=0, dtype=torch.qint8)
  # quant_a = torch.quantize_per_tensor(a)
  q_a = quant_a.int_repr().numpy()

  quant_rotate_a = torch.quantize_per_tensor(rotate_a, scale=1/32, zero_point=0, dtype=torch.qint8)
  # quant_rotate_a = torch.quantize_per_tensor(rotate_a)
  q_rotate_a = quant_rotate_a.int_repr().numpy()

  # print("a: ", a)
  # print("quant a: ", quant_a)
  # print("= ", np.vectorize(np.binary_repr)(q_a, 16))
  # print("rotate a: ", rotate_a)
  # print("quant rotate a: ", quant_rotate_a)
  # print("= ", np.vectorize(np.binary_repr)(q_rotate_a, 16))
  # print("quant rotated a: ", bit_q_rotate_a)
  return rotate_a

def proj(mat, w):
  return torch.matmul(mat, w)

def kv(k, v):
  # k = torch.tensor([[0.4, 0.3, 0.2, 0.1]])
  # v = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]])
  v = v.reshape(-1, 1)
  kv = k * v

  # print(kv)
  return kv

def split_head(a, h):
  '''
    Split head for matrix a
  '''
  # a = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
  # h = 2
  dim_per_head = int(a.shape[-1]/h)
  # print(a)
  # print(a.reshape((-1, dim_per_head)))
  return a.reshape((-1, dim_per_head))

def add_state(kv, prev_s, gamma):
  # print(f"s shape: {prev_s.shape}")
  # print(f"gamma shame: {gamma.shape}")
  return kv+(prev_s*gamma)

def sq(s, q):
  """
    Multiply state and Q
  """
  return torch.matmul(s, q)