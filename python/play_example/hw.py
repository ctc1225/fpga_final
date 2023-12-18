from operation import rotation, proj, kv, split_head, add_state, sq
import numpy as np
import torch

H     = 2
D_e   = 8
D_v   = 16
D_k   = int(D_e/H)
D_h   = int(D_v/H)
NUM_OF_TOKEN = 2
torch.manual_seed(1999)

q_weight_file_path = "../torchscale/weight/q_weight"
k_weight_file_path = "../torchscale/weight/k_weight"
v_weight_file_path = "../torchscale/weight/v_weight"
Q_file_path = "data/Q.txt"
K_file_path = "data/K.txt"
V_file_path = "data/V.txt"
w_Q = torch.from_numpy(np.loadtxt(q_weight_file_path).astype(np.float32))
w_K = torch.from_numpy(np.loadtxt(k_weight_file_path).astype(np.float32))
w_V = torch.from_numpy(np.loadtxt(v_weight_file_path).astype(np.float32))

def write_1darray(file_path, arr):
  c = arr.shape
  with open(f'{file_path}', "w") as file: # for hardware use
    for i in range(c):
      file.write(str(arr[i]))
      if i != c-1: file.write("\n")

def write_2darray(file_path, arr):
  (r, c) = arr.shape
  with open(f'{file_path}', "w") as file: # for hardware use
    for i in range(r):
      file.write("{ ")
      for j in range(c):
        file.write(str(arr[i][j]))
        if j != c-1: file.write(", ")
      file.write(" },\n") if i != r-1 else file.write("}\n")

def forward(X, sin, cos, prev_state, decay):
  # print(f"x: {X}")
  # print(f'prev state: {prev_state}')

  K = proj(X, w_K)
  Q = proj(X, w_Q)
  V = proj(X, w_V)
  # write_2darray(K_file_path, K)
  # write_2darray(Q_file_path, Q)
  # write_2darray(V_file_path, V)

  # print(f"wk: {w_K}")
  print(f"K: {K}")
  # print(f"projected Q: {Q}")

  K = split_head(K, H)
  Q = split_head(Q, H)
  V = split_head(V, H)

  # print("q: ", Q)
  # print(f"sin: {sin}")
  # print(f"cos: {cos}")
  K = rotation(K, sin, cos)
  Q = rotation(Q, sin, cos)
  # print("rotate k: ", K)

  previous_kv = prev_state["prev_kv"]
  # previous_scale = prev_state["scale"]

  k_v = torch.rand(H, D_h, D_k)
  state = torch.ones(H, D_h, D_k)
  out = torch.ones(H, 1, D_h)
  
  # print("k: ", K)
  # print("v: ", V)
  for h in range(H):
    k_v[h] = kv(K[h], V[h])
    state[h] = add_state(k_v[h], previous_kv[h], decay[h])
    out[h] = sq(state[h], Q[h])
  # print("kv: ", k_v)
  # print(f"state: {state}")
  # print(f"state: {state}")
  # print(f"Q: {Q}")
  # update state
  prev_state["prev_kv"] = state
  print(f'output: {out}')
  # print(f'state: {state}')
  return out

def full_op():
  state = {}
  state["scale"] = 0
  state["prev_kv"] = torch.zeros(H, D_h, D_k)
  tokens = torch.tensor([
    [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8],
    [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
  ])
  sin = torch.tensor([ 
    [-8.4147e-01, -8.4147e-01, -1.0000e-04, -1.0000e-04],
    [0, 0, 0, 0]
  ])
  cos = torch.tensor([
    [0.5403, 0.5403, 1.0000, 1.0000],
    [1, 1, 1, 1]  
  ])
  decay = torch.tensor([
    [0.9688, 0.9844],
    [0.9688, 0.9844]
  ])

  output = torch.randn(NUM_OF_TOKEN, H, 1, D_h)
  for t in range(NUM_OF_TOKEN):
    print(f"forwarding {t} token")
    output[t] = forward(tokens[t], sin[t], cos[t], state, decay[t])
  # print(f'output: {output}')
    

def test_rotate():
  a = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
  sin = 0.6
  cos = 0.8
  rotate_a = rotation(a, sin, cos)
  print("a: ", a)
  print("rotate a: ", rotate_a)

def test_kv():
  k = torch.tensor([[0.4, 0.3, 0.2, 0.1]])
  v = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]])
  k_v = kv(k, v)
  print("k: ", k)
  print("v: ", v)
  print("kv: ", kv)

def test_add_state():
  prev_state = torch.tensor([[1, 2, 3], [2, 4, 6]])/10
  kv = torch.tensor([[1, 2, 3], [2, 4, 6]])/10
  print("prev_state: ", prev_state)
  print("kv: ", kv)
  print("result: ", add_state(kv, prev_state, gamma))

if __name__ == "__main__":
  full_op()
  # test_rotate()