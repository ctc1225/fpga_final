"""
@author : Hyunwoong
@when : 2019-10-25
@homepage : https://github.com/gusdnd852
"""
from torch import nn
import torch
import time
import sys
import numpy as np
import math

torch.manual_seed(1999)

class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # transpose
        # print(f'q: {q}')
        print(f'k: {k}')
        # print(f'scale: {math.sqrt(d_tensor)}')
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product
        print(f'QK^T scale: {score}')
        # print(f'score dim: {score.shape}')

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)
        print(f'score: {score}')
        # print(f'score dim after softmax: {score.shape}')

        # 4. multiply with Value
        # print(f'v dim: {v.shape}')
        v = score @ v
        # print(f'output dim: {v.shape}')

        return v, score


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_concat = nn.Linear(d_model, d_model, bias=False)

    def forward(self, q, k, v, mask=None):
        # 1. dot product with weight matrices
        # print(f'input x: {q}')
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        # print(f'weight k: {self.w_k.weight}')
        # print(f'k after projection: {k}')

        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v, mask=mask)

        # 4. concat and pass to linear layer
        out = self.concat(out)
        # print(f'output: {out}')
        out = self.w_concat(out)
        print(f'projected output: {out}')

        # 5. visualize attention map
        # TODO : we should implement visualization

        return out

    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor


if __name__ == '__main__':

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

    def write_token(file_path, tokens):
        l = tokens.shape[0]
        with open(f'{file_path}.txt', "w") as file:
            for i in range(l):
                file.write("%.5f" % (float(tokens[i])))
                file.write('\n')

    D_MODEL = 128
    HEAD = 2
    NUM_OF_LAYER = 1
    NUM_OF_TOKEN = 0
    if len(sys.argv) == 1: NUM_OF_TOKEN = 1
    else: NUM_OF_TOKEN = int(sys.argv[1])
    device = 'cuda' if torch.cuda.is_available() else 'cuda'
    model = MultiHeadAttention(D_MODEL, HEAD)
    x = torch.tensor([[[0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8]]])
    x = torch.rand(1, 1, D_MODEL)-0.5
    
    # trainable_params = sum(
    #     p.numel() for p in model.parameters() if p.requires_grad
    # )*NUM_OF_LAYER
    # print(f'params: {trainable_params}')

    export_weight = True
    if export_weight:
        q_weight_file_path = "./mha_weight/q_weight"
        k_weight_file_path = "./mha_weight/k_weight"
        v_weight_file_path = "./mha_weight/v_weight"
        out_weight_file_path = "./mha_weight/out_weight"
        token_file_path     = "./mha_weight/tokens"
        write_weight(q_weight_file_path, torch.transpose(model.w_q.weight, 0, 1))
        write_weight(k_weight_file_path, torch.transpose(model.w_k.weight, 0, 1))
        write_weight(v_weight_file_path, torch.transpose(model.w_v.weight, 0, 1))
        write_weight(out_weight_file_path, torch.transpose(model.w_concat.weight, 0, 1))
        write_token(token_file_path, x[0][0])




    # for name, parameter in model.named_parameters():
    #     if not parameter.requires_grad:
    #         continue
    #     params = parameter.numel()
    #     print(f'name: {name}, params: {params}')

    # print(f"--- Running MHA with D_MODEL={D_MODEL}, HEAD={HEAD}, # of layers={NUM_OF_LAYER}, # of token={NUM_OF_TOKEN} ---")
    # print(f"# of parameters: {trainable_params}")
    
    y = x
    y = model(y, y, y)
    x = torch.cat((x, y), 1)
    start_time = time.time()
    for i in range (NUM_OF_TOKEN-1):
        for l in range(NUM_OF_LAYER):
            y = model(y, x, x)
        x = torch.cat((x, y), 1)
    # print("execution time: %s seconds" % (time.time() - start_time))
    # print("%s" % (time.time() - start_time))
    # print(f"{torch.cuda.memory_allocated()}")
    # print(model(x, x, x))