"""
Run auto-regressive inference on multi-head attention (single layer)
Measure computation time and GPU memory usage
"""
from torch import nn
import torch
import time
import sys
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
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with Value
        v = score @ v

        return v, score

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)
        self.K_cache = None
        self.V_cache = None

    def forward(self, q, k, v, mask=None):
        # 1. dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # use KV cache
        if self.K_cache == None:
            self.K_cache = k
        else:
            self.K_cache = torch.concat((self.K_cache, k), 1)

        if self.V_cache == None:
            self.V_cache = v
        else:
            self.V_cache = torch.concat((self.V_cache, v), 1)

        k = self.K_cache
        v = self.V_cache

        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v, mask=mask)

        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)

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
    D_MODEL = 8
    HEAD = 2
    device = 'cuda'
    mha = MultiHeadAttention(D_MODEL, HEAD).to(device)
    mha.eval()
    trainable_params = sum(
        p.numel() for p in mha.parameters() if p.requires_grad
    )
    # print(f'params: {trainable_params}')

    # send # of tokens through parameter, else set to one
    NUM_OF_TOKEN = 0
    if len(sys.argv) == 1: NUM_OF_TOKEN = 1
    else: NUM_OF_TOKEN = int(sys.argv[1])

    # print out parameter
    # for name, parameter in mha.named_parameters():
    #     if not parameter.requires_grad:
    #         continue
    #     params = parameter.numel()
    #     print(f'name: {name}, params: {params}')

    # print(f"--- Running MHA with D_MODEL={D_MODEL}, HEAD={HEAD}, # of token={NUM_OF_TOKEN} ---")
    x = torch.randn(1, 1, D_MODEL).to(device)
    y = x
    start_time = time.time()
    for i in range (NUM_OF_TOKEN):
        y = mha(y, y, y)
        # print(f'output: {y}')
    # print("execution time: %s seconds" % (time.time() - start_time))
    print("%s" % (time.time() - start_time))  # elapsed time
    print(f"{torch.cuda.memory_allocated()}") # total GPU memory usage