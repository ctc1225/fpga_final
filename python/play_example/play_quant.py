import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class dense(nn.Module):
  def __init__(
    self,
    in_dim,
    out_dim
  ):
    super().__init__()
    self.in_dim = in_dim
    self.out_dim = out_dim
    self.fc = nn.Linear(self.in_dim, self.out_dim, bias=False)

    self.reset_parameters()

  def reset_parameters(self):
    # nn.init.uniform_(self.fc.weight)
    nn.init.ones_(self.fc.weight)
    # print(self.fc.weight)

  def forward(self, x):
    x = self.fc(x)
    return x


if __name__ == "__main__":
  in_dim = 4
  out_dim = 8
  x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

  model = dense(in_dim, out_dim)
  # torch.save(model.state_dict(), 'ones_weights.pth')
  model.load_state_dict(torch.load('ones_weights.pth'))

  model_int8 = torch.ao.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
  )

  with torch.no_grad():
    print("original model weight:", model.fc.weight)
    print("quantized model weight:", model_int8.fc.weight().int_repr())


    out = model(x)
    out_int8 = model_int8(x)
    print("original output: ", out)
    print("quantized output: ", out_int8)

    print("weight zero point & scale: ", model_int8.fc.weight().q_zero_point(), model_int8.fc.weight().q_scale())

    # y = torch.quantize_per_tensor(x, scale=1/128, zero_point=0, dtype=torch.qint8)
    # print(y.int_repr())
    # out = model(y)
    # print("original output: ", out)


  from torch.fx import symbolic_trace
  # Symbolic tracing frontend - captures the semantics of the model
  symbolic_traced : torch.fx.GraphModule = symbolic_trace(model)

  # High-level intermediate representation (IR) - Graph representation
  print(symbolic_traced.graph)
