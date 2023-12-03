from retnet import RetNetModel
from config import RetNetConfig
import torch

config = RetNetConfig(decoder_layers=2, decoder_embed_dim=256, decoder_retention_heads=8, decoder_ffn_embed_dim=512, recurrent_chunk_size=2)
embed_tokens = torch.nn.Embedding(100, config.decoder_embed_dim)
model = RetNetModel(config, embed_tokens=embed_tokens)

model = torch.ao.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# device = 'cuda'
device = 'cpu'

model = model.to(device)

input_ids = torch.LongTensor([[1,2,1,2]]).to(device)



# parallel inference
# model.chunkwise_recurrent = False
# parallel_state, _ = model(input_ids, features_only=True)

# recurrent inference
incremental_state = {}
rnn_state = []
for i in range(input_ids.shape[1]):
    # print("input: ", input_ids[:, :i+1])
    print("input size: ", input_ids[:, :i+1].shape)
    rnn_h, _ = model(input_ids[:, :i+1], incremental_state=incremental_state, features_only=True)
    rnn_state.append(rnn_h)
rnn_state = torch.cat(rnn_state, dim=1)

print("----------------------------------")
print("printing quantized model")
print(model.layers[0].retention.q_proj.weight())
print(model.layers[0].retention.q_proj.weight().int_repr())

# chunkwise recurrent inference
# model.chunkwise_recurrent = True
# chunk_state, _ = model(input_ids, features_only=True)

# compare
# print('parallel vs recurrent', torch.allclose(parallel_state, rnn_state, atol=1e-5))
# print('parallel vs chunkwise', torch.allclose(parallel_state, chunk_state, atol=1e-5))
