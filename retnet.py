import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

MODEL_NAME = "isek-ai/SDPrompt-RetNet-300M"

DEVICE = "cuda"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
).to(DEVICE)

for (name, param) in model.named_parameters():
    if param.requires_grad:
        print (name, param.data)


streamer = TextStreamer(tokenizer)

prompt = "<s>1girl"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

_ = model.generate(
    inputs["input_ids"],
    max_new_tokens=256,
    do_sample=True,
    top_p=0.9,
    top_k=20,
    temperature=0.9,
    streamer=streamer,
)

# PTDQ
# model_int8 = torch.ao.quantization.quantize_dynamic(
#     model,  # the original model
#     {torch.nn.Linear},  # a set of layers to dynamically quantize
#     dtype=torch.qint8)  # the target dtype for quantized weights

# print(model_int8(model))

# PTSQ
model.eval()

model.qconfig = torch.ao.quantization.get_default_qconfig('x86')
model_int8 = torch.ao.quantization.convert(model)
print(model_int8(model))

