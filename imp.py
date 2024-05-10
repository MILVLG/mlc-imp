import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

torch.set_default_device("cuda")

#Create model
model = AutoModelForCausalLM.from_pretrained(
    "/data2/wangmy/mlc-imp/dist/models/imp-v1-3b_224", 
    torch_dtype=torch.float16, 
    device_map="auto",
    trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("/data2/wangmy/mlc-imp/dist/models/imp-v1-3b_224", trust_remote_code=True)

#Set inputs
text = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\nWhat is this? ASSISTANT:"
image = Image.open("/data1/zhenglh/llama.cpp/imp-pic/2.jpg")

input_ids = tokenizer(text, return_tensors='pt').input_ids
image_tensor = model.image_preprocess(image)

#Generate the answer
output_ids = model.generate(
    input_ids,
    max_new_tokens=100,
    images=image_tensor,
    use_cache=True)[0]
print(tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip())
