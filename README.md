# 😈 MLC-Imp

This repository contains an efficient deployment solution for Imp models on mobile devices, which is based on the [MLC-LLM](https://github.com/mlc-ai/mlc-llm) framework and referred to the [MLC-MiniCPM](https://github.com/OpenBMB/mlc-MiniCPM) project.
## Model Description

Note that our default Imp-3B model uses an input image size of 384x384, which results in a 729 visual embeddings and takes too much time and memory for mobile devices. Therefore, we use an Imp variant with a reduced image size of 196x196. More details can be referred to our [technical report](https://arxiv.org/abs/2405.12107)). The models run on android are quantized to `4-bit`, which takes 1.9GB stoarage.

## Running MLC-Imp on Android
We provide two ways to run MLC-Imp on mobile phones with Android system. YOu can use directly install our precompiled `.apk` file or compile it from scatrch by yourself.

The solution for IOS system is still one the way. 

### 1. Use precompiled APK
0. Download precompiled ImpChat apk from [here](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EUEBOigiw91Hs8a3Psu9SYIBu4BqpzXYZ6SjqLL5h0d7SA?download=1)

1. Accept camera & photo permission
<img src="assets/permission.jpg" width="250">

2. Download model: (1) Press the download button (2) Wait for the progress bar to fill up (3) Start chat 
<div>
    <img src="assets/download1.jpg" width="350">
    <img src="assets/download2.jpg" width="350">
    <img src="assets/download3.jpg" width="350">
</div>

3. Chat with Imp: (1) Wait for model initialization until "Ready to chat" pop up. (2) Upload an image from the gallary or take a photo using the camera (3) Wait until "process image done" show up. (4) Enter your question to begin a conversation.
    - Chat mode: **Text** or **Vision** are both support
    - Note：image process may take some time.
<div>
    <img src="assets/chat1.jpg" width="200">
    <img src="assets/chat2.jpg" width="200">
    <img src="assets/chat3.jpg" width="200">

</div>

4. Demo 

<div>
    <img src="assets/demo1_480p.gif" width="300">
    <img src="assets/demo2_480p.gif" width="300">
</div>

### 2. Compile APK from scartch

#### Prepare MLC environment
1. Follow https://llm.mlc.ai/docs/install/tvm.html to Install TVM Unity Compiler

2. Follow https://llm.mlc.ai/docs/install/mlc_llm.html to Install MLC LLM Python Package

3. Follow https://llm.mlc.ai/docs/deploy/android.html to prepare android requirements.

#### Compile libarieis for Android

Download the model [checkpoint](https://huggingface.co/MILVLG/Imp-v1.5-3B-196) into the `dist/models` folder.

```
# covert the model weights from fp-16 to 4-bit
mlc_llm convert_weight --model-type imp ./dist/models/imp-v1.5-3B-196 --quantization q4f16_1 -o ./dist/imp-v1.5-3B-196-q4f16_1

# generate config
mlc_llm gen_config ./dist/models/imp-v1.5-3B-196 --quantization q4f16_1 --conv-template imp -o ./dist/imp-v1.5-3B-196-q4f16_1

# compile for android
mlc_llm compile ./dist/imp-v1.5-3B-196-q4f16_1/mlc-chat-config.json --device android -o ./dist/libs/imp-v1.5-3B-196-q4f16_1-android.tar

cd ./android/library
./prepare_libs.sh
```

### Build Android application
Go to `android/` and use Android Studio to build the app. (Follow https://llm.mlc.ai/docs/deploy/android.html)

## Run MLC-Imp on Linux/Windows
Alternatively, we can also use the MLC-Imp on Linux/Windows servers.

### Prepare MLC-Imp Enviroment

1. Follow https://llm.mlc.ai/docs/install/tvm.html to Install TVM Unity Compiler

2. Follow https://llm.mlc.ai/docs/install/mlc_llm.html to Install MLC LLM Python Package

### Compile libarieis for servers

Download the model [checkpoint](https://huggingface.co/MILVLG/Imp-v1.5-3B-196) and put it into the `dist/models` folder.

use Vulkan as an example:
```
# covert imp model into 4bit
mlc_llm convert_weight --model-type imp ./dist/models/imp-v1.5-3B_196 --quantization q4f16_1 -o ./dist/imp-v1.5-3B_196-q4f16_1

# generate config
mlc_llm gen_config ./dist/models/imp-v1.5-3B-196 --quantization q4f16_1 --conv-template imp -o ./dist/imp-v1.5-3B-196-q4f16_1

# compile to vulkan
mlc_llm compile ./dist/imp-v1.5-3B-196-q4f16_1/mlc-chat-config.json --device vulkan -o ./dist/libs/imp-v1.5-3B-196-q4f16_1-vulkan.so
```

Then, you can use the following example python script to use MLC-Imp on the corresponding server

```
import tvm
from mlc_llm import ChatModule
from PIL import Image

from transformers.image_utils import (
    ChannelDimension,
    PILImageResampling,
    to_numpy_array,
)
from transformers.image_transforms import (
    convert_to_rgb,
    normalize,
    rescale,
    resize,
    to_channel_dimension_format,
)
from functools import partial, reduce
from transformers.image_processing_utils import BatchFeature

def load_image(image_file):
    from io import BytesIO
    import requests
    from PIL import Image

    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image
    
def simple_image_processor(
        images, 
        image_mean=(0.5, 0.5, 0.5), 
        image_std=(0.5, 0.5, 0.5), 
        size=(196, 196), 
        resample=PILImageResampling.BICUBIC, 
        rescale_factor=1 / 255, 
        data_format=ChannelDimension.FIRST,
        return_tensors="pt"
    ):

    if isinstance(images, Image.Image):
        images = [images]
    else:
        assert isinstance(images, list)
    
    transforms = [
        convert_to_rgb,
        to_numpy_array,
        partial(resize, size=size, resample=resample, data_format=data_format),
        partial(rescale, scale=rescale_factor, data_format=data_format),
        partial(normalize, mean=image_mean, std=image_std, data_format=data_format),
        partial(to_channel_dimension_format, channel_dim=data_format, input_channel_dim=data_format),
    ]

    images = reduce(lambda x, f: [*map(f, x)], transforms, images)
    data = {"pixel_values": images}
    
    return BatchFeature(data=data, tensor_type=return_tensors)

image_path = "./assets/bus.jpg"
image_tensor = load_image(image_path)
image_features = tvm.nd.array(
    simple_image_processor(image_tensor)['pixel_values'].numpy().astype("float32"),
    device=tvm.runtime.ndarray.vulkan(),
)
cm = ChatModule(model="./dist/imp-v1.5-3B-196-q4f16_1", model_lib_path="./dist/libs/imp-v1.5-3B-196-q4f16_1-vulkan.so")

output = cm.generate(
    prompt="<image>\nWhat are the colors of the bus in the image?",
    pixel_values=image_features
)
print(output)

```
