import tvm
from mlc_llm import ChatModule
from mlc_llm.callback import StreamToStdout
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
        size=(384, 384), 
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

image_path = "/data1/zhenglh/llama.cpp/imp-pic/1.jpg"
# image_path = "/data2/wangmy/images/bus.jpg"
image_tensor = load_image(image_path)
image_features = tvm.nd.array(
    simple_image_processor(image_tensor)['pixel_values'].numpy().astype("float32"),
    device=tvm.runtime.ndarray.vulkan(),
)
cm = ChatModule(model="/data2/wangmy/mlc-imp/dist/imp-v1-3b-q4f16_1", model_lib_path="/data2/wangmy/mlc-imp/dist/libs/imp-v1-3b-q4f16_1-vulkan.so")

output = cm.generate(
    prompt="<image>\nWhat is this?",
    # prompt="write me a poem.",
    pixel_values=image_features
)
print(output)
