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

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result
    
def simple_image_processor(
        images, 
        image_mean=(0.5, 0.5, 0.5), 
        image_std=(0.5, 0.5, 0.5), 
        size=(224, 224), 
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

    new_images = []
    for image in images:
        image = expand2square(image, tuple(int(x*255) for x in image_mean))
        new_images.append(image)
    images=new_images

    images = reduce(lambda x, f: [*map(f, x)], transforms, images)
    data = {"pixel_values": images}
    
    return BatchFeature(data=data, tensor_type=return_tensors)

image_path = "/data1/zhenglh/llama.cpp/imp-pic/1.jpg"
# image_path = "/data2/wangmy/images/2.jpg"
image_tensor = load_image(image_path)
image_features = tvm.nd.array(
    simple_image_processor(image_tensor)['pixel_values'].numpy().astype("float32"),
    device=tvm.runtime.ndarray.vulkan(),
)
print(image_features)
cm = ChatModule(model="/data2/wangmy/mlc-imp/dist/imp-v1-3b_224-q4f16_1", model_lib_path="/data2/wangmy/mlc-imp/dist/libs/imp-v1-3b_224-q4f16_1-vulkan.so")

output = cm.generate(
    # prompt="<image>\nWhat are the colors of the bird in the image?",
    prompt="<image>\nWhat is this?",
    pixel_values=image_features
)
print(output)
