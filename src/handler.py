import os

from diffusers import AutoPipelineForText2Image
from diffusers import AutoPipelineForImage2Image
import torch
from diffusers.utils import load_image

import runpod
from runpod.serverless.utils import rp_upload, rp_cleanup
from runpod.serverless.utils.rp_validator import validate
from rp_schemas import INPUT_SCHEMA

pipe = AutoPipelineForText2Image.from_pretrained("kandinsky-community/kandinsky-3", variant="fp16", torch_dtype=torch.float16)
pipe.enable_sequential_cpu_offload()

pipe_1_2 = AutoPipelineForImage2Image.from_pretrained("kandinsky-community/kandinsky-3", variant="fp16", torch_dtype=torch.float16)
pipe_1_2.enable_sequential_cpu_offload()

def _save_and_upload_images(image, job_id):
    os.makedirs(f"/{job_id}", exist_ok=True)
    image_path = os.path.join(f"/{job_id}", "image.png")
    image.save(image_path)
    image_url = rp_upload.upload_image(job_id, image_path)
    rp_cleanup.clean([f"/{job_id}"])
    return [image_url]



def generate_image(job):

    job_input = job["input"]

    validated_input = validate(job_input, INPUT_SCHEMA)

    if 'errors' in validated_input:
        return {"error": validated_input['errors']}
    validated_input = validated_input['validated_input']

    generator = torch.Generator(device="cpu").manual_seed(0)

    init_image = None
    if job_input.get('init_image', None) is not None:
        init_image = load_image(job_input['init_image'])

    image_urls = []

    

    if init_image is None:
        image = pipe(validated_input['prompt'], num_inference_steps=25, generator=generator).images[0]
    else:
        image = pipe_1_2(validated_input['prompt'], image=image, strength=0.75, num_inference_steps=25, generator=generator).images[0]

    image_urls = _save_and_upload_images(image, job['id'])

    return {"image_url": image_urls[0]} if len(image_urls) == 1 else {"images": image_urls}

runpod.serverless.start({"handler": generate_image})
