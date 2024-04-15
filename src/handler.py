import base64
import io
import torch
from PIL import Image
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
from diffusers.utils import load_image

import runpod
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.utils import rp_cleanup
from rp_schemas import INPUT_SCHEMA

# Initialize diffusers pipelines
pipe = AutoPipelineForText2Image.from_pretrained("kandinsky-community/kandinsky-3", variant="fp16", torch_dtype=torch.float16)
pipe.enable_sequential_cpu_offload()

pipe_1_2 = AutoPipelineForImage2Image.from_pretrained("kandinsky-community/kandinsky-3", variant="fp16", torch_dtype=torch.float16)
pipe_1_2.enable_sequential_cpu_offload()

def image_to_base64(image):
    """Converts a PIL Image object to a base64 encoded string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def generate_image(job):
    """Generate an image based on the input prompt and return base64-encoded image(s) in the response."""
    job_input = job["input"]

    # Validate input against the schema
    validated_input = validate(job_input, INPUT_SCHEMA)
    if 'errors' in validated_input:
        return {"error": validated_input['errors']}
    validated_input = validated_input['validated_input']

    # Access input parameters
    prompt = validated_input['prompt']
    num_images_per_prompt = validated_input.get('num_images_per_prompt', 1)  # Default to 1 if not provided

    # Set up Torch generator
    generator = torch.Generator(device="cpu").manual_seed(0)

    # Initialize image_urls list
    image_urls = []

    # Load initial image if provided
    init_image = None
    if job_input.get('init_image'):
        init_image = load_image(job_input['init_image'])

    # Generate images based on the input prompt
    for _ in range(num_images_per_prompt):
        if init_image is None:
            image = pipe(prompt, num_inference_steps=25, generator=generator).images[0]
        else:
            image = pipe_1_2(prompt, image=init_image, strength=0.75, num_inference_steps=25, generator=generator).images[0]

        # Convert image to base64
        image_base64 = image_to_base64(image)
        image_urls.append(image_base64)

    # Clean up temporary resources
    rp_cleanup.clean([f"/{job['id']}"])

    # Prepare response based on the number of images generated
    if num_images_per_prompt == 1:
        response = {"image_url": image_urls[0]}  # Single image base64
    else:
        response = {"images": image_urls}  # Multiple images base64 (in a list)

    # Return the response containing base64-encoded image(s)
    return response

# Start RunPod serverless execution
runpod.serverless.start({"handler": generate_image})