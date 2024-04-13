'''
Fetches and caches the Kandinsky models.
'''

import torch
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image

def get_kandinsky_pipelines():
    pipe_1_1 = AutoPipelineForText2Image.from_pretrained("kandinsky-community/kandinsky-3", variant="fp16", torch_dtype=torch.float16)
    pipe_1_1.enable_sequential_cpu_offload()

    pipe_1_2 = AutoPipelineForImage2Image.from_pretrained("kandinsky-community/kandinsky-3", variant="fp16", torch_dtype=torch.float16)
    pipe_1_2.enable_sequential_cpu_offload()

    return (pipe_1_1), (pipe_1_2)

if __name__ == "__main__":
    get_kandinsky_pipelines()