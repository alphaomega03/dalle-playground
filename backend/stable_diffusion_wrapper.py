from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import torch
import os

class StableDiffusionWrapper:
    def __init__(self) -> None:
        repo_id = "stabilityai/stable-diffusion-2-base"
        # torch.set_num_threads(8)
        # num_cores = os.cpu_count()
        # print(f"--> Using {num_cores} cores")
        # torch.set_num_threads(num_cores)
        # Set the device to use the first GPU
        torch.cuda.set_device(0)

        # Enable memory growth on the selected GPU
        torch.cuda.set_memory_growth(True)
        pipe = DiffusionPipeline.from_pretrained(
            repo_id, revision="fp16",
            torch_dtype=torch.float16
        )

        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config)
        self.pipe = pipe.to("cuda")

            
    def generate_images(self, text_prompt: str, num_images: int):
        prompt = [text_prompt] * num_images
        images = self.pipe(prompt, num_inference_steps=10).images
        return images
