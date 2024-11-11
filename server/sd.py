import logging
import os
import time
from argparse import Namespace
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple

import habana_frameworks.torch.hpu as hthpu
import numpy as np
import requests
import torch
from diffusers import ControlNetModel
from optimum.habana.diffusers import (
    GaudiDDIMScheduler,
    GaudiStableDiffusionControlNetPipeline,
)
from optimum.habana.utils import set_seed
from PIL import Image
import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HPUModelManager:
    def __init__(self, base_config: Namespace):
        self.base_config = base_config
        self.loaded_models = {}
        self.warm_up_stats = {}
        self.memory_stats = {}
        if not hthpu.is_available():
            raise RuntimeError("HPU is not available")
        hthpu.reset_peak_memory_stats()
        mem_stats = hthpu.memory_stats()
        total_memory = mem_stats["Limit"]
        logger.info(f"Using HPU device: {hthpu.get_device_name()}")
        logger.info(f"Total HPU memory: {total_memory / 1024**3:.2f} GB")
        logger.info(f"Initial memory in use: {mem_stats['InUse'] / 1024**3:.2f} GB")

    def preprocess_image(self, model_path: str, image: Image.Image) -> Image.Image:
        """Preprocess image based on ControlNet type"""
        image_np = np.array(image)
        if "canny" in model_path.lower():
            image_edges = cv2.Canny(image_np, 100, 200)
            return Image.fromarray(image_edges)

        elif "hed" in model_path.lower():
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            return Image.fromarray(edges)

        elif "depth" in model_path.lower():
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            return Image.fromarray(gray)

        else:
            raise ValueError(f"Unknown ControlNet type in {model_path}")

    def get_memory_stats(self) -> dict:
        """Get current HPU memory statistics"""
        mem_stats = hthpu.memory_stats()
        current_allocated = hthpu.memory_allocated()
        max_allocated = hthpu.max_memory_allocated()
        return {
            "allocated": current_allocated,
            "max_allocated": max_allocated,
            "in_use": mem_stats["InUse"],
            "max_in_use": mem_stats["MaxInUse"],
            "limit": mem_stats["Limit"],
            "active_allocs": mem_stats["ActiveAllocs"],
            "utilization": (current_allocated / mem_stats["Limit"] * 100),
        }

    def print_memory_status(self):
        """Print detailed memory status"""
        stats = self.get_memory_stats()
        logger.info(
            f"""
        Memory Status:
        - Current Allocated: {stats['allocated'] / 1024**3:.2f} GB
        - Max Allocated: {stats['max_allocated'] / 1024**3:.2f} GB
        - In Use: {stats['in_use'] / 1024**3:.2f} GB
        - Max In Use: {stats['max_in_use'] / 1024**3:.2f} GB
        - Memory Limit: {stats['limit'] / 1024**3:.2f} GB
        - Active Allocations: {stats['active_allocs']}
        - Utilization: {stats['utilization']:.2f}%
        """
        )

    def load_model(self, model_path: str) -> bool:
        """Load model into HPU memory"""
        try:
            logger.info(f"\nLoading model: {model_path}")
            initial_memory = self.get_memory_stats()
            logger.info(f"Memory before loading:")
            self.print_memory_status()
            scheduler = GaudiDDIMScheduler.from_pretrained(
                self.base_config.model_name_or_path,
                subfolder="scheduler",
                timestep_spacing=self.base_config.timestep_spacing,
            )
            controlnet = ControlNetModel.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16 if self.base_config.bf16 else torch.float32,
            )
            pipeline = GaudiStableDiffusionControlNetPipeline.from_pretrained(
                self.base_config.model_name_or_path,
                controlnet=controlnet,
                scheduler=scheduler,
                use_habana=True,
                use_hpu_graphs=True,
                gaudi_config="Habana/stable-diffusion",
                torch_dtype=torch.bfloat16 if self.base_config.bf16 else torch.float32,
            )
            pipeline.to("hpu")
            self.loaded_models[model_path] = pipeline
            post_load_memory = self.get_memory_stats()
            self.memory_stats[model_path] = {
                "initial": initial_memory,
                "after_load": post_load_memory,
                "memory_increase": post_load_memory["allocated"]
                - initial_memory["allocated"],
            }
            logger.info(
                f"""
            Model loaded successfully:
            - Memory increase: {self.memory_stats[model_path]['memory_increase'] / 1024**3:.2f} GB
            - Total utilization: {post_load_memory['utilization']:.2f}%
            """
            )
            return True
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {str(e)}")
            return False

    def warm_up_model(self, model_path: str, num_warmup_steps: int = 2):
        """Warm up a loaded model with dummy inference"""
        try:
            logger.info(f"Warming up model: {model_path}")
            pipeline = self.loaded_models[model_path]
            control_image = Image.new("RGB", (512, 512))
            prompt = "test warm up"
            warm_up_times = []
            for _ in range(num_warmup_steps):
                start_time = time.time()
                # Run inference with processed control image
                _ = pipeline(
                    prompt,
                    image=control_image,
                    num_inference_steps=2,  # Use minimal steps for warm-up
                    guidance_scale=7.5,
                ).images[0]

                end_time = time.time()
                warm_up_times.append(end_time - start_time)
            self.warm_up_stats[model_path] = {
                "average_time": sum(warm_up_times) / len(warm_up_times),
                "min_time": min(warm_up_times),
                "max_time": max(warm_up_times),
            }
            logger.info(
                f"""
            Warm-up completed for {model_path}:
            - Average time: {self.warm_up_stats[model_path]['average_time']:.2f}s
            - Min time: {self.warm_up_stats[model_path]['min_time']:.2f}s
            - Max time: {self.warm_up_stats[model_path]['max_time']:.2f}s
            """
            )

            return True

        except Exception as e:
            logger.error(f"Warm-up failed for {model_path}: {str(e)}")
            return False
