import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any

from ray import serve
from starlette.responses import Response
from starlette.requests import Request
from argparse import Namespace
import base64
from io import BytesIO
from PIL import Image
import logging

from sd import HPUModelManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@serve.deployment(
    ray_actor_options={"num_cpus": 10, "resources": {"HPU": 1}},
    num_replicas=1, # use 8 to distribute to 8 cards
    max_ongoing_requests=2,
    max_queued_requests=10000,
)
class ControlNetServer:
    def __init__(self):
        logger.info("Initializing ControlNetServer")
        try:
            self.config = Namespace(
                model_name_or_path="stabilityai/stable-diffusion-2-1",
                prompt="",
                num_inference_steps=50,
                guidance_scale=7.5,
                batch_size=1,
                bf16=True,
                timestep_spacing="linspace",
            )
            self.manager = HPUModelManager(self.config)
            # Load controlnet models as sd21 is the model, using the controlnets for it
            self.controlnet_models = [
                "thibaud/controlnet-sd21-canny-diffusers",
                "thibaud/controlnet-sd21-depth-diffusers",
                "thibaud/controlnet-sd21-hed-diffusers",
                "thibaud/controlnet-sd21-openpose-diffusers"

            ]
            for model_path in self.controlnet_models:
                if not self.manager.load_model(model_path):
                    logger.error(f"Failed to load model {model_path}")
                    raise RuntimeError(f"Failed to load model {model_path}")
                if not self.manager.warm_up_model(model_path, num_warmup_steps=2):
                    logger.error(f"Failed to warm up model {model_path}")
                    raise RuntimeError(f"Failed to warm up model {model_path}")

            logger.info("ControlNetServer initialized successfully")
            self.executor = ThreadPoolExecutor(
                max_workers=5
            ) 

        except Exception as e:
            logger.error(f"Exception during server initialization: {e}")
            raise

    async def __call__(self, request: Request) -> Response:
        logger.info("Received a request")
        try:
            data = await request.json()
            logger.info(f"Request data: {data}")

            control_type = data.get("control_type", "canny").lower()
            prompt = data.get("prompt", "")
            image_data = data.get("image", None)
            if not image_data:
                logger.error("No image provided in the request")
                return Response(
                    content="No image provided",
                    media_type="text/plain",
                    status_code=400,
                )
            try:
                image_bytes = base64.b64decode(image_data)
                input_image = Image.open(BytesIO(image_bytes)).convert("RGB")
                input_image = input_image.resize((512, 512))
                logger.info("Image decoded and resized")
            except Exception as e:
                logger.error(f"Invalid image data: {e}")
                return Response(
                    content=f"Invalid image data: {str(e)}",
                    media_type="text/plain",
                    status_code=400,
                )
            model_map = {
                "canny": "thibaud/controlnet-sd21-canny-diffusers",
                "depth": "thibaud/controlnet-sd21-depth-diffusers",
                "hed": "thibaud/controlnet-sd21-hed-diffusers",
            }
            model_path = model_map.get(control_type)
            if not model_path:
                logger.error(f"Invalid control type: {control_type}")
                return Response(
                    content="Invalid control type. Must be one of: canny, depth, hed",
                    media_type="text/plain",
                    status_code=400,
                )
            logger.info(f"Using control type: {control_type}, model path: {model_path}")l
            pipeline = self.manager.loaded_models[model_path]
            control_image = self.manager.preprocess_image(model_path, input_image)
            logger.info("Image preprocessed")
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor, 
                self.run_inference,
                pipeline,
                prompt,
                control_image,
            )
            logger.info("Image generated")
            img_byte_arr = BytesIO()
            result.save(img_byte_arr, format="PNG")
            img_byte_arr = img_byte_arr.getvalue()
            logger.info("Returning generated image")
            return Response(content=img_byte_arr, media_type="image/png")
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            return Response(
                content=f"Error generating image: {str(e)}",
                media_type="text/plain",
                status_code=500,
            )

    def run_inference(self, pipeline, prompt, control_image):
        """Run the inference synchronously in an executor."""
        return pipeline(
            prompt=prompt,
            image=control_image,
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale,
        ).images[0]


entrypoint = ControlNetServer.bind()
