import requests
import base64
from PIL import Image
from io import BytesIO
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def generate_image(image_path, prompt, control_type="canny"):
    data = {
        "control_type": control_type,
        "prompt": prompt,
        "image": encode_image(image_path),
    }

    try:
        response = requests.post("http://localhost:8000/", json=data, timeout=600)
        logger.info(f"Response Status Code: {response.status_code}")
        if response.status_code == 200:
            # Save the generated image
            image = Image.open(BytesIO(response.content))
            output_path = f"output_{control_type}.png"
            image.save(output_path)
            logger.info(f"Generated image saved as {output_path}")
        else:
            logger.error(f"Error: {response.status_code}")
            logger.error(f"Response: {response.text}")
    except requests.exceptions.Timeout:
        logger.error("Request timed out")
    except Exception as e:
        logger.error(f"Error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images using ControlNet")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument(
        "--prompt", required=True, help="Text prompt for image generation"
    )
    parser.add_argument(
        "--control_type",
        default="canny",
        choices=["canny", "depth", "hed"],
        help="Type of ControlNet to use",
    )
    args = parser.parse_args()
    generate_image(args.image, args.prompt, args.control_type)
