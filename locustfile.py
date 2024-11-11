from locust import HttpUser, TaskSet, task, between
import base64
import json
import os
import random


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


class UserBehavior(TaskSet):
    @task
    def generate_image(self):
        control_type = ["canny", "depth", "hed"]
        data = {
            "control_type": random.choice(control_type),
            "prompt": "A futuristic-looking woman",
            "image": self.encoded_image,
        }
        headers = {"Content-Type": "application/json"}
        self.client.post("/", data=json.dumps(data), headers=headers)

    def on_start(self):
        # Preload and encode the image once
        self.encoded_image = encode_image("input.png")


class WebsiteUser(HttpUser):
    tasks = [UserBehavior]
    wait_time = between(1, 4)
