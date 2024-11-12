# Note to users: Before using this Dockerfile, check your Gaudi driver version using 'hl-smi'
# Then refer to the support matrix at https://docs.habana.ai/en/latest/Support_Matrix/Support_Matrix.html
# to ensure you use a compatible docker image version.
# This Dockerfile uses 1.17.0 base image which is compatible with driver version 1.17.x

FROM vault.habana.ai/gaudi-docker/1.17.0/ubuntu22.04/habanalabs/pytorch-installer-2.3.1:latest

# Install required packages
RUN apt-get update && apt-get install -y \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
# Note: optimum-habana 1.14.1 is compatible with Gaudi software 1.18
RUN pip install --no-cache-dir \
    "optimum-habana==1.14.1"  \
    "opencv-python==4.10.0.84" \
    "compel==2.0.3" ray[serve] "huggingface_hub"

COPY sd.py /app/sd.py
COPY serve.py /app/serve.py
COPY start_serving.sh /app/
WORKDIR /app

# Set entry point to start serving
ENTRYPOINT ["bash", "start_serving.sh"]
