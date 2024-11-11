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
    optimum-habana==1.14.1 \
    opencv-python \
    compel \
    ray[serve]

COPY server.py /app/server.py
COPY start_Serving.sh /app/
WORKDIR /app

# Set entry point to start serving
ENTRYPOINT ["python3", "-m", "start_Serving"]

# Usage instructions:
# 1. Build: docker build -t gaudi-server .
# 2. Run: docker run -it --network host --device=/dev/hl* --security-opt seccomp=unconfined gaudi-server
#
# Note: Using host network (--network host) means you don't need to explicitly expose ports
# The container will use the host's network stack directly
