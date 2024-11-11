# Flash ControlNet

A high-performance server that pre-loads multiple ControlNet models into HPU memory for instant access, eliminating model switching delays.

## Features

- Zero model swap time
- Multiple ControlNet flavors (Canny, Depth, HED)
- HPU-optimized for speed
- Concurrent request handling
- Built-in memory management

## Setup

### Prerequisites
- Habana HPU
- Python 3.8+
- Ray Serve
- Optimum Habana
- PIL, OpenCV

### Quick Start

```bash
git clone <repo-url>
cd flash-controlnet
pip install -r requirements.txt
ray start --head
serve run serve:entrypoint
```
## Server Setup and Deployment

### Project Structure
```bash
flash-controlnet/
├── Dockerfile.locust          # Dockerfile for load testing
├── Readme.md                  # This documentation
├── client.py                  # Client code for testing
├── locustfile.py             # Load testing configuration
└── server/                   # Server components
    ├── Dockerfile           # Server Dockerfile
    ├── sd.py               # Stable Diffusion implementation
    ├── serve.py            # Ray Serve implementation
    └── start_serving.sh    # Server startup script
```
### Server Deployment

1. **Check Compatibility**
```   
# Check your Gaudi driver version
hl-smi
```   
**Note:** Ensure your docker image version matches your driver version using the Support Matrix: https://docs.habana.ai/en/latest/Support_Matrix/Support_Matrix.html

2. **Build Server Container**
```
cd server
docker build -t flash-controlnet-server .
```
3. **Run Server**
```
docker run -it --network host \
--device=/dev/hl* \
--security-opt seccomp=unconfined \
flash-controlnet-server
```
### Testing

1. **Single Client Test**
```
python client.py
```
2. **Load Testing Setup**
```
# Build load testing container
docker build -f Dockerfile.locust -t flash-controlnet-locust .

# Run load testing container
docker run -it --network host flash-controlnet-locust
```
3. **Access Load Testing Dashboard**
   - Open http://localhost:8089 in your browser
   - Set number of users and spawn rate
   - Start the test

### Monitoring and Logs

- Check server logs:
````
docker logs <server-container-id>
```
- Monitor HPU usage:
```
hl-smi
```
### Common Issues and Troubleshooting

1. **Device Access Issues**
   - Check if Gaudi devices are properly detected: hl-smi

2. **Container Network Issues**
   - Verify host network access: curl localhost:8000
   - Check if Ray Serve is running: ray status

3. **Memory Issues**
   - Monitor HPU memory usage: hl-smi
   - Check system logs: dmesg | grep -i gaudi


## Notes
 - HPU memory is pre-allocated for all models
 - Models are pre-warmed for optimal first-request performance
 - Concurrent requests are handled via ThreadPoolExecutor
