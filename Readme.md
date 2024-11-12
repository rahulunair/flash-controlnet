# Flash ControlNet

A high-performance server that pre-loads multiple ControlNet models into HPU memory for instant access, eliminating model switching delays.

## Features

- Zero model swap time
- Multiple ControlNet flavors (Canny, Depth, HED)
- HPU-optimized for speed
- Concurrent request handling
- Built-in memory management

## Server Setup and Deployment
### Server Deployment

1. **Check Compatibility**
```   
# Check your Gaudi driver version (validated with 1.17.x driver)
hl-smi
```

2. **Build Server Container**
```
cd server
docker build -t flash-controlnet-server .
```
3. **Run Server**
```
mkdir -p ./models && \
docker run -it \
  --runtime=habana \
  -e HABANA_VISIBLE_DEVICES=all \
  -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
  --cap-add=sys_nice \
  --net=host \
  --ipc=host \
  -v $(pwd)/models:/root/.cache/huggingface/ \
  flash-controlnet-server
```
This would initiate ray serve and after few seconds you should see something like:

```bash
ServeReplica:default:ControlNetServer pid=892) INFO:sd:
(ServeReplica:default:ControlNetServer pid=892)             Warm-up completed for thibaud/controlnet-sd21-openpose-diffusers:
(ServeReplica:default:ControlNetServer pid=892)             - Average time: 1.36s
(ServeReplica:default:ControlNetServer pid=892)             - Min time: 0.13s
(ServeReplica:default:ControlNetServer pid=892)             - Max time: 2.58s
(ServeReplica:default:ControlNetServer pid=892)
(ServeReplica:default:ControlNetServer pid=892) INFO:serve:ControlNetServer initialized successfully
INFO 2024-11-12 23:37:34,417 serve 594 client.py:312 - Application 'default' is ready at http://127.0.0.1:8000/.
INFO 2024-11-12 23:37:34,418 serve 594 api.py:502 - Deployed app 'default' successfully.
```


**Note:** Ensure your docker image version matches your driver version using the Support Matrix: https://docs.habana.ai/en/latest/Support_Matrix/Support_Matrix.html . The code here has been tested on synapse AI version 1.17.

### Testing

1. **Single Client Test**

Now that the server is up, we can use the client to interact with the ControlNet server to generate AI-modified images.

Open a new terminal and follow the below steps:

## Setup

Build the Docker image:
```bash
# cd to flash_controlnet repo:
cd flash_controlnet
docker build -t sd_client -f Dockerfile.client .
```

Run the client container:
```bash
docker run -it --net=host sd_client
```

## Usage

The client supports three different control types: canny, depth, and hed. Basic command structure:

```bash
python client.py --image <input_image> --prompt "<text_prompt>" [--control_type <type>]
```

### Parameters

- `--image`: Path to input image (required)
- `--prompt`: Text description for image generation (required)
- `--control_type`: Type of control net (optional, defaults to canny)
  - Options: canny, depth, hed

### Examples

Download a sample image:

```bash
wget https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png
```

1. Using Canny edge detection (default):
```bash
python client.py --image input_image_vermeer.png --prompt "A fancy pancy image"
```

2. Using Depth estimation:
```bash
python client.py --image input_image_vermeer.png --prompt "A fancy pancy image" --control_type depth
```

3. Using HED edge detection:
```bash
python client.py --image input_image_vermeer.png --prompt "A fancy pancy image" --control_type hed
```

### Output

Generated images will be saved in the current directory with names corresponding to the control type used:
- `output_canny.png`
- `output_depth.png`
- `output_hed.png`

A successful run will show:
- Response Status Code: 200
- Confirmation message about saved image

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
