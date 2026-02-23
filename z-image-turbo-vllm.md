# README: vLLM-Omni Installation and Usage Guide

This document provides a complete set of instructions for installing and using vLLM-Omni with multimodal models (image generation and TTS) on a 24GB GPU system. Each command is preserved exactly as provided, with English annotations added for clarity.

---

## **Prerequisites**
- A system with a 24GB GPU card. (3090/4090)
- Ensure you have necessary permissions (e.g., `sudo` for system-level installations).

---

## **Step 1: Install Modelscope and Download Model**
```bash
# Install the modelscope library
pip install modelscope

# Download the Tongyi-MAI/Z-Image-Turbo model to a local directory named "Z-Image-Turbo"
modelscope download Tongyi-MAI/Z-Image-Turbo --local_dir="Z-Image-Turbo"
```

---

## **Step 2: Set Up Python Environment with UV**
```bash
# Install uv (a fast Python package installer)
pip install uv

# Deactivate any existing virtual environment
deactivate

# Remove any existing .venv directory
rm -rf .venv

# Create a new virtual environment using Python 3.12
uv venv --python 3.12

# Activate the virtual environment
source .venv/bin/activate

# Set the package index to Tsinghua University mirror for faster downloads in China
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## **Step 3: Install vLLM and Dependencies**
```bash
# Upgrade and install vLLM (allow pre-release versions)
# uv pip install --upgrade vllm
uv pip install --prerelease=allow vllm --extra-index-url https://wheels.vllm.ai/2d5be1dd5ce2e44dfea53ea03ff61143da5137eb
```

---

## **Step 4: Install FlashInfer for CUDA Optimization**
```bash
# Detect CUDA version and set environment variable (default to 12.4 if not found)
export FLASHINFER_CUDA_TAG="$(python3 -c 'import torch; print((torch.version.cuda or "12.4").replace(".", ""))')"
echo $FLASHINFER_CUDA_TAG

# Alternative: Manual installation of FlashInfer components (commented out in original)
# uv pip install --upgrade --force-reinstall \
#   "flashinfer-python==0.6.3" \
#   "flashinfer-cubin==0.6.3" \
#   "flashinfer-jit-cache==0.6.3" \
#   --extra-index-url "${FLASHINFER_CUDA_TAG}" 

# Instead, download and install a specific FlashInfer JIT cache wheel (version 0.6.4)
wget https://github.com/flashinfer-ai/flashinfer/releases/download/v0.6.4/flashinfer_jit_cache-0.6.4+cu128-cp39-abi3-manylinux_2_28_x86_64.whl
# Or copy from existing location if already downloaded:
cp work/flashinfer_jit_cache-0.6.4+cu128-cp39-abi3-manylinux_2_28_x86_64.whl .
uv pip install flashinfer_jit_cache-0.6.4+cu128-cp39-abi3-manylinux_2_28_x86_64.whl

# Install FlashInfer Python and Cubin components (version 0.6.3)
uv pip install --upgrade --force-reinstall \
  "flashinfer-python==0.6.3" \
  "flashinfer-cubin==0.6.3" \
  --extra-index-url "${FLASHINFER_CUDA_TAG}"
```

---

## **Step 5: Install Additional CUDA and Numerical Libraries**
```bash
# Install NVIDIA CUDA BLAS library for CUDA 12
uv pip install --upgrade --force-reinstall "nvidia-cublas-cu12==12.9.1.4"

# Install specific version of NumPy
uv pip install --upgrade --force-reinstall "numpy==2.2.6"
```

---

## **Step 6: Install vLLM-Omni (Multimodal Extension)**
```bash
# Clone the vLLM-Omni repository (Note: Original URL had error, corrected to GitHub)
git clone https://github.com/vllm-project/vllm-omni.git

# Navigate to directory and install in editable mode
cd vllm-omni
uv pip install -e .
cd ..

# Downgrade NumPy to a version below 2.0 (required for compatibility)
uv pip install "numpy<2"
```

---

## **Step 7: Start vLLM Server for Image Generation**
```bash
# Start the server with the Z-Image-Turbo model
# --omni: Enable multimodal support
# --host 0.0.0.0: Allow external connections
# --port 8091: Use port 8091
# --gpu_memory_utilization 0.2: Use 20% of GPU memory
# --cpu-offload-gb 40: Offload 40GB to CPU memory
uv run vllm serve Z-Image-Turbo --omni --host 0.0.0.0 --port 8091 --gpu_memory_utilization 0.2 --cpu-offload-gb 40
```

---

## **Step 8: Test Image Generation with cURL**
After starting the server, test from another machine (replace `<server_ip>` with the actual internal IP):

### **Example 1: Generate an image of a dragon**
```bash
curl -X POST http://<server_ip>:8091/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a dragon laying over the spine of the Green Mountains of Vermont",
    "size": "832x480",
    "num_inference_steps": 20,
    "seed": 42
  }' | jq -r '.data[0].b64_json' | base64 -d > dragon.png
```

### **Example 2: Generate an image of coffee (using chat completions API)**
```bash
curl -s http://<server_ip>:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "a cup of coffee on the table"}
    ],
    "extra_body": {
      "height": 480,
      "width": 832,
      "num_inference_steps": 20,
      "guidance_scale": 4.0,
      "seed": 42
    }
  }' | jq -r '.choices[0].message.content[0].image_url.url' | cut -d',' -f2 | base64 -d > coffee_3.png
```

### **Example 3: Generate a Chinese ink landscape painting**
```bash
curl -s http://<server_ip>:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "水墨山水，远山淡影，留白意境"}
    ],
    "extra_body": {
      "height": 480,
      "width": 832,
      "num_inference_steps": 20,
      "guidance_scale": 4.0,
      "seed": 42
    }
  }' | jq -r '.choices[0].message.content[0].image_url.url' | cut -d',' -f2 | base64 -d > landscape_3.png
```

### **Example 4: Generate a portrait based on detailed description**
```bash
curl -s http://<server_ip>:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "王翔戴着深蓝色细框窄边细眼镜，镜片后深邃的眼眸泛着微光，浅灰色高领帽衫垂落的衣角随着他垂眸时的呼吸微微起伏，露出锁骨处若隐若现的皮肤光泽。他指尖无意识地摩挲着帽衫褶皱，腕表带滑过手背时带起一阵若有似无的香气，垂落的发丝扫过锁骨凹陷处，喉结在吞咽动作中泛起微红，睫毛在眼下投出细碎阴影，仿佛随时会从镜片后溢出灼热的凝视。"}
    ],
    "extra_body": {
      "height": 480,
      "width": 832,
      "num_inference_steps": 20,
      "guidance_scale": 4.0,
      "seed": 42
    }
  }' | jq -r '.choices[0].message.content[0].image_url.url' | cut -d',' -f2 | base64 -d > man.png
```

---

## **Step 9: Install jq for JSON Processing**
```bash
# Install jq (command-line JSON processor) if not present
sudo apt-get install jq
```

---

## **Step 10: Set Up Text-to-Speech (TTS) Server**
First, stop the image generation server if running, then start the TTS server:

### **Option A: Using Custom Voice Model**
```bash
# Set environment variables
export FLASHINFER_DISABLE_VERSION_CHECK=1
export HF_ENDPOINT=https://hf-mirror.com

# Start server with Qwen3 TTS CustomVoice model
vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --stage-configs-path vllm-omni/vllm_omni/model_executor/stage_configs/qwen3_tts.yaml \
  --omni --port 8091 --trust-remote-code --enforce-eager
```

### **Option B: Using Base TTS Model (commented in original)**
```bash
export FLASHINFER_DISABLE_VERSION_CHECK=1
export HF_ENDPOINT=https://hf-mirror.com

vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --stage-configs-path vllm-omni/vllm_omni/model_executor/stage_configs/qwen3_tts.yaml \
  --omni \
  --port 8091 \
  --trust-remote-code \
  --enforce-eager
```

### **Test TTS Generation**
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "input": "这是一个有关王翔和小白猫的故事。",
    "task_type": "VoiceDesign",
    "instructions": "man"
  }' --output story.wav
```

---

## **Step 11: Integration with ComfyUI (Optional)**
```bash
# Do NOT clone again (as per instruction "不准" meaning "do not")
# Instead, copy the ComfyUI custom node from the existing vllm-omni clone
cp -r vllm-omni/apps/ComfyUI-vLLM-Omni ComfyUI/custom_nodes
```

---

## **Notes:**
- All commands are preserved exactly as provided in the original text.
- Replace `<server_ip>` with the actual internal IP address of your server. (ifconfig)
- The TTS server uses port 8091, same as the image generation server. Ensure only one is running at a time, or change the port for one of them.
- The `--extra-index-url` in some commands may require a valid URL; adjust as needed based on your package source configuration.
