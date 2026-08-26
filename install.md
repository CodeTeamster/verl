## 1. Conda 环境与国内源
VLLM 版本：
```bash
conda create -n verl-vllm python=3.12 -y
conda activate verl-vllm
pip install --upgrade pip
```

SGLang 版本：
```bash
conda create -n verl-sglang python=3.12 -y
conda activate verl-sglang
pip install --upgrade pip
```

## 2. 项目依赖
VLLM 版本：
```bash
pip install -r requirements-vllm.txt
pip install -e . --no-deps
```

SGLang 版本：
```bash
pip install -r requirements-sglang.txt
pip install -e . --no-deps
```

## 3. Torch 与 FlashAttention
验证 CUDA 与 ABI：
```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.compiled_with_cxx11_abi())
PY
```

安装对应版本 FlashAttention2：
```bash
pip install --no-deps \
  'https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl'
```

## 4. 测试安装
VLLM 版本：
```bash
INFER_BACKEND=vllm bash ./lab/test_qwen2_5_1_5b_fsdp.sh
```

SGLang 版本：
```bash
INFER_BACKEND=sglang bash ./lab/test_qwen2_5_1_5b_fsdp.sh
```