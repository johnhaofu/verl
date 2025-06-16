#!/bin/bash

USE_MEGATRON=${USE_MEGATRON:-1}
USE_SGLANG=${USE_SGLANG:-1}

export MAX_JOBS=32

echo "=== verl手动安装脚本 ==="
echo "USE_MEGATRON: $USE_MEGATRON"
echo "USE_SGLANG: $USE_SGLANG"

echo "1. install inference frameworks and pytorch they need"
if [ $USE_SGLANG -eq 1 ]; then
    echo "安装SGLang..."
    pip install "sglang[all]==0.4.6.post1" --no-cache-dir --find-links https://flashinfer.ai/whl/cu124/torch2.6/flashinfer-python && pip install torch-memory-saver --no-cache-dir
fi

echo "安装vLLM和PyTorch..."
pip install --no-cache-dir "vllm==0.8.5.post1" "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" "tensordict==0.6.2" torchdata

echo "2. install basic packages"
echo "安装基础Python包..."
pip install "transformers[hf_xet]>=4.51.0" accelerate datasets peft hf-transfer \
    "numpy<2.0.0" "pyarrow>=15.0.0" pandas \
    ray[default] codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler \
    pytest py-spy pyext pre-commit ruff

pip install "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1"

echo "3. install FlashAttention and FlashInfer (手动模式)"
echo "注意：请确保您已经手动下载了以下wheel文件到当前目录："
echo "  - FlashAttention: flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
echo "  - FlashInfer: flashinfer_python-0.2.2.post1+cu124torch2.6-cp38-abi3-linux_x86_64.whl"

# 检查FlashAttention wheel文件
if [ -f "flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl" ]; then
    echo "找到FlashAttention wheel文件，开始安装..."
    pip install --no-cache-dir flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
else
    echo "错误：未找到FlashAttention wheel文件"
    echo "请手动下载到当前目录："
    echo "下载链接：https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
    echo "或使用命令：wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
    exit 1
fi

# 检查FlashInfer wheel文件
if [ -f "flashinfer_python-0.2.2.post1+cu124torch2.6-cp38-abi3-linux_x86_64.whl" ]; then
    echo "找到FlashInfer wheel文件，开始安装..."
    pip install --no-cache-dir flashinfer_python-0.2.2.post1+cu124torch2.6-cp38-abi3-linux_x86_64.whl
else
    echo "错误：未找到FlashInfer wheel文件"
    echo "请手动下载到当前目录："
    echo "下载链接：https://github.com/flashinfer-ai/flashinfer/releases/download/v0.2.2.post1/flashinfer_python-0.2.2.post1+cu124torch2.6-cp38-abi3-linux_x86_64.whl"
    echo "或使用命令：wget https://github.com/flashinfer-ai/flashinfer/releases/download/v0.2.2.post1/flashinfer_python-0.2.2.post1+cu124torch2.6-cp38-abi3-linux_x86_64.whl"
    exit 1
fi

if [ $USE_MEGATRON -eq 1 ]; then
    echo "4. install TransformerEngine and Megatron (手动模式)"
    echo "注意：请确保您已经手动下载了以下仓库："
    echo "  - TransformerEngine: https://github.com/NVIDIA/TransformerEngine/archive/refs/tags/v2.2.tar.gz"
    echo "  - Megatron-LM: https://github.com/NVIDIA/Megatron-LM/archive/refs/tags/core_v0.12.0rc3.tar.gz"
    
    # 检查手动下载的目录
    if [ -d "TransformerEngine-2.2" ] || [ -d "TransformerEngine" ]; then
        echo "找到TransformerEngine目录，开始安装..."
        if [ -d "TransformerEngine-2.2" ]; then
            cd TransformerEngine-2.2
        else
            cd TransformerEngine
        fi
        echo "安装TransformerEngine中，这可能需要很长时间..."
        NVTE_FRAMEWORK=pytorch pip install --no-deps -v .
        cd ..
    else
        echo "错误：未找到TransformerEngine目录"
        echo "请下载并解压 TransformerEngine v2.2 到当前目录"
        echo "下载命令：wget https://github.com/NVIDIA/TransformerEngine/archive/refs/tags/v2.2.tar.gz"
        echo "解压命令：tar -xzf v2.2.tar.gz"
        exit 1
    fi
    
    if [ -d "Megatron-LM-core_v0.12.0rc3" ] || [ -d "Megatron-LM" ]; then
        echo "找到Megatron-LM目录，开始安装..."
        if [ -d "Megatron-LM-core_v0.12.0rc3" ]; then
            cd Megatron-LM-core_v0.12.0rc3
        else
            cd Megatron-LM
        fi
        pip install --no-deps .
        cd ..
    else
        echo "错误：未找到Megatron-LM目录"
        echo "请下载并解压 Megatron-LM core_v0.12.0rc3 到当前目录"
        echo "下载命令：wget https://github.com/NVIDIA/Megatron-LM/archive/refs/tags/core_v0.12.0rc3.tar.gz"
        echo "解压命令：tar -xzf core_v0.12.0rc3.tar.gz"
        exit 1
    fi
fi

echo "5. May need to fix opencv"
pip install opencv-python
pip install opencv-fixer && \
    python -c "from opencv_fixer import AutoFix; AutoFix()"

if [ $USE_MEGATRON -eq 1 ]; then
    echo "6. Install cudnn python package (avoid being overridden)"
    pip install nvidia-cudnn-cu12==9.8.0.87
fi

echo "=== 安装完成 ==="
echo "请运行以下命令验证安装："
echo "python -c \"import verl; print('verl安装成功!')\""
echo "python -c \"import torch; print(f'CUDA可用: {torch.cuda.is_available()}')\"" 