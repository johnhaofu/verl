#!/bin/bash

echo "=== Python环境修复脚本 ==="

# 检查当前Python版本
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '(?<=Python )\d+\.\d+')
echo "当前Python版本: $PYTHON_VERSION"

if [[ "$PYTHON_VERSION" > "3.11" ]]; then
    echo "警告: Python 3.12+ 可能与某些包不兼容"
    echo "建议使用Python 3.10或3.11"
    
    # 检查是否有其他Python版本可用
    if command -v python3.10 &> /dev/null; then
        echo "发现Python 3.10，建议使用: python3.10 -m venv verl_env"
    elif command -v python3.11 &> /dev/null; then
        echo "发现Python 3.11，建议使用: python3.11 -m venv verl_env"
    else
        echo "建议安装Python 3.10: apt install python3.10 python3.10-venv"
    fi
fi

echo "如果要继续使用当前Python版本，pyext包会被跳过。"
echo "这通常不会影响verl的核心功能。" 