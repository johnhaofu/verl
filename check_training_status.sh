#!/bin/bash

# 检查训练状态和寻找可用检查点的脚本
# 在 AutoDL 服务器上运行

echo "=== 检查训练状态和可用检查点 ==="
echo

# 1. 检查训练进程
echo "1. 检查训练进程是否还在运行:"
TRAINING_PID=$(pgrep -f "verl.trainer.main_ppo")
if [ -n "$TRAINING_PID" ]; then
    echo "   ✅ 训练进程还在运行 (PID: $TRAINING_PID)"
    echo "   使用 'ps aux | grep verl.trainer.main_ppo' 查看详细信息"
else
    echo "   ❌ 训练进程已结束"
fi
echo

# 2. 检查日志文件
echo "2. 检查训练日志:"
if [ -f "/root/autodl-tmp/verl/verl_demo.log" ]; then
    echo "   ✅ 找到日志文件: /root/autodl-tmp/verl/verl_demo.log"
    echo "   最后几行日志:"
    tail -n 5 /root/autodl-tmp/verl/verl_demo.log | sed 's/^/      /'
else
    echo "   ❌ 未找到日志文件"
fi
echo

# 3. 寻找检查点目录
echo "3. 寻找训练检查点:"
CHECKPOINT_DIRS=(
    "/root/autodl-tmp/verl/outputs"
    "/root/autodl-tmp/verl/checkpoints"
    "/root/autodl-tmp/verl"
    "/root/data"
    "/root"
)

FOUND_CHECKPOINTS=()

for dir in "${CHECKPOINT_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        # 查找 global_step_ 目录
        while IFS= read -r -d '' checkpoint; do
            FOUND_CHECKPOINTS+=("$checkpoint")
        done < <(find "$dir" -name "global_step_*" -type d -print0 2>/dev/null)
    fi
done

if [ ${#FOUND_CHECKPOINTS[@]} -eq 0 ]; then
    echo "   ❌ 未找到任何训练检查点"
    echo "   提示: 检查点通常保存在以下位置之一:"
    echo "        - /root/autodl-tmp/verl/outputs/global_step_XXX/"
    echo "        - /root/autodl-tmp/verl/checkpoints/global_step_XXX/"
else
    echo "   ✅ 找到 ${#FOUND_CHECKPOINTS[@]} 个检查点:"
    
    # 按时间排序并显示
    for checkpoint in $(printf '%s\n' "${FOUND_CHECKPOINTS[@]}" | sort -V); do
        STEP_NUM=$(basename "$checkpoint" | sed 's/global_step_//')
        echo "      - $checkpoint (步数: $STEP_NUM)"
        
        # 检查子目录结构
        if [ -d "$checkpoint/actor" ]; then
            echo "        └── actor/ (演员模型)"
            if [ -d "$checkpoint/actor/huggingface" ]; then
                echo "            └── huggingface/ ✅ (可用于推理)"
            fi
        fi
        if [ -d "$checkpoint/critic" ]; then
            echo "        └── critic/ (评论员模型)"
        fi
    done
    
    # 推荐最新检查点
    LATEST_CHECKPOINT=$(printf '%s\n' "${FOUND_CHECKPOINTS[@]}" | sort -V | tail -1)
    echo
    echo "   📌 推荐使用最新检查点: $LATEST_CHECKPOINT"
fi
echo

# 4. 检查可用的推理方式
echo "4. 可用的推理方式:"
echo "   方式1: 使用 verl 原生推理框架"
echo "         ./verl_inference.sh"
echo
echo "   方式2: 使用简单 Python 脚本"
echo "         python3 inference_script.py"
echo
echo "   方式3: 使用基础模型（如果没有检查点）"
echo "         从 Qwen/Qwen2.5-0.5B-Instruct 开始"
echo

# 5. 检查环境状态
echo "5. 环境检查:"
if [ -f "/root/autodl-tmp/verl/verl_env/bin/activate" ]; then
    echo "   ✅ 虚拟环境可用: /root/autodl-tmp/verl/verl_env/"
else
    echo "   ❌ 虚拟环境未找到"
fi

# 检查 GPU 状态
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "   GPU 状态:"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | sed 's/^/      GPU /'
else
    echo "   ❌ nvidia-smi 不可用"
fi
echo

echo "=== 检查完成 ==="
echo
echo "💡 下一步操作建议:"
if [ ${#FOUND_CHECKPOINTS[@]} -gt 0 ]; then
    echo "   1. 使用 ./verl_inference.sh 进行推理"
    echo "   2. 或者修改 inference_script.py 中的 CHECKPOINT_PATH"
else
    echo "   1. 确认训练是否完成或设置了正确的保存路径"
    echo "   2. 如果需要，可以使用基础模型进行推理"
fi 