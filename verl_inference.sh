#!/bin/bash

# 使用 verl 的 main_generation 进行推理
# 需要在 AutoDL 服务器上运行

# 进入工作目录并激活环境
cd /root/autodl-tmp/verl
source verl_env/bin/activate

# 设置环境变量
export HF_ENDPOINT=https://hf-mirror.com
export WANDB_MODE=offline

# 找到最新的检查点
CHECKPOINT_BASE="/root/autodl-tmp/verl/outputs"
LATEST_CHECKPOINT=$(find $CHECKPOINT_BASE -name "global_step_*" -type d | sort -V | tail -1)

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "No checkpoint found, using base model..."
    MODEL_PATH="Qwen/Qwen2.5-0.5B-Instruct"
else
    echo "Found checkpoint: $LATEST_CHECKPOINT"
    # 检查是否有 huggingface 子目录
    if [ -d "$LATEST_CHECKPOINT/actor/huggingface" ]; then
        MODEL_PATH="$LATEST_CHECKPOINT/actor/huggingface"
        echo "Using HuggingFace format checkpoint: $MODEL_PATH"
    else
        echo "No HuggingFace format found, using base model..."
        MODEL_PATH="Qwen/Qwen2.5-0.5B-Instruct"
    fi
fi

# 创建测试数据文件
cat > /tmp/test_prompts.json << 'EOF'
[
    [{"role": "user", "content": "Solve this step by step: If a train travels 120 miles in 2 hours, what is its average speed?"}],
    [{"role": "user", "content": "Calculate: What is 15% of 240?"}],
    [{"role": "user", "content": "A rectangle has length 8 cm and width 5 cm. What is its area and perimeter?"}],
    [{"role": "user", "content": "If x + 5 = 12, what is the value of x?"}],
    [{"role": "user", "content": "Sarah has 24 apples. She gives away 1/3 of them. How many apples does she have left?"}]
]
EOF

# 转换 JSON 为 parquet 格式
python3 << 'EOF'
import json
import pandas as pd
import numpy as np

# 读取 JSON 数据
with open('/tmp/test_prompts.json', 'r') as f:
    prompts = json.load(f)

# 转换为 DataFrame
df = pd.DataFrame({
    'prompt': prompts  # 直接使用 chat 格式
})

# 保存为 parquet
df.to_parquet('/tmp/test_data.parquet', index=False)
print(f"Created test dataset with {len(df)} samples")
EOF

# 设置输出路径
OUTPUT_PATH="/root/autodl-tmp/verl/inference_results.parquet"

echo "Starting inference with model: $MODEL_PATH"
echo "Output will be saved to: $OUTPUT_PATH"

# 运行推理
python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=1 \
    data.path=/tmp/test_data.parquet \
    data.prompt_key=prompt \
    data.n_samples=1 \
    data.batch_size=5 \
    data.output_path=$OUTPUT_PATH \
    model.path=$MODEL_PATH \
    rollout.name=vllm \
    rollout.temperature=0.7 \
    rollout.top_k=-1 \
    rollout.top_p=0.9 \
    rollout.prompt_length=512 \
    rollout.response_length=256 \
    rollout.tensor_model_parallel_size=1 \
    rollout.gpu_memory_utilization=0.8

echo "Inference completed!"

# 查看结果
if [ -f "$OUTPUT_PATH" ]; then
    echo "Results saved to: $OUTPUT_PATH"
    
    # 使用 Python 显示结果
    python3 << EOF
import pandas as pd
import json

# 读取结果
df = pd.read_parquet('$OUTPUT_PATH')

print("=== 推理结果 ===")
for i, row in df.iterrows():
    print(f"\n问题 {i+1}:")
    print(f"Q: {row['prompt'][0]['content']}")
    print(f"A: {row['responses'][0]}")
    print("-" * 50)

# 保存为更易读的格式
results = []
for i, row in df.iterrows():
    results.append({
        "question": row['prompt'][0]['content'],
        "answer": row['responses'][0]
    })

with open('/root/autodl-tmp/verl/inference_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n结果已保存到:")
print(f"- Parquet: $OUTPUT_PATH")
print(f"- JSON: /root/autodl-tmp/verl/inference_results.json")
EOF
else
    echo "Error: Output file not found!"
fi

# 清理临时文件
rm -f /tmp/test_prompts.json /tmp/test_data.parquet

echo "Inference script completed!" 