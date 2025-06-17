#!/bin/bash

# 修正版推理脚本 - 正确处理 verl 原生检查点
# 需要在 AutoDL 服务器上运行

cd /root/autodl-tmp/verl
source verl_env/bin/activate

export HF_ENDPOINT=https://hf-mirror.com
export WANDB_MODE=offline

echo "=== 检查点结构分析 ==="

# 1. 查找检查点
CHECKPOINT_BASE="/root/autodl-tmp/verl"
LATEST_CHECKPOINT=$(find $CHECKPOINT_BASE -name "global_step_*" -type d | sort -V | tail -1)

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "❌ 未找到检查点，使用基础模型"
    MODEL_PATH="Qwen/Qwen2.5-0.5B-Instruct"
    USE_CHECKPOINT=false
else
    echo "✅ 找到检查点: $LATEST_CHECKPOINT"
    
    # 2. 详细查看检查点结构
    echo "检查点目录结构:"
    ls -la "$LATEST_CHECKPOINT" | sed 's/^/  /'
    
    if [ -d "$LATEST_CHECKPOINT/actor" ]; then
        echo "Actor 目录结构:"
        ls -la "$LATEST_CHECKPOINT/actor" | sed 's/^/    /'
    fi
    
    # 3. 检查不同的可能路径
    POSSIBLE_PATHS=(
        "$LATEST_CHECKPOINT/actor/huggingface"
        "$LATEST_CHECKPOINT/actor"
        "$LATEST_CHECKPOINT/huggingface"
        "$LATEST_CHECKPOINT"
    )
    
    MODEL_PATH=""
    USE_CHECKPOINT=false
    
    for path in "${POSSIBLE_PATHS[@]}"; do
        if [ -f "$path/config.json" ] && [ -f "$path/pytorch_model.bin" -o -f "$path/model.safetensors" ]; then
            echo "✅ 找到 HuggingFace 格式模型: $path"
            MODEL_PATH="$path"
            USE_CHECKPOINT=true
            break
        elif [ -f "$path/config.json" ]; then
            echo "⚠️  找到配置文件但缺少模型文件: $path"
        fi
    done
    
    if [ "$USE_CHECKPOINT" = false ]; then
        echo "❌ 未找到可用的 HuggingFace 格式模型"
        echo "📝 检查点内容详情:"
        find "$LATEST_CHECKPOINT" -type f -name "*.json" -o -name "*.bin" -o -name "*.safetensors" | head -10 | sed 's/^/    /'
        
        # 4. 尝试使用 verl 原生推理
        echo ""
        echo "🔧 尝试使用 verl 原生检查点格式进行推理..."
        MODEL_PATH="$LATEST_CHECKPOINT"
        USE_VERL_NATIVE=true
    fi
fi

echo "最终使用模型路径: $MODEL_PATH"
echo "使用检查点: $USE_CHECKPOINT"
echo ""

# 5. 创建测试数据
cat > /tmp/test_prompts.json << 'EOF'
[
    [{"role": "user", "content": "Solve this step by step: If a train travels 120 miles in 2 hours, what is its average speed?"}],
    [{"role": "user", "content": "Calculate: What is 15% of 240?"}],
    [{"role": "user", "content": "A rectangle has length 8 cm and width 5 cm. What is its area and perimeter?"}],
    [{"role": "user", "content": "If x + 5 = 12, what is the value of x?"}],
    [{"role": "user", "content": "Sarah has 24 apples. She gives away 1/3 of them. How many apples does she have left?"}]
]
EOF

python3 << 'EOF'
import json
import pandas as pd

with open('/tmp/test_prompts.json', 'r') as f:
    prompts = json.load(f)

df = pd.DataFrame({'prompt': prompts})
df.to_parquet('/tmp/test_data.parquet', index=False)
print(f"Created test dataset with {len(df)} samples")
EOF

OUTPUT_PATH="/root/autodl-tmp/verl/inference_results.parquet"

# 6. 根据检查点类型选择推理方式
if [ "$USE_CHECKPOINT" = true ]; then
    echo "🚀 使用 HuggingFace 格式检查点进行推理..."
    
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

elif [ -n "$USE_VERL_NATIVE" ]; then
    echo "🔧 尝试使用 verl 原生格式..."
    echo "⚠️  这需要特殊的加载方式，暂时使用基础模型进行对比测试"
    
    # 使用基础模型作为对比
    python3 -m verl.trainer.main_generation \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node=1 \
        data.path=/tmp/test_data.parquet \
        data.prompt_key=prompt \
        data.n_samples=1 \
        data.batch_size=5 \
        data.output_path=$OUTPUT_PATH \
        model.path="Qwen/Qwen2.5-0.5B-Instruct" \
        rollout.name=vllm \
        rollout.temperature=0.7 \
        rollout.top_k=-1 \
        rollout.top_p=0.9 \
        rollout.prompt_length=512 \
        rollout.response_length=256 \
        rollout.tensor_model_parallel_size=1 \
        rollout.gpu_memory_utilization=0.8
        
    echo ""
    echo "📋 下面是转换检查点为 HuggingFace 格式的方法:"
    echo "1. 查看 verl 训练配置中的 trainer.save_hf_model 选项"
    echo "2. 或者运行专门的转换脚本"
    
else
    echo "🎯 使用基础模型进行推理..."
    
    python3 -m verl.trainer.main_generation \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node=1 \
        data.path=/tmp/test_data.parquet \
        data.prompt_key=prompt \
        data.n_samples=1 \
        data.batch_size=5 \
        data.output_path=$OUTPUT_PATH \
        model.path="Qwen/Qwen2.5-0.5B-Instruct" \
        rollout.name=vllm \
        rollout.temperature=0.7 \
        rollout.top_k=-1 \
        rollout.top_p=0.9 \
        rollout.prompt_length=512 \
        rollout.response_length=256 \
        rollout.tensor_model_parallel_size=1 \
        rollout.gpu_memory_utilization=0.8
fi

echo ""
echo "=== 推理完成 ==="

if [ -f "$OUTPUT_PATH" ]; then
    echo "✅ 结果保存到: $OUTPUT_PATH"
    
    python3 << EOF
import pandas as pd
import json

df = pd.read_parquet('$OUTPUT_PATH')

print("=== 推理结果 ===")
for i, row in df.iterrows():
    print(f"\n问题 {i+1}:")
    print(f"Q: {row['prompt'][0]['content']}")
    print(f"A: {row['responses'][0]}")
    print("-" * 50)

# 保存为 JSON
results = []
for i, row in df.iterrows():
    results.append({
        "question": row['prompt'][0]['content'],
        "answer": row['responses'][0]
    })

with open('/root/autodl-tmp/verl/inference_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n结果已保存:")
print(f"- Parquet: $OUTPUT_PATH")
print(f"- JSON: /root/autodl-tmp/verl/inference_results.json")
EOF
else
    echo "❌ 推理失败，未找到输出文件"
fi

# 清理
rm -f /tmp/test_prompts.json /tmp/test_data.parquet

echo ""
echo "=== 总结 ==="
if [ "$USE_CHECKPOINT" = true ]; then
    echo "✅ 成功使用训练后的检查点进行推理"
elif [ -n "$USE_VERL_NATIVE" ]; then
    echo "⚠️  检查点存在但需要转换为 HuggingFace 格式"
    echo "💡 建议：重新训练时设置保存 HuggingFace 格式"
else
    echo "ℹ️  使用基础模型进行推理（未找到可用检查点）"
fi 