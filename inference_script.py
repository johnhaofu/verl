#!/usr/bin/env python3
"""
使用训练好的 verl PPO 模型进行推理预测
"""
import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_trained_model(checkpoint_path, device="cuda"):
    """
    加载训练好的模型
    
    Args:
        checkpoint_path: 检查点路径，可以是：
                        1. HuggingFace格式的模型路径
                        2. verl训练保存的检查点路径
    """
    print(f"Loading model from {checkpoint_path}")
    
    # 尝试从HuggingFace格式加载
    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        print("Successfully loaded HuggingFace format model")
        return model, tokenizer
    except Exception as e:
        print(f"Failed to load as HuggingFace model: {e}")
        
    # 如果是verl checkpoint，需要转换
    # 这里假设检查点包含huggingface子目录
    hf_path = os.path.join(checkpoint_path, "huggingface")
    if os.path.exists(hf_path):
        tokenizer = AutoTokenizer.from_pretrained(hf_path)
        model = AutoModelForCausalLM.from_pretrained(
            hf_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        print("Successfully loaded converted HuggingFace model")
        return model, tokenizer
    
    raise ValueError(f"Cannot load model from {checkpoint_path}")

def prepare_math_prompts():
    """
    准备数学问题测试用例
    """
    test_prompts = [
        "Solve this step by step: If a train travels 120 miles in 2 hours, what is its average speed?",
        "Calculate: What is 15% of 240?",
        "A rectangle has length 8 cm and width 5 cm. What is its area and perimeter?",
        "If x + 5 = 12, what is the value of x?",
        "Sarah has 24 apples. She gives away 1/3 of them. How many apples does she have left?"
    ]
    
    # 转换为chat格式
    chat_prompts = []
    for prompt in test_prompts:
        chat_prompts.append([
            {"role": "user", "content": prompt}
        ])
    
    return chat_prompts

@torch.no_grad()
def generate_responses(model, tokenizer, prompts, max_length=512, temperature=0.7, top_p=0.9):
    """
    生成回复
    """
    responses = []
    
    for prompt in prompts:
        # 应用chat template
        input_text = tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            tokenize=False
        )
        
        # 编码输入
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # 生成回复
        output = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        
        # 解码回复
        response = tokenizer.decode(
            output[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        responses.append({
            "prompt": prompt[0]["content"],
            "response": response.strip()
        })
        
        print(f"Q: {prompt[0]['content']}")
        print(f"A: {response.strip()}")
        print("-" * 50)
    
    return responses

def main():
    # 配置
    # 修改这个路径为你的检查点路径
    CHECKPOINT_PATH = "/root/autodl-tmp/verl/outputs/global_step_XXX/actor/huggingface"  # 需要替换实际路径
    
    # 如果没有训练检查点，使用基础模型
    if not os.path.exists(CHECKPOINT_PATH):
        print("Checkpoint not found, using base model...")
        CHECKPOINT_PATH = "Qwen/Qwen2.5-0.5B-Instruct"
    
    # 加载模型
    model, tokenizer = load_trained_model(CHECKPOINT_PATH)
    
    # 准备测试prompts
    test_prompts = prepare_math_prompts()
    
    # 生成回复
    print("Starting inference...")
    responses = generate_responses(
        model, 
        tokenizer, 
        test_prompts,
        max_length=256,
        temperature=0.7,
        top_p=0.9
    )
    
    # 保存结果
    df = pd.DataFrame(responses)
    df.to_csv("inference_results.csv", index=False)
    df.to_json("inference_results.json", orient="records", indent=2)
    
    print(f"Inference completed! Results saved to inference_results.csv and inference_results.json")
    print(f"Generated {len(responses)} responses")

if __name__ == "__main__":
    main() 