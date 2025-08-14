#!/usr/bin/env python3
"""
Qwen3 模型微调示例
使用 4bit 量化 + 混合精度 + LoRA 微调
"""

import torch
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import numpy as np
import os
#Qwen/Qwen3-4B-Thinking-2507
#Qwen/Qwen3-4B-Instruct-2507
def setup_model_and_tokenizer(model_name="Qwen/Qwen3-4B-Thinking-2507"):
    """设置模型和分词器 - 针对显存深度优化"""
    print(f"🔄 正在加载模型: {model_name}")
    
    # 清理GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # 4bit 量化配置 - 针对16G显存深度优化
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # 4-bit NormalFloat，最佳精度
        bnb_4bit_compute_dtype=torch.bfloat16,  # 计算时使用 BF16，精度更好
        bnb_4bit_use_double_quant=True,  # 双重量化节省显存
        llm_int8_threshold=6.0,
        
        # 额外的量化优化
        bnb_4bit_storage_dtype=torch.uint8,  # 存储时使用8位
        load_in_8bit=False,  # 不使用8bit量化，4bit更节省显存
        
        # 新增OOM防护参数
        bnb_4bit_quant_storage=torch.uint8,  # 量化存储使用uint8
        quant_method="bitsandbytes",  # 明确指定量化方法
    )
    
    # 加载模型 - 添加OOM防护
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",  # 自动设备映射
        trust_remote_code=True,
        low_cpu_mem_usage=True,  # 低CPU内存使用
        torch_dtype=torch.bfloat16,  # 指定数据类型
    )
    
    # 准备模型用于 kbit 训练 - 添加OOM防护
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    
    # 设置 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("✅ 模型和分词器加载完成")
    return model, tokenizer

def setup_lora(model):
    """设置 LoRA 适配器 - 针对显存深度优化"""
    print("🔄 正在配置 LoRA 适配器（深度显存优化版）")
    
    # 清理GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # LoRA 配置 - 针对16G显存深度优化
    lora_config = LoraConfig(
        r=6,  # 进一步减小LoRA矩阵的秩以节省显存（从8降到4）
        lora_alpha=12,  # 相应调整缩放因子（alpha = 2*r）
        target_modules=[  # 精选目标模块，最小化显存占用
            "q_proj", "v_proj", "ffn.w1"  # 只对最重要的注意力模块应用LoRA
        ],
        lora_dropout=0.1,  # 降低dropout以减少计算开销
        bias="none",  # 不训练偏置以节省参数
        task_type="CAUSAL_LM",  # 因果语言模型任务
        
        # 额外的优化参数
        modules_to_save=[],  # 不保存额外模块以节省显存
        use_rslora=True,  # 使用Rank-Stabilized LoRA提高训练稳定性
        loftq_config=None,  # 不使用LoFTQ以节省计算资源
        
        # 新增OOM防护参数
        init_lora_weights="gaussian",  # 使用高斯初始化
        fan_in_fan_out=False,  # 禁用fan-in fan-out以节省计算
    )
    
    # 应用 LoRA 适配器 - 添加OOM防护
    model = get_peft_model(model, lora_config)
    
    # 启用梯度检查点以进一步节省显存
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    
    # 设置模型为训练模式
    model.train()
    
    # 打印可训练参数比例
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_percentage = 100 * trainable_params / total_params
    
    print(f"✅ LoRA 配置完成")
    print(f"   - 可训练参数: {trainable_params:,}")
    print(f"   - 总参数: {total_params:,}")
    print(f"   - 可训练比例: {trainable_percentage:.2f}%")
    
    return model

def load_qa_data_from_files():
    """从 data 目录加载 QA 数据"""
    print("🔄 正在从 data 目录加载 QA 数据")
    
    qa_data = []
    
    # 定义数据文件路径
    data_files = [
        "data/raw/39786qa.md",
        "data/raw/qA.md"
    ]
    
    for file_path in data_files:
        if os.path.exists(file_path):
            print(f"📖 正在读取文件: {file_path}")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # 解析 QA 数据
                if file_path == "data/raw/39786qa.md":
                    # 解析 39786qa.md 格式 (Q1: ... A1: ...)
                    import re
                    qa_pairs = re.findall(r'Q(\d+):\s*(.*?)\s*A\1:\s*(.*?)(?=Q\d+:|$)', content, re.DOTALL)
                    for q_num, question, answer in qa_pairs:
                        question = question.strip()
                        answer = answer.strip()
                        if question and answer:
                            qa_data.append({
                                "instruction": question,
                                "input": "",
                                "output": answer
                            })
                elif file_path == "data/raw/qA.md":
                    # 解析 qA.md 格式 (### Q1: ... A1: ...)
                    import re
                    qa_pairs = re.findall(r'###\s*Q(\d+):\s*(.*?)\s*A\1:\s*(.*?)(?=###\s*Q\d+:|$)', content, re.DOTALL)
                    for q_num, question, answer in qa_pairs:
                        question = question.strip()
                        answer = answer.strip()
                        if question and answer:
                            qa_data.append({
                                "instruction": question,
                                "input": "",
                                "output": answer
                            })
                
                print(f"✅ 从 {file_path} 成功解析 {len([q for q in qa_pairs if q[1].strip() and q[2].strip()])} 条 QA 数据")
                
            except Exception as e:
                print(f"⚠️  读取文件 {file_path} 时出错: {e}")
        else:
            print(f"⚠️  文件不存在: {file_path}")
    
    print(f"📊 总共加载了 {len(qa_data)} 条 QA 数据")
    return qa_data

def prepare_dataset(tokenizer, dataset_path=None, max_length=256):
    """准备训练数据集 - 针对内存深度优化"""
    print("🔄 正在准备数据集（深度内存优化版）")
    
    # 清理内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 优先从 data 目录加载 QA 数据
    qa_data = load_qa_data_from_files()
    
    if qa_data:
        # 使用加载的 QA 数据
        from datasets import Dataset
        dataset = Dataset.from_list(qa_data)
        print("✅ 使用从 data 目录加载的 QA 数据")
    else:
        # 如果没有找到 QA 数据，创建示例数据
        print("⚠️  未找到 QA 数据，使用示例数据")
        sample_data = [
            {
                "instruction": "介绍一下人工智能",
                "input": "",
                "output": "人工智能（AI）是计算机科学分支，创建执行人类智能任务的系统。"
            },
            {
                "instruction": "什么是机器学习？",
                "input": "",
                "output": "机器学习是AI子领域，让计算机从数据中学习和改进。"
            },
            {
                "instruction": "解释深度学习",
                "input": "",
                "output": "深度学习使用多层神经网络模拟人脑，自动学习特征表示。"
            },
            {
                "instruction": "什么是神经网络？",
                "input": "",
                "output": "神经网络是模仿生物神经网络的数学模型，通过调整权重学习关系。"
            },
            {
                "instruction": "简述自然语言处理",
                "input": "",
                "output": "NLP是AI分支，专注计算机与人类语言交互，包括文本分析和机器翻译。"
            }
        ]
        
        # 转换为 Hugging Face 数据集格式
        from datasets import Dataset
        dataset = Dataset.from_list(sample_data)
    
    # 如果提供了外部数据集路径，则加载外部数据集
    if dataset_path is not None:
        external_dataset = load_dataset(dataset_path, split="train")
        dataset = dataset.concatenate(external_dataset)
        print(f"✅ 合并外部数据集，总共 {len(dataset)} 条样本")
    
    # 格式化数据为 Qwen 格式 - 简化格式以减少token数量
    def format_example(example):
        """格式化单个示例"""
        instruction = example["instruction"]
        input_text = example["input"]
        output = example["output"]
        
        # 构建简化的 Qwen 格式对话
        if input_text:
            prompt = f"用户: {instruction} {input_text}\n助手: {output}"
        else:
            prompt = f"用户: {instruction}\n助手: {output}"
        
        return {"text": prompt}
    
    # 应用格式化
    formatted_dataset = dataset.map(format_example)
    
    # 分词函数 - 深度优化内存使用
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=False,  # 不在分词时填充，在数据整理器中处理
            max_length=max_length,  # 进一步减小最大长度以节省显存（从384降到256）
            return_tensors=None
        )
    
    # 应用分词 - 使用更小的批次大小以减少内存使用
    tokenized_dataset = formatted_dataset.map(
        tokenize_function,
        batched=True,
        batch_size=4,  # 进一步减小批次大小以节省内存
        remove_columns=formatted_dataset.column_names,
        num_proc=1,  # 使用单进程以避免内存问题
    )
    
    # 过滤过短的样本以提高训练质量
    def filter_short_samples(example):
        """过滤过短的样本"""
        return len(example["input_ids"]) >= 16  # 最小长度为16个token
    
    filtered_dataset = tokenized_dataset.filter(filter_short_samples)
    
    print(f"✅ 数据集准备完成，共 {len(filtered_dataset)} 条样本（过滤后）")
    return filtered_dataset

def setup_training_args(output_dir="./qwen3-finetuned"):
    """设置训练参数 - 针对16G显存深度优化"""
    print("🔄 正在配置训练参数（深度显存优化版）")
    
    # 清理GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    training_args = TrainingArguments(
        # 基本参数 - 深度优化批次大小以适应16G显存
        output_dir=output_dir,
        per_device_train_batch_size=4,  # 最小批次大小以节省显存
        gradient_accumulation_steps=16,  # 增加梯度累积步数以保持有效批次大小
        learning_rate=5e-5,  # 进一步降低学习率以提高稳定性
        num_train_epochs=100,  # 增加训练轮次以提升模型效果（从2轮增加到5轮）
        
        # 优化器参数 - 使用最内存高效的优化器
        optim="paged_adamw_8bit",  # 8bit Adam 优化器，节省显存
        weight_decay=0.01,
        warmup_ratio=0.2,  # 增加预热比例以提高稳定性
        
        # 混合精度 - 优先使用BF16如果支持，否则使用FP16
        fp16=False,  # 关闭FP16
        bf16=True,   # 优先使用BF16，精度更好且在某些GPU上更快
        
        # 保存和日志 - 大幅减少保存频率以减少IO开销和显存使用
        logging_steps=10,  # 进一步减少日志频率
        save_steps=10,    # 进一步减少保存频率
        save_total_limit=2,  # 只保留最新的1个检查点以节省磁盘空间
        logging_dir="./logs",
        
        # 评估参数
        eval_strategy="no",
        
        # 其他参数
        report_to="tensorboard",
        run_name="qwen3-lora-finetuning-deep-optimized",
        
        # 内存优化 - 启用所有可用的内存优化技术
        gradient_checkpointing=True,  # 梯度检查点，节省显存但增加计算时间
        ddp_find_unused_parameters=False,
        
        # 数据加载优化 - 进一步优化以减少内存使用
        dataloader_num_workers=4,  # 减少工作线程以节省内存
        dataloader_pin_memory=True,  # 固定内存以加速数据传输
        
        # 内存和性能优化
        per_device_eval_batch_size=2,  # 评估时也使用小批次
        max_grad_norm=0.5,  # 降低梯度裁剪阈值以提高稳定性
        seed=42,  # 设置随机种子以保证可重复性
        
        # 额外的内存优化
        fp16_opt_level="O1",  # 混合精度优化级别
        tf32=True,  # 如果GPU支持，使用TF32精度
        
        # 新增OOM防护参数
        dataloader_drop_last=True,  # 丢弃最后一个不完整的批次以避免内存问题
        remove_unused_columns=True,  # 移除未使用的列以节省内存
        push_to_hub=False,  # 不推送到hub以节省网络和内存
        local_rank=-1,  # 单GPU训练
        fp16_full_eval=False,  # 评估时不使用FP16以节省显存
    )
    
    print("✅ 训练参数配置完成（深度显存优化版）")
    return training_args

def train_model(model, tokenizer, tokenized_dataset, training_args):
    """训练模型"""
    print("🚀 开始训练模型")
    
    # 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # 不使用掩码语言建模
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,  # 使用 processing_class 替代已弃用的 tokenizer
    )
    
    # 开始训练
    print("📊 训练开始...")
    trainer.train()
    
    print("✅ 训练完成")
    return trainer

def save_model(trainer, output_dir="./qwen3-finetuned"):
    """保存模型"""
    print(f"💾 正在保存模型到: {output_dir}")
    
    # 保存模型
    trainer.save_model(output_dir)
    
    # 保存分词器 - 使用 processing_class 替代已弃用的 tokenizer
    if hasattr(trainer, 'processing_class') and trainer.processing_class is not None:
        trainer.processing_class.save_pretrained(output_dir)
    elif hasattr(trainer, 'tokenizer') and trainer.tokenizer is not None:
        trainer.tokenizer.save_pretrained(output_dir)
    else:
        print("⚠️  警告: 未找到分词器，跳过分词器保存")
    
    print(f"✅ 模型保存完成: {output_dir}")

def test_inference(model_path, test_prompt="密钥管理是什么"):
    """测试模型推理"""
    print("🧪 正在测试模型推理")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 加载基础模型
    #Qwen/Qwen3-4B-Thinking-2507
    #Qwen/Qwen3-4B-Instruct-2507
    base_model_name = "Qwen/Qwen3-4B-Thinking-2507"
    
    # 4bit 量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    
    # 加载PEFT适配器
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, model_path)
    
    # 设置为评估模式
    model.eval()
    
    # 准备输入
    prompt = f"<|im_start|>user\n{test_prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # 移动到 GPU
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        model.to("cuda")
    
    # 生成回答
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 解码输出
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"📝 测试结果:")
    print(f"问题: {test_prompt}")
    print(f"回答: {response}")
    
    return response

def monitor_gpu_memory():
    """监控GPU显存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"📊 GPU显存使用: {allocated:.1f}GB 已分配, {cached:.1f}GB 已缓存, 总共 {total:.1f}GB")
        return allocated, cached, total
    return 0, 0, 0

def cleanup_gpu_memory():
    """清理GPU内存以避免OOM"""
    if torch.cuda.is_available():
        print("🧹 清理GPU内存...")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("✅ GPU内存清理完成")

def check_memory_safety(required_memory_gb=2.0):
    """检查是否有足够的显存继续训练"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        available = total - allocated
        
        if available < required_memory_gb:
            print(f"⚠️  警告: 可用显存不足 ({available:.1f}GB < {required_memory_gb}GB)")
            cleanup_gpu_memory()
            # 再次检查
            allocated = torch.cuda.memory_allocated() / 1024**3
            available = total - allocated
            if available < required_memory_gb:
                print(f"❌ 错误: 显存仍然不足 ({available:.1f}GB < {required_memory_gb}GB)")
                return False
        return True
    return True  # CPU模式下总是返回True

def main():
    """主函数 - 针对16G显存深度优化版本"""
    print("🚀 Qwen3 模型微调示例（16G显存深度优化版）")
    print("📋 功能: 4bit 量化 + BF16 + LoRA + 梯度累积 + OOM防护")
    print("🎯 目标: 在16G显存限制下安全稳定地完成训练")
    print()
    
    # 检查 GPU
    if torch.cuda.is_available():
        print(f"✅ GPU 可用: {torch.cuda.get_device_name()}")
        print(f"✅ GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # 设置GPU优化
        cleanup_gpu_memory()
        torch.backends.cudnn.benchmark = True  # 启用cuDNN基准测试
        torch.backends.cuda.matmul.allow_tf32 = True  # 允许TF32矩阵乘法
        print("✅ GPU 优化已启用")
    else:
        print("⚠️  GPU 不可用，将使用 CPU 训练（速度较慢）")
    
    print()
    
    # 初始显存监控和检查
    monitor_gpu_memory()
    if not check_memory_safety(required_memory_gb=3.0):
        print("❌ 初始显存检查失败，无法开始训练")
        return
    
    print()
    
    try:
        # 1. 设置模型和分词器 - 添加OOM防护
        print("📋 步骤 1: 加载模型和分词器")
        #Qwen/Qwen3-4B-Thinking-2507
        model_name = "Qwen/Qwen3-4B-Thinking-2507"#"Qwen/Qwen3-4B-Instruct-2507"
        model, tokenizer = setup_model_and_tokenizer(model_name)
        
        # 检查显存
        monitor_gpu_memory()
        if not check_memory_safety(required_memory_gb=2.5):
            print("❌ 模型加载后显存不足")
            return
        
        print()
        
        # 2. 设置 LoRA - 添加OOM防护
        print("📋 步骤 2: 配置 LoRA 适配器")
        model = setup_lora(model)
        
        # 检查显存
        monitor_gpu_memory()
        if not check_memory_safety(required_memory_gb=2.0):
            print("❌ LoRA配置后显存不足")
            return
        
        print()
        
        # 3. 准备数据集 - 添加OOM防护
        print("📋 步骤 3: 准备数据集")
        tokenized_dataset = prepare_dataset(tokenizer)
        
        # 清理内存并检查
        cleanup_gpu_memory()
        monitor_gpu_memory()
        if not check_memory_safety(required_memory_gb=1.5):
            print("❌ 数据集准备后显存不足")
            return
        
        print()
        
        # 4. 设置训练参数
        print("📋 步骤 4: 配置训练参数")
        training_args = setup_training_args()
        
        print()
        
        # 5. 训练模型 - 添加OOM防护
        print("📋 步骤 5: 开始训练模型")
        trainer = train_model(model, tokenizer, tokenized_dataset, training_args)
        
        print()
        
        # 6. 保存模型 - 添加OOM防护
        print("📋 步骤 6: 保存模型")
        cleanup_gpu_memory()
        save_model(trainer)
        
        print()
        
        # 7. 测试推理 - 添加OOM防护
        print("📋 步骤 7: 测试模型推理")
        cleanup_gpu_memory()
        test_inference("./qwen3-finetuned")
        
        print("\n🎉 微调完成！")
        print("📁 模型保存在: ./qwen3-finetuned")
        print("📊 训练日志在: ./logs")
        
        # 最终显存状态
        print("\n📊 最终显存状态:")
        monitor_gpu_memory()
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("❌ CUDA OOM 错误: 显存不足")
            print("💡 建议:")
            print("   1. 重启程序释放显存")
            print("   2. 进一步减小批次大小或序列长度")
            print("   3. 考虑使用更小的模型")
            cleanup_gpu_memory()
        else:
            print(f"❌ 训练过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 最终清理
        print("\n🧹 执行最终清理...")
        cleanup_gpu_memory()
        print("✅ 清理完成")

if __name__ == "__main__":
    main()
