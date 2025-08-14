#!/usr/bin/env python3
"""
Qwen3 模型导出程序
支持导出微调模型、GGUF 格式和 ONNX 格式
"""

import os
import sys
import json
import shutil
import argparse
import subprocess
import torch
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig
)
from peft import PeftModel
import numpy as np
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_export.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelExporter:
    def __init__(self, base_model_name="Qwen/Qwen3-4B-Thinking-2507"):
        self.base_model_name = base_model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"使用设备: {self.device}")
        
    def check_dependencies(self):
        """检查必要的依赖"""
        required_packages = [
            'torch', 'transformers', 'peft', 'accelerate', 
            'bitsandbytes', 'sentencepiece', 'protobuf'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.error(f"缺少必要的依赖包: {missing_packages}")
            logger.info("请运行: pip install " + " ".join(missing_packages))
            return False
        
        # 检查 GGUF 和 ONNX 相关工具
        gguf_tools = ['llama.cpp', 'quantize']
        onnx_tools = ['onnxruntime']
        
        logger.info("正在检查 GGUF 和 ONNX 工具...")
        for tool in gguf_tools:
            if not shutil.which(tool):
                logger.warning(f"GGUF 工具 '{tool}' 未找到，请安装 llama.cpp")
        
        for tool in onnx_tools:
            try:
                __import__(tool.replace('-', '_'))
            except ImportError:
                logger.warning(f"ONNX 工具 '{tool}' 未找到，请运行: pip install {tool}")
        
        return True
    
    def load_finetuned_model(self, checkpoint_path):
        """加载微调后的模型"""
        logger.info(f"正在加载微调模型: {checkpoint_path}")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint 路径不存在: {checkpoint_path}")
        
        # 加载基础模型配置
        config = AutoConfig.from_pretrained(self.base_model_name, trust_remote_code=True)
        
        # 加载基础模型
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            config=config,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # 加载 LoRA 适配器
        if os.path.exists(os.path.join(checkpoint_path, "adapter_config.json")):
            logger.info("检测到 LoRA 适配器，正在加载...")
            model = PeftModel.from_pretrained(model, checkpoint_path)
            model = model.merge_and_unload()  # 合并适配器
            logger.info("LoRA 适配器已合并")
        else:
            logger.info("未检测到 LoRA 适配器，使用基础模型")
        
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path if os.path.exists(os.path.join(checkpoint_path, "tokenizer.json")) 
            else self.base_model_name,
            trust_remote_code=True
        )
        
        # 设置 pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info("模型加载完成")
        return model, tokenizer
    
    def export_merged_model(self, checkpoint_path, output_dir):
        """导出合并后的模型"""
        logger.info(f"正在导出合并模型到: {output_dir}")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载模型和分词器
        model, tokenizer = self.load_finetuned_model(checkpoint_path)
        
        # 保存模型
        model.save_pretrained(output_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)
        
        # 保存元数据
        metadata = {
            "base_model": self.base_model_name,
            "checkpoint_path": checkpoint_path,
            "export_time": str(torch.datetime.now()),
            "model_type": "merged",
            "framework": "pytorch"
        }
        
        with open(os.path.join(output_dir, "export_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"合并模型导出完成: {output_dir}")
        return output_dir
    
    def export_to_gguf(self, model_path, output_dir, quantization_methods=None):
        """导出为 GGUF 格式"""
        if quantization_methods is None:
            quantization_methods = ["q4_k_m", "q5_k_m", "q8_0"]
        
        logger.info(f"正在导出 GGUF 格式到: {output_dir}")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 检查 llama.cpp 是否可用
        if not shutil.which("llama.cpp"):
            logger.error("llama.cpp 未找到，请先安装 llama.cpp")
            logger.info("安装方法: git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && make")
            return None
        
        # 转换为 GGUF
        f16_path = os.path.join(output_dir, "model.f16.gguf")
        convert_cmd = [
            "python", "-m", "llama_cpp.convert",
            model_path,
            "--outfile", f16_path,
            "--outtype", "f16"
        ]
        
        logger.info(f"执行转换命令: {' '.join(convert_cmd)}")
        try:
            result = subprocess.run(convert_cmd, check=True, capture_output=True, text=True)
            logger.info("F16 GGUF 转换完成")
        except subprocess.CalledProcessError as e:
            logger.error(f"GGUF 转换失败: {e}")
            logger.error(f"错误输出: {e.stderr}")
            return None
        
        # 量化
        gguf_files = []
        for quant_method in quantization_methods:
            quant_path = os.path.join(output_dir, f"model.{quant_method}.gguf")
            quant_cmd = [
                "llama.cpp/quantize",
                f16_path,
                quant_path,
                quant_method
            ]
            
            logger.info(f"执行量化命令: {' '.join(quant_cmd)}")
            try:
                result = subprocess.run(quant_cmd, check=True, capture_output=True, text=True)
                gguf_files.append(quant_path)
                logger.info(f"量化完成: {quant_method}")
            except subprocess.CalledProcessError as e:
                logger.error(f"量化失败 {quant_method}: {e}")
                continue
        
        # 保存元数据
        metadata = {
            "source_model": model_path,
            "quantization_methods": quantization_methods,
            "gguf_files": gguf_files,
            "export_time": str(torch.datetime.now()),
            "model_type": "gguf"
        }
        
        with open(os.path.join(output_dir, "gguf_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"GGUF 导出完成，生成了 {len(gguf_files)} 个量化版本")
        return output_dir
    
    def export_to_onnx(self, model_path, output_dir, opset_version=17):
        """导出为 ONNX 格式"""
        logger.info(f"正在导出 ONNX 格式到: {output_dir}")
        
        try:
            import onnx
            import onnxruntime as ort
        except ImportError:
            logger.error("ONNX 相关库未安装，请运行: pip install onnx onnxruntime")
            return None
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载模型和分词器
        model, tokenizer = self.load_finetuned_model(model_path)
        
        # 设置为评估模式
        model.eval()
        
        # 准备示例输入
        max_length = 512
        dummy_input = torch.randint(0, tokenizer.vocab_size, (1, max_length)).to(self.device)
        attention_mask = torch.ones_like(dummy_input)
        
        # 导出 ONNX
        onnx_path = os.path.join(output_dir, "model.onnx")
        
        logger.info("正在导出 ONNX 模型...")
        try:
            torch.onnx.export(
                model,
                (dummy_input, attention_mask),
                onnx_path,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input_ids', 'attention_mask'],
                output_names=['logits'],
                dynamic_axes={
                    'input_ids': {0: 'batch_size', 1: 'sequence'},
                    'attention_mask': {0: 'batch_size', 1: 'sequence'},
                    'logits': {0: 'batch_size', 1: 'sequence'}
                }
            )
            logger.info("ONNX 导出完成")
        except Exception as e:
            logger.error(f"ONNX 导出失败: {e}")
            return None
        
        # 验证 ONNX 模型
        logger.info("正在验证 ONNX 模型...")
        try:
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            logger.info("ONNX 模型验证通过")
        except Exception as e:
            logger.error(f"ONNX 模型验证失败: {e}")
            return None
        
        # 测试 ONNX 推理
        logger.info("正在测试 ONNX 推理...")
        try:
            ort_session = ort.InferenceSession(onnx_path)
            
            # 准备测试输入
            test_input = torch.randint(0, tokenizer.vocab_size, (1, 128)).to(self.device)
            test_mask = torch.ones_like(test_input)
            
            # ONNX 推理
            ort_inputs = {
                'input_ids': test_input.cpu().numpy(),
                'attention_mask': test_mask.cpu().numpy()
            }
            ort_outputs = ort_session.run(None, ort_inputs)
            
            # PyTorch 推理（对比）
            with torch.no_grad():
                pt_outputs = model(test_input, attention_mask=test_mask)
            
            logger.info("ONNX 推理测试完成")
        except Exception as e:
            logger.error(f"ONNX 推理测试失败: {e}")
            return None
        
        # 保存分词器配置
        tokenizer.save_pretrained(output_dir)
        
        # 保存元数据
        metadata = {
            "source_model": model_path,
            "onnx_path": onnx_path,
            "opset_version": opset_version,
            "export_time": str(torch.datetime.now()),
            "model_type": "onnx",
            "input_names": ['input_ids', 'attention_mask'],
            "output_names": ['logits']
        }
        
        with open(os.path.join(output_dir, "onnx_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"ONNX 导出完成: {onnx_path}")
        return output_dir
    
    def export_all_formats(self, checkpoint_path, output_base_dir):
        """导出所有格式"""
        logger.info(f"开始导出所有格式，基础目录: {output_base_dir}")
        
        results = {}
        
        # 1. 导出合并的 PyTorch 模型
        try:
            merged_dir = os.path.join(output_base_dir, "merged_model")
            results["merged"] = self.export_merged_model(checkpoint_path, merged_dir)
        except Exception as e:
            logger.error(f"合并模型导出失败: {e}")
            results["merged"] = None
        
        # 2. 导出 GGUF 格式
        try:
            gguf_dir = os.path.join(output_base_dir, "gguf_model")
            results["gguf"] = self.export_to_gguf(
                results["merged"] or checkpoint_path, 
                gguf_dir
            )
        except Exception as e:
            logger.error(f"GGUF 导出失败: {e}")
            results["gguf"] = None
        
        # 3. 导出 ONNX 格式
        try:
            onnx_dir = os.path.join(output_base_dir, "onnx_model")
            results["onnx"] = self.export_to_onnx(
                results["merged"] or checkpoint_path, 
                onnx_dir
            )
        except Exception as e:
            logger.error(f"ONNX 导出失败: {e}")
            results["onnx"] = None
        
        # 保存导出摘要
        summary = {
            "checkpoint_path": checkpoint_path,
            "output_base_dir": output_base_dir,
            "export_results": results,
            "export_time": str(torch.datetime.now()),
            "base_model": self.base_model_name
        }
        
        with open(os.path.join(output_base_dir, "export_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info("所有格式导出完成")
        return results
    
    def find_checkpoints(self, base_dir="./qwen3-finetuned"):
        """查找可用的 checkpoint"""
        checkpoints = []
        
        if not os.path.exists(base_dir):
            logger.warning(f"基础目录不存在: {base_dir}")
            return checkpoints
        
        # 查找 checkpoint-* 目录
        for item in os.listdir(base_dir):
            if item.startswith("checkpoint-") and os.path.isdir(os.path.join(base_dir, item)):
                checkpoint_path = os.path.join(base_dir, item)
                checkpoints.append(checkpoint_path)
                logger.info(f"发现 checkpoint: {checkpoint_path}")
        
        # 按数字排序
        checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
        
        return checkpoints

def main():
    parser = argparse.ArgumentParser(description="Qwen3 模型导出程序")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint 路径")
    parser.add_argument("--output", type=str, default="./exported_models", help="输出目录")
    parser.add_argument("--base-dir", type=str, default="./qwen3-finetuned", help="微调模型基础目录")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3-4B-Thinking-2507", help="基础模型名称")
    parser.add_argument("--formats", nargs="+", choices=["merged", "gguf", "onnx", "all"], 
                       default=["all"], help="导出格式")
    parser.add_argument("--list-checkpoints", action="store_true", help="列出可用的 checkpoints")
    
    args = parser.parse_args()
    
    # 创建导出器
    exporter = ModelExporter(args.base_model)
    
    # 检查依赖
    if not exporter.check_dependencies():
        return 1
    
    # 列出 checkpoints
    if args.list_checkpoints:
        checkpoints = exporter.find_checkpoints(args.base_dir)
        if checkpoints:
            print("可用的 checkpoints:")
            for i, cp in enumerate(checkpoints, 1):
                print(f"  {i}. {cp}")
        else:
            print("未找到可用的 checkpoints")
        return 0
    
    # 确定 checkpoint 路径
    checkpoint_path = args.checkpoint
    if not checkpoint_path:
        checkpoints = exporter.find_checkpoints(args.base_dir)
        if not checkpoints:
            logger.error("未找到可用的 checkpoints，请指定 --checkpoint 参数")
            return 1
        checkpoint_path = checkpoints[-1]  # 使用最新的 checkpoint
        logger.info(f"使用最新的 checkpoint: {checkpoint_path}")
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 执行导出
    try:
        if "all" in args.formats:
            results = exporter.export_all_formats(checkpoint_path, args.output)
        else:
            results = {}
            if "merged" in args.formats:
                merged_dir = os.path.join(args.output, "merged_model")
                results["merged"] = exporter.export_merged_model(checkpoint_path, merged_dir)
            
            if "gguf" in args.formats:
                gguf_dir = os.path.join(args.output, "gguf_model")
                results["gguf"] = exporter.export_to_gguf(
                    results.get("merged", checkpoint_path), 
                    gguf_dir
                )
            
            if "onnx" in args.formats:
                onnx_dir = os.path.join(args.output, "onnx_model")
                results["onnx"] = exporter.export_to_onnx(
                    results.get("merged", checkpoint_path), 
                    onnx_dir
                )
        
        # 输出结果摘要
        logger.info("导出完成！结果摘要:")
        for format_name, path in results.items():
            if path:
                logger.info(f"  {format_name.upper()}: {path}")
            else:
                logger.info(f"  {format_name.upper()}: 导出失败")
        
        logger.info(f"详细日志保存在: model_export.log")
        
    except Exception as e:
        logger.error(f"导出过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
