# Qwen3 模型导出程序使用指南

这个程序可以将微调后的 Qwen3 模型导出为多种格式，包括 PyTorch 合并模型、GGUF 格式和 ONNX 格式。

## 功能特性

- **PyTorch 合并模型**: 将 LoRA 适配器与基础模型合并，生成完整的 PyTorch 模型
- **GGUF 格式**: 生成适用于 llama.cpp 的 GGUF 格式模型，支持多种量化级别
- **ONNX 格式**: 生成 ONNX 格式模型，支持跨平台部署

## 安装依赖

### 1. 基础 Python 依赖

```bash
pip install torch transformers peft accelerate bitsandbytes sentencepiece protobuf
```

### 2. GGUF 相关依赖

```bash
# 安装 llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# 安装 Python 转换工具
pip install -r requirements.txt
```

### 3. ONNX 相关依赖

```bash
pip install onnx onnxruntime
```

### 4. 完整安装脚本

创建并运行安装脚本：

```bash
# 创建安装脚本
cat > install_export_deps.sh << 'EOF'
#!/bin/bash

echo "正在安装模型导出依赖..."

# 基础依赖
echo "安装基础 Python 依赖..."
pip install torch transformers peft accelerate bitsandbytes sentencepiece protobuf

# ONNX 依赖
echo "安装 ONNX 依赖..."
pip install onnx onnxruntime

# llama.cpp
echo "安装 llama.cpp..."
if [ ! -d "llama.cpp" ]; then
    git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp
    make
    pip install -r requirements.txt
    cd ..
else
    echo "llama.cpp 已存在，跳过安装"
fi

echo "依赖安装完成！"
EOF

# 运行安装脚本
chmod +x install_export_deps.sh
./install_export_deps.sh
```

## 使用方法

### 1. 列出可用的 Checkpoints

```bash
python model_exporter.py --list-checkpoints
```

### 2. 导出所有格式

```bash
# 使用默认设置
python model_exporter.py

# 指定输出目录
python model_exporter.py --output ./my_exported_models

# 指定基础模型
python model_exporter.py --base-model Qwen/Qwen3-4B-Instruct-2507
```

### 3. 导出特定格式

```bash
# 只导出合并的 PyTorch 模型
python model_exporter.py --formats merged

# 只导出 GGUF 格式
python model_exporter.py --formats gguf

# 只导出 ONNX 格式
python model_exporter.py --formats onnx

# 导出多种格式
python model_exporter.py --formats merged gguf
```

### 4. 指定 Checkpoint

```bash
# 使用特定的 checkpoint
python model_exporter.py --checkpoint ./qwen3-finetuned/checkpoint-240

# 使用不同的基础目录
python model_exporter.py --base-dir ./my_finetuned_models --checkpoint checkpoint-300
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--checkpoint` | 自动查找 | 指定要导出的 checkpoint 路径 |
| `--output` | `./exported_models` | 输出目录 |
| `--base-dir` | `./qwen3-finetuned` | 微调模型基础目录 |
| `--base-model` | `Qwen/Qwen3-4B-Thinking-2507` | 基础模型名称 |
| `--formats` | `all` | 导出格式：merged, gguf, onnx, all |
| `--list-checkpoints` | False | 列出可用的 checkpoints |

## 输出结构

导出完成后，输出目录结构如下：

```
exported_models/
├── merged_model/              # 合并的 PyTorch 模型
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   └── export_metadata.json
├── gguf_model/                # GGUF 格式模型
│   ├── model.f16.gguf        # F16 精度版本
│   ├── model.q4_k_m.gguf     # Q4_K_M 量化版本
│   ├── model.q5_k_m.gguf     # Q5_K_M 量化版本
│   ├── model.q8_0.gguf       # Q8_0 量化版本
│   └── gguf_metadata.json
├── onnx_model/                # ONNX 格式模型
│   ├── model.onnx
│   ├── tokenizer.json
│   └── onnx_metadata.json
├── export_summary.json       # 导出摘要
└── model_export.log          # 详细日志
```

## 使用示例

### 示例 1：完整导出流程

```bash
# 1. 首先运行微调训练（如果还没有训练）
python qwen3_finetuning_example.py

# 2. 列出可用的 checkpoints
python model_exporter.py --list-checkpoints

# 3. 导出所有格式
python model_exporter.py --output ./final_models
```

### 示例 2：只导出 GGUF 格式用于部署

```bash
python model_exporter.py \
    --checkpoint ./qwen3-finetuned/checkpoint-240 \
    --formats gguf \
    --output ./deployment_models
```

### 示例 3：导出 ONNX 格式用于推理服务

```bash
python model_exporter.py \
    --formats onnx \
    --output ./onnx_models
```

## 故障排除

### 1. 依赖相关问题

**问题**: `ModuleNotFoundError: No module named 'onnx'`
**解决**: `pip install onnx onnxruntime`

**问题**: `llama.cpp not found`
**解决**: 安装 llama.cpp
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make
```

### 2. 内存问题

**问题**: CUDA out of memory
**解决**: 
- 确保有足够的 GPU 内存
- 关闭其他占用 GPU 的程序
- 使用 CPU 模式（较慢）

### 3. Checkpoint 问题

**问题**: `Checkpoint path does not exist`
**解决**: 
- 检查路径是否正确
- 先运行 `--list-checkpoints` 查看可用的 checkpoints
- 确保已经完成了模型微调训练

### 4. GGUF 转换问题

**问题**: GGUF 转换失败
**解决**: 
- 确保 llama.cpp 已正确安装
- 检查模型文件是否完整
- 查看详细日志 `model_export.log`

## 性能建议

1. **GPU 内存**: 建议至少 16GB GPU 内存用于大型模型导出
2. **存储空间**: 确保有足够的磁盘空间（通常需要原始模型大小的 3-5 倍）
3. **并行处理**: 可以分别导出不同格式以减少内存压力
4. **量化选择**: 
   - `q4_k_m`: 平衡质量和大小，推荐用于大多数场景
   - `q8_0`: 高质量，适合需要最佳性能的场景
   - `q5_k_m`: 介于 q4 和 q8 之间

## 支持的模型

目前支持以下基础模型：
- `Qwen/Qwen3-4B-Thinking-2507`
- `Qwen/Qwen3-4B-Instruct-2507`
- 其他 Qwen3 系列模型（需要测试兼容性）

## 技术细节

### GGUF 量化方法

- `q4_k_m`: 4-bit K-quantization, medium
- `q5_k_m`: 5-bit K-quantization, medium  
- `q8_0`: 8-bit quantization

### ONNX 导出参数

- Opset 版本: 17
- 支持动态批处理和序列长度
- 包含输入输出元数据

## 日志和调试

程序会生成详细的日志文件 `model_export.log`，包含：
- 依赖检查结果
- 模型加载过程
- 导出进度和状态
- 错误信息和堆栈跟踪

如需调试，请查看此日志文件。
