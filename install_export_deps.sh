#!/bin/bash

echo "=========================================="
echo "Qwen3 模型导出程序依赖安装脚本"
echo "=========================================="
echo

# 检查 Python 是否可用
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到 python3，请先安装 Python 3.8+"
    exit 1
fi

echo "✅ Python 版本: $(python3 --version)"
echo

# 检查 pip 是否可用
if ! command -v pip3 &> /dev/null; then
    echo "❌ 错误: 未找到 pip3，请先安装 pip"
    exit 1
fi

echo "✅ Pip 版本: $(pip3 --version)"
echo

# 检查 git 是否可用
if ! command -v git &> /dev/null; then
    echo "❌ 错误: 未找到 git，请先安装 git"
    exit 1
fi

echo "✅ Git 版本: $(git --version)"
echo

# 检查 make 是否可用
if ! command -v make &> /dev/null; then
    echo "❌ 错误: 未找到 make，请先安装 build-essential 或类似工具包"
    exit 1
fi

echo "✅ Make 可用"
echo

echo "🔄 开始安装基础 Python 依赖..."
pip3 install torch transformers peft accelerate bitsandbytes sentencepiece protobuf

if [ $? -eq 0 ]; then
    echo "✅ 基础 Python 依赖安装完成"
else
    echo "❌ 基础 Python 依赖安装失败"
    exit 1
fi

echo

echo "🔄 开始安装 ONNX 依赖..."
pip3 install onnx onnxruntime

if [ $? -eq 0 ]; then
    echo "✅ ONNX 依赖安装完成"
else
    echo "❌ ONNX 依赖安装失败"
    exit 1
fi

echo

echo "🔄 开始安装 llama.cpp..."
if [ ! -d "llama.cpp" ]; then
    echo "正在克隆 llama.cpp 仓库..."
    git clone https://github.com/ggerganov/llama.cpp
    
    if [ $? -eq 0 ]; then
        echo "✅ llama.cpp 克隆完成"
    else
        echo "❌ llama.cpp 克隆失败"
        exit 1
    fi
    
    cd llama.cpp
    
    echo "正在编译 llama.cpp..."
    make
    
    if [ $? -eq 0 ]; then
        echo "✅ llama.cpp 编译完成"
    else
        echo "❌ llama.cpp 编译失败"
        exit 1
    fi
    
    echo "正在安装 llama.cpp Python 依赖..."
    if [ -f "requirements.txt" ]; then
        pip3 install -r requirements.txt
        if [ $? -eq 0 ]; then
            echo "✅ llama.cpp Python 依赖安装完成"
        else
            echo "❌ llama.cpp Python 依赖安装失败"
            exit 1
        fi
    else
        echo "⚠️  未找到 requirements.txt，跳过 Python 依赖安装"
    fi
    
    cd ..
else
    echo "✅ llama.cpp 已存在，跳过安装"
fi

echo

echo "🔄 正在验证安装..."

# 验证 Python 包
echo "验证 Python 包..."
python3 -c "
import torch
import transformers
import peft
import accelerate
import bitsandbytes
import sentencepiece
import protobuf
print('✅ 所有基础 Python 包导入成功')
"

if [ $? -eq 0 ]; then
    echo "✅ 基础 Python 包验证通过"
else
    echo "❌ 基础 Python 包验证失败"
    exit 1
fi

# 验证 ONNX 包
echo "验证 ONNX 包..."
python3 -c "
import onnx
import onnxruntime
print('✅ ONNX 包导入成功')
"

if [ $? -eq 0 ]; then
    echo "✅ ONNX 包验证通过"
else
    echo "❌ ONNX 包验证失败"
    exit 1
fi

# 验证 llama.cpp
if [ -f "llama.cpp/main" ]; then
    echo "✅ llama.cpp 编译文件存在"
else
    echo "❌ llama.cpp 编译文件不存在"
    exit 1
fi

echo
echo "=========================================="
echo "🎉 所有依赖安装完成！"
echo "=========================================="
echo
echo "现在可以使用模型导出程序："
echo "  python3 model_exporter.py --list-checkpoints"
echo "  python3 model_exporter.py"
echo
echo "更多信息请参考 MODEL_EXPORT_README.md"
echo "=========================================="
