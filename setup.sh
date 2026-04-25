#!/bin/bash
# setup.sh - 为 DeepSeek CLI 创建虚拟环境并安装依赖

set -e  # 遇到错误立即退出

echo "🔧 正在创建 Python 虚拟环境..."

# 检查 Python3 是否可用
if ! command -v python3 &> /dev/null; then
    echo "❌ 未找到 python3，请先安装 Python 3.8+。"
    exit 1
fi

# 创建虚拟环境（若已存在则跳过）
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ 虚拟环境创建成功：venv/"
else
    echo "⚠️  虚拟环境已存在，跳过创建。"
fi

# 激活虚拟环境（此处在脚本内直接使用 venv/bin/python 和 venv/bin/pip）
echo "📦 正在安装依赖包..."

# 设置国内镜像源
PIP_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"
PIP_TRUSTED_HOST="pypi.tuna.tsinghua.edu.cn"

echo "📦 正在使用清华镜像源安装依赖..."
venv/bin/pip install --upgrade pip -i $PIP_INDEX --trusted-host $PIP_TRUSTED_HOST
venv/bin/pip install \
    tiktoken \
    typer \
    "python-dotenv>=1.0.0" \
    "openai>=1.0.0" \
    rich \
    prompt-toolkit \
    -i $PIP_INDEX --trusted-host $PIP_TRUSTED_HOST

echo ""
echo "✅ 所有依赖安装完成！"
echo ""
echo "👉 使用以下命令激活虚拟环境："
echo "   source venv/bin/activate"
echo ""
echo "👉 运行 DeepSeek CLI："
echo "   python main.py chat -i"