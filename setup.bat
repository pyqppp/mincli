@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================
echo   DeepSeek CLI 安装脚本 (Windows)
echo ========================================

:: 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python，请安装 Python 3.8+ 并添加到 PATH。
    pause
    exit /b 1
)

:: 创建虚拟环境
if not exist "venv" (
    echo 正在创建虚拟环境...
    python -m venv venv
    if errorlevel 1 (
        echo [错误] 创建虚拟环境失败。
        pause
        exit /b 1
    )
    echo 虚拟环境创建成功。
) else (
    echo 虚拟环境已存在，跳过创建。
)

:: 激活虚拟环境
echo 正在激活虚拟环境...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [错误] 激活虚拟环境失败。
    pause
    exit /b 1
)

:: 安装依赖 - 使用清华镜像
echo 正在安装依赖包（使用清华镜像）...
pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
pip install tiktoken typer "python-dotenv>=1.0.0" "openai>=1.0.0" rich prompt-toolkit -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn

if errorlevel 1 (
    echo [错误] 安装依赖失败，请检查网络连接。
    pause
    exit /b 1
)

echo.
echo ========================================
echo   安装完成！
echo   请创建 .env 文件并填入 DEEPSEEK_API_KEY
echo   以后使用前请激活虚拟环境：
echo   venv\Scripts\activate
echo   运行：
echo   python main.py chat -i
echo ========================================
pause