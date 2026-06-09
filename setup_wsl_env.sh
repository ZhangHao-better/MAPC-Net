#!/bin/bash
# WSL Ubuntu 环境一键配置脚本 - ASCFormer 项目
# 在 WSL Ubuntu 终端中执行: bash setup_wsl_env.sh

set -e  # 遇到错误立即停止

echo "=========================================="
echo "ASCFormer WSL 环境自动配置脚本"
echo "=========================================="

# 1. 验证 GPU
echo ""
echo "[1/7] 验证 GPU 直通..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    echo "✓ GPU 直通成功"
else
    echo "⚠ nvidia-smi 未找到，请确保 Windows 端安装了支持 WSL 的 NVIDIA 驱动"
fi

# 2. 配置 APT 镜像
echo ""
echo "[2/7] 配置 APT 国内镜像（清华源）..."
sudo cp /etc/apt/sources.list /etc/apt/sources.list.bak 2>/dev/null || true
UBUNTU_VERSION=$(lsb_release -cs)
sudo bash -c "cat >/etc/apt/sources.list" << EOF
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ ${UBUNTU_VERSION} main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ ${UBUNTU_VERSION}-updates main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ ${UBUNTU_VERSION}-backports main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ ${UBUNTU_VERSION}-security main restricted universe multiverse
EOF
sudo apt update
echo "✓ APT 镜像配置完成"

# 3. 安装基础工具
echo ""
echo "[3/7] 安装基础编译工具..."
sudo apt install -y build-essential wget git vim curl
echo "✓ 基础工具安装完成"

# 4. 安装 Miniforge（如果未安装 conda）
echo ""
echo "[4/7] 检查并安装 Miniforge..."
if ! command -v conda &> /dev/null; then
    echo "正在下载 Miniforge..."
    wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh
    bash /tmp/miniforge.sh -b -p $HOME/miniforge3
    rm /tmp/miniforge.sh
    
    # 初始化 conda
    eval "$($HOME/miniforge3/bin/conda shell.bash hook)"
    conda init bash
    
    # 配置 conda 镜像
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
    conda config --set show_channel_urls yes
    
    echo "✓ Miniforge 安装完成"
else
    echo "✓ Conda 已安装，跳过"
    eval "$(conda shell.bash hook)"
fi

# 5. 配置 pip 镜像
echo ""
echo "[5/7] 配置 pip 国内镜像..."
mkdir -p ~/.pip
cat > ~/.pip/pip.conf << 'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
timeout = 120
EOF
echo "✓ pip 镜像配置完成"

# 6. 复制项目到 WSL 文件系统
echo ""
echo "[6/7] 复制项目到 WSL 文件系统..."
if [ ! -d "$HOME/RTM-main" ]; then
    echo "正在从 Windows 复制项目文件..."
    cp -r /mnt/c/zh/RTM-main/RTM-main $HOME/RTM-main
    echo "✓ 项目复制完成: $HOME/RTM-main"
else
    echo "✓ 项目已存在: $HOME/RTM-main"
fi

# 7. 创建 conda 环境并安装依赖
echo ""
echo "[7/7] 创建 conda 环境并安装依赖..."
cd $HOME/RTM-main/ASCFormer

# 创建环境
if conda env list | grep -q "^rtm "; then
    echo "环境 rtm 已存在，跳过创建"
else
    conda create -n rtm python=3.8 -y
    echo "✓ conda 环境创建完成"
fi

# 激活环境
eval "$(conda shell.bash hook)"
conda activate rtm

echo ""
echo "正在安装 PyTorch 2.0.0 + CUDA 11.8..."
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

echo ""
echo "正在安装 OpenMMLab 依赖..."
pip install -U openmim
mim install "mmengine==0.7.0"
mim install "mmcv==2.0.0"

echo ""
echo "正在安装项目依赖..."
pip install -r requirements.txt || echo "⚠ 部分依赖安装失败，可能需要手动处理 natten"

echo ""
echo "正在安装项目本身..."
pip install -v -e .

echo ""
echo "=========================================="
echo "环境配置完成！"
echo "=========================================="
echo ""
echo "下一步操作："
echo "1. 重新打开终端或执行: source ~/.bashrc"
echo "2. 激活环境: conda activate rtm"
echo "3. 进入项目: cd ~/RTM-main/ASCFormer"
echo "4. 验证环境:"
echo "   python -c \"import torch, mmcv, mmseg; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())\""
echo ""
echo "如果 natten 安装失败，请查看 WINDOWS_WORKAROUND.md"
echo "或运行以下命令安装 CUDA Toolkit："
echo "  sudo apt install -y cuda-toolkit-11-8"
echo "  export CUDA_HOME=/usr/local/cuda-11.8"
echo "  export PATH=\$CUDA_HOME/bin:\$PATH"
echo "  export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH"
echo "  pip install natten==0.14.6 --verbose"
echo ""
echo "推荐使用 VS Code + Remote-WSL 扩展进行开发！"
