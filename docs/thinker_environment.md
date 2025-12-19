# thinker环境配置

## 需安装以下工具及版本（与linger保持统一）
- gcc-10.2.0
- cmake-3.20.1
- cuda/11.7.1-cudnn-v8.5.0

## python依赖包安装
建议采用虚拟环境安装
```Shell
conda create -n thinker python==3.12.0
conda activate thinker
pip install -U pip
cat requirements.txt | xargs -n 1 pip install
```
