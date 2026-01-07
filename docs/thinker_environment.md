# thinker环境配置

## 需安装以下工具及版本（与linger一致）
- gcc/10.2.0
- cmake/4.2.1
- cuda/11.2

## python依赖包安装
建议采用虚拟环境安装
```Shell
conda create -n thinker python==3.10
conda activate thinker
pip install -U pip
cat requirements.txt | xargs -n 1 pip install
```
