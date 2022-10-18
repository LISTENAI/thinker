
# thinker环境配置

## 方案1，采用下列脚本，自动安装需要的python库
```Shell
conda create -n thinker python==3.8.5
conda activate thinker
pip install -U pip
cat requirements.txt | xargs -n 1 pip install
```
若无法执行，则采用方案2直接安装需要的库文件

## 方案2，需要常见的库如下, 内网配置好pip源
* onnx       :  pip install onnx
* sympy==1.8 :  pip install sympy --ignore-installed（忽略已安装,一般python已自动安装低版本）
