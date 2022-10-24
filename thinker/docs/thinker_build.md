# 源码编译方式

## 下载源码
```Shell
  mkdir thinker
  git clone https://github.com/LISTENAI/thinker/thinker.git
```
## 配置本地环境
```Shell
conda create -n thinker python==3.8.5
conda activate thinker
pip install -U pip
cat requirements.txt | xargs -n 1 pip install
```
若无法执行，则采用手动安装requirements中的各种库

查看已创建的环境
```Shell
conda info --env
```

查看当前环境中已安装的包
```Shell
conda list
```
删除不要的环境有两种方式：
```Shell
conda activate base(或者 conda deactivate xxx)
conda remove -n xxx --all
```
## x86_linux编译
  * gcc版本最好为5.4.0及以上
  * 修改script/x86_linux.sh和test/auto_test.sh脚本中的**CMAKE**的路径, 版本建议为3.0及以上
  * 执行编译脚本
  ```Shell
  bash scripts/x86_linux.sh
  ```
