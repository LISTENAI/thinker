
# thinker环境配置

## 采用下列脚本，自动安装需要的python库
```Shell
conda create -n thinker python==3.8.5
conda activate thinker
pip install -U pip
cat requirements.txt | xargs -n 1 pip install
```
若无法执行，则采用手动安装requirements中的各种库

删除不要的环境有两种方式：
```Shell
conda activate base(或者 conda deactivate xxx)
conda remove -n xxx --all
```

查看已创建的环境
```Shell
conda info --env
```

查看当前环境中已安装的包
```Shell
conda list
```