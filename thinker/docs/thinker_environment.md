
# pip安装方式

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

## 安装thinker
pip install pythinker

即可使用离线工具tpacker对模型进行打包