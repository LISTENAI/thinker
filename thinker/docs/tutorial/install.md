## 环境配置

源码和pip源安装方式需要在本地配置环境，**镜像安装方式可跳过**  
推荐使用conda创建独立的虚拟环境，以下介绍虚拟环境如何配置

### 创建虚拟环境
```Shell
conda create -n thinker-env python==3.8.5
conda activate thinker-env
pip install -U pip
cat requirements.txt | xargs -n 1 pip install
```
若无法执行，则采用手动安装requirements中的各种依赖项

### 查看已创建的环境
```Shell
conda info --env
```

### 查看当前环境中已安装的包
```Shell
conda list
```

### 删除不要的环境
```Shell
conda activate base(或者 conda deactivate xxx)
conda remove -n xxx --all
```

## thinker安装
三种方式，任选一种
### 源码安装方式
git的安装和设置请自行配置
``` Shell
git clone git@github.com:LISTENAI/thinker.git
cd thinker && sh ./script/x86_linux.sh
```

### pip包安装方式
``` shell
pip install pythinker==1.1.0
```

### docker镜像安装方式
1、docker安装  
如果未安装docker工具，请参考以下链接进行安装（建议使用Centos系统）。如果已安装则直接进行权限验证
* [Ubuntu](https://docs.docker.com/engine/install/ubuntu/)
* [Debian](https://docs.docker.com/engine/install/debian/)
* [Centos](https://docs.docker.com/engine/install/centos/)
* [其他 LINUX 发行版](https://docs.docker.com/engine/install/binaries/)  

安装完成后，运行下面的命令查看版本，验证是否安装成功并有足够权限
```Shell
$ docker version
```
如果出现"Got permission denied"权限报错，说明当前用户权限不够，需要添加权限  
docker 需要用户具有 sudo 权限，但不宜直接使用root用户进行操作  
为了避免每次命令都输入sudo，可以建立 docker 用户组并把用户加入  
```Shell
$ sudo groupadd docker  # 添加docker用户组
$ sudo gpasswd -a $USER docker   # 将登陆用户加入到docker用户组中
$ newgrp docker     # 更新用户组
$ docker ps    # 测试docker命令是否可以使用sudo正常使用
```
再次执行"docker version"命令，发现不再出现"Got permission denied"权限报错，继续下一步。

2、启动docker服务
```Shell
$ sudo systemctl start docker # systemctl 命令的用法
```

3、拉取镜像并加载  
1）、拉取镜像
```shell
docker pull listenai/thinker:1.1.0
```

2）、运行容器
```shell
docker container run -it listenai/thinker:1.1.0 /bin/bash
```

如果一切正常，运行上面的命令以后，就会返回一个命令行提示符。
```shell
root@66d80f4aaf1e:/LISTENAI#
```

这表示你已经在容器里面了，返回的提示符就是容器内部的 Shell 提示符。这里进入thinker目录，能够执行thinker相关命令。

3）、其它可能的操作  
* 查询容器ID
```Shell
$ docker ps 
```
* 从宿主机拷文件到容器里面
docker cp 要拷贝的文件路径 容器名：要拷贝到容器里面对应的路径
```Shell
$ docker cp model 2ef7893f06bc:thinker  
```

* 从容器里面拷文件到宿主机
docker cp 容器名：要拷贝的文件在容器里面的路径       要拷贝到宿主机的相应路径
```Shell
$ docker cp 2ef7893f06bc:/models /opt
```

* 终止运行的容器文件
```Shell
$ docker container kill [containID] 
```
* 移除停止的容器文件
```Shell
$ docker container rm [containID] 
```

容器内部的退出  
|  方式  |  结果       |  再次启动  |
| ----   | ----        |----   |
|exit     |退出后,容器消失并销毁，ps查不到|docker start 容器名/容器id|
|ctrl + D     |退出后,容器消失并销毁，ps查不到|docker start 容器名/容器id|
|先按 ctrl + p,再按 ctrl + q  |退出后,容器后台运行，ps能查到|docker start 容器名/容器id|  
  
## thinker安装验证
### 1、分析打包工具的验证
``` Shell
tpacker -g model/xx.onnx
```
能够正常对模型进行分析打包或者对模型的路径及分析过程报错，即可认为安装成功。
如果提示找不到tpacker指令，则表明安装未成功

### 2、测试用例和thinker库编译
gcc版本为5.4.0及以上,修改script/x86_linux.sh和test/auto_test.sh脚本中的**CMAKE**的路径, 版本建议为3.0及以上  
执行编译脚本  
```Shell
bash scripts/x86_linux.sh
```
