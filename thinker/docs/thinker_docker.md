# docker镜像安装
## docker的安装
* [Mac](https://docs.docker.com/desktop/install/mac-install/)
* [Windows](https://docs.docker.com/desktop/install/windows-install/)
* [Ubuntu](https://docs.docker.com/engine/install/ubuntu/)
* [Debian](https://docs.docker.com/engine/install/debian/)
* [Centos](https://docs.docker.com/engine/install/centos/)
* [Fedora](https://docs.docker.com/engine/install/fedora/)
* [其他 LINUX 发行版](https://docs.docker.com/engine/install/binaries/)
安装完成后，运行下面的命令，验证是否安装成功
```Shell
$ docker version
```
docker 需要用户具有 sudo 权限，为了避免每次命令都输入sudo，可以把用户加入 docker 用户组
```Shell
$ sudo groupadd docker  # 添加docker用户组
$ sudo gpasswd -a $USER docker   # 将登陆用户加入到docker用户组中
$ newgrp docker     # 更新用户组
$ docker ps    # 测试docker命令是否可以使用sudo正常使用
```
检查是否成功：

执行"docker version"命令，发现不再出现"Got permission denied"权限报错

命令行运行docker命令的时候，需要本机有 Docker 服务。安装下载 docker 并用下面的命令启动。
```Shell
$ sudo service  start docker # service  命令的用法
$ sudo systemctl start docker # systemctl 命令的用法
```

## 方案1，下载安装 docker 镜像
在 hub.docker.com 上注册登录后, 执行命令, 输入用户名和密码
```Shell
$ docker login # 
```

登录成功后 , 拉取已经制作好的 image 到本地。
```Shell
$ docker pull bzcai2022:thinker:0.1.0
```
运行容器
```Shell
$ docker container run -it bzcai2022/thinker:0.1.0 /bin/bash
``` 

如果一切正常，运行上面的命令以后，就会返回一个命令行提示符。
```Shell
root@66d80f4aaf1e:/thinker#
```  

这表示你已经在容器里面了，返回的提示符就是容器内部的 Shell 提示符。能够执行命令。
```Shell
root@66d80f4aaf1e:/thinker# ./scripts/x86_linux.sh
```
## 方案2 , 编写Dockerfile , 生成docker容器
使用本地 Dockerfile , 创建 image 文件. 
```Shell
$ docker image build -t thinker:0.1.0 . (.表示当前路径)
```
(注: x86_linux.sh 脚本中 CMAKE_ROOT 根据 anoconda 路径修改 , module load gcc 可以注释掉)


从 image 文件生成容器
```Shell
$ docker container run -it thinker:0.1.0 /bin/bash
```
能够执行命令。
```Shell
root@66d80f4aaf1e:/thinker# ./scripts/x86_linux.sh
```
## 容器的退出
image 文件生成的容器实例，本身也是一个文件，称为容器文件。查看容器文件
```Shell
$ docker container ls # 列出本机正在运行的容器
$ docker container ls --all # 列出本机所有容器，包括终止运行的容器
```
终止运行的容器文件
```Shell
$ docker container kill [containID] 
```

#### 容器内部的退出
|  方式  |  结果       |  再次启动  |
| ----   | ----        |----   |
|exit     |退出后,容器消失并销毁，ps查不到|docker start容器名/容器id|
|ctrl + D     |退出后,容器消失并销毁，ps查不到|docker start容器名/容器id|
|先按 ctrl + p,再按 ctrl + q  |退出后,容器后台运行，ps能查到|docker start容器名/容器id|

***

## 容器与宿主机的文件交互
查询容器ID
```Shell
$ docker ps 
```
从宿主机拷文件到容器里面
* docker cp 要拷贝的文件路径 容器名：要拷贝到容器里面对应的路径
```Shell
$ docker cp model 2ef7893f06bc:thinker  
```
从容器里面拷文件到宿主机
* docker cp 容器名：要拷贝的文件在容器里面的路径       要拷贝到宿主机的相应路径
```Shell
$ docker cp 2ef7893f06bc:/models /opt
```
