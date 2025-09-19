官网：https://docs.docker.com/

教程1：https://cloud.tencent.com/developer/article/1885678

教程2：https://www.ruanyifeng.com/blog/2018/02/docker-tutorial.html

教程3：https://yeasy.gitbook.io/docker_practice/



# 概念

Docker是一个虚拟环境容器，可以将你的开发环境、代码、配置文件等一并打包到这个容器中，并发布和应用到任意平台中。



Linux 容器不是模拟一个完整的操作系统，而是对进程进行隔离。或者说，在正常进程的外面套了一个[保护层](https://opensource.com/article/18/1/history-low-level-container-runtimes)。对于容器里面的进程来说，它接触到的各种资源都是虚拟的，从而实现与底层系统的隔离。



Docker 的主要用途，目前有三大类。

**（1）提供一次性的环境。**比如，本地测试他人的软件、持续集成的时候提供单元测试和构建的环境。

**（2）提供弹性的云服务。**因为 Docker 容器可以随开随关，很适合动态扩容和缩容。

**（3）组建微服务架构。**通过多个容器，一台机器可以跑多个服务，因此在本机就可以模拟出微服务架构。



# 安装

Docker 是一个开源的商业产品，有两个版本：社区版（Community Edition，缩写为 CE）和企业版（Enterprise Edition，缩写为 EE）。企业版包含了一些收费服务，个人开发者一般用不到。下面的介绍都针对社区版。



Linux：https://docs.docker.com/engine/install/

windows：https://zhuanlan.zhihu.com/p/441965046

​					https://zhuanlan.zhihu.com/p/600936178



```shell
# 安装是否成功
docker version
# 服务端启动
sudo service docker start
```



# Image

Docker 把应用程序及其依赖，打包在 **image** 文件里面。用于生成 Docker 容器，image 文件可以看作是容器的模板。

实际开发中，一个 image 文件往往通过继承另一个 image 文件，加上一些个性化设置而生成。



为了方便共享，image 文件制作完成后，可以上传到网上的仓库。

Docker 的官方仓库 [Docker Hub](https://hub.docker.com/) 是最重要、最常用的 image 仓库。

```shell
# 列出本机的所有 image 文件。
docker image ls

# 删除 image 文件
docker image rm [imageName]
```

更改使用镜像仓库

```
打开/etc/default/docker文件（需要sudo权限），在文件的底部加上一行。
DOCKER_OPTS="--registry-mirror=https://registry.docker-cn.com"

重启 Docker 服务。
sudo service docker restart
```

```shell
# 下载镜像
docker image pull hello-world

# 运行镜像，生成容器
docker container run hello-world
```



```bash
#  image 生成容器
docker container run -p 8000:3000 -it koa-demo:0.0.1 /bin/bash
```

```
-p参数：容器的 3000 端口映射到本机的 8000 端口。

-it参数：容器的 Shell 映射到当前的 Shell，然后你在本机窗口输入的命令，就会传入容器。

koa-demo:0.0.1：image 文件的名字（如果有标签，还需要提供标签，默认是 latest 标签）。

/bin/bash：容器启动以后，内部第一个执行的命令。这里是启动 Bash，保证用户可以使用 Shell。
```



# Container

**image 文件生成的容器实例，本身也是一个文件，称为容器文件。**关闭容器并不会删除容器文件，只是容器停止运行而已。

```bash
# 列出本机正在运行的容器
$ docker container ls

# 列出本机所有容器，包括终止运行的容器
$ docker container ls --all
```

```bash
# 容器生成
docker container run -p 8000:3000 -it koa-demo:0.0.1 /bin/bash

# 退出容器界面,不终止容器
ctrl+p+q

# 进入容器界面
docker container exec -it [containerID] /bin/bash
docker attach container_name/container_id

# 容器停止
docker container stop [containerID]
# 容器强行终止
docker container kill [containID]

# 容器启动
docker container start [containerID]

# 删除容器文件
docker container rm [containerID]
```

```bash
# 查看 docker 容器的输出
docker container logs [containerID]

# 从正在运行的 Docker 容器里面，将文件拷贝到本机
docker container cp [containID]:[/path/to/file] ./
```



# Dockerfile

https://zhuanlan.zhihu.com/p/430989391

https://zhuanlan.zhihu.com/p/385402838



Docker 根据 Dockerfile 的文件配置生成 image 文件。



首先要有一个项目

```bash
git clone https://github.com/ruanyf/koa-demos.git
cd koa-demos
```

忽略文件，不进入 image 。根目录新建一个文本文件`.dockerignore`，写入下面的内容。

```
.git
node_modules
npm-debug.log
```

在项目的根目录下，新建一个文本文件 Dockerfile，写入以下内容。

```bash
FROM node:8.4
COPY ./app
WORKDIR /app
RUN npm install --registry=https://registry.npm.taobao.org
EXPOSE 3000
CMD node demos/01.js
```

```
FROM node:8.4：		该 image 文件继承官方的 node image，冒号表示标签，这里标签是8.4，即8.4版本的 node。

COPY ./app：			将当前目录下的所有文件（除了.dockerignore排除的路径），都拷贝进入 image 文件的/app目录。

WORKDIR /app：		指定接下来的工作路径为/app。

RUN npm install：	在/app目录下，运行npm install命令安装依赖。注意，安装后所有的依赖，都将打包进入 image 文件。

EXPOSE 3000：		将容器 3000 端口暴露出来， 允许外部连接这个端口。

CMD node：			容器启动后自动执行node demos/01.js。一个 Dockerfile 可以包含多个RUN命令，但是只能有一个CMD命令。
					 指定了CMD命令以后，docker container run命令就不能附加命令了（比如前面的/bin/bash），否则它会覆盖CMD命令。
```

Dockerfile 创建 image 文件。

```bash
docker image build -t koa-demo:0.0.1 .
```

将自己自定义的镜像发布（push）到DockerHub上。

```shell
# 访问 https://hub.docker.com/ 注册

# 登录DockerHub，输入用户名、密码
docker login

# 标注用户名和版本
docker image tag koa-demos:0.0.1 ruanyf/koa-demos:0.0.1

# 推送镜像到DockerHub
docker image push [username]/[repository]:[tag]
```



# 微服务实战

https://www.ruanyifeng.com/blog/2018/02/docker-wordpress-tutorial.html



# 生成镜像

容器转化为镜像

```shell
docker commit -m "centos with git" -a "qixianhu" 72f1a8a0e394 xianhu/centos:git
docker save -o centos.tar xianhu/centos:git    # 保存镜像
docker load -i centos.tar    # 加载镜像
#-m指定说明信息
#-a指定用户信息
#72f1a8a0e394代表容器的id
#xianhu/centos:git指定目标镜像的用户名、仓库名和 tag 信息
```





















