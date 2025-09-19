教程：https://liaoxuefeng.com/books/git/introduction/index.html

项目中使用Git四种流程：https://www.cnblogs.com/quartz/p/12814064.html



# 安装

Git：https://zhuanlan.zhihu.com/p/242540359



GUI：https://zhuanlan.zhihu.com/p/666417763



# 简介

![1543302039138](https://raw.githubusercontent.com/zhanghongyang42/images/main/1543302039138.png)



本地仓库：狭义的本地仓库特指版本库

​					广义的本地仓库包括 版本库，工作区，暂存区。



版本库（）：.git文件夹。这里的版本就是一个点，可以认为你每次commit都是一个版本。里面包括暂存区，master分支，指针HEAD

工作区：包含.git文件夹的目录，即可视的文件都位于工作区

暂存区：工作区到版本库的一个缓冲区域，位于.git中，工作区多次add到暂存区



忽略文件：忽略文件不保存到版本库。在Git工作区的根目录下创建一个特殊的.gitignore文件，然后把要忽略的文件名填进去



# 分支

### 分支作用

1. 并行开发：使用分支可以让团队成员同时在不同的分支上开发功能或修复bug，而不会相互干扰或影响彼此的工作。
2. 功能开发：分支可以用于开发新功能。当一个新功能需要多次提交才能完成时，使用分支可以保持主分支的代码稳定，而新功能开发不会影响其他人的工作。
3. 代码审查：使用分支，可以让其他团队成员更容易地审查你的更改。他们可以查看你的分支，并在合并到主分支之前提出建议或要求进行修改。
4. 版本发布：分支可以用于管理软件的版本发布。例如，创建一个单独的分支来发布稳定版本。新功能开发可以继续在主分支上进行开发。
5. 修复bug：当代码中出现问题时，你可以创建一个分支来修复bug，而不会影响主分支的稳定性。修复完成后，可以将修复的代码合并回主分支。



### 分支操作

1. 查看分支：

   ```
   git branch
   ```

   此命令会显示所有本地分支，当前活动分支前会有一个星号(*)。

   

2. 创建分支：

   ```
   git branch <branch-name>
   ```

   使用此命令可以创建一个新分支，其中`<branch-name>`是你要创建的分支名称。

   

3. 切换分支：

   ```
   git checkout <branch-name>
   ```

   此命令用于切换到指定的分支，其中`<branch-name>`是你要切换到的分支名称。

   切换分支，就是切换仓库。

   切换分支或者新建并切换分支时，工作区未保存的更改内容将会丢失，所以切换分支时，要将工作区的内容保存后再切换。

   切换分支时，暂存区的内容不会变化，可以提交到新分支，但是不建议这样做，也要清理好暂存区再切换分支。

   

4. 创建并切换到新分支：

   ```
   git checkout -b <branch-name>
   ```

   此命令将创建一个新分支并立即切换到该分支，其中`<branch-name>`是你要创建的分支名称。

   

5. 删除分支：

   ```
   git branch -d <branch-name>
   ```

   删除指定的分支，分支的所有commit都已经合并到当前分支或者丢弃后才能成功删除。如果存在未合并的更改，需要使用`-D`选项来强制删除。

   

6. 合并分支：

   ```
   git merge <branch-name>
   ```

   此命令将指定分支的更改合并到当前分支，其中`<branch-name>`是你要合并的分支名称。

   分支冲突时，手动解决分支冲突再合并。

   

7. 重命名分支：

   ```
   git branch -m <old-branch-name> <new-branch-name>使用此命令可以重命名分支，将`<old-branch-name>`重命名为`<new-branch-name>`。
   ```

   

8. 查看所有分支（本地和远程）：

   ```
   git branch -a
   ```

   此命令会显示所有本地和远程分支。



# 版本回退

### 工作区回退

命令：git checkout -- <文件名>

若文件已修改但尚未 git add，该命令可将其还原为版本库中最新提交的内容。

若文件已 git add，之后又进行了修改，该命令可将其回退至暂存区状态，撤销后续修改。



### 清空暂存区

git reset HEAD file



### 版本回退

版本库会维护所有commit过的版本，版本回退，会同时改变 暂存区+工作区。



查看所有分支的操作记录

```
git reflog 
```



查看当前分支所有commit记录

```
git log –pretty=oneline
```



回退到指定版本号

```
git reset --hard 版本号 
```



# -----------------------



# 远程仓库

可以自建远程仓库服务器，也可以使用github，gitlab等远程仓库。

一般的远程仓库提供ssh和http两种方式连接交互，ssh需要生成ssh密钥，http需要账号密码。



为了避免分支冲突，一定要先pull，确认没有冲突，再push。



1. 查看远程仓库：

   ```
   git remote -v
   ```

   此命令会显示所有远程仓库及其对应的URL。

   

2. 删除远程仓库：

   ```
   git remote rm <remote-name>
   ```

   使用此命令可以删除一个远程仓库，其中`<remote-name>`是你要删除的远程仓库名称。

   

3. 添加远程仓库：

   ```
   git remote add <remote-name> <remote-url>
   git remote add origin https://github.com/zhanghongyang42/远程仓库名.git
   ```

   使用此命令添加一个新的远程仓库，其中`<remote-name>`是远程仓库的名称（通常为origin），`<remote-url>`是远程仓库的URL。

   适用于本地已经有仓库，建一个远程空仓库，然后将本地仓库和远程仓库关联。

   

4. 克隆远程仓库：

   ```
   git clone <remote-url>
   git clone https://github.com/zhanghongyang42/远程仓库名
   ```

   使用此命令可以将远程仓库克隆到本地计算机，其中`<remote-url>`是远程仓库的URL。

   这种方法，本地要有远程的账号密码或者ssh公钥才能使用。ssh公钥配置暂时省略。

   ```
   需要的时候要输入 gitlab或者github的账户名和密码
   
   清除gitlab或者github的账户名和密码
   git credential-manager uninstall
   
   永久记住密码
   git config credential.helper store
   ```

   

5. 拉取远程仓库的变更并合并到当前分支：

   ```
   git pull <remote-name> <remote-branch-name>
   ```

   使用此命令可以从远程仓库拉取最新的更改，并将其合并到当前分支。

   `git pull`：该命令实际上是 `git fetch` 和 `git merge` 的组合。它会从远程仓库获取最新更改，并自动将这些更改合并到当前分支

   

6. 推送本地分支到远程仓库：

   ```
   git push <remote-name> <branch-name>
   ```

   将指定的本地分支推送到远程仓库，其中`<remote-name>`是远程仓库的名称（通常为origin），`<branch-name>`是你要推送的本地分支名称。



# 分支映射

1. 查看远程分支：

   ```
   git branch -r
   ```

   此命令会显示所有远程分支。

   

2. 查看映射关系

   ```
   git branch -vv
   ```

   

3. 设置本地分支跟踪远程分支：

   ```
   git branch --set-upstream-to=<remote-name>/<remote-branch-name> <local-branch-name>
   ```

   此命令将指定的本地分支与远程分支关联，以便在执行`git pull`和`git push`时自动同步。

   

4. 取消本地分支的跟踪远程分支：

   ```
   git branch --unset-upstream <local-branch-name>
   ```

   使用此命令可以取消指定本地分支对远程分支的跟踪。`<local-branch-name>`是你要取消跟踪的本地分支名称。



- 快捷命令：创建一个和远程分支相同本地分支，并跟踪远程分支：

  ```
  git checkout --track <remote-name>/<remote-branch-name>
  ```

  使用此命令可以将本地分支与远程分支关联起来，实现跟踪。

  

- 快捷命令：创建一个和本地分支相同远程分支，并跟踪远程分支：

  推送本地分支到远程仓库并设置跟踪关系：

  ```
  git push -u <remote-name> <local-branch-name>
  ```

  使用此命令可以将本地分支推送到远程仓库，并设置本地分支与远程分支的跟踪关系。

  

- 快捷命令：创建一个和远程分支相同本地分支，但是不会跟踪远程分支：

  ```
  git checkout -b <local-branch-name> <remote-name>/<remote-branch-name>
  git checkout -b my-feature origin/master
  ```



# -----------------------



# IDEA集成

### 本地推送项目到远程仓库



##### 准备工作

在开始之前，确保你已经安装了IntelliJ IDEA，并且你有一个GitHub账号。如果你还没有GitHub账号，你可以在[GitHub官网](https://link.juejin.cn?target=https%3A%2F%2Fgithub.com)上注册一个。



##### 第一步：在IDEA中配置Git

1. 打开IntelliJ IDEA。
2. 点击 `File` > `Settings` (对于macOS是 `IntelliJ IDEA` > `Preferences`)。
3. 在设置窗口中选择 `Version Control` > `Git`。
4. 确认`Git`已正确安装，并且IDEA可以定位到`git.exe`的路径（对于macOS或Linux系统是`git`）。
5. 如果需要，点击 `Test` 按钮来验证设置是否正确。
6. 点击 `OK` 保存并关闭设置窗口。



##### 第二步：在IDEA中配置GitHub账户

1. 在设置窗口中，选择 `Version Control` > `GitHub`。
2. 点击 `Add account` 或 `+` 符号添加你的GitHub账户。
3. 输入你的GitHub用户名和密码，或者使用token方式登录。
4. 如果启用了双因素认证，你需要提供一个个人访问令牌。
5. 点击 `OK` 来保存账户信息。



##### 第三步：将本地项目变为Git仓库

1. 打开你想要推送到GitHub的项目。
2. 点击 `VCS` 菜单并选择 `Import into Version Control` > `Create Git Repository`。
3. 在弹出的窗口中选择项目根目录，点击 `OK` 创建Git仓库。



##### 第四步：添加文件到Git仓库

1. 在项目窗口中，右键点击你想要添加到仓库的文件或文件夹。
2. 选择 `Git` > `Add`。
3. 被选中的文件现在会显示为绿色，表示它们已被添加到Git仓库。



##### 第五步：提交更改到本地仓库

1. 点击 `VCS` > `Commit` (或使用快捷键 `Ctrl+K` / `Cmd+K`)。
2. 在弹出的窗口中，填写提交信息。
3. 确认你要提交的文件，然后点击 `Commit` 按钮。



##### 第六步：创建GitHub仓库

1. 点击 `VCS` > `Import into Version Control` > `Share Project on GitHub`。
2. 在弹出的窗口中，输入你的GitHub仓库名称和描述。
3. 点击 `Share`。



##### 第七步：推送更改到GitHub

1. IntelliJ IDEA将自动推送你的代码到新创建的GitHub仓库。
2. 如果需要手动推送，可以点击 `VCS` > `Git` > `Push` (或使用快捷键 `Ctrl+Shift+K` / `Cmd+Shift+K`)。
3. 在弹出的窗口中，确认推送的分支和目标，然后点击 `Push`。



##### 注意事项

- 在推送之前，请确保你的GitHub仓库是空的，没有初始化的README文件或其他文件。
- 如果遇到任何认证问题，请回到第二步，确保你的GitHub账户配置正确。

通过以上步骤，你应该能够成功地将一个本地项目推送到GitHub。如果在操作过程中遇到问题，你可以查看IntelliJ IDEA的帮助文档，或者在GitHub的帮助页面上寻找解决方案。



### 从远程仓库克隆

​	关闭工程后，在idea的欢迎页上有“Check out from version control”下拉框，选择git

![1543401467964](https://raw.githubusercontent.com/zhanghongyang42/images/main/1543401467964.png)

![1543401508478](https://raw.githubusercontent.com/zhanghongyang42/images/main/1543401508478.png)



![1543401651635](https://raw.githubusercontent.com/zhanghongyang42/images/main/1543401651635.png)

> 使用idea选择克隆后, 会出现如下内容, 一![1543401862519](https://raw.githubusercontent.com/zhanghongyang42/images/main/1543401862519.png)直下一步即可



# Github使用

- 在GitHub上，可以任意Fork开源仓库；
- 自己拥有Fork后的仓库的读写权限；
- 可以clone，然后修改Fork后的仓库，然后push修改
- 可以把自己修改过的Fork后的仓库 pull request给官方仓库来贡献代码。

