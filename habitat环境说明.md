# habitat环境配置说明
参考来源：  
1.NaVILA复现踩坑记录 https://zhuanlan.zhihu.com/p/1942233936064385661  
2.navid复现记录
https://zhuanlan.zhihu.com/p/1892632890208146583  
3.Navid复现记录
https://zhuanlan.zhihu.com/p/28568886663  
4.论文阅读及复现笔记之——《NaVILA: Legged Robot Vision-Language-Action Model for Navigation》
https://kwanwaipang.github.io/NaVILA/#%E5%AE%9E%E9%AA%8C%E9%85%8D%E7%BD%AE    

## 一、创建conda环境  
```bash
conda create -n vlnce3 python=3.8
conda activate vlnce3
```
## 二、接着安装habitat-sim 0.1.7，基本上不可能通过命令安装，下载安装包离线安装  
离线包下载地址
https://anaconda.org/aihabitat/habitat-sim/files?version=0.1.7  

```bash
conda install habitat-sim-0.1.7-py3.8_headless_linux_856d4b08c1a2632626bf0d205bf46471a99502b7.tar.bz2
```
## 三、下载habitat-lab
```bash
git clone --branch v0.1.7 git@github.com:facebookresearch/habitat-lab.git
or
git clone --branch v0.1.7 https://github.com/facebookresearch/habitat-lab.git

cd habitat-lab
# installs both habitat and habitat_baselines
python -m pip install -r requirements.txt
# 注意，下面其中的tensorflow==1.13.1似乎已经不支持安装了，改为tensorflow>=2.8.0或者直接注释掉，看下面的注释
python -m pip install -r habitat_baselines/rl/requirements.txt

python -m pip install -r habitat_baselines/rl/ddppo/requirements.txt

python setup.py develop --all
```
上面安装可能遇到的问题：   

### 3.1  一般会遇到tensorflow的问题

建议修改habitat_baselines/rl/requirements.txt。参考issueQuestion about env set up · Issue #10 · jzhzhang/NaVid-VLN-CE[https://link.zhihu.com/?target=https%3A//github.com/jzhzhang/NaVid-VLN-CE/issues/10],直接注释掉。
```bash
moviepy>=1.0.1
torch==2.0.1
# full tensorflow required for tensorboard video support
# tensorflow==2.8.0
tb-nightly
```

### 3.2 安装tb-nightly报错  

ERROR: Could not find a version that satisfies the requirement tb-nightly (from versions: none)
ERROR: No matching distribution found for tb-nightly

解决方案：

使用下面这行安装指令：
```bash
pip install tb_nightly==2.14.0a20230808 -i https://mirrors.aliyun.com/pypi/simple
```

备注：tb_nightly是tensorboard的早期版本，阿里云镜像可以正常安装


### 3.3 运行 python setup.py develop --all 报错

<details>
<summary>点击查看代码 </summary>

```bash
(vlnce3) guest@gpu3-labot:~/ZBD/habitat-lab$ python setup.py develop --all
/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/__init__.py:80: _DeprecatedInstaller: setuptools.installer and fetch_build_eggs are deprecated.
!!

        ********************************************************************************
        Requirements should be satisfied by a PEP 517 installer.
        If you are using pip, you can try `pip install --use-pep517`.
        ********************************************************************************

!!
  dist.fetch_build_eggs(dist.setup_requires)
running develop
/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/command/develop.py:42: EasyInstallDeprecationWarning: easy_install command is deprecated.
!!

        ********************************************************************************
        Please avoid running ``setup.py`` and ``easy_install``.
        Instead, use pypa/build, pypa/installer or other
        standards-based tools.

        See https://github.com/pypa/setuptools/issues/917 for details.
        ********************************************************************************

!!
  easy_install.initialize_options(self)
setup.py:57: SetuptoolsDeprecationWarning: setup.py install is deprecated.
!!

        ********************************************************************************
        Please avoid running ``setup.py`` directly.
        Instead, use pypa/build, pypa/installer or other
        standards-based tools.

        See https://blog.ganssle.io/articles/2021/10/setup-py-deprecated.html for details.
        ********************************************************************************

!!
  super().initialize_options()
running egg_info
creating habitat.egg-info
writing habitat.egg-info/PKG-INFO
writing dependency_links to habitat.egg-info/dependency_links.txt
writing requirements to habitat.egg-info/requires.txt
writing top-level names to habitat.egg-info/top_level.txt
writing manifest file 'habitat.egg-info/SOURCES.txt'
reading manifest file 'habitat.egg-info/SOURCES.txt'
reading manifest template 'MANIFEST.in'
adding license file 'LICENSE'
writing manifest file 'habitat.egg-info/SOURCES.txt'
running build_ext
Creating /home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/habitat.egg-link (link to .)
Adding habitat 0.1.7 to easy-install.pth file

Installed /home/guest/ZBD/habitat-lab
Processing dependencies for habitat==0.1.7
Searching for lmdb>=0.98
Reading https://pypi.org/simple/lmdb/
Downloading https://files.pythonhosted.org/packages/bd/2c/982cb5afed533d0cb8038232b40c19b5b85a2d887dec74dfd39e8351ef4b/lmdb-1.7.5-py3-none-any.whl#sha256=fc344bb8bc0786c87c4ccb19b31f09a38c08bd159ada6f037d669426fea06f03
Best match: lmdb 1.7.5
Processing lmdb-1.7.5-py3-none-any.whl
Installing lmdb-1.7.5-py3-none-any.whl to /home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages
Adding lmdb 1.7.5 to easy-install.pth file

Installed /home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/lmdb-1.7.5-py3.8.egg
Searching for webdataset==0.1.40
Reading https://pypi.org/simple/webdataset/
Downloading https://files.pythonhosted.org/packages/16/bc/8b98d07eb97a51584ff305b468a13628f3905964d597c62513f6beacb4a4/webdataset-0.1.40-py3-none-any.whl#sha256=f20e3f1143395a321fad7d6cc10e07c4b269f0270ef6fdc1d4ed94421bce1bf0
Best match: webdataset 0.1.40
Processing webdataset-0.1.40-py3-none-any.whl
Installing webdataset-0.1.40-py3-none-any.whl to /home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages
Adding webdataset 0.1.40 to easy-install.pth file
detected new path './lmdb-1.7.5-py3.8.egg'

Installed /home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/webdataset-0.1.40-py3.8.egg
Searching for setuptools
Reading https://pypi.org/simple/setuptools/
Downloading https://files.pythonhosted.org/packages/a3/dc/17031897dae0efacfea57dfd3a82fdd2a2aeb58e0ff71b77b87e44edc772/setuptools-80.9.0-py3-none-any.whl#sha256=062d34222ad13e0cc312a4c02d73f059e86a4acbfbdea8f8f76b28c99f306922
Best match: setuptools 80.9.0
Processing setuptools-80.9.0-py3-none-any.whl
Installing setuptools-80.9.0-py3-none-any.whl to /home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages
Adding setuptools 80.9.0 to easy-install.pth file
detected new path './webdataset-0.1.40-py3.8.egg'

Installed /home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools-80.9.0-py3.8.egg
Searching for cffi>=0.8
Reading https://pypi.org/simple/cffi/
Downloading https://files.pythonhosted.org/packages/eb/56/b1ba7935a17738ae8453301356628e8147c79dbb825bcbc73dc7401f9846/cffi-2.0.0.tar.gz#sha256=44d1b5909021139fe36001ae048dbdde8214afa20200eda0f64c068cac5d5529
Best match: cffi 2.0.0
Processing cffi-2.0.0.tar.gz
Writing /tmp/easy_install-xt612p02/cffi-2.0.0/setup.cfg
Running cffi-2.0.0/setup.py -q bdist_egg --dist-dir /tmp/easy_install-xt612p02/cffi-2.0.0/egg-dist-tmp-7u28optt
compiling '_configtest.c':
__thread int some_threadlocal_variable_42;

gcc -pthread -B /home/guest/anaconda3/envs/vlnce3/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -fwrapv -O2 -Wall -fPIC -O2 -isystem /home/guest/anaconda3/envs/vlnce3/include -fPIC -O2 -isystem /home/guest/anaconda3/envs/vlnce3/include -fPIC -c _configtest.c -o _configtest.o
success!
removing: _configtest.c _configtest.o
compiling '_configtest.c':
int main(void) { __sync_synchronize(); return 0; }

gcc -pthread -B /home/guest/anaconda3/envs/vlnce3/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -fwrapv -O2 -Wall -fPIC -O2 -isystem /home/guest/anaconda3/envs/vlnce3/include -fPIC -O2 -isystem /home/guest/anaconda3/envs/vlnce3/include -fPIC -c _configtest.c -o _configtest.o
gcc -pthread -B /home/guest/anaconda3/envs/vlnce3/compiler_compat _configtest.o -o _configtest
success!
removing: _configtest.c _configtest.o _configtest
Traceback (most recent call last):
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/sandbox.py", line 158, in save_modules
    yield saved
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/sandbox.py", line 200, in setup_context
    yield
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/sandbox.py", line 261, in run_setup
    _execfile(setup_script, ns)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/sandbox.py", line 48, in _execfile
    exec(code, globals, locals)
  File "/tmp/easy_install-xt612p02/cffi-2.0.0/setup.py", line 185, in <module>
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/__init__.py", line 103, in setup
    return distutils.core.setup(**attrs)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/core.py", line 158, in setup
    dist.parse_config_files()
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/dist.py", line 632, in parse_config_files
    pyprojecttoml.apply_configuration(self, filename, ignore_option_errors)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/config/pyprojecttoml.py", line 70, in apply_configuration
    config = read_configuration(filepath, True, ignore_option_errors, dist)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/config/pyprojecttoml.py", line 135, in read_configuration
    validate(subset, filepath)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/config/pyprojecttoml.py", line 59, in validate
    raise ValueError(f"{error}\n{summary}") from None
ValueError: invalid pyproject.toml config: `project.license`.
configuration error: `project.license` must be valid exactly by one definition (2 matches found):

    - keys:
        'file': {type: string}
      required: ['file']
    - keys:
        'text': {type: string}
      required: ['text']


During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "setup.py", line 87, in <module>
    setuptools.setup(
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/__init__.py", line 103, in setup
    return distutils.core.setup(**attrs)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/core.py", line 184, in setup
    return run_commands(dist)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/core.py", line 200, in run_commands
    dist.run_commands()
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/dist.py", line 970, in run_commands
    self.run_command(cmd)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/dist.py", line 974, in run_command
    super().run_command(command)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/dist.py", line 989, in run_command
    cmd_obj.run()
  File "setup.py", line 69, in run
    super().run()
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/command/develop.py", line 36, in run
    self.install_for_development()
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/command/develop.py", line 128, in install_for_development
    self.process_distribution(None, self.dist, not self.no_deps)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/command/easy_install.py", line 785, in process_distribution
    distros = WorkingSet([]).resolve(
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/pkg_resources/__init__.py", line 889, in resolve
    dist = self._resolve_dist(
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/pkg_resources/__init__.py", line 925, in _resolve_dist
    dist = best[req.key] = env.best_match(
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/pkg_resources/__init__.py", line 1256, in best_match
    return self.obtain(req, installer)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/pkg_resources/__init__.py", line 1292, in obtain
    return installer(requirement) if installer else None
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/command/easy_install.py", line 709, in easy_install
    return self.install_item(spec, dist.location, tmpdir, deps)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/command/easy_install.py", line 734, in install_item
    dists = self.install_eggs(spec, download, tmpdir)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/command/easy_install.py", line 931, in install_eggs
    return self.build_and_install(setup_script, setup_base)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/command/easy_install.py", line 1203, in build_and_install
    self.run_setup(setup_script, setup_base, args)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/command/easy_install.py", line 1189, in run_setup
    run_setup(setup_script, args)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/sandbox.py", line 264, in run_setup
    raise
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/contextlib.py", line 131, in __exit__
    self.gen.throw(type, value, traceback)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/sandbox.py", line 200, in setup_context
    yield
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/contextlib.py", line 131, in __exit__
    self.gen.throw(type, value, traceback)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/sandbox.py", line 171, in save_modules
    saved_exc.resume()
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/sandbox.py", line 145, in resume
    raise exc.with_traceback(self._tb)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/sandbox.py", line 158, in save_modules
    yield saved
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/sandbox.py", line 200, in setup_context
    yield
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/sandbox.py", line 261, in run_setup
    _execfile(setup_script, ns)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/sandbox.py", line 48, in _execfile
    exec(code, globals, locals)
  File "/tmp/easy_install-xt612p02/cffi-2.0.0/setup.py", line 185, in <module>
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/__init__.py", line 103, in setup
    return distutils.core.setup(**attrs)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/core.py", line 158, in setup
    dist.parse_config_files()
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/dist.py", line 632, in parse_config_files
    pyprojecttoml.apply_configuration(self, filename, ignore_option_errors)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/config/pyprojecttoml.py", line 70, in apply_configuration
    config = read_configuration(filepath, True, ignore_option_errors, dist)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/config/pyprojecttoml.py", line 135, in read_configuration
    validate(subset, filepath)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/config/pyprojecttoml.py", line 59, in validate
    raise ValueError(f"{error}\n{summary}") from None
ValueError: invalid pyproject.toml config: `project.license`.
configuration error: `project.license` must be valid exactly by one definition (2 matches found):

    - keys:
        'file': {type: string}
      required: ['file']
    - keys:
        'text': {type: string}
      required: ['text']
```
</details>  

原因分析：

Setuptools 版本过高： 你的环境中自动下载了 setuptools 80.9.0（非常新的版本）。新版 Setuptools 对 pyproject.toml 配置文件的校验非常严格。

构建方式过旧： 你使用的 python setup.py develop 是旧式的安装命令，它触发了 easy_install 机制去编译 cffi 库。

冲突点： cffi 2.0.0（最新版）的配置文件在被旧式安装流程调用新版 Setuptools 解析时，触发了 project.license 的校验错误。

解决方案：

```bash
pip install "cffi==1.17.1"
pip install "msgpack==1.0.0"
python setup.py develop --all
# 似乎还可以降级 Setuptools
# pip install "setuptools<70.0.0"
```

## 四、navid安装

```bash
cd ..
git clone git@github.com:jzhzhang/NaVid-VLN-CE.git
or 
git clone https://github.com/jzhzhang/NaVid-VLN-CE.git

cd NaVid-VLN-CE

pip install -r requirements.txt -i https://pypi.python.org/simple

# 评论区有人说用清华源好像会出问题
# pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 在运行上面的一行代码之前，建议修改requirements.txt

fairscale==0.4.0
numpy==1.21.6
```
可能会报的错
### 4.1 Could not find a version that satisfies the requirement puccinialin (from versions: none)

<details>
<summary>点击查看报错代码 </summary>

```bash
(vlnce3) guest@gpu3-labot:~/ZBD/NaVid-VLN-CE$ pip install -r requirements.txt -i https://pypi.org/simple
Collecting fastapi
  Downloading fastapi-0.124.4-py3-none-any.whl (113 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 113.3/113.3 kB 755.5 kB/s eta 0:00:00
Collecting gradio==3.35.2
  Downloading gradio-3.35.2-py3-none-any.whl (19.7 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 19.7/19.7 MB 5.6 MB/s eta 0:00:00
Collecting markdown2[all]
  Downloading markdown2-2.5.1-py2.py3-none-any.whl (48 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 48.4/48.4 kB 3.6 MB/s eta 0:00:00
Collecting numpy==1.23.5
  Downloading numpy-1.23.5-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (17.1 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 17.1/17.1 MB 6.2 MB/s eta 0:00:00
Requirement already satisfied: requests in /home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages (from -r requirements.txt (line 5)) (2.32.4)
Collecting sentencepiece
  Downloading sentencepiece-0.2.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.3 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.3/1.3 MB 5.3 MB/s eta 0:00:00
Collecting tokenizers>=0.12.1
  Downloading tokenizers-0.21.0.tar.gz (343 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 343.0/343.0 kB 2.7 MB/s eta 0:00:00
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Installing backend dependencies ... error
  error: subprocess-exited-with-error
  
  × pip subprocess to install backend dependencies did not run successfully.
  │ exit code: 1
  ╰─> [3 lines of output]
      ERROR: Ignored the following versions that require a different python version: 0.1.0 Requires-Python >=3.9; 0.1.1 Requires-Python >=3.9; 0.1.2 Requires-Python >=3.9; 0.1.3 Requires-Python >=3.9; 0.1.4 Requires-Python >=3.9; 0.1.5 Requires-Python >=3.9; 0.1.6 Requires-Python >=3.9; 0.1.7 Requires-Python >=3.9; 0.1.8 Requires-Python >=3.9
      ERROR: Could not find a version that satisfies the requirement puccinialin (from versions: none)
      ERROR: No matching distribution found for puccinialin
      [end of output]
  
  note: This error originates from a subprocess, and is likely not a problem with pip.
error: subprocess-exited-with-error

× pip subprocess to install backend dependencies did not run successfully.
│ exit code: 1
╰─> See above for output.

note: This error originates from a subprocess, and is likely not a problem with pip.

```

</details>

问题原因  
根据你的报错日志，问题的核心在于 tokenizers 库的最新版本（0.21.0） 与你的 Python 3.8环境不兼容。
1. Python 3.8 环境：你的日志显示你正在使用 Python 3.8 (/python3.8/site-packages)。
2. Tokenizers 更新：tokenizers 库在最近的 0.21.0 版本中放弃了对 Python 3.8 的支持（通常要求 3.9+）。
3. 构建依赖失败：pip 尝试为你下载最新的 tokenizers-0.21.0 源码包进行编译，但该版本的构建工具（puccinialin）强制要求 Python 3.9+，因此导致报错：Could not find a version that satisfies the requirement puccinialin。

解决方案  
最直接的解决方法是强制安装一个支持 Python 3.8 的旧版 tokenizers
可以在requirements.txt中注释上面的tokenizers>=0.12.1，解开下面的tokenizers==0.13.3
```bash
fastapi
gradio==3.35.2
markdown2[all]
numpy==1.23.5
requests
sentencepiece
#tokenizers>=0.12.1
torch==2.0.1
torchvision==0.15.2
uvicorn
shortuuid
peft==0.4.0
transformers==4.31.0
accelerate==0.21.0
bitsandbytes==0.41.0
scikit-learn==1.2.2
sentencepiece==0.1.99
einops==0.6.1
einops-exts==0.0.4
timm==0.6.13
gradio_client==0.2.9
fairscale
decord
absl-py
braceexpand
simplejson
dtw
fastdtw
gym==0.17.3



# accelerate==0.21.0
# transformers==4.31.0
# xformers==0.0.22
tokenizers==0.13.3
# torchvision==0.15.2
# google-auth==2.23.3
# google-auth-oauthlib==1.0.0
# google-pasta==0.2.0
# protobuf==4.24.4
```

## 五、数据集下载

### 5.1 Matterport3D (MP3D) dataset for habitat  
 
直接下载链接：http://kaldir.vc.in.tum.de/matterport/v1/tasks//mp3d_habitat.zip  

官方python下载脚本
```bash
# requires running with python 2.7
python download_mp.py --task habitat -o data/scene_datasets/mp3d/
```
download_mp.py完整代码  

<details>
<summary>点击查看代码 </summary>

```bash
#!/usr/bin/env python
# Downloads MP public data release
# Run with ./download_mp.py (or python download_mp.py on Windows)
# -*- coding: utf-8 -*-
import argparse
import collections
import os
import tempfile
import urllib

BASE_URL = 'http://kaldir.vc.cit.tum.de/matterport/'
RELEASE = 'v1/scans'
RELEASE_TASKS = 'v1/tasks/'
RELEASE_SIZE = '1.3TB'
TOS_URL = BASE_URL + 'MP_TOS.pdf'
FILETYPES = [
    'cameras',
    'matterport_camera_intrinsics',
    'matterport_camera_poses',
    'matterport_color_images',
    'matterport_depth_images',
    'matterport_hdr_images',
    'matterport_mesh',
    'matterport_skybox_images',
    'undistorted_camera_parameters',
    'undistorted_color_images',
    'undistorted_depth_images',
    'undistorted_normal_images',
    'house_segmentations',
    'region_segmentations',
    'image_overlap_data',
    'poisson_meshes',
    'sens'
]
TASK_FILES = {
    'keypoint_matching_data': ['keypoint_matching/data.zip'],
    'keypoint_matching_models': ['keypoint_matching/models.zip'],
    'surface_normal_data': ['surface_normal/data_list.zip'],
    'surface_normal_models': ['surface_normal/models.zip'],
    'region_classification_data': ['region_classification/data.zip'],
    'region_classification_models': ['region_classification/models.zip'],
    'semantic_voxel_label_data': ['semantic_voxel_label/data.zip'],
    'semantic_voxel_label_models': ['semantic_voxel_label/models.zip'],
    'minos': ['mp3d_minos.zip'],
    'gibson': ['mp3d_for_gibson.tar.gz'],
    'habitat': ['mp3d_habitat.zip'],
    'pixelsynth': ['mp3d_pixelsynth.zip'],
    'igibson': ['mp3d_for_igibson.zip'],
    'mp360': ['mp3d_360/data_00.zip', 'mp3d_360/data_01.zip', 'mp3d_360/data_02.zip', 'mp3d_360/data_03.zip', 'mp3d_360/data_04.zip', 'mp3d_360/data_05.zip', 'mp3d_360/data_06.zip']
}


def get_release_scans(release_file):
    scan_lines = urllib.urlopen(release_file)
    scans = []
    for scan_line in scan_lines:
        scan_id = scan_line.rstrip('\n')
        scans.append(scan_id)
    return scans


def download_release(release_scans, out_dir, file_types):
    print('Downloading MP release to ' + out_dir + '...')
    for scan_id in release_scans:
        scan_out_dir = os.path.join(out_dir, scan_id)
        download_scan(scan_id, scan_out_dir, file_types)
    print('Downloaded MP release.')


def download_file(url, out_file):
    out_dir = os.path.dirname(out_file)
    if not os.path.isfile(out_file):
        print '\t' + url + ' > ' + out_file
        fh, out_file_tmp = tempfile.mkstemp(dir=out_dir)
        f = os.fdopen(fh, 'w')
        f.close()
        urllib.urlretrieve(url, out_file_tmp)
        os.rename(out_file_tmp, out_file)
    else:
        print('WARNING: skipping download of existing file ' + out_file)

def download_scan(scan_id, out_dir, file_types):
    print('Downloading MP scan ' + scan_id + ' ...')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    for ft in file_types:
        url = BASE_URL + RELEASE + '/' + scan_id + '/' + ft + '.zip'
        out_file = out_dir + '/' + ft + '.zip'
        download_file(url, out_file)
    print('Downloaded scan ' + scan_id)


def download_task_data(task_data, out_dir):
    print('Downloading MP task data for ' + str(task_data) + ' ...')
    for task_data_id in task_data:
        if task_data_id in TASK_FILES:
            file = TASK_FILES[task_data_id]
            for filepart in file:
                url = BASE_URL + RELEASE_TASKS + '/' + filepart
                localpath = os.path.join(out_dir, filepart)
                localdir = os.path.dirname(localpath)
                if not os.path.isdir(localdir):
                    os.makedirs(localdir)
                    download_file(url, localpath)
                    print('Downloaded task data ' + task_data_id)


def main():
    parser = argparse.ArgumentParser(description=
        '''
        Downloads MP public data release.
        Example invocation:
          python download_mp.py -o base_dir --id ALL --type object_segmentations --task_data semantic_voxel_label_data semantic_voxel_label_models
        The -o argument is required and specifies the base_dir local directory.
        After download base_dir/v1/scans is populated with scan data, and base_dir/v1/tasks is populated with task data.
        Unzip scan files from base_dir/v1/scans and task files from base_dir/v1/tasks/task_name.
        The --type argument is optional (all data types are downloaded if unspecified).
        The --id ALL argument will download all house data. Use --id house_id to download specific house data.
        The --task_data argument is optional and will download task data and model files.
        ''',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-o', '--out_dir', required=True, help='directory in which to download')
    parser.add_argument('--task_data', default=[], nargs='+', help='task data files to download. Any of: ' + ','.join(TASK_FILES.keys()))
    parser.add_argument('--id', default='ALL', help='specific scan id to download or ALL to download entire dataset')
    parser.add_argument('--type', nargs='+', help='specific file types to download. Any of: ' + ','.join(FILETYPES))
    args = parser.parse_args()

    print('By pressing any key to continue you confirm that you have agreed to the MP terms of use as described at:')
    print(TOS_URL)
    print('***')
    print('Press any key to continue, or CTRL-C to exit.')
    key = raw_input('')

    release_file = BASE_URL + RELEASE + '.txt'
    release_scans = get_release_scans(release_file)
    file_types = FILETYPES

    # download task data
    if args.task_data:
        if set(args.task_data) & set(TASK_FILES.keys()):  # download task data
            out_dir = os.path.join(args.out_dir, RELEASE_TASKS)
            download_task_data(args.task_data, out_dir)
        else:
            print('ERROR: Unrecognized task data id: ' + args.task_data)
        print('Done downloading task_data for ' + str(args.task_data))
        key = raw_input('Press any key to continue on to main dataset download, or CTRL-C to exit.')

    # download specific file types?
    if args.type:
        if not set(args.type) & set(FILETYPES):
            print('ERROR: Invalid file type: ' + file_type)
            return
        file_types = args.type

    if args.id and args.id != 'ALL':  # download single scan
        scan_id = args.id
        if scan_id not in release_scans:
            print('ERROR: Invalid scan id: ' + scan_id)
        else:
            out_dir = os.path.join(args.out_dir, RELEASE, scan_id)
            download_scan(scan_id, out_dir, file_types)
    elif 'minos' not in args.task_data and args.id == 'ALL' or args.id == 'all':  # download entire release
        if len(file_types) == len(FILETYPES):
            print('WARNING: You are downloading the entire MP release which requires ' + RELEASE_SIZE + ' of space.')
        else:
            print('WARNING: You are downloading all MP scans of type ' + file_types[0])
        print('Note that existing scan directories will be skipped. Delete partially downloaded directories to re-download.')
        print('***')
        print('Press any key to continue, or CTRL-C to exit.')
        key = raw_input('')
        out_dir = os.path.join(args.out_dir, RELEASE)
        download_release(release_scans, out_dir, file_types)

if __name__ == "__main__": main()

```

</details>

### 5.2 R2R与RxR数据集下载

| Dataset | Download Link | Extract Path | Size |
|---------|--------------|--------------|------|
| R2R_VLNCE_v1-3 | [Google Drive](https://drive.google.com/file/d/1T9SjqZWyR2PCLSXYkFckfDeIs6Un0Rjm/view) | `data/datasets/R2R_VLNCE_v1-3` | 3 MB |
| R2R_VLNCE_v1-3_preprocessed | [Google Drive](https://drive.google.com/file/d/1fo8F4NKgZDH-bPSdVU3cONAkt5EW-tyr/view) | `data/datasets/R2R_VLNCE_v1-3_preprocessed` | 250 MB |

RxR数据集：https://drive.google.com/file/d/145xzLjxBaNTbVgBfQ8e9EsBAV8W-SM0t/view  

### 数据集目录与路径
数据集目录
```bash
data
├── datasets
│   ├── R2R_VLNCE_v1-3
│   ├── R2R_VLNCE_v1-3_preprocessed
│   ├── RxR_VLNCE_v0
├── scene_datasets
    └── mp3d
```
修改NaVid-VLN-CE/VLN_CE/habitat_extensions/config/vlnce_task_navid_r2r.yaml
```bash
NDTW:
  GT_PATH: /XXX/XXX/XXX/NaVid-VLN-CE/data/datasets/R2R_VLNCE_v1-3_preprocessed/{split}/{split}_gt.json.gz 
DATASET:
  TYPE: VLN-CE-v1 # for R2R 
  SPLIT: val_unseen
  DATA_PATH: /XXX/XXX/XXX/NaVid-VLN-CE/data/datasets/R2R_VLNCE_v1-3_preprocessed/{split}/{split}_gt.json.gz  # episodes dataset
  SCENES_DIR: /XXX/XXX/XXX/NaVid-VLN-CE/data/scene_datasets/ # scene datasets
```
修改NaVid-VLN-CE/VLN_CE/habitat_extensions/config/vlnce_task_navid_rxr.yaml
```bash
NDTW:
    GT_PATH: /XXX/XXX/XXX/NaVid-VLN-CE/data/datasets/RxR_VLNCE_v0/{split}/{split}_{role}_gt.json.gz
DATASET:
  TYPE: RxR-VLN-CE-v1
  SPLIT: val_unseen
  ROLES: [guide]  # "*", "guide", "follower"
  LANGUAGES: [en-US, en-IN]  #  "*", "te-IN", "hi-IN", "en-US", "en-IN"
  DATA_PATH: /XXX/XXX/XXX/NaVid-VLN-CE/data/datasets/RxR_VLNCE_v0/{split}/{split}_{role}.json.gz
  SCENES_DIR: /XXX/XXX/XXX/NaVid-VLN-CE/data/scene_datasets/
```
## 六、权重下载
（1）vision encoder下载  
https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth  
（2）navid权重下载  
https://huggingface.co/Jzzhang/

模型路径：

```bash
navid_ws
├── habitat-lab
├── NaVid-VLN-CE
│   ├── navid
│   ├── VLN_CE
│   ├── model_zoo
│   │   ├── eva_vit_g.pth
│   │   ├── navid-7b-full-224-video-fps-1-grid-2-r2r-rxr-training-split
```

## 七、运行
修改eval_navid_vlnce.sh
```bash
CHUNKS=2 # GPU numbers
MODEL_PATH="" # model wieght
CONFIG_PATH="" # task configuration configure, see script for an example
SAVE_PATH="" #  results
```
运行
```bash
bash eval_navid_vlnce.sh
```
停止
```bash
bash kill_navid_eval.sh
```
结果在指定的SAVE_PATH中，里面有log和运行的gif图，比较直观。这个命令可以监控结果。
```bash
watch -n 1 python  analyze_results.py --path SAVE_PATH
```

### 运行eval_navid_vlnce.sh时可能报的错

可能的报错如下所示：

```bash
fatal error: preload.h: No such file or directory

fatal error: lmdb.h: No such file or directory
```
完整的错误信息如下：
<details>
<summary>点击查看代码 </summary>

```bash
/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/lmdb-1.7.5-py3.8.egg/lmdb/__pycache__/lmdb_cffi.c:271:14: fatal error: lmdb.h: No such file or directory
  271 |     #include "lmdb.h"
      |              ^~~~~~~~
compilation terminated.
/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/lmdb-1.7.5-py3.8.egg/lmdb/__pycache__/lmdb_cffi.c:271:14: fatal error: lmdb.h: No such file or directory
  271 |     #include "lmdb.h"
      |              ^~~~~~~~

compilation terminated.
/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/lmdb-1.7.5-py3.8.egg/lmdb/__pycache__/lmdb_cffi.c:271:14: fatal error: lmdb.h: No such file or directory
  271 |     #include "lmdb.h"
      |              ^~~~~~~~
compilation terminated.
Traceback (most recent call last):
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/lmdb-1.7.5-py3.8.egg/lmdb/__init__.py", line 42, in <module>
Traceback (most recent call last):
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/lmdb-1.7.5-py3.8.egg/lmdb/__init__.py", line 42, in <module>
Traceback (most recent call last):
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/lmdb-1.7.5-py3.8.egg/lmdb/__init__.py", line 42, in <module>
    from lmdb.cpython import *
    from lmdb.cpython import *
ModuleNotFoundError: No module named 'lmdb.cpython'
ModuleNotFoundError: No module named 'lmdb.cpython'

During handling of the above exception, another exception occurred:


During handling of the above exception, another exception occurred:

Traceback (most recent call last):
Traceback (most recent call last):
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/unixccompiler.py", line 188, in _compile
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/unixccompiler.py", line 188, in _compile
    from lmdb.cpython import *
ModuleNotFoundError: No module named 'lmdb.cpython'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
    self.spawn(compiler_so + cc_args + [src, '-o', obj] + extra_postargs)
    self.spawn(compiler_so + cc_args + [src, '-o', obj] + extra_postargs)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/unixccompiler.py", line 188, in _compile
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/ccompiler.py", line 1041, in spawn
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/ccompiler.py", line 1041, in spawn
    self.spawn(compiler_so + cc_args + [src, '-o', obj] + extra_postargs)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/ccompiler.py", line 1041, in spawn
    spawn(cmd, dry_run=self.dry_run, **kwargs)
    spawn(cmd, dry_run=self.dry_run, **kwargs)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/spawn.py", line 68, in spawn
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/spawn.py", line 68, in spawn
    raise DistutilsExecError(f"command {cmd!r} failed with exit code {exitcode}")
    raise DistutilsExecError(f"command {cmd!r} failed with exit code {exitcode}")
distutils.errors.DistutilsExecError: command '/usr/bin/gcc' failed with exit code 1
distutils.errors.DistutilsExecError: command '/usr/bin/gcc' failed with exit code 1
    spawn(cmd, dry_run=self.dry_run, **kwargs)

During handling of the above exception, another exception occurred:


During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/spawn.py", line 68, in spawn
Traceback (most recent call last):
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/ffiplatform.py", line 48, in _build
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/ffiplatform.py", line 48, in _build
    raise DistutilsExecError(f"command {cmd!r} failed with exit code {exitcode}")
    dist.run_command('build_ext')
    dist.run_command('build_ext')
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/dist.py", line 967, in run_command
distutils.errors.DistutilsExecError: command '/usr/bin/gcc' failed with exit code 1

During handling of the above exception, another exception occurred:

  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/dist.py", line 967, in run_command
Traceback (most recent call last):
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/ffiplatform.py", line 48, in _build
    dist.run_command('build_ext')
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/dist.py", line 967, in run_command
    super().run_command(command)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/dist.py", line 988, in run_command
    super().run_command(command)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/dist.py", line 988, in run_command
    super().run_command(command)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/dist.py", line 988, in run_command
    cmd_obj.run()
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/command/build_ext.py", line 91, in run
    cmd_obj.run()
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/command/build_ext.py", line 91, in run
    _build_ext.run(self)
    _build_ext.run(self)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/command/build_ext.py", line 359, in run
    cmd_obj.run()
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/command/build_ext.py", line 359, in run
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/command/build_ext.py", line 91, in run
    _build_ext.run(self)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/command/build_ext.py", line 359, in run
    self.build_extensions()
    self.build_extensions()
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/command/build_ext.py", line 479, in build_extensions
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/command/build_ext.py", line 479, in build_extensions
    self.build_extensions()
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/command/build_ext.py", line 479, in build_extensions
    self._build_extensions_serial()
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/command/build_ext.py", line 505, in _build_extensions_serial
    self._build_extensions_serial()
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/command/build_ext.py", line 505, in _build_extensions_serial
    self._build_extensions_serial()
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/command/build_ext.py", line 505, in _build_extensions_serial
    self.build_extension(ext)
    self.build_extension(ext)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/command/build_ext.py", line 252, in build_extension
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/command/build_ext.py", line 252, in build_extension
    _build_ext.build_extension(self, ext)
    self.build_extension(ext)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/command/build_ext.py", line 560, in build_extension
    _build_ext.build_extension(self, ext)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/command/build_ext.py", line 252, in build_extension
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/command/build_ext.py", line 560, in build_extension
    _build_ext.build_extension(self, ext)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/command/build_ext.py", line 560, in build_extension
    objects = self.compiler.compile(
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/ccompiler.py", line 600, in compile
    objects = self.compiler.compile(
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/ccompiler.py", line 600, in compile
    objects = self.compiler.compile(
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/ccompiler.py", line 600, in compile
    self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/unixccompiler.py", line 190, in _compile
    self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/unixccompiler.py", line 190, in _compile
    raise CompileError(msg)
    raise CompileError(msg)
distutils.errors.CompileError: command '/usr/bin/gcc' failed with exit code 1
    self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)

During handling of the above exception, another exception occurred:

distutils.errors.CompileError: command '/usr/bin/gcc' failed with exit code 1
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/unixccompiler.py", line 190, in _compile
Traceback (most recent call last):

During handling of the above exception, another exception occurred:

  File "run.py", line 5, in <module>
Traceback (most recent call last):
  File "run.py", line 5, in <module>
    raise CompileError(msg)
distutils.errors.CompileError: command '/usr/bin/gcc' failed with exit code 1

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "run.py", line 5, in <module>
    from VLN_CE.vlnce_baselines.config.default import get_config
  File "/data/DataLACP/zhangbodong/ZBD/NaVid-VLN-CE/VLN_CE/__init__.py", line 1, in <module>
    from VLN_CE.vlnce_baselines.config.default import get_config
  File "/data/DataLACP/zhangbodong/ZBD/NaVid-VLN-CE/VLN_CE/__init__.py", line 1, in <module>
    from VLN_CE.vlnce_baselines.config.default import get_config
  File "/data/DataLACP/zhangbodong/ZBD/NaVid-VLN-CE/VLN_CE/__init__.py", line 1, in <module>
    from . import habitat_extensions
  File "/data/DataLACP/zhangbodong/ZBD/NaVid-VLN-CE/VLN_CE/habitat_extensions/__init__.py", line 1, in <module>
    from . import habitat_extensions
  File "/data/DataLACP/zhangbodong/ZBD/NaVid-VLN-CE/VLN_CE/habitat_extensions/__init__.py", line 1, in <module>
    from . import habitat_extensions
  File "/data/DataLACP/zhangbodong/ZBD/NaVid-VLN-CE/VLN_CE/habitat_extensions/__init__.py", line 1, in <module>
    from VLN_CE.habitat_extensions import actions, measures, obs_transformers, sensors
  File "/data/DataLACP/zhangbodong/ZBD/NaVid-VLN-CE/VLN_CE/habitat_extensions/obs_transformers.py", line 11, in <module>
    from VLN_CE.habitat_extensions import actions, measures, obs_transformers, sensors
  File "/data/DataLACP/zhangbodong/ZBD/NaVid-VLN-CE/VLN_CE/habitat_extensions/obs_transformers.py", line 11, in <module>
    from VLN_CE.habitat_extensions import actions, measures, obs_transformers, sensors
  File "/data/DataLACP/zhangbodong/ZBD/NaVid-VLN-CE/VLN_CE/habitat_extensions/obs_transformers.py", line 11, in <module>
    from habitat_baselines.common.baseline_registry import baseline_registry
  File "/data/DataLACP/zhangbodong/ZBD/habitat-lab/habitat_baselines/__init__.py", line 9, in <module>
    from habitat_baselines.common.baseline_registry import baseline_registry
  File "/data/DataLACP/zhangbodong/ZBD/habitat-lab/habitat_baselines/__init__.py", line 9, in <module>
    from habitat_baselines.common.baseline_registry import baseline_registry
  File "/data/DataLACP/zhangbodong/ZBD/habitat-lab/habitat_baselines/__init__.py", line 9, in <module>
    from habitat_baselines.il.trainers.eqa_cnn_pretrain_trainer import (
  File "/data/DataLACP/zhangbodong/ZBD/habitat-lab/habitat_baselines/il/trainers/eqa_cnn_pretrain_trainer.py", line 17, in <module>
    from habitat_baselines.il.trainers.eqa_cnn_pretrain_trainer import (
  File "/data/DataLACP/zhangbodong/ZBD/habitat-lab/habitat_baselines/il/trainers/eqa_cnn_pretrain_trainer.py", line 17, in <module>
    from habitat_baselines.il.trainers.eqa_cnn_pretrain_trainer import (
    from habitat_baselines.il.data.eqa_cnn_pretrain_data import (
  File "/data/DataLACP/zhangbodong/ZBD/habitat-lab/habitat_baselines/il/trainers/eqa_cnn_pretrain_trainer.py", line 17, in <module>
  File "/data/DataLACP/zhangbodong/ZBD/habitat-lab/habitat_baselines/il/data/eqa_cnn_pretrain_data.py", line 5, in <module>
    from habitat_baselines.il.data.eqa_cnn_pretrain_data import (
  File "/data/DataLACP/zhangbodong/ZBD/habitat-lab/habitat_baselines/il/data/eqa_cnn_pretrain_data.py", line 5, in <module>
    import lmdb
    from habitat_baselines.il.data.eqa_cnn_pretrain_data import (
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/lmdb-1.7.5-py3.8.egg/lmdb/__init__.py", line 48, in <module>
  File "/data/DataLACP/zhangbodong/ZBD/habitat-lab/habitat_baselines/il/data/eqa_cnn_pretrain_data.py", line 5, in <module>
    from lmdb.cffi import *
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/lmdb-1.7.5-py3.8.egg/lmdb/cffi.py", line 372, in <module>
    _lib = _ffi.verify(_CFFI_VERIFY,
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/api.py", line 468, in verify
    import lmdb
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/lmdb-1.7.5-py3.8.egg/lmdb/__init__.py", line 48, in <module>
    from lmdb.cffi import *
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/lmdb-1.7.5-py3.8.egg/lmdb/cffi.py", line 372, in <module>
    lib = self.verifier.load_library()
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/verifier.py", line 105, in load_library
    self._compile_module()
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/verifier.py", line 201, in _compile_module
    _lib = _ffi.verify(_CFFI_VERIFY,
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/api.py", line 468, in verify
    import lmdb
    outputfilename = ffiplatform.compile(tmpdir, self.get_extension())
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/ffiplatform.py", line 20, in compile
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/lmdb-1.7.5-py3.8.egg/lmdb/__init__.py", line 48, in <module>
    outputfilename = _build(tmpdir, ext, compiler_verbose, debug)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/ffiplatform.py", line 54, in _build
    lib = self.verifier.load_library()
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/verifier.py", line 105, in load_library
    from lmdb.cffi import *
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/lmdb-1.7.5-py3.8.egg/lmdb/cffi.py", line 372, in <module>
    raise VerificationError('%s: %s' % (e.__class__.__name__, e))
    self._compile_module()
cffi.VerificationError: CompileError: command '/usr/bin/gcc' failed with exit code 1
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/verifier.py", line 201, in _compile_module
    _lib = _ffi.verify(_CFFI_VERIFY,
    outputfilename = ffiplatform.compile(tmpdir, self.get_extension())
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/ffiplatform.py", line 20, in compile
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/api.py", line 468, in verify
    outputfilename = _build(tmpdir, ext, compiler_verbose, debug)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/ffiplatform.py", line 54, in _build
    raise VerificationError('%s: %s' % (e.__class__.__name__, e))
    lib = self.verifier.load_library()
cffi.VerificationError: CompileError: command '/usr/bin/gcc' failed with exit code 1
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/verifier.py", line 105, in load_library
    self._compile_module()
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/verifier.py", line 201, in _compile_module
    outputfilename = ffiplatform.compile(tmpdir, self.get_extension())
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/ffiplatform.py", line 20, in compile
    outputfilename = _build(tmpdir, ext, compiler_verbose, debug)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/ffiplatform.py", line 54, in _build
    raise VerificationError('%s: %s' % (e.__class__.__name__, e))
cffi.VerificationError: CompileError: command '/usr/bin/gcc' failed with exit code 1
/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/lmdb-1.7.5-py3.8.egg/lmdb/__pycache__/lmdb_cffi.c:271:14: fatal error: lmdb.h: No such file or directory
  271 |     #include "lmdb.h"
      |              ^~~~~~~~
compilation terminated.
Traceback (most recent call last):
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/lmdb-1.7.5-py3.8.egg/lmdb/__init__.py", line 42, in <module>
    from lmdb.cpython import *
ModuleNotFoundError: No module named 'lmdb.cpython'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/unixccompiler.py", line 188, in _compile
    self.spawn(compiler_so + cc_args + [src, '-o', obj] + extra_postargs)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/ccompiler.py", line 1041, in spawn
    spawn(cmd, dry_run=self.dry_run, **kwargs)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/spawn.py", line 68, in spawn
    raise DistutilsExecError(f"command {cmd!r} failed with exit code {exitcode}")
distutils.errors.DistutilsExecError: command '/usr/bin/gcc' failed with exit code 1

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/ffiplatform.py", line 48, in _build
    dist.run_command('build_ext')
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/dist.py", line 967, in run_command
    super().run_command(command)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/dist.py", line 988, in run_command
    cmd_obj.run()
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/command/build_ext.py", line 91, in run
    _build_ext.run(self)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/command/build_ext.py", line 359, in run
    self.build_extensions()
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/command/build_ext.py", line 479, in build_extensions
    self._build_extensions_serial()
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/command/build_ext.py", line 505, in _build_extensions_serial
    self.build_extension(ext)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/command/build_ext.py", line 252, in build_extension
    _build_ext.build_extension(self, ext)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/command/build_ext.py", line 560, in build_extension
    objects = self.compiler.compile(
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/ccompiler.py", line 600, in compile
    self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/unixccompiler.py", line 190, in _compile
    raise CompileError(msg)
distutils.errors.CompileError: command '/usr/bin/gcc' failed with exit code 1

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "run.py", line 5, in <module>
    from VLN_CE.vlnce_baselines.config.default import get_config
  File "/data/DataLACP/zhangbodong/ZBD/NaVid-VLN-CE/VLN_CE/__init__.py", line 1, in <module>
    from . import habitat_extensions
  File "/data/DataLACP/zhangbodong/ZBD/NaVid-VLN-CE/VLN_CE/habitat_extensions/__init__.py", line 1, in <module>
    from VLN_CE.habitat_extensions import actions, measures, obs_transformers, sensors
  File "/data/DataLACP/zhangbodong/ZBD/NaVid-VLN-CE/VLN_CE/habitat_extensions/obs_transformers.py", line 11, in <module>
    from habitat_baselines.common.baseline_registry import baseline_registry
  File "/data/DataLACP/zhangbodong/ZBD/habitat-lab/habitat_baselines/__init__.py", line 9, in <module>
    from habitat_baselines.il.trainers.eqa_cnn_pretrain_trainer import (
  File "/data/DataLACP/zhangbodong/ZBD/habitat-lab/habitat_baselines/il/trainers/eqa_cnn_pretrain_trainer.py", line 17, in <module>
    from habitat_baselines.il.data.eqa_cnn_pretrain_data import (
  File "/data/DataLACP/zhangbodong/ZBD/habitat-lab/habitat_baselines/il/data/eqa_cnn_pretrain_data.py", line 5, in <module>
    import lmdb
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/lmdb-1.7.5-py3.8.egg/lmdb/__init__.py", line 48, in <module>
    from lmdb.cffi import *
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/lmdb-1.7.5-py3.8.egg/lmdb/cffi.py", line 372, in <module>
    _lib = _ffi.verify(_CFFI_VERIFY,
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/api.py", line 468, in verify
    lib = self.verifier.load_library()
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/verifier.py", line 105, in load_library
    self._compile_module()
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/verifier.py", line 201, in _compile_module
    outputfilename = ffiplatform.compile(tmpdir, self.get_extension())
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/ffiplatform.py", line 20, in compile
    outputfilename = _build(tmpdir, ext, compiler_verbose, debug)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/ffiplatform.py", line 54, in _build
    raise VerificationError('%s: %s' % (e.__class__.__name__, e))
cffi.VerificationError: CompileError: command '/usr/bin/gcc' failed with exit code 1
(vlnce3) guest@gpu3-labot:/data/DataLACP/zhangbodong/ZBD/NaVid-VLN-CE$ bash eval_navid_vlnce.sh
0
1
2
3
/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/lmdb-1.7.5-py3.8.egg/lmdb/__pycache__/lmdb_cffi.c:271:14: fatal error: lmdb.h: No such file or directory
  271 |     #include "lmdb.h"
      |              ^~~~~~~~
compilation terminated.
/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/lmdb-1.7.5-py3.8.egg/lmdb/__pycache__/lmdb_cffi.c:271:14: fatal error: lmdb.h: No such file or directory
  271 |     #include "lmdb.h"
      |              ^~~~~~~~
compilation terminated.
/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/lmdb-1.7.5-py3.8.egg/lmdb/__pycache__/lmdb_cffi.c:271:14: fatal error: lmdb.h: No such file or directory
  271 |     #include "lmdb.h"
      |              ^~~~~~~~
compilation terminated.
Traceback (most recent call last):
Traceback (most recent call last):
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/lmdb-1.7.5-py3.8.egg/lmdb/__init__.py", line 42, in <module>
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/lmdb-1.7.5-py3.8.egg/lmdb/__init__.py", line 42, in <module>
Traceback (most recent call last):
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/lmdb-1.7.5-py3.8.egg/lmdb/__init__.py", line 42, in <module>
    from lmdb.cpython import *
    from lmdb.cpython import *
ModuleNotFoundError: No module named 'lmdb.cpython'
ModuleNotFoundError: No module named 'lmdb.cpython'

During handling of the above exception, another exception occurred:


During handling of the above exception, another exception occurred:

Traceback (most recent call last):
Traceback (most recent call last):
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/unixccompiler.py", line 188, in _compile
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/unixccompiler.py", line 188, in _compile
    self.spawn(compiler_so + cc_args + [src, '-o', obj] + extra_postargs)
    self.spawn(compiler_so + cc_args + [src, '-o', obj] + extra_postargs)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/ccompiler.py", line 1041, in spawn
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/ccompiler.py", line 1041, in spawn
    from lmdb.cpython import *
ModuleNotFoundError: No module named 'lmdb.cpython'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/unixccompiler.py", line 188, in _compile
/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/lmdb-1.7.5-py3.8.egg/lmdb/__pycache__/lmdb_cffi.c:271:14: fatal error: lmdb.h: No such file or directory
  271 |     #include "lmdb.h"
      |              ^~~~~~~~
compilation terminated.
    spawn(cmd, dry_run=self.dry_run, **kwargs)
    spawn(cmd, dry_run=self.dry_run, **kwargs)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/spawn.py", line 68, in spawn
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/spawn.py", line 68, in spawn
    self.spawn(compiler_so + cc_args + [src, '-o', obj] + extra_postargs)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/ccompiler.py", line 1041, in spawn
    raise DistutilsExecError(f"command {cmd!r} failed with exit code {exitcode}")
    raise DistutilsExecError(f"command {cmd!r} failed with exit code {exitcode}")
distutils.errors.DistutilsExecError: command '/usr/bin/gcc' failed with exit code 1
distutils.errors.DistutilsExecError: command '/usr/bin/gcc' failed with exit code 1

During handling of the above exception, another exception occurred:


During handling of the above exception, another exception occurred:

Traceback (most recent call last):
Traceback (most recent call last):
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/ffiplatform.py", line 48, in _build
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/ffiplatform.py", line 48, in _build
    dist.run_command('build_ext')
    dist.run_command('build_ext')
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/dist.py", line 967, in run_command
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/dist.py", line 967, in run_command
    spawn(cmd, dry_run=self.dry_run, **kwargs)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/spawn.py", line 68, in spawn
    raise DistutilsExecError(f"command {cmd!r} failed with exit code {exitcode}")
distutils.errors.DistutilsExecError: command '/usr/bin/gcc' failed with exit code 1

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/ffiplatform.py", line 48, in _build
    super().run_command(command)
    super().run_command(command)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/dist.py", line 988, in run_command
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/dist.py", line 988, in run_command
    dist.run_command('build_ext')
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/dist.py", line 967, in run_command
    cmd_obj.run()
    cmd_obj.run()
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/command/build_ext.py", line 91, in run
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/command/build_ext.py", line 91, in run
    super().run_command(command)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/dist.py", line 988, in run_command
    _build_ext.run(self)
    _build_ext.run(self)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/command/build_ext.py", line 359, in run
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/command/build_ext.py", line 359, in run
    self.build_extensions()
    self.build_extensions()
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/command/build_ext.py", line 479, in build_extensions
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/command/build_ext.py", line 479, in build_extensions
    cmd_obj.run()
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/command/build_ext.py", line 91, in run
    _build_ext.run(self)
    self._build_extensions_serial()
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/command/build_ext.py", line 359, in run
    self._build_extensions_serial()
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/command/build_ext.py", line 505, in _build_extensions_serial
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/command/build_ext.py", line 505, in _build_extensions_serial
    self.build_extensions()
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/command/build_ext.py", line 479, in build_extensions
    self.build_extension(ext)
    self.build_extension(ext)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/command/build_ext.py", line 252, in build_extension
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/command/build_ext.py", line 252, in build_extension
    _build_ext.build_extension(self, ext)
    self._build_extensions_serial()
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/command/build_ext.py", line 560, in build_extension
    _build_ext.build_extension(self, ext)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/command/build_ext.py", line 505, in _build_extensions_serial
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/command/build_ext.py", line 560, in build_extension
    objects = self.compiler.compile(
    self.build_extension(ext)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/ccompiler.py", line 600, in compile
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/command/build_ext.py", line 252, in build_extension
    objects = self.compiler.compile(
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/ccompiler.py", line 600, in compile
    _build_ext.build_extension(self, ext)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/command/build_ext.py", line 560, in build_extension
    self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/unixccompiler.py", line 190, in _compile
    self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/unixccompiler.py", line 190, in _compile
    raise CompileError(msg)
    objects = self.compiler.compile(
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/ccompiler.py", line 600, in compile
distutils.errors.CompileError: command '/usr/bin/gcc' failed with exit code 1
    raise CompileError(msg)

During handling of the above exception, another exception occurred:

distutils.errors.CompileError: command '/usr/bin/gcc' failed with exit code 1
Traceback (most recent call last):

During handling of the above exception, another exception occurred:

  File "run.py", line 5, in <module>
Traceback (most recent call last):
  File "run.py", line 5, in <module>
    self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/unixccompiler.py", line 190, in _compile
    raise CompileError(msg)
distutils.errors.CompileError: command '/usr/bin/gcc' failed with exit code 1

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "run.py", line 5, in <module>
    from VLN_CE.vlnce_baselines.config.default import get_config
  File "/data/DataLACP/zhangbodong/ZBD/NaVid-VLN-CE/VLN_CE/__init__.py", line 1, in <module>
    from VLN_CE.vlnce_baselines.config.default import get_config
  File "/data/DataLACP/zhangbodong/ZBD/NaVid-VLN-CE/VLN_CE/__init__.py", line 1, in <module>
    from VLN_CE.vlnce_baselines.config.default import get_config
  File "/data/DataLACP/zhangbodong/ZBD/NaVid-VLN-CE/VLN_CE/__init__.py", line 1, in <module>
    from . import habitat_extensions
    from . import habitat_extensions
  File "/data/DataLACP/zhangbodong/ZBD/NaVid-VLN-CE/VLN_CE/habitat_extensions/__init__.py", line 1, in <module>
  File "/data/DataLACP/zhangbodong/ZBD/NaVid-VLN-CE/VLN_CE/habitat_extensions/__init__.py", line 1, in <module>
    from . import habitat_extensions
  File "/data/DataLACP/zhangbodong/ZBD/NaVid-VLN-CE/VLN_CE/habitat_extensions/__init__.py", line 1, in <module>
    from VLN_CE.habitat_extensions import actions, measures, obs_transformers, sensors
  File "/data/DataLACP/zhangbodong/ZBD/NaVid-VLN-CE/VLN_CE/habitat_extensions/obs_transformers.py", line 11, in <module>
    from VLN_CE.habitat_extensions import actions, measures, obs_transformers, sensors
  File "/data/DataLACP/zhangbodong/ZBD/NaVid-VLN-CE/VLN_CE/habitat_extensions/obs_transformers.py", line 11, in <module>
    from VLN_CE.habitat_extensions import actions, measures, obs_transformers, sensors
  File "/data/DataLACP/zhangbodong/ZBD/NaVid-VLN-CE/VLN_CE/habitat_extensions/obs_transformers.py", line 11, in <module>
    from habitat_baselines.common.baseline_registry import baseline_registry
  File "/data/DataLACP/zhangbodong/ZBD/habitat-lab/habitat_baselines/__init__.py", line 9, in <module>
    from habitat_baselines.common.baseline_registry import baseline_registry
  File "/data/DataLACP/zhangbodong/ZBD/habitat-lab/habitat_baselines/__init__.py", line 9, in <module>
    from habitat_baselines.common.baseline_registry import baseline_registry
  File "/data/DataLACP/zhangbodong/ZBD/habitat-lab/habitat_baselines/__init__.py", line 9, in <module>
    from habitat_baselines.il.trainers.eqa_cnn_pretrain_trainer import (
  File "/data/DataLACP/zhangbodong/ZBD/habitat-lab/habitat_baselines/il/trainers/eqa_cnn_pretrain_trainer.py", line 17, in <module>
    from habitat_baselines.il.trainers.eqa_cnn_pretrain_trainer import (
  File "/data/DataLACP/zhangbodong/ZBD/habitat-lab/habitat_baselines/il/trainers/eqa_cnn_pretrain_trainer.py", line 17, in <module>
    from habitat_baselines.il.trainers.eqa_cnn_pretrain_trainer import (
  File "/data/DataLACP/zhangbodong/ZBD/habitat-lab/habitat_baselines/il/trainers/eqa_cnn_pretrain_trainer.py", line 17, in <module>
    from habitat_baselines.il.data.eqa_cnn_pretrain_data import (
  File "/data/DataLACP/zhangbodong/ZBD/habitat-lab/habitat_baselines/il/data/eqa_cnn_pretrain_data.py", line 5, in <module>
    from habitat_baselines.il.data.eqa_cnn_pretrain_data import (
    from habitat_baselines.il.data.eqa_cnn_pretrain_data import (
  File "/data/DataLACP/zhangbodong/ZBD/habitat-lab/habitat_baselines/il/data/eqa_cnn_pretrain_data.py", line 5, in <module>
    import lmdb
  File "/data/DataLACP/zhangbodong/ZBD/habitat-lab/habitat_baselines/il/data/eqa_cnn_pretrain_data.py", line 5, in <module>
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/lmdb-1.7.5-py3.8.egg/lmdb/__init__.py", line 48, in <module>
    from lmdb.cffi import *
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/lmdb-1.7.5-py3.8.egg/lmdb/cffi.py", line 372, in <module>
    _lib = _ffi.verify(_CFFI_VERIFY,
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/api.py", line 468, in verify
    import lmdb
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/lmdb-1.7.5-py3.8.egg/lmdb/__init__.py", line 48, in <module>
    lib = self.verifier.load_library()
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/verifier.py", line 105, in load_library
    from lmdb.cffi import *
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/lmdb-1.7.5-py3.8.egg/lmdb/cffi.py", line 372, in <module>
    self._compile_module()
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/verifier.py", line 201, in _compile_module
    outputfilename = ffiplatform.compile(tmpdir, self.get_extension())
    _lib = _ffi.verify(_CFFI_VERIFY,
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/ffiplatform.py", line 20, in compile
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/api.py", line 468, in verify
    import lmdb
    outputfilename = _build(tmpdir, ext, compiler_verbose, debug)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/lmdb-1.7.5-py3.8.egg/lmdb/__init__.py", line 48, in <module>
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/ffiplatform.py", line 54, in _build
    raise VerificationError('%s: %s' % (e.__class__.__name__, e))
    from lmdb.cffi import *
    lib = self.verifier.load_library()
cffi.VerificationError: CompileError: command '/usr/bin/gcc' failed with exit code 1
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/lmdb-1.7.5-py3.8.egg/lmdb/cffi.py", line 372, in <module>
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/verifier.py", line 105, in load_library
    self._compile_module()
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/verifier.py", line 201, in _compile_module
    _lib = _ffi.verify(_CFFI_VERIFY,
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/api.py", line 468, in verify
    outputfilename = ffiplatform.compile(tmpdir, self.get_extension())
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/ffiplatform.py", line 20, in compile
    outputfilename = _build(tmpdir, ext, compiler_verbose, debug)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/ffiplatform.py", line 54, in _build
    lib = self.verifier.load_library()
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/verifier.py", line 105, in load_library
    raise VerificationError('%s: %s' % (e.__class__.__name__, e))
cffi.VerificationError: CompileError: command '/usr/bin/gcc' failed with exit code 1
    self._compile_module()
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/verifier.py", line 201, in _compile_module
    outputfilename = ffiplatform.compile(tmpdir, self.get_extension())
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/ffiplatform.py", line 20, in compile
    outputfilename = _build(tmpdir, ext, compiler_verbose, debug)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/ffiplatform.py", line 54, in _build
    raise VerificationError('%s: %s' % (e.__class__.__name__, e))
cffi.VerificationError: CompileError: command '/usr/bin/gcc' failed with exit code 1
Traceback (most recent call last):
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/lmdb-1.7.5-py3.8.egg/lmdb/__init__.py", line 42, in <module>
    from lmdb.cpython import *
ModuleNotFoundError: No module named 'lmdb.cpython'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/unixccompiler.py", line 188, in _compile
    self.spawn(compiler_so + cc_args + [src, '-o', obj] + extra_postargs)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/ccompiler.py", line 1041, in spawn
    spawn(cmd, dry_run=self.dry_run, **kwargs)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/spawn.py", line 68, in spawn
    raise DistutilsExecError(f"command {cmd!r} failed with exit code {exitcode}")
distutils.errors.DistutilsExecError: command '/usr/bin/gcc' failed with exit code 1

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/ffiplatform.py", line 48, in _build
    dist.run_command('build_ext')
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/dist.py", line 967, in run_command
    super().run_command(command)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/dist.py", line 988, in run_command
    cmd_obj.run()
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/command/build_ext.py", line 91, in run
    _build_ext.run(self)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/command/build_ext.py", line 359, in run
    self.build_extensions()
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/command/build_ext.py", line 479, in build_extensions
    self._build_extensions_serial()
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/command/build_ext.py", line 505, in _build_extensions_serial
    self.build_extension(ext)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/command/build_ext.py", line 252, in build_extension
    _build_ext.build_extension(self, ext)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/command/build_ext.py", line 560, in build_extension
    objects = self.compiler.compile(
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/ccompiler.py", line 600, in compile
    self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/unixccompiler.py", line 190, in _compile
    raise CompileError(msg)
distutils.errors.CompileError: command '/usr/bin/gcc' failed with exit code 1

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "run.py", line 5, in <module>
    from VLN_CE.vlnce_baselines.config.default import get_config
  File "/data/DataLACP/zhangbodong/ZBD/NaVid-VLN-CE/VLN_CE/__init__.py", line 1, in <module>
    from . import habitat_extensions
  File "/data/DataLACP/zhangbodong/ZBD/NaVid-VLN-CE/VLN_CE/habitat_extensions/__init__.py", line 1, in <module>
    from VLN_CE.habitat_extensions import actions, measures, obs_transformers, sensors
  File "/data/DataLACP/zhangbodong/ZBD/NaVid-VLN-CE/VLN_CE/habitat_extensions/obs_transformers.py", line 11, in <module>
    from habitat_baselines.common.baseline_registry import baseline_registry
  File "/data/DataLACP/zhangbodong/ZBD/habitat-lab/habitat_baselines/__init__.py", line 9, in <module>
    from habitat_baselines.il.trainers.eqa_cnn_pretrain_trainer import (
  File "/data/DataLACP/zhangbodong/ZBD/habitat-lab/habitat_baselines/il/trainers/eqa_cnn_pretrain_trainer.py", line 17, in <module>
    from habitat_baselines.il.data.eqa_cnn_pretrain_data import (
  File "/data/DataLACP/zhangbodong/ZBD/habitat-lab/habitat_baselines/il/data/eqa_cnn_pretrain_data.py", line 5, in <module>
    import lmdb
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/lmdb-1.7.5-py3.8.egg/lmdb/__init__.py", line 48, in <module>
    from lmdb.cffi import *
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/lmdb-1.7.5-py3.8.egg/lmdb/cffi.py", line 372, in <module>
    _lib = _ffi.verify(_CFFI_VERIFY,
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/api.py", line 468, in verify
    lib = self.verifier.load_library()
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/verifier.py", line 105, in load_library
    self._compile_module()
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/verifier.py", line 201, in _compile_module
    outputfilename = ffiplatform.compile(tmpdir, self.get_extension())
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/ffiplatform.py", line 20, in compile
    outputfilename = _build(tmpdir, ext, compiler_verbose, debug)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/ffiplatform.py", line 54, in _build
    raise VerificationError('%s: %s' % (e.__class__.__name__, e))
cffi.VerificationError: CompileError: command '/usr/bin/gcc' failed with exit code 1

这是第一个报错

(vlnce3) guest@gpu3-labot:/data/DataLACP/zhangbodong/ZBD/NaVid-VLN-CE$ conda install -c conda-forge lmdb -y
Collecting package metadata (current_repodata.json): done
Solving environment: \ 
The environment is inconsistent, please check the package plan carefully
The following packages are causing the inconsistency:

  - <unknown>/linux-64::habitat-sim==0.1.7=py3.8_headless_linux_856d4b08c1a2632626bf0d205bf46471a99502b7
failed with initial frozen solve. Retrying with flexible solve.
Solving environment: failed with repodata from current_repodata.json, will retry with next repodata source.
Collecting package metadata (repodata.json): failed

CondaHTTPError: HTTP 000 CONNECTION FAILED for url <https://conda.anaconda.org/conda-forge/linux-64/repodata.json>
Elapsed: -

An HTTP error occurred when trying to retrieve this URL.
HTTP errors are often intermittent, and a simple retry will get you on your way.
'https//conda.anaconda.org/conda-forge/linux-64'


(vlnce3) guest@gpu3-labot:/data/DataLACP/zhangbodong/ZBD/NaVid-VLN-CE$ bash eval_navid_vlnce.sh
0
1
2
3
/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/lmdb-1.7.5-py3.8.egg/lmdb/__pycache__/lmdb_cffi.c:272:14: fatal error: preload.h: No such file or directory
  272 |     #include "preload.h"
      |              ^~~~~~~~~~~
compilation terminated.
/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/lmdb-1.7.5-py3.8.egg/lmdb/__pycache__/lmdb_cffi.c:272:14: fatal error: preload.h: No such file or directory
  272 |     #include "preload.h"
      |              ^~~~~~~~~~~
compilation terminated.
/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/lmdb-1.7.5-py3.8.egg/lmdb/__pycache__/lmdb_cffi.c:272:14: fatal error: preload.h: No such file or directory
  272 |     #include "preload.h"
      |              ^~~~~~~~~~~
/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/lmdb-1.7.5-py3.8.egg/lmdb/__pycache__/lmdb_cffi.c:272:14: fatal error: preload.h: No such file or directory
  272 |     #include "preload.h"
      |              ^~~~~~~~~~~
compilation terminated.
compilation terminated.
Traceback (most recent call last):
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/lmdb-1.7.5-py3.8.egg/lmdb/__init__.py", line 42, in <module>
Traceback (most recent call last):
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/lmdb-1.7.5-py3.8.egg/lmdb/__init__.py", line 42, in <module>
    from lmdb.cpython import *
ModuleNotFoundError: No module named 'lmdb.cpython'
    from lmdb.cpython import *

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
ModuleNotFoundError: No module named 'lmdb.cpython'
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/unixccompiler.py", line 188, in _compile

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/unixccompiler.py", line 188, in _compile
    self.spawn(compiler_so + cc_args + [src, '-o', obj] + extra_postargs)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/ccompiler.py", line 1041, in spawn
    self.spawn(compiler_so + cc_args + [src, '-o', obj] + extra_postargs)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/ccompiler.py", line 1041, in spawn
    spawn(cmd, dry_run=self.dry_run, **kwargs)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/spawn.py", line 68, in spawn
    spawn(cmd, dry_run=self.dry_run, **kwargs)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/spawn.py", line 68, in spawn
    raise DistutilsExecError(f"command {cmd!r} failed with exit code {exitcode}")
    raise DistutilsExecError(f"command {cmd!r} failed with exit code {exitcode}")
distutils.errors.DistutilsExecError: command '/usr/bin/gcc' failed with exit code 1

During handling of the above exception, another exception occurred:

distutils.errors.DistutilsExecError: command '/usr/bin/gcc' failed with exit code 1

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
Traceback (most recent call last):
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/ffiplatform.py", line 48, in _build
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/ffiplatform.py", line 48, in _build
    dist.run_command('build_ext')
    dist.run_command('build_ext')
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/dist.py", line 967, in run_command
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/dist.py", line 967, in run_command
    super().run_command(command)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/dist.py", line 988, in run_command
    super().run_command(command)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/dist.py", line 988, in run_command
    cmd_obj.run()
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/command/build_ext.py", line 91, in run
    cmd_obj.run()
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/command/build_ext.py", line 91, in run
    _build_ext.run(self)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/command/build_ext.py", line 359, in run
    _build_ext.run(self)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/command/build_ext.py", line 359, in run
    self.build_extensions()
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/command/build_ext.py", line 479, in build_extensions
    self.build_extensions()
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/command/build_ext.py", line 479, in build_extensions
    self._build_extensions_serial()
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/command/build_ext.py", line 505, in _build_extensions_serial
    self._build_extensions_serial()
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/command/build_ext.py", line 505, in _build_extensions_serial
    self.build_extension(ext)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/command/build_ext.py", line 252, in build_extension
    self.build_extension(ext)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/command/build_ext.py", line 252, in build_extension
    _build_ext.build_extension(self, ext)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/command/build_ext.py", line 560, in build_extension
    _build_ext.build_extension(self, ext)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/command/build_ext.py", line 560, in build_extension
    objects = self.compiler.compile(
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/ccompiler.py", line 600, in compile
    objects = self.compiler.compile(
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/ccompiler.py", line 600, in compile
    self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/unixccompiler.py", line 190, in _compile
    self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)
    raise CompileError(msg)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/unixccompiler.py", line 190, in _compile
distutils.errors.CompileError: command '/usr/bin/gcc' failed with exit code 1

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "run.py", line 5, in <module>
    raise CompileError(msg)
distutils.errors.CompileError: command '/usr/bin/gcc' failed with exit code 1

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "run.py", line 5, in <module>
    from VLN_CE.vlnce_baselines.config.default import get_config
  File "/data/DataLACP/zhangbodong/ZBD/NaVid-VLN-CE/VLN_CE/__init__.py", line 1, in <module>
    from VLN_CE.vlnce_baselines.config.default import get_config
  File "/data/DataLACP/zhangbodong/ZBD/NaVid-VLN-CE/VLN_CE/__init__.py", line 1, in <module>
    from . import habitat_extensions
  File "/data/DataLACP/zhangbodong/ZBD/NaVid-VLN-CE/VLN_CE/habitat_extensions/__init__.py", line 1, in <module>
    from . import habitat_extensions
  File "/data/DataLACP/zhangbodong/ZBD/NaVid-VLN-CE/VLN_CE/habitat_extensions/__init__.py", line 1, in <module>
    from VLN_CE.habitat_extensions import actions, measures, obs_transformers, sensors
  File "/data/DataLACP/zhangbodong/ZBD/NaVid-VLN-CE/VLN_CE/habitat_extensions/obs_transformers.py", line 11, in <module>
    from VLN_CE.habitat_extensions import actions, measures, obs_transformers, sensors
  File "/data/DataLACP/zhangbodong/ZBD/NaVid-VLN-CE/VLN_CE/habitat_extensions/obs_transformers.py", line 11, in <module>
    from habitat_baselines.common.baseline_registry import baseline_registry
  File "/data/DataLACP/zhangbodong/ZBD/habitat-lab/habitat_baselines/__init__.py", line 9, in <module>
    from habitat_baselines.common.baseline_registry import baseline_registry
  File "/data/DataLACP/zhangbodong/ZBD/habitat-lab/habitat_baselines/__init__.py", line 9, in <module>
    from habitat_baselines.il.trainers.eqa_cnn_pretrain_trainer import (
  File "/data/DataLACP/zhangbodong/ZBD/habitat-lab/habitat_baselines/il/trainers/eqa_cnn_pretrain_trainer.py", line 17, in <module>
    from habitat_baselines.il.trainers.eqa_cnn_pretrain_trainer import (
  File "/data/DataLACP/zhangbodong/ZBD/habitat-lab/habitat_baselines/il/trainers/eqa_cnn_pretrain_trainer.py", line 17, in <module>
    from habitat_baselines.il.data.eqa_cnn_pretrain_data import (
  File "/data/DataLACP/zhangbodong/ZBD/habitat-lab/habitat_baselines/il/data/eqa_cnn_pretrain_data.py", line 5, in <module>
    from habitat_baselines.il.data.eqa_cnn_pretrain_data import (
  File "/data/DataLACP/zhangbodong/ZBD/habitat-lab/habitat_baselines/il/data/eqa_cnn_pretrain_data.py", line 5, in <module>
    import lmdb
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/lmdb-1.7.5-py3.8.egg/lmdb/__init__.py", line 48, in <module>
    from lmdb.cffi import *
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/lmdb-1.7.5-py3.8.egg/lmdb/cffi.py", line 372, in <module>
    _lib = _ffi.verify(_CFFI_VERIFY,
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/api.py", line 468, in verify
    import lmdb
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/lmdb-1.7.5-py3.8.egg/lmdb/__init__.py", line 48, in <module>
    lib = self.verifier.load_library()
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/verifier.py", line 105, in load_library
    from lmdb.cffi import *
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/lmdb-1.7.5-py3.8.egg/lmdb/cffi.py", line 372, in <module>
    self._compile_module()
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/verifier.py", line 201, in _compile_module
    _lib = _ffi.verify(_CFFI_VERIFY,
    outputfilename = ffiplatform.compile(tmpdir, self.get_extension())
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/api.py", line 468, in verify
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/ffiplatform.py", line 20, in compile
    outputfilename = _build(tmpdir, ext, compiler_verbose, debug)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/ffiplatform.py", line 54, in _build
    raise VerificationError('%s: %s' % (e.__class__.__name__, e))
    lib = self.verifier.load_library()
cffi.VerificationError: CompileError: command '/usr/bin/gcc' failed with exit code 1
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/verifier.py", line 105, in load_library
    self._compile_module()
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/verifier.py", line 201, in _compile_module
    outputfilename = ffiplatform.compile(tmpdir, self.get_extension())
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/ffiplatform.py", line 20, in compile
    outputfilename = _build(tmpdir, ext, compiler_verbose, debug)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/ffiplatform.py", line 54, in _build
    raise VerificationError('%s: %s' % (e.__class__.__name__, e))
cffi.VerificationError: CompileError: command '/usr/bin/gcc' failed with exit code 1
Traceback (most recent call last):
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/lmdb-1.7.5-py3.8.egg/lmdb/__init__.py", line 42, in <module>
Traceback (most recent call last):
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/lmdb-1.7.5-py3.8.egg/lmdb/__init__.py", line 42, in <module>
    from lmdb.cpython import *
ModuleNotFoundError: No module named 'lmdb.cpython'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/unixccompiler.py", line 188, in _compile
    from lmdb.cpython import *
ModuleNotFoundError: No module named 'lmdb.cpython'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/unixccompiler.py", line 188, in _compile
    self.spawn(compiler_so + cc_args + [src, '-o', obj] + extra_postargs)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/ccompiler.py", line 1041, in spawn
    self.spawn(compiler_so + cc_args + [src, '-o', obj] + extra_postargs)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/ccompiler.py", line 1041, in spawn
    spawn(cmd, dry_run=self.dry_run, **kwargs)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/spawn.py", line 68, in spawn
    raise DistutilsExecError(f"command {cmd!r} failed with exit code {exitcode}")
    spawn(cmd, dry_run=self.dry_run, **kwargs)
distutils.errors.DistutilsExecError: command '/usr/bin/gcc' failed with exit code 1

During handling of the above exception, another exception occurred:

  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/spawn.py", line 68, in spawn
Traceback (most recent call last):
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/ffiplatform.py", line 48, in _build
    raise DistutilsExecError(f"command {cmd!r} failed with exit code {exitcode}")
    dist.run_command('build_ext')
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/dist.py", line 967, in run_command
distutils.errors.DistutilsExecError: command '/usr/bin/gcc' failed with exit code 1

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/ffiplatform.py", line 48, in _build
    dist.run_command('build_ext')
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/dist.py", line 967, in run_command
    super().run_command(command)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/dist.py", line 988, in run_command
    super().run_command(command)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/dist.py", line 988, in run_command
    cmd_obj.run()
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/command/build_ext.py", line 91, in run
    _build_ext.run(self)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/command/build_ext.py", line 359, in run
    cmd_obj.run()
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/command/build_ext.py", line 91, in run
    _build_ext.run(self)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/command/build_ext.py", line 359, in run
    self.build_extensions()
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/command/build_ext.py", line 479, in build_extensions
    self.build_extensions()
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/command/build_ext.py", line 479, in build_extensions
    self._build_extensions_serial()
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/command/build_ext.py", line 505, in _build_extensions_serial
    self._build_extensions_serial()
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/command/build_ext.py", line 505, in _build_extensions_serial
    self.build_extension(ext)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/command/build_ext.py", line 252, in build_extension
    self.build_extension(ext)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/command/build_ext.py", line 252, in build_extension
    _build_ext.build_extension(self, ext)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/command/build_ext.py", line 560, in build_extension
    _build_ext.build_extension(self, ext)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/command/build_ext.py", line 560, in build_extension
    objects = self.compiler.compile(
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/ccompiler.py", line 600, in compile
    objects = self.compiler.compile(
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/ccompiler.py", line 600, in compile
    self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/unixccompiler.py", line 190, in _compile
    self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/setuptools/_distutils/unixccompiler.py", line 190, in _compile
    raise CompileError(msg)
distutils.errors.CompileError: command '/usr/bin/gcc' failed with exit code 1

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "run.py", line 5, in <module>
    raise CompileError(msg)
distutils.errors.CompileError: command '/usr/bin/gcc' failed with exit code 1

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "run.py", line 5, in <module>
    from VLN_CE.vlnce_baselines.config.default import get_config
  File "/data/DataLACP/zhangbodong/ZBD/NaVid-VLN-CE/VLN_CE/__init__.py", line 1, in <module>
    from VLN_CE.vlnce_baselines.config.default import get_config
  File "/data/DataLACP/zhangbodong/ZBD/NaVid-VLN-CE/VLN_CE/__init__.py", line 1, in <module>
    from . import habitat_extensions
    from . import habitat_extensions
  File "/data/DataLACP/zhangbodong/ZBD/NaVid-VLN-CE/VLN_CE/habitat_extensions/__init__.py", line 1, in <module>
  File "/data/DataLACP/zhangbodong/ZBD/NaVid-VLN-CE/VLN_CE/habitat_extensions/__init__.py", line 1, in <module>
    from VLN_CE.habitat_extensions import actions, measures, obs_transformers, sensors
  File "/data/DataLACP/zhangbodong/ZBD/NaVid-VLN-CE/VLN_CE/habitat_extensions/obs_transformers.py", line 11, in <module>
    from VLN_CE.habitat_extensions import actions, measures, obs_transformers, sensors
  File "/data/DataLACP/zhangbodong/ZBD/NaVid-VLN-CE/VLN_CE/habitat_extensions/obs_transformers.py", line 11, in <module>
    from habitat_baselines.common.baseline_registry import baseline_registry
  File "/data/DataLACP/zhangbodong/ZBD/habitat-lab/habitat_baselines/__init__.py", line 9, in <module>
    from habitat_baselines.common.baseline_registry import baseline_registry
  File "/data/DataLACP/zhangbodong/ZBD/habitat-lab/habitat_baselines/__init__.py", line 9, in <module>
    from habitat_baselines.il.trainers.eqa_cnn_pretrain_trainer import (
  File "/data/DataLACP/zhangbodong/ZBD/habitat-lab/habitat_baselines/il/trainers/eqa_cnn_pretrain_trainer.py", line 17, in <module>
    from habitat_baselines.il.trainers.eqa_cnn_pretrain_trainer import (
  File "/data/DataLACP/zhangbodong/ZBD/habitat-lab/habitat_baselines/il/trainers/eqa_cnn_pretrain_trainer.py", line 17, in <module>
    from habitat_baselines.il.data.eqa_cnn_pretrain_data import (
  File "/data/DataLACP/zhangbodong/ZBD/habitat-lab/habitat_baselines/il/data/eqa_cnn_pretrain_data.py", line 5, in <module>
    from habitat_baselines.il.data.eqa_cnn_pretrain_data import (
  File "/data/DataLACP/zhangbodong/ZBD/habitat-lab/habitat_baselines/il/data/eqa_cnn_pretrain_data.py", line 5, in <module>
    import lmdb
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/lmdb-1.7.5-py3.8.egg/lmdb/__init__.py", line 48, in <module>
    from lmdb.cffi import *
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/lmdb-1.7.5-py3.8.egg/lmdb/cffi.py", line 372, in <module>
    _lib = _ffi.verify(_CFFI_VERIFY,
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/api.py", line 468, in verify
    import lmdb
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/lmdb-1.7.5-py3.8.egg/lmdb/__init__.py", line 48, in <module>
    from lmdb.cffi import *
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/lmdb-1.7.5-py3.8.egg/lmdb/cffi.py", line 372, in <module>
    lib = self.verifier.load_library()
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/verifier.py", line 105, in load_library
    self._compile_module()
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/verifier.py", line 201, in _compile_module
    _lib = _ffi.verify(_CFFI_VERIFY,
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/api.py", line 468, in verify
    outputfilename = ffiplatform.compile(tmpdir, self.get_extension())
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/ffiplatform.py", line 20, in compile
    outputfilename = _build(tmpdir, ext, compiler_verbose, debug)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/ffiplatform.py", line 54, in _build
    lib = self.verifier.load_library()
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/verifier.py", line 105, in load_library
    raise VerificationError('%s: %s' % (e.__class__.__name__, e))
cffi.VerificationError: CompileError: command '/usr/bin/gcc' failed with exit code 1
    self._compile_module()
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/verifier.py", line 201, in _compile_module
    outputfilename = ffiplatform.compile(tmpdir, self.get_extension())
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/ffiplatform.py", line 20, in compile
    outputfilename = _build(tmpdir, ext, compiler_verbose, debug)
  File "/home/guest/anaconda3/envs/vlnce3/lib/python3.8/site-packages/cffi/ffiplatform.py", line 54, in _build
    raise VerificationError('%s: %s' % (e.__class__.__name__, e))
cffi.VerificationError: CompileError: command '/usr/bin/gcc' failed with exit code 1

这是我第二个报错。
```
</details>  


问题原因：   

1. 系统中缺少 LMDB 库的底层依赖（系统级的 LMDB 开发包），导致 Python 的 lmdb 模块在编译 C 扩展时失败 

2. lmdb 1.7.5 + preload 机制 + 无系统开发环境 的已知硬冲突  

    lmdb 1.7.x 默认启用了 “preload” 特性

    该特性 依赖 glibc 内部头文件，在 无 sudo / 无完整 libc-dev 的服务器 上 必然失败，即使你自己编译了 liblmdb，也 解决不了 preload.h


解决方法：  
彻底绕开 preload：降级 lmdb 到 1.4.x

```bash
# 完全卸载当前 lmdb（包括 egg）
conda activate vlnce3
pip uninstall lmdb -y
# 确认干净
python - << 'EOF'
import lmdb
EOF
# 应该直接报 ModuleNotFoundError

# 安装不会触发 preload 的稳定版本
pip install lmdb==1.4.1
# 验证
python - << 'EOF'
import lmdb
print("lmdb OK:", lmdb.__version__)
EOF
# 正确结果
lmdb OK: 1.4.1
# 此时：
# 不会再编译 lmdb_cffi.c
# 不会再出现 preload.h
# 不会再调用 gcc
```