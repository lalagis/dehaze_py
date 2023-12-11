## 遥感影像去雾的几种算法

### 概述

读取一张遥感影像，选择不同的经典去雾算法，实时观察效果，考察处理时间上的差异，以及提供保存处理后影像(.tif)的能力。

### 架构

- GDAL 读写各格式遥感影像
- opencv, numpy 实现算法
- [flet](https://flet.dev/)(flutter)构建现代 ui 界面，图片编译为 base64 展示，同时支持打包三端桌面应用程序

### 工作流

1. 安装

   windows[PaddlePaddle-gpu(>=2.5.0)](https://www.paddlepaddle.org.cn/install/quick)以及对应版本的[CUDA](https://developer.nvidia.com/cuda-11-8-0-download-archive), [cuDNN](https://developer.nvidia.com/rdp/cudnn-download)(登录后申请资格即可下载)

   mac [PaddlePaddle(>=2.5.0)](https://www.paddlepaddle.org.cn/install/quick)(无法运行 NAFNET 预测)

   ```zsh
   pip install paddlepaddle==2.5.2 # mac
   pip install paddlepaddle-gpu==2.5.2 # windows
   ```

2. 安装[PaddleRS](https://github.com/PaddlePaddle/PaddleRS)

   ```zsh
   git clone https://github.com/PaddlePaddle/PaddleRS
   cd PaddleRS
   git checkout develop
   pip install .
   ```

   2.1 若在执行`pip install`时下载依赖缓慢或超时，可以在`setup.py`相同目录下新建`setup.cfg`，并输入以下内容，则可通过清华源进行加速下载：

   ```yml
   [easy_install]
   index-url=https://pypi.tuna.tsinghua.edu.cn/simple
   ```

3. 安装相关依赖 GDAL, flet

   ```zsh
   pip install -r requirements.txt
   ```

   3.1 windows 无法成功安装GDAL并报`无法打开包括文件: “gdal.h”: No such file or directory`错误时，访问https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal，下载对应python版本的包，并在终端安装。

   ```zsh
   pip install GDAL-3.4.3-cp311-cp311-win_amd64.whl
   ```

4. mac可能需要重新安装一下 numpy, wheel

   ```zsh
   pip install --no-cache-dir --force-reinstall wheel numpy GDAL
   ```

5. 运行`main.py`

6. NAFNET所用的数据集为遥感影像去云数据集[RICE](https://github.com/BUPTLdy/RICE_DATASET)，可以下载我们训练所得的[模型](https://drive.google.com/file/d/1yR_OOSGZMc6C8OG8JS343m4j4J78BEbB/view?usp=sharing)，并修改`utils/algorithm.py`下的路径。

7. 警告 Can not use `conditional_random_field`. Please install pydensecrf first. 

   无法直接`pip install pydensecrf`，直接拉最新的仓库安装

   ```zsh
   pip install git+https://github.com/lucasb-eyer/pydensecrf.git
   ```

8. 打包为对应平台的桌面app

   安装pyinstaller

   ```zsh
   pip install pyinstaller
   ```

   如需打包到桌面应用程序运行到请注释掉NAFNET预测内容：

   ```python
   # from paddlers.tasks import load_model
   # from paddlers import transforms as T
   
   # NAFNET预测
   # model = load_model("./data/best_model")
   # eval_transforms = [
   #     # 验证阶段与训练阶段的数据归一化方式必须相同
   #     T.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
   # ]
   
   
   def nafnetPredict(img):
       # result = model.predict(img, T.Compose(eval_transforms))["res_map"]
       # return result
       return img
   ```

   

   打包 --name 名称 --icon (.png)图标路径

   ```zsh
   flet pack main.py --name dehazer --icon icon.svg --product-name dehazer --product-version 0.1 --copyright lalagis © 2023  
   ```

   

### 参考

[PaddleRS](https://github.com/PaddlePaddle/PaddleRS)飞桨遥感 Toolkit
