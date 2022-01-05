
# 隐特征网络（Implict Feature Networks）

> 隐函数在特征空间中用于形状重建和补全 <br />
> [Julian Chibane](http://virtualhumans.mpi-inf.mpg.de/people/Chibane.html), [Thiemo Alldieck](http://virtualhumans.mpi-inf.mpg.de/people/alldieck.html), [Gerard Pons-Moll](http://virtualhumans.mpi-inf.mpg.de/people/pons-moll.html)

![Teaser](teaser.gif)

[论文](https://virtualhumans.mpi-inf.mpg.de/papers/chibane20ifnet/chibane20ifnet.pdf) - 
[补充材料](https://virtualhumans.mpi-inf.mpg.de/papers/chibane20ifnet/chibane20ifnet_supp.pdf) -
[项目代码](https://virtualhumans.mpi-inf.mpg.de/ifnets/) -
[Arxiv](https://arxiv.org/abs/2003.01456) -
[Video](https://youtu.be/cko07jINRZg) -
Published in CVPR 2020.

## 引用

如果你发现我们的代码或者论文对你们的项目有用，请考虑引用：

    @inproceedings{chibane20ifnet,
        title = {Implicit Functions in Feature Space for 3D Shape Reconstruction and Completion},
        author = {Chibane, Julian and Alldieck, Thiemo and Pons-Moll, Gerard},
        booktitle = {{IEEE} Conference on Computer Vision and Pattern Recognition (CVPR)},
        month = {jun},
        organization = {{IEEE}},
        year = {2020},
    }

## Install

本项目需要安装了CUDA 9.0 的 Linux 系统。

文件 `if-net_env.yml` 包含了项目需要的所有 Python 依赖，为了方便安装这些依赖可以使用 [anaconda](https://www.anaconda.com/) ：

```shell
conda env create -f if-net_env.yml
conda activate if-net
```

请克隆这个仓库，并在你的终端中浏览它，它的定位假设了随后的所有命令。

> 这个项目使用了 [^Mescheder et. al. CVPR'19] 的 [Occupancy Networks](https://github.com/autonomousvision/occupancy_networks)  的库
> 以及 [^Xu et. al. NeurIPS'19] 的 [DISN](https://github.com/Xharlie/DISN) 预处理的  ShapeNet 数据集，请在你使用我们的代码中也引用它们。

安装所需要的这些库：

```shell
cd data_processing/libmesh/
python setup.py build_ext --inplace
cd ../libvoxelize/
python setup.py build_ext --inplace
cd ../..
```

## 数据准备

全部准备好数据大约需要 800 GB。下载 [Xu et. al. NeurIPS'19] 预处理的 [ShapeNet](https://www.shapenet.org/) 数据： [数据地址](https://drive.google.com/drive/folders/1QGhDW335L7ra31uw5U-0V7hB-viA0JXr)，并保存在 `shapenet` 文件夹中。

现在抽取文件到 `shapenet\data` ：

```shell
ls shapenet/*.tar.gz |xargs -n1 -i tar -xf {} -C shapenet/data/
```

接下来，用于 IF-Nets 的输入和训练的点样本将被创建。下面三个命令可以并行地在多台机器上执行，并且可以显著地增加准备的速度。
首先，数据被转换为 `.off`-格式，并且使用下面的命令缩放：

```shell
python data_processing/convert_to_scaled_off.py
```

用于体素超分辨率的输入数据使用下面的命令创建：

```shell
python data_processing/voxelize.py -res 32
```

使用 `-res 32` 指定 32<sup>3</sup> 和 `-res 128` 指定 128<sup>3</sup> 分辨率。

点云补全的输入数据的创建：

```shell
python data_processing/voxelized_pointcloud_sampling.py -res 128 -num_points 300
```

使用 `-num_points 300` 用于 300个点的点云，使用 `-num_points 3000` 用于3000个点的点云。

训练的输入点和对应的基准占位值由下面的命令生成：

```shell
python data_processing/boundary_sampling.py -sigma 0.1
python data_processing/boundary_sampling.py -sigma 0.01
```

其中 `-sigma` 指定的是加入到曲面抽样中的正态分布位移的标准方差。

为了移除不能被预处理（小于 15 个面片）的网格可以执行下面的命令：

```shell
python data_processing/filter_corrupted.py -file 'voxelization_32.npy' -delete
```

输入的数据可以使用下面的命令将体素转换为 `.off`-格式从而可视化：

```shell
python data_processing/create_voxel_off.py -res 32
```

将？转换为 `.off`-格式从而可视化：

```shell
python data_processing/create_pc_off.py -res 128 -num_points 300
```

其中 `-res` 和 `-num_points` 匹配着来自前一步的值。

## 训练

IF-Nets的训练由下面的命令开始：

```shell
python train.py -std_dev 0.1 0.01 -res 32 -m ShapeNet32Vox -batch_size 6
```

其中 `-std_dev` 表示使用的 `sigma` 值， `-res` 表示输入的分辨率（32<sup>3</sup> 或者 128<sup>3</sup>）， `-m` 表示 IF-Net 模型的设置：

- ShapeNet32Vox for 32<sup>3</sup> voxel Super-Resolution experiment
- ShapeNet128Vox for 128<sup>3</sup> voxel Super-Resolution experiment
- ShapeNetPoints for Point Cloud Completion experiments
- SVR for 3D Single-View Reconstruction

其中 `-batch_size` 表示一个批处理中输入的不同的网格的数目，每从此网格包含5万个点样本（=6 用于较小的 GPU）。

如果你希望使用点云训练可以加上 `-pointcloud` 和 `-pc_samples` 确认使用的点样本的数目，即： `-pc_samples 3000`。
考虑使用最高可能的 `batch_size` 从而加速训练。

在 `experiments/` 目录中，你可以找到一个实验目录，实验目录包含了模型的检查点、验证最小值的检查点和包含了 tensorboard 汇总信息的目录，这个 tensorboard 可以使用下面的命令启动：

```shell
tensorboard --logdir experiments/YOUR_EXPERIMENT/summary/ --host 0.0.0.0
```

## 生成

```shell
python generate.py -std_dev 0.1 0.01 -res 32 -m ShapeNet32Vox -checkpoint 10 -batch_points 400000
```

上述命令生成了来自于 ShapeNet 的测试样本的重建（训练阶段没有重建）到一个目录中
```experiments/YOUR_EXPERIMENT/evaluation_CHECKPOINT_@256/generation```.
使用 `-checkpoint` 可以选择 IF-Nets 的检查点。使用具有最小验证错误值的模型可以使用
`-batch_points` 即刻指定匹配 GPU 内存的点的数目（40万用于小GPU）。
请在训练期间加入所有的参数。
> 生成的脚本可以同时运行在多台机器上以显著地增加生成的速度。而且，在你的 GPU 上尽可能地使用最大的批处理大小的值。

## 评估

```shell
python data_processing/evaluate.py -reconst -generation_path experiments/iVoxels_dist-0.5_0.5_sigmas-0.1_0.01_v32_mShapeNet32Vox/evaluation_10_@256/generation/
```

上述命令用于评估每一次重建，其中 `-generation_path` 是在前一步中生成重建对象的对象。
> 上述的评估脚本可以并行地运行在多台机器上从而显著地增加生成的速度。

```shell
python data_processing/evaluate.py -voxels -res 32
```

上述命令用于评估输入的质量。对于体素网格使用 `-voxels` 增加 `-res` 去指定输入的分辨率，而对于点云使用 `-pc` 增加 `-points` 指定点的数目。

所有重建和输入的定量评估采用下面的命令收集起来，并且存入 `experiment/YOUR_EXPERIMENT/evaluation_CHECKPOINT_@256` 目录：

```shell
python data_processing/evaluate_gather.py -voxel_input -res 32 -generation_path experiments/iVoxels_dist-0.5_0.5_sigmas-0.1_0.01_v32_mShapeNet32Vox/evaluation_10_@256/generation/
```

其中， `-voxel_input` 表示体素超分辨率实验，同时紧跟 `-res` 表示输入的分辨率 或者 `-pc_input` 表示点云实例，同时紧跟 `-points` 表示使用的点的数目。

## 预训练模型

[预训练模型的下载地址](https://nextcloud.mpi-klsb.mpg.de/index.php/s/rdBogFjm3LSxYGy).

## 联系方式

可以通过 Email 地址联系 [Julian Chibane](http://virtualhumans.mpi-inf.mpg.de/people/Chibane.html) 了解代码中的问题和注释。（参见论文）

## 版本

Copyright (c) 2020 Julian Chibane, Max-Planck-Gesellschaft

Please read carefully the following terms and conditions and any accompanying documentation before you download and/or use this software and associated documentation files (the "Software").

The authors hereby grant you a non-exclusive, non-transferable, free of charge right to copy, modify, merge, publish, distribute, and sublicense the Software for the sole purpose of performing non-commercial scientific research, non-commercial education, or non-commercial artistic projects.

Any other use, in particular any use for commercial purposes, is prohibited. This includes, without limitation, incorporation in a commercial product, use in a commercial service, or production of other artefacts for commercial purposes.
For commercial inquiries, please see above contact information.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

You understand and agree that the authors are under no obligation to provide either maintenance services, update services, notices of latent defects, or corrections of defects with regard to the Software. The authors nevertheless reserve the right to update, modify, or discontinue the Software at any time.

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. You agree to cite the `Implicit Functions in Feature Space for 3D Shape Reconstruction and Completion` paper in documents and papers that report on research using this Software.
