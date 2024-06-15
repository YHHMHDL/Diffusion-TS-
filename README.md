# 代码目录
## Diffusion-TS
### 1.Turorial_0.ipynb——无条件时间序列生成

**包括训练、采样过程，用PCA、t-SNE、kernel三种图形展示生成序列与真实序列的符合程度。**

### 2.Turorial_1.ipynb——条件时间序列生成，插补任务（imputation）

**包括训练、采样过程，红色点为观测数据，蓝色点为缺失数据，绿线为预测数据。**

### 3.Turorial_2.ipynb——条件时间序列生成，预测任务（forecasting）

**包括训练、采样过程，绘制了History、Ground Truth和Prediction数据曲线。**



# Experiments

### 1.metric_pytorch.ipynb

**计算Context-FID Score和Correlational Score**

### 2.metric_tensorflow.ipynb

**计算Discriminative score和Predictive Score**

### 3.mujoco_imputation.ipynb

 **MuJoCo数据集上的插补实验，分别计算缺失70%、80%、90%数据时的模型MSE。**

### 4.solar_nips_forecasting.ipynb

**Solar_nips数据集上的预测实验，绘制了History、Ground Truth和Prediction数据曲线。**

### 5.view_interpretability.ipynb

**在人工合成数据集上进行解纠缠验证。（DISENTANGLEMENT VALIDATION ON SYNTHETIC DATASET）；**

**可视化了合成时间序列的真实trend、季节性模式，以及模型学习到的trend、season、residual。**

## 原文环境要求
torch==2.0.1
einops==0.6.0
ema-pytorch==0.2.1
matplotlib==3.6.0
pandas==1.5.0
scikit-learn==1.1.2
scipy==1.11.1
seaborn==0.12.2
tqdm==4.64.1
dm-control==1.0.12
dm-env==1.6
dm-tree==0.1.8
mujoco==2.3.4
gluonts==0.12.6

urllib3 ==1.22
keras   ==3.3.3

## 本地电脑所用配置
1.
torch——pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

2.
pip install gluonts

pip install mxnet   

//as gluonts relies on mxnet install MXnet using
