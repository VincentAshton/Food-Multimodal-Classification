# Food_Multimodal

这是一个深度学习课程大作业项目，用于完成带有图像和英文文本描述的 5 类食物分类任务。

## 项目任务

本项目需要比较两种分类方法：

1. 仅图像输入分类：image -> CNN/ResNet18 -> 5 分类
2. 图像 + 文本联合输入分类：image -> ResNet18 特征，text -> Embedding + LSTM 特征，拼接后进行 5 分类

## 类别

- baozi
- cupcake
- french_fries
- rice
- youtiao

## 数据结构

```text
data/
  train/
    images/
    metadata.csv
  val/
    images/
    metadata.csv
  test/
    images/
    metadata.csv
```

每个 `metadata.csv` 包含三列：

```text
image_path,label,text
```

其中：

- `image_path`：图片相对于当前 split 的 `images/` 目录的路径
- `label`：食物类别名称
- `text`：图片对应的英文文本描述

## 项目结构

```text
src/
  config.py              # 统一配置文件
  check_data.py          # 数据完整性检查与类别分布统计
  dataset.py             # PyTorch Dataset 与图像预处理
  text_utils.py          # 英文文本分词、词表、padding 工具
  models.py              # 仅图像模型和多模态模型
  train_image.py         # 仅图像模型训练入口
  train_multimodal.py    # 图像 + 文本模型训练入口
  evaluate.py            # 测试集评估入口
  plot_curves.py         # 训练曲线绘制脚本
outputs/
  checkpoints/           # 保存模型权重
  logs/                  # 保存训练日志
  figures/               # 保存曲线图
```

## 当前阶段

项目已经搭建基础代码框架，并在关键位置加入中文注释。后续可以继续完善测试集评估、
实验对比表格、混淆矩阵和训练曲线分析。
