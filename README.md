# 手写数学公式识别 (Handwritten Mathematical Formula Recognition)

这是一个基于深度学习的手写数学公式识别项目。该项目旨在将手写的数学公式图像转化为相应的 LaTeX 代码或 MathML 格式。

## 📋 目录
- [简介](#简介)
- [环境依赖](#环境依赖)
- [数据集](#数据集)
- [快速开始](#快速开始)
- [训练](#训练)
- [测试与评估](#测试与评估)
- [结果展示](#结果展示)

## 📖 简介
本项目使用 [请在此处填写模型名称，例如：Encoder-Decoder with Attention] 架构来实现端到端的公式识别。
主要特性：
- 支持不定长公式序列识别
- 输出标准的 LaTeX 语法
- 如果有预训练模型，可直接进行推理

## 📦 环境依赖

请确保您的环境满足以下要求：

- Python >= 3.6
- PyTorch >= 1.7
- CUDA (如果需要 GPU 训练)

安装依赖包：
```bash
pip install -r requirements.txt
```

## 💾 数据集

本项目通常使用 **CROHME** (Competition on Recognition of Online Handwritten Mathematical Expressions) 数据集或类似格式的数据。

1. **下载数据**：请下载 CROHME 数据集或准备自己的数据。
2. **数据预处理**：
   将数据解压到 `data/` 目录下。
   运行预处理脚本（如果有）：
   ```bash
   python preprocess.py --data_path ./data/
   ```
   *注意：请根据实际的数据路径修改配置。*

## 🚀 快速开始 (Inference)

如果你已经下载了预训练模型，可以使用 `inference.py` 进行单张图片的测试。

```bash
python inference.py --image_path ./test_images/sample.png --model_path ./checkpoints/best_model.pth
```

## 🏋️‍♂️ 训练

配置好 `config.yaml` 或相关参数后，运行以下命令开始训练：

```bash
python train.py --config config.yaml
```

**参数说明**：
- `--batch_size`: 批大小
- `--epoch`: 训练轮数
- `--lr`: 学习率

## 📊 测试与评估

在测试集上评估模型性能（如计算 BLEU score 或 Expression Match rate）：

```bash
python eval.py --model_path ./checkpoints/best_model.pth --dataset test
```

## 📝 结果展示

| 输入图片 | 识别结果 (LaTeX) |
| :---: | :--- |
| <img src="docs/sample1.png" width="200"> | `x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}` |
| <img src="docs/sample2.png" width="200"> | `\int_{0}^{\infty} e^{-x^2} dx` |

## 🤝 贡献
欢迎提交 Issue 或 Pull Request 来改进本项目。

## 📄 许可证
[MIT License](LICENSE)
