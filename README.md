# -coze- · 手写公式识别 API（早期部署版）

手写数学公式识别的 **Flask API 服务**：接收图片 URL，返回 LaTeX 识别结果。

> ⚠️ **建议归档本仓库**：本项目与 `math-ocr-api`、`Handwritten-mathematical-formula-recognition` 内容高度重复（同一套模型代码），且 `math-ocr-api` 代码更规范。保留一个 API 仓库即可。

## ✨ 功能特性

- 🌐 `POST /recognize`：传入 `{"image": "图片URL"}`，服务端下载图片并识别
- 📝 返回 LaTeX 序列（JSON）
- 🧠 Encoder-Decoder 模型（DenseNet + GRU + Attention）
- 🎯 针对 **RTX 5090** 做了 CUDA 优化（cudnn.benchmark；无 GPU 时自动降级 CPU）
- 🔤 词表从 `dictionary.txt` 加载（112 tokens）

## 🚀 快速开始

```bash
pip install flask torch torchvision pillow requests
python api_server.py
```

### 调用

```bash
curl -X POST http://localhost:5000/recognize \
  -H "Content-Type: application/json" \
  -d '{"image": "https://example.com/formula.png"}'
```

## 📁 项目结构

```
-coze-/
├── api_server.py       # Flask 入口与识别逻辑
├── encoder.py          # DenseNet 编码器
├── decoder.py          # GRU 解码器
├── encoder_decoder.py  # 模型封装
├── dictionary.txt      # 词表
├── requirements.txt    # 依赖
├── Procflie            # ⚠️ 文件名拼写错误（应为 Procfile）
└── README.md
```

## ⚠️ 已知问题

1. **`Procflie` 拼写错误**：Render 等平台要求文件名是 `Procfile`，拼错会导致启动失败，请重命名。
2. **模型权重缺失**：代码加载 `model_weights.pkl`，仓库未包含，需自行放入。
3. **与 `math-ocr-api` 重复**：建议本仓库归档/删除，主页上保留主项目 + `math-ocr-api` 即可。

## 📄 许可

未指定开源许可（默认保留所有权利）。
