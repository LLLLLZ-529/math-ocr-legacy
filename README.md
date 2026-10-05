适用于coze的手写公式识别 API

手写数学公式识别的 **Flask API 服务**：接收图片 URL，返回 LaTeX 识别结果。

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
├── Procfile            
└── README.md
```


## 📄 许可

未指定开源许可（默认保留所有权利）。
