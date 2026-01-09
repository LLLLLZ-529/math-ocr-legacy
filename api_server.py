from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import sys
import os
import requests

sys.path.append('.')

# 尝试导入项目模块
try:
    from encoder_decoder import Encoder_Decoder
    from package.utils import gen_sample_bidirection
    print("✅ 项目模块导入成功")
except Exception as e:
    print(f"❌ 导入失败: {e}")

app = Flask(__name__)

# ============ 5090 显卡初始化 ============
# 强制使用 CUDA
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"🔥 检测到顶级显卡: {torch.cuda.get_device_name(0)}")
    # 针对 5090 的优化设置
    torch.backends.cudnn.benchmark = True 
else:
    print("⚠️ 未检测到显卡，将降级使用 CPU")

def load_system():
    try:
        # 1. 加载词表
        with open('dictionary.txt', 'r', encoding='utf-8') as f:
            tokens = [line.strip() for line in f if line.strip()][:112]
        idx2word = {i: tok for i, tok in enumerate(tokens)}

        # 2. 模型参数 (严格匹配你之前的 Size Mismatch 修复)
        params = {
            'growthRate': 24, 'reduction': 0.5, 'bottleneck': True, 'use_dropout': True,
            'L2R': 1, 'R2L': 1, 'D': 684, 'n': 512, 'm': 256, 'K': 112,
            'dim_attention': 512, 'input_channels': 1
        }
        
        # 3. 加载模型
        model = Encoder_Decoder(params)
        if os.path.exists('model_weights.pkl'):
            checkpoint = torch.load('model_weights.pkl', map_location=DEVICE)
            state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
            model.load_state_dict(state_dict, strict=False)
            print("✅ 权重已加载至 RTX 5090 显存")
        
        model = model.to(DEVICE)
        model.eval()
        return model, idx2word, params
    except Exception as e:
        print(f"❌ 系统初始化失败: {e}")
        return None, None, None

MODEL, IDX2WORD, PARAMS = load_system()

# ============ 核心识别逻辑 ============
def recognize_latex(image_bytes):
    try:
        # 1. 图片预处理
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((64, 64)), # 根据模型训练尺寸调整
            transforms.ToTensor(),
        ])
        img = Image.open(io.BytesIO(image_bytes)).convert('L')
        # 【关键】将张量移动到显卡
        input_tensor = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            # 2. 调用 gen_sample_bidirection
            # 注意: 根据你的源码，gpu_flag 必须为 True 才能让它内部执行 .cuda()
            result = gen_sample_bidirection(
                MODEL, 
                input_tensor, 
                PARAMS, 
                gpu_flag=True, 
                k=1, 
                maxlen=150, 
                idx_decoder=1
            )
            
            # 3. 结果解包 (根据你 utils.py 返回 4 个值的逻辑)
            samples = result[0] if isinstance(result, (tuple, list)) else result
            if not samples: return "识别失败"
            
            best_sample = samples[0]
            tokens = []
            for idx in best_sample:
                word = IDX2WORD.get(int(idx), "")
                if word in ('<eos>', '<EOS>', ''): break
                if not word.startswith('<PAD'):
                    tokens.append(word)
            return " ".join(tokens)
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"5090 推理报错: {str(e)}"

# ============ API 接口 ============
@app.route('/recognize', methods=['POST'])
def api():
    try:
        data = request.json
        if data and 'image' in data:
            image_url = data['image']
            img_content = requests.get(image_url, timeout=10).content
            latex = recognize_latex(img_content)
            print(f"✨ 结果: {latex}")
            return jsonify({"success": True, "latex_formula": latex})
        return jsonify({"success": False, "error": "No image URL"}), 400
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
