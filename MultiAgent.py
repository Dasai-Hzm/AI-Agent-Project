import os
import torch
import numpy as np
from PIL import Image
import cv2
from transformers import CLIPModel, CLIPProcessor

LOCAL_MODEL_DIR = input("请输入你的CLIP模型路径：")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


try:
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
except Exception as e:
    print(f"加载模型失败，请检查本地文件是否完整: {e}")
    print(
        "必须包含以下文件: config.json, pytorch_model.bin, preprocessor_config.json, tokenizer_config.json, vocab.json, merges.txt")
    exit()

video_path = input("请输入你的视频文件路径：")
output_frame_path =input("请输入你的最佳匹配帧保存路径：")

user_text = input("请输入一段英文描述（例如：a waitress standing in front of a restaurant）：")
if not user_text.strip():
    print("输入不能为空！")
    exit()

texts = [user_text]

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"无法打开视频文件: {video_path}")
    exit()

best_match_score = -1
best_match_frame = None
best_match_text = ""


frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    print(f"Processing frame {frame_count}...")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

    try:
        inputs = processor(
            text=texts,
            images=[pil_image],
            return_tensors="pt",
            padding=True
        ).to(device)
    except RuntimeError as e:
        print(f"预处理失败: {e}")
        continue

    with torch.no_grad():
        try:
            outputs = model(**inputs)
        except RuntimeError as e:
            print(f"推理错误: {e}")
            continue

    logits_per_image = outputs.logits_per_image.cpu().numpy()

    if len(texts) == 1:
        current_match_score = logits_per_image[0][0]
    else:
        probs = np.exp(logits_per_image) / np.sum(np.exp(logits_per_image), axis=1, keepdims=True)
        current_match_score = probs[0][0]

    current_match_text = texts[0]

    if current_match_score > best_match_score:
        best_match_score = current_match_score
        best_match_frame = frame
        best_match_text = current_match_text
        print(f"更新最佳匹配: '{best_match_text}' (分数: {best_match_score:.4f})")

cap.release()

if best_match_frame is not None:
    print(f"\n最佳匹配描述: '{best_match_text}' (分数: {best_match_score:.4f})")
    cv2.imshow("Best Matching Frame", best_match_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    save_frame = input("是否保存最佳匹配帧？(y/n): ").strip().lower()
    if save_frame == 'y':
        cv2.imwrite(output_frame_path, best_match_frame)
        print(f"最佳匹配帧已保存到: {output_frame_path}")
else:
    print("未找到匹配的帧。")