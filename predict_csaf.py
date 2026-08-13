"""
predict_csaf.py
Drop-in replacement for predict.py, used by app.py via subprocess.
Runs CSAF-Net (or any trained variant) on a single image and returns
JSON with prediction, confidence, crack type, and visualization images --
same output contract as the old predict.py so app.py needs minimal changes.

Usage (called by app.py):
    python predict_csaf.py <image_path> [--variant full_csaf]
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import json
import base64
from io import BytesIO

import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms

from csaf_net import build_model, get_device

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def preprocess(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    tensor = transform(image).unsqueeze(0)
    return tensor, image


def load_model(variant="full_csaf", checkpoint_dir="checkpoints"):
    device = get_device()
    model = build_model(variant=variant).to(device)
    ckpt_path = os.path.join(checkpoint_dir, f"{variant}_best.pth")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model, device


def to_base64(img_array):
    pil_img = Image.fromarray(img_array)
    buffered = BytesIO()
    pil_img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def analyze_crack(image):
    """Same Canny + Hough crack-type analysis as the original predict.py."""
    img_cv = np.array(image)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10)

    annotated_img = img_cv.copy()
    crack_type = "Fine Crack"

    if lines is not None:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            if x2 - x1 != 0:
                angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                angles.append(angle)

        if angles:
            avg_angle = np.mean(angles)
            if avg_angle < 20:
                crack_type = "Horizontal Crack"
            elif avg_angle > 70:
                crack_type = "Vertical Crack"
            elif avg_angle > 45:
                crack_type = "Diagonal Crack"
            else:
                crack_type = "Corner Crack"

    return crack_type, edges, annotated_img


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: predict_csaf.py <image_path> [--variant NAME]"}))
        sys.exit(1)

    image_path = sys.argv[1]
    variant = "full_csaf"
    if "--variant" in sys.argv:
        variant = sys.argv[sys.argv.index("--variant") + 1]

    try:
        model, device = load_model(variant=variant)
        img_tensor, image = preprocess(image_path)
        img_tensor = img_tensor.to(device)

        with torch.no_grad():
            logits, _ = model(img_tensor)
            prediction = torch.sigmoid(logits).item()

        result = {
            "prediction": prediction,
            "variant": variant,
            "crack_type": None,
            "canny_img": None,
            "annotated_img": None,
        }

        crack_type, edges, annotated_img = analyze_crack(image)
        result["canny_img"] = to_base64(edges)
        result["annotated_img"] = to_base64(annotated_img)

        if prediction > 0.5:
            result["crack_type"] = crack_type

        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
