import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['GRPC_ENABLE_FORK_SUPPORT'] = 'False'
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import json
import numpy as np
import cv2
import base64
from PIL import Image
from io import BytesIO


def preprocess(image_path, size=64):
    image = Image.open(image_path).convert("RGB")
    img = np.array(image)
    img_resized = cv2.resize(img, (size, size))
    img_resized = img_resized.astype(np.float32) / 255.0
    return np.expand_dims(img_resized, axis=0), image


def load_crack_model():
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.load_weights("cnn_wall_crack_model.h5")
    return model


def to_base64(img_array, is_gray=False):
    pil_img = Image.fromarray(img_array)
    buffered = BytesIO()
    pil_img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def analyze_crack(image):
    img_cv = np.array(image)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)

    # Apply Canny Edge Detection
    edges = cv2.Canny(gray, 50, 150)

    # Run Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10)

    annotated_img = img_cv.copy()
    crack_type = "Fine Crack"

    if lines is not None:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Draw Hough lines on annotated image - FIXED: Use (0, 0, 255) for red in RGB format
            cv2.line(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Red lines in RGB

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
    if len(sys.argv) != 2:
        sys.exit(1)

    image_path = sys.argv[1]

    try:
        model = load_crack_model()
        img_array, image = preprocess(image_path)
        prediction = float(model.predict(img_array, verbose=0)[0][0])

        result = {
            "prediction": prediction,
            "crack_type": None,
            "canny_img": None,
            "annotated_img": None
        }

        # We always return the Canny and Annotated images for visualization
        crack_type, edges, annotated_img = analyze_crack(image)
        result["canny_img"] = to_base64(edges)
        result["annotated_img"] = to_base64(annotated_img)

        if prediction > 0.5:
            result["crack_type"] = crack_type

        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)