import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

dataset_path = "DIP/Negative"
img_size = (64, 64)

crack_images = glob(os.path.join(dataset_path, "crack", "*.jpg"))[:500]
no_crack_images = glob(os.path.join(dataset_path, "no_crack", "*.jpg"))[:500]

print(f"Crack images: {len(crack_images)}")
print(f"No Crack images: {len(no_crack_images)}")

image_paths = crack_images + no_crack_images
labels = [1] * len(crack_images) + [0] * len(no_crack_images)

train_paths, test_paths, train_labels, test_labels = train_test_split(
    image_paths, labels, test_size=0.2, random_state=42)
train_paths, val_paths, train_labels, val_labels = train_test_split(
    train_paths, train_labels, test_size=0.2, random_state=42)

def load_images(paths, labels):
    images = []
    for img_path in paths:
        img = cv2.imread(img_path)
        img = cv2.resize(img, img_size)
        img = img.astype(np.float32) / 255.0
        images.append(img)
    return np.array(images), np.array(labels)

print("Loading images...")
X_train, y_train = load_images(train_paths, train_labels)
X_val, y_val = load_images(val_paths, val_labels)
X_test, y_test = load_images(test_paths, test_labels)
print("Images loaded!")

datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen.fit(X_train)

def build_cnn():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

cnn_model = build_cnn()
history_cnn = cnn_model.fit(
    datagen.flow(X_train, y_train, batch_size=16),
    validation_data=(X_val, y_val),
    epochs=10
)

cnn_model.save("cnn_wall_crack_model.h5")
print("CNN Model saved!")

def plot_metrics(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

plot_metrics(history_cnn)

y_pred = (cnn_model.predict(X_test) > 0.5).astype(int)
print(f"\nAccuracy: {np.mean(y_pred.flatten() == y_test):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=['No Crack', 'Crack']))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Crack', 'Crack'],
            yticklabels=['No Crack', 'Crack'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()