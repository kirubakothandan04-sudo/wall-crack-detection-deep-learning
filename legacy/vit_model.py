import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, Subset
from transformers import ViTForImageClassification, ViTFeatureExtractor
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

dataset_path = "DIP/Negative"

model_name = "google/vit-base-patch16-224-in21k"
print("Loading ViT model...")
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(
    model_name,
    num_labels=2
)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

print("Loading dataset...")
full_dataset = ImageFolder(root=dataset_path, transform=transform)

# Use only 500 images to save memory
indices = list(range(min(500, len(full_dataset))))
dataset = Subset(full_dataset, indices)
print(f"Using {len(dataset)} images")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(
    dataset, [train_size, val_size]
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
criterion = torch.nn.CrossEntropyLoss()

train_accuracies = []
train_losses = []

print("\nStarting Training...")
epochs = 3
for epoch in range(epochs):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device, dtype=torch.long)

        inputs = feature_extractor(
            images,
            return_tensors="pt",
            do_rescale=False
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(**inputs).logits
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_acc = 100 * correct / total
    train_accuracies.append(train_acc)
    train_losses.append(total_loss)
    print(f"Epoch [{epoch+1}/{epochs}], "
          f"Loss: {total_loss:.4f}, "
          f"Accuracy: {train_acc:.2f}%")

torch.save(model.state_dict(), "vit_wall_crack.pth")
print("\nViT Model saved as vit_wall_crack.pth")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_accuracies, marker='o', label='Train Accuracy')
plt.title('ViT Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(train_losses, marker='o',
         color='orange', label='Train Loss')
plt.title('ViT Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

print("\nEvaluating...")
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device, dtype=torch.long)
        inputs = feature_extractor(
            images,
            return_tensors="pt",
            do_rescale=False
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs).logits
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
print(f"\nFinal Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(
    all_labels, all_preds,
    target_names=['No Crack', 'Crack']
))

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Crack', 'Crack'],
            yticklabels=['No Crack', 'Crack'])
plt.title('ViT Confusion Matrix')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()