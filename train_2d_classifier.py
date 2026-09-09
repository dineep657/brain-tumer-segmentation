import os
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image, ImageEnhance

torch.set_num_threads(os.cpu_count() or 4)

CLASSES = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']

# 1. Residual Convolutional Block
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(4, channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(4, channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.conv(x))

# 2. Upgraded Lightweight 2D ResNet-like Multi-Class Classifier
class LightweightClassifier2D(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.prep = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2) # 64x64
        )
        self.res1 = ResBlock(16)
        
        self.down1 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2, bias=False),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True) # 32x32
        )
        self.res2 = ResBlock(32)
        
        self.down2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2, bias=False),
            nn.GroupNorm(4, 64),
            nn.ReLU(inplace=True) # 16x16
        )
        self.res3 = ResBlock(64)
        
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.prep(x)
        x = self.res1(x)
        x = self.down1(x)
        x = self.res2(x)
        x = self.down2(x)
        x = self.res3(x)
        x = self.avgpool(x)
        logits = self.classifier(x)
        return logits

# 3. High-Confidence Dataset with Rich Medical Data Augmentations
class MRIClassificationDataset(Dataset):
    def __init__(self, root_dir, augment=False):
        self.root_dir = root_dir
        self.augment = augment
        self.samples = []
        
        for c_idx, cname in enumerate(CLASSES):
            cdir = os.path.join(root_dir, cname)
            if os.path.exists(cdir):
                for fname in sorted(os.listdir(cdir)):
                    if fname.endswith('.png'):
                        self.samples.append((os.path.join(cdir, fname), c_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('L').resize((128, 128), Image.Resampling.BILINEAR)
        
        if self.augment:
            if random.random() > 0.5:
                img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                angle = random.uniform(-12.0, 12.0)
                img = img.rotate(angle, resample=Image.Resampling.BILINEAR)
            if random.random() > 0.5:
                factor = random.uniform(0.8, 1.2)
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(factor)
        
        img_arr = np.array(img, dtype=np.float32)
        min_val, max_val = img_arr.min(), img_arr.max()
        if max_val > min_val:
            normalized_img = (img_arr - min_val) / (max_val - min_val + 1e-8)
        else:
            normalized_img = img_arr / 255.0
            
        tensor = torch.from_numpy(normalized_img).unsqueeze(0) # (1, 128, 128)
        return tensor, torch.tensor(label, dtype=torch.long)

def train_classifier(epochs: int = 8, batch_size: int = 128, lr: float = 1.5e-3):
    os.makedirs("models", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Fast High-Confidence ResNet Classifier Training on device: {device} (CPU Threads: {torch.get_num_threads()})")
    
    train_dataset = MRIClassificationDataset("classification_dataset/train", augment=True)
    val_dataset = MRIClassificationDataset("classification_dataset/val", augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = LightweightClassifier2D(num_classes=len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_acc = 0.0
    model_save_path = os.path.join("models", "brain_tumor_classifier_2d.pth")
    
    print(f"Beginning Fast High-Confidence ResNet Training ({epochs} Epochs, Batch Size {batch_size})...")
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        correct_train = 0
        
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct_train += (preds == labels).sum().item()
            
        scheduler.step()
        train_loss /= len(train_dataset)
        train_acc = (correct_train / len(train_dataset)) * 100.0
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(dim=1)
                correct_val += (preds == labels).sum().item()
                
        val_loss /= len(val_dataset)
        val_acc = (correct_val / len(val_dataset)) * 100.0
        
        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} (Acc: {train_acc:.2f}%) | Val Loss: {val_loss:.4f} (Acc: {val_acc:.2f}%)")
            
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'state_dict': model.state_dict(),
                'classes': CLASSES
            }, model_save_path)
            
    total_time = time.time() - start_time
    print(f"\nResNet Classifier training completed in {total_time:.2f} seconds.")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Saved trained PyTorch ResNet Classifier checkpoint to: {model_save_path}")

if __name__ == "__main__":
    train_classifier()
