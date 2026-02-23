import os
import json
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from collections import defaultdict

# =================== CONFIGURATION ===================
ROOT_DIR = './classification_frames'
ANNOT_TRAIN = os.path.join(ROOT_DIR, 'annotations_train.json')
ANNOT_VAL = os.path.join(ROOT_DIR, 'annotations_val.json')
ANNOT_TEST = os.path.join(ROOT_DIR, 'annotations_test.json')
BATCH_SIZE = 32
EPOCHS = 10
MODEL_SAVE_PATH = 'nitymed_resnet18.pth'

# =================== DATASET CLASS ===================
class NITYMEDFramesDataset(Dataset):
    def __init__(self, frames_root, annotations_json, classes=None, transform=None):
        self.frames_root = frames_root
        self.transform = transform
        with open(annotations_json, 'r') as f:
            self.annotations = json.load(f)
        self.frame_files = list(self.annotations.keys())
        self.label_set = set(self.annotations[f]["driver_state"] for f in self.frame_files)
        self.classes = sorted(classes) if classes else sorted(list(self.label_set))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        # assumes annotation keys are actual relative paths to frames
        self.frame_name_to_path = {f: f for f in self.frame_files}
    def __len__(self):
        return len(self.frame_files)
    def __getitem__(self, idx):
        frame_file = self.frame_files[idx]
        img_path = self.frame_name_to_path[frame_file]
        label = self.annotations[frame_file]["driver_state"]
        label_idx = self.class_to_idx[label]
        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, label_idx

# =================== DATA TRANSFORMS ===================
means = [0.485, 0.456, 0.406]
stds = [0.229, 0.224, 0.225]
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(means, stds),
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(means, stds),
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(means, stds),
    ]),
}

# =================== LOAD CLASSES ===================
def get_all_classes(*annot_paths):
    all_labels = set()
    for p in annot_paths:
        with open(p, 'r') as f:
            anns = json.load(f)
            for x in anns.values():
                if 'driver_state' in x:
                    all_labels.add(x['driver_state'])
    return sorted(list(all_labels))

classes = get_all_classes(ANNOT_TRAIN, ANNOT_VAL, ANNOT_TEST)
NUM_CLASSES = len(classes)

# =================== MAIN LOGIC ===================
def main():
    # Datasets and loaders
    train_dataset = NITYMEDFramesDataset(ROOT_DIR, ANNOT_TRAIN, classes, data_transforms['train'])
    val_dataset   = NITYMEDFramesDataset(ROOT_DIR, ANNOT_VAL, classes, data_transforms['val'])
    test_dataset  = NITYMEDFramesDataset(ROOT_DIR, ANNOT_TEST, classes, data_transforms['test'])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights='IMAGENET1K_V1')  # for torch>=0.13
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    best_acc = 0.0
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        # --- TRAIN ---
        model.train()
        running_loss, running_corrects = 0.0, 0
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += (preds == labels).sum().item()
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = running_corrects / len(train_loader.dataset)
        print(f"  Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")

        # --- VALIDATE ---
        model.eval()
        val_loss, val_corrects = 0.0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += (preds == labels).sum().item()
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects / len(val_loader.dataset)
        print(f"  Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  Saved new best model with Acc: {best_acc:.4f}")

    print("Training finished.")
    print(f"Best validation accuracy: {best_acc:.4f}")

    # ========== TESTING ==========
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.eval()
    correct = 0
    n = 0
    per_class = defaultdict(lambda: [0, 0])  # correct, total
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            n += len(labels)
            for l, p in zip(labels.cpu(), preds.cpu()):
                per_class[classes[l]][0] += (p == l).item()
                per_class[classes[l]][1] += 1
    acc = correct / n
    print(f"Test accuracy: {acc:.4f}")
    for cls in classes:
        if per_class[cls][1] > 0:
            print(f"Class {cls}: {(per_class[cls][0] / per_class[cls][1]):.4f}")

    # ========== INFERENCE EXAMPLE ==========
    def predict_image(filename, model, classes):
        trans = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ])
        image = Image.open(filename).convert('RGB')
        image_t = trans(image).unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            output = model(image_t)
            _, pred = torch.max(output, 1)
        return classes[pred.item()]
    # Example usage (uncomment to test):
    # print(predict_image('./classification_frames/P1043127_720/frame461.jpg', model, classes))

if __name__ == "__main__":
    main()
