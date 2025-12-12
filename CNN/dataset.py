import numpy as np
import csv
from torch.utils.data import Dataset
import torch

# import torch
# from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


class DataSet(Dataset):

    def __init__(self, images, labels=None, transform=None):
        self.transform = transform
        self.images = images.astype(np.float32)
        if labels is not None:
            self.labels = np.array(labels, dtype=np.int64)
        else:
            self.labels = None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]  # (H, W, C), float32 in [0,1] grâce à preprocess_images
        label = self.labels[idx] if self.labels is not None else None

        if self.transform is not None:
            # Pas de astype ici, ToPILImage gère très bien float [0,1]
            img = self.transform(img)
        else:
            img = torch.from_numpy(np.transpose(img, (2,0,1))).float()

        if label is not None:
            label = torch.tensor(label, dtype=torch.long)
            return img, label
        return img

    
    @staticmethod
    def preprocess_images(images):
        processed = images.copy().astype(np.float32)
    
        # Si les valeurs sont dans [0, 255], normaliser vers [0, 1]
        if processed.max() > 1:
            processed = processed / 255.0
        
        return processed
    
    @staticmethod
    def stratified_split(X, y, test_size=0.2, seed=42):

        np.random.seed(seed)

        train_idx = []
        val_idx = []

        for cls in np.unique(y):
            cls_idx = np.where(y == cls)[0]
            np.random.shuffle(cls_idx)

            #at leat 1 element in val 
            n_val = max(1, int(len(cls_idx) * test_size))

            val_idx.extend(cls_idx[:n_val])
            train_idx.extend(cls_idx[n_val:])
        
        return X[train_idx], X[val_idx], y[train_idx], y[val_idx]

    @staticmethod
    def train_model(model, train_loader, val_loader, criterion, optimizer,
                     scheduler=None, epochs=50, device='cpu', patience=10):
        
        model = model.to(device)

        history = {
            "train_loss" : [],
            "val_loss" : [],
            "train_acc" : [],
            "val_acc" : []
        }

        best_val_acc = 0
        best_model_state = None
        epochs_without_improvement = 0

        for epoch in range(epochs):

            model.train()

            train_loss = 0
            correct = 0
            total = 0

            for images, labels in train_loader:

                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(images)

                loss = criterion(outputs, labels)

                loss.backward()

                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            train_loss /= len(train_loader)
            train_acc = correct / total

            model.eval()

            val_loss = 0
            correct = 0
            total = 0

            with torch.no_grad():

                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            
            val_loss /= len(val_loader)
            val_acc = correct / total

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)

            if scheduler:
                scheduler.step(val_loss)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()  # Sauvegarder les poids
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            if (epoch + 1) % 5 == 0:
                print(f"   Epoch {epoch+1:3d}/{epochs} | "
                    f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            
            if epochs_without_improvement >= patience:
                print(f"\n   ⚠️ Early stopping à l'epoch {epoch+1} (pas d'amélioration depuis {patience} epochs)")
                break
        
        if best_model_state:
            model.load_state_dict(best_model_state)
            print(f"\n   ✅ Modèle restauré à la meilleure epoch (val_acc = {best_val_acc:.4f})")
        
        return model, history
    
    @staticmethod
    def write_preds_to_csv(preds) :

        with open("CNN_1", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(['ID', 'Label'])
            for i in range(len(preds)) :
                writer.writerow([i+1, preds[i]])
            

        




        

