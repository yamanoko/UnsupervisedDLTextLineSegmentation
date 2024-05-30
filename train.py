import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from sklearn.metrics import accuracy_score
from dataset import CustomDataset
from pretrain_model import PretrainModel

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set up dataset
dataset = CustomDataset('Dataset_with_label/first_half', 'Dataset_with_label/second_half', 'Dataset_with_label/paired_patches.csv')

# Split dataset into training set and validation set
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Set up dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Load model
model = PretrainModel().to(device)

# Set up loss function and optimizer
criterion = BCEWithLogitsLoss()
optimizer = Adam(model.parameters())

# Training and validation loop
for epoch in range(1):  # 100 epochs
    for step, ((first_img, second_img), labels) in enumerate(train_dataloader):
        model.train()
        first_img, second_img, labels = first_img.to(device), second_img.to(device), labels.to(device)
        # Forward pass
        outputs = model(first_img, second_img)
        loss = criterion(outputs, labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print validation loss every 100 steps
        if (step + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                val_preds = []
                val_labels = []
                for (val_first_img, val_second_img), val_labels_batch in val_dataloader:
                    val_first_img, val_second_img, val_labels_batch = val_first_img.to(device), val_second_img.to(device), val_labels_batch.to(device)
                    val_outputs = model(val_first_img, val_second_img)
                    val_loss = criterion(val_outputs, val_labels_batch)
                    val_preds.append(torch.round(torch.sigmoid(val_outputs)).cpu().numpy())
                    val_labels.append(val_labels_batch.cpu().numpy())
                val_preds = np.concatenate(val_preds)
                val_labels = np.concatenate(val_labels)
                val_acc = accuracy_score(val_labels, val_preds)
                print(f'Step {step+1}, Validation Loss: {val_loss.item()}, Validation Accuracy: {val_acc}')
                
    # Print final validation loss at the end of each epoch
    model.eval()
    with torch.no_grad():
        val_preds = []
        val_labels = []
        for (val_first_img, val_second_img), val_labels_batch in val_dataloader:
            val_first_img, val_second_img, val_labels_batch = val_first_img.to(device), val_second_img.to(device), val_labels_batch.to(device)
            val_outputs = model(val_first_img, val_second_img)
            val_loss = criterion(val_outputs, val_labels_batch)
            val_preds.append(torch.round(torch.sigmoid(val_outputs)).cpu().numpy())
            val_labels.append(val_labels_batch.cpu().numpy())
        val_preds = np.concatenate(val_preds)
        val_labels = np.concatenate(val_labels)
        val_acc = accuracy_score(val_labels, val_preds)
        print(f'Epoch {epoch+1}, Loss: {loss.item()}, Validation Loss: {val_loss.item()}, Validation Accuracy: {val_acc}')

# Save model
torch.save(model.state_dict(), 'pretrain_model.pth')
