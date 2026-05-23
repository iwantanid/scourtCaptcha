import os
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_train_val_loaders

# 1. Model Definition
class CaptchaCNN(nn.Module):
    def __init__(self):
        super(CaptchaCNN, self).__init__()
        
        # Convolutional Layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> 32 x 20 x 60
            nn.Dropout(0.1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> 64 x 10 x 30
            nn.Dropout(0.1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> 128 x 5 x 15
            nn.Dropout(0.1)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> 256 x 2 x 7
            nn.Dropout(0.15)
        )
        
        # Dense Layer
        self.fc = nn.Sequential(
            nn.Linear(256 * 2 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        
        # Six Output Heads (one for each digit position)
        self.head1 = nn.Linear(512, 10)
        self.head2 = nn.Linear(512, 10)
        self.head3 = nn.Linear(512, 10)
        self.head4 = nn.Linear(512, 10)
        self.head5 = nn.Linear(512, 10)
        self.head6 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        
        out1 = self.head1(x)
        out2 = self.head2(x)
        out3 = self.head3(x)
        out4 = self.head4(x)
        out5 = self.head5(x)
        out6 = self.head6(x)
        
        return out1, out2, out3, out4, out5, out6

# 2. Evaluation Helper
def evaluate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct_chars = 0
    correct_sequences = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, targets, _ in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            out1, out2, out3, out4, out5, out6 = model(inputs)
            
            # Loss
            loss = (
                criterion(out1, targets[:, 0]) +
                criterion(out2, targets[:, 1]) +
                criterion(out3, targets[:, 2]) +
                criterion(out4, targets[:, 3]) +
                criterion(out5, targets[:, 4]) +
                criterion(out6, targets[:, 5])
            )
            val_loss += loss.item()
            
            # Predictions
            pred1 = torch.argmax(out1, dim=1)
            pred2 = torch.argmax(out2, dim=1)
            pred3 = torch.argmax(out3, dim=1)
            pred4 = torch.argmax(out4, dim=1)
            pred5 = torch.argmax(out5, dim=1)
            pred6 = torch.argmax(out6, dim=1)
            
            preds = torch.stack([pred1, pred2, pred3, pred4, pred5, pred6], dim=1)
            
            # Character-level accuracy
            correct_chars += (preds == targets).sum().item()
            
            # Sequence-level accuracy (all 6 digits must be correct)
            correct_seq_mask = (preds == targets).all(dim=1)
            correct_sequences += correct_seq_mask.sum().item()
            
            total_samples += inputs.size(0)
            
    avg_loss = val_loss / len(dataloader)
    char_acc = correct_chars / (total_samples * 6.0) * 100.0
    seq_acc = correct_sequences / total_samples * 100.0
    
    return avg_loss, char_acc, seq_acc

# 3. Main Training Loop
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        print(f"Using device: {device}")
    
    # Load data
    train_loader, val_loader, _ = get_train_val_loaders(batch_size=64)
    
    # Initialize model
    model = CaptchaCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Scheduler: reduce LR when validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    epochs = 120
    best_val_seq_acc = 0.0
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captcha_model.pth")
    
    print("Starting training...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        for inputs, targets, _ in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            out1, out2, out3, out4, out5, out6 = model(inputs)
            
            # Loss sum
            loss = (
                criterion(out1, targets[:, 0]) +
                criterion(out2, targets[:, 1]) +
                criterion(out3, targets[:, 2]) +
                criterion(out4, targets[:, 3]) +
                criterion(out5, targets[:, 4]) +
                criterion(out6, targets[:, 5])
            )
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Evaluate
        avg_val_loss, char_acc, seq_acc = evaluate(model, val_loader, criterion, device)
        
        # Step scheduler
        scheduler.step(avg_val_loss)
        
        print(f"Epoch [{epoch:03d}/{epochs:03d}] - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Char Acc: {char_acc:.2f}% | Val Seq Acc: {seq_acc:.2f}%")
        
        # Save best model
        if seq_acc > best_val_seq_acc:
            best_val_seq_acc = seq_acc
            torch.save(model.state_dict(), model_path)
            print(f"--> Saved best model with validation sequence accuracy: {seq_acc:.2f}%")
            
        # Early stopping once we hit our >= 99.0% validation sequence accuracy target
        if best_val_seq_acc >= 99.0:
            print(f"Successfully reached the target validation sequence accuracy ({best_val_seq_acc:.2f}%)! Stopping early.")
            break
            
    print(f"Training completed! Best Validation Sequence Accuracy: {best_val_seq_acc:.2f}%")
    
if __name__ == "__main__":
    main()
