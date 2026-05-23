import os
import torch
import torch.nn as nn
from dataset import get_train_val_loaders
from train import CaptchaCNN, evaluate

def main():
    device = torch.device("cpu")
    train_loader, val_loader, _ = get_train_val_loaders(batch_size=64)
    
    model = CaptchaCNN().to(device)
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captcha_model.pth")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    criterion = nn.CrossEntropyLoss()
    avg_val_loss, char_acc, seq_acc = evaluate(model, val_loader, criterion, device)
    
    print(f"--- Saved Model Evaluation ---")
    # Print the exact metric name so we can read it easily
    print(f"Validation Loss: {avg_val_loss:.4f}")
    print(f"Character Accuracy: {char_acc:.2f}%")
    print(f"Validation Sequence Accuracy: {seq_acc:.2f}%")

if __name__ == "__main__":
    main()
