from tqdm.auto import tqdm
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import wandb
import gc

# Set matrix multiplication precision for better performance
if hasattr(torch, 'set_float32_matmul_precision'):
    torch.set_float32_matmul_precision('medium')

wandb.login(key="7f46816d45e3df192c3053bab59032e9d710fef4")

def data_generation(dataset_path, num_classes=10, data_augmentation=False, batch_size=32):
    
    # Mean and standard deviation values calculated from function get_mean_and_std on training dataset
    mean = [0.4708, 0.4596, 0.3891]
    std = [0.1951, 0.1892, 0.1859]

    # Define transformations for training and testing data
    augment_transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.RandomHorizontalFlip(), 
        transforms.RandomRotation(30), 
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
    ])

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
        ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
    ])

    # Load datasets
    train_dataset = datasets.ImageFolder(root = dataset_path + "train", transform=train_transform)
    test_dataset = datasets.ImageFolder(root = dataset_path + "val", transform=test_transform)
    
    # Split train dataset into train and validation sets
    train_data_class = dict()
    for c in range(num_classes):
        train_data_class[c] = [i for i, label in enumerate(train_dataset.targets) if label == c]

    val_data_indices = []
    val_ratio = 0.2  # 20% for validation
    for class_indices in train_data_class.values():
        num_val = int(len(class_indices) * val_ratio)
        val_data_indices.extend(random.sample(class_indices, num_val))

    # Create training and validation datasets
    train_data = torch.utils.data.Subset(train_dataset, [i for i in range(len(train_dataset)) if i not in val_data_indices])
    val_data = torch.utils.data.Subset(train_dataset, val_data_indices)

    # Number of workers for data loading (adjust based on CPU cores)
    num_workers = 4
    
    # Create optimized data loaders with pinned memory for faster GPU transfer
    train_loader = DataLoader(
        train_data, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_data, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )

    if data_augmentation:
        augmented_dataset = datasets.ImageFolder(root = dataset_path + "train", transform=augment_transform)
        augmented_loader = DataLoader(
            augmented_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        )
        train_loader = torch.utils.data.ConcatDataset([train_loader.dataset, augmented_loader.dataset])
        train_loader = DataLoader(
            train_loader, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        )

    # Get class names
    classpath = pathlib.Path(dataset_path + "train")
    class_names = sorted([j.name.split('/')[-1] for j in classpath.iterdir() if j.name != ".DS_Store"])

    return train_loader, val_loader, test_loader, class_names

def trainCNN(device, train_loader, val_loader, test_loader, model, num_epochs=10, optimizer="Adam"):    
    criterion = nn.CrossEntropyLoss()
    if optimizer == "Adam":
        opt_func = optim.Adam(model.parameters(), lr=0.001)
    
    # Initialize gradient scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    for epoch in tqdm(range(num_epochs)):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for inputs, labels in tqdm(train_loader):
            # Move inputs and labels to device with non-blocking transfer
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            opt_func.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            
            # Mixed precision forward pass
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(inputs)  # Forward pass
                loss = criterion(outputs, labels)  # Compute the loss
            
            # Scale gradients and optimize with mixed precision
            scaler.scale(loss).backward()  # Backward pass
            scaler.step(opt_func)  # Update parameters
            scaler.update()  # Update scaler

            # Calculate metrics
            with torch.no_grad():
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)
                running_loss += loss.item() * inputs.size(0)
        
        # Calculate epoch metrics
        loss = running_loss / len(train_loader.dataset)
        accuracy = total_correct / total_samples
        print(f"Epoch [{epoch+1}/{num_epochs}], Accuracy: {accuracy * 100:.2f}%, Loss: {loss:.4f}")
        wandb.log({'accuracy': accuracy, 'loss': loss})

        # Validation
        model.eval()
        with torch.no_grad():
            val_total_correct = 0
            val_total_samples = 0
            val_running_loss = 0.0
            
            for val_inputs, val_labels in tqdm(val_loader):
                val_inputs = val_inputs.to(device, non_blocking=True)
                val_labels = val_labels.to(device, non_blocking=True)
                
                # Mixed precision validation
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    val_outputs = model(val_inputs)
                    val_loss = criterion(val_outputs, val_labels)

                _, val_predicted = torch.max(val_outputs, 1)
                val_total_correct += (val_predicted == val_labels).sum().item()
                val_total_samples += val_labels.size(0)
                val_running_loss += val_loss.item() * val_inputs.size(0)

            val_loss = val_running_loss / len(val_loader.dataset)
            val_accuracy = val_total_correct / val_total_samples
            print(f"Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {val_accuracy * 100:.2f}%, Validation Loss: {val_loss:.4f}")
            wandb.log({'val_accuracy': val_accuracy, 'val_loss': val_loss})

        # Clear cache to avoid memory pressure
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        # Test accuracy evaluation at final epoch
        if epoch == num_epochs-1:
            model.eval()
            with torch.no_grad():
                test_total_correct = 0
                test_total_samples = 0
                test_running_loss = 0.0
                
                for test_inputs, test_labels in tqdm(test_loader):
                    test_inputs = test_inputs.to(device, non_blocking=True)
                    test_labels = test_labels.to(device, non_blocking=True)
                    
                    # Mixed precision testing
                    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                        test_outputs = model(test_inputs)
                        test_loss = criterion(test_outputs, test_labels)
    
                    _, test_predicted = torch.max(test_outputs, 1)
                    test_total_correct += (test_predicted == test_labels).sum().item()
                    test_total_samples += test_labels.size(0)
                    test_running_loss += test_loss.item() * test_inputs.size(0)
    
                test_loss = test_running_loss / len(test_loader.dataset)
                test_accuracy = test_total_correct / test_total_samples
                print(f"Test Accuracy: {test_accuracy * 100:.2f}%, Test Loss: {test_loss:.4f}")

def feature_extraction(model, device):
    for params in model.parameters():
        params.requires_grad = False

def freeze_till_k(model, device, k):
    # Counter to track the number of frozen layers
    frozen_layers = 0
    
    for param in model.parameters():
        # Freeze layers up to the k-th layer
        if frozen_layers < k:
            param.requires_grad = False
            frozen_layers += 1
        else:
            # Stop freezing layers after k-th layer
            break

def no_freezing(model, device):
    for params in model.parameters():
        params.requires_grad = True

def main():
    dataset_path = '/kaggle/input/nature/inaturalist_12K/'  

    # You can increase batch size for better GPU utilization if memory allows
    data_augmentation = True
    batch_size = 64  # Increased from 32 for better GPU utilization
    num_classes = 10
    fine_tuning_method = 2
    k = 12

    def train():
        with wandb.init(project="Testing_3") as run:
            config = wandb.config
            run_name = "aug_" + str(data_augmentation) + "_bs_" + str(batch_size) + "_fine_tune_" + str(fine_tuning_method) + "_num_freeze_layer_all"
            if fine_tuning_method != 1:
                run_name = "aug_" + str(data_augmentation) + "_bs_" + str(batch_size) + "_fine_tune_" + str(fine_tuning_method) + "_num_freeze_layer_" + str(k)
            elif fine_tuning_method == 3:
                run_name = "aug_" + str(data_augmentation) + "_bs_" + str(batch_size) + "_fine_tune_" + str(fine_tuning_method) + "_num_freeze_layer_none"

            wandb.run.name = run_name
            
            train_loader, val_loader, test_loader, class_names = data_generation(
                dataset_path, 
                num_classes=10, 
                data_augmentation=data_augmentation, 
                batch_size=batch_size
            )
            
            print("Train: ", len(train_loader))
            print("Val: ", len(val_loader))
            print("Test: ", len(test_loader))
    
            # Properly detect and use CUDA device for Kaggle
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print("Device: ", device)
        
            # Use weights parameter instead of pretrained for newer PyTorch versions
            if hasattr(models.googlenet, 'pretrained'):
                model = models.googlenet(pretrained=True)
            else:
                model = models.googlenet(weights='IMAGENET1K_V1')
                
            model.to(device)
            
            # Enable model compilation for PyTorch 2.0+ if available
            if hasattr(torch, 'compile') and torch.cuda.is_available():
                #model = torch.compile(model)
                print("Using compiled model for faster training")

            if fine_tuning_method == 1:
                feature_extraction(model, device)
                model.fc = nn.Linear(model.fc.in_features, num_classes)
                model.to(device)
                trainCNN(device, train_loader, val_loader, test_loader, model, num_epochs=5, optimizer="Adam")
            
            elif fine_tuning_method == 2:
                freeze_till_k(model, device, k)
                model.fc = nn.Linear(model.fc.in_features, num_classes)
                model.to(device)
                trainCNN(device, train_loader, val_loader, test_loader, model, num_epochs=5, optimizer="Adam")

            else:
                feature_extraction(model, device)
                model.fc = nn.Linear(model.fc.in_features, num_classes)
                model.to(device)
                trainCNN(device, train_loader, val_loader, test_loader, model, num_epochs=5, optimizer="Adam")
                
            # Clean up GPU memory when done
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    train()
    wandb.finish()
    
if __name__ == "__main__":
    main()