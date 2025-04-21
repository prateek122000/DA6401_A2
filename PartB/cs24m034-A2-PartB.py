from tqdm.auto import tqdm
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset
import pathlib
import wandb
import gc

# Configure matrix multiplication precision for better performance
if hasattr(torch, 'set_float32_matmul_precision'):
    torch.set_float32_matmul_precision('medium')

# Initialize Weights & Biases for experiment tracking
wandb.login(key="7f46816d45e3df192c3053bab59032e9d710fef4")

def get_transforms(mean, std, augment=False):
    """Create image transformations pipeline"""
    base_transforms = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
    ]
    
    if augment:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            *base_transforms[1:]
        ])
    
    return transforms.Compose(base_transforms)

def split_dataset(dataset, num_classes, validation_ratio=0.2):
    """Divide dataset into training and validation sets with balanced classes"""
    class_indices = {cls: [] for cls in range(num_classes)}
    
    for idx, label in enumerate(dataset.targets):
        class_indices[label].append(idx)
    
    validation_indices = []
    for indices in class_indices.values():
        val_count = int(len(indices) * validation_ratio)
        validation_indices += random.sample(indices, val_count)
    
    train_indices = list(set(range(len(dataset))) - set(validation_indices))
    return Subset(dataset, train_indices), Subset(dataset, validation_indices)

def prepare_data_loaders(dataset_path, num_classes=10, augment_data=False, batch_size=32):
    """Prepare data loaders for training, validation, and testing"""
    # Precomputed normalization parameters
    channel_means = [0.4708, 0.4596, 0.3891]
    channel_stds = [0.1951, 0.1892, 0.1859]

    # Create transformation pipelines
    train_transform = get_transforms(channel_means, channel_stds)
    test_transform = get_transforms(channel_means, channel_stds)
    augment_transform = get_transforms(channel_means, channel_stds, augment=True)

    # Load datasets
    train_dataset = datasets.ImageFolder(
        root=f"{dataset_path}train", 
        transform=train_transform
    )
    test_dataset = datasets.ImageFolder(
        root=f"{dataset_path}val", 
        transform=test_transform
    )

    # Split into training and validation
    train_subset, val_subset = split_dataset(train_dataset, num_classes)

    # Configure data loader parameters
    loader_config = {
        "batch_size": batch_size,
        "num_workers": 4,
        "pin_memory": True,
        "persistent_workers": True
    }

    # Create data loaders
    train_loader = DataLoader(train_subset, shuffle=True, **loader_config)
    val_loader = DataLoader(val_subset, shuffle=False, **loader_config)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_config)

    # Add augmented data if enabled
    if augment_data:
        augmented_data = datasets.ImageFolder(
            root=f"{dataset_path}train", 
            transform=augment_transform
        )
        combined_data = ConcatDataset([train_subset, augmented_data])
        train_loader = DataLoader(combined_data, shuffle=True, **loader_config)

    # Extract class names from directory structure
    class_names = sorted(
        entry.name for entry in pathlib.Path(f"{dataset_path}train").iterdir() 
        if entry.is_dir() and entry.name != ".DS_Store"
    )

    return train_loader, val_loader, test_loader, class_names

def train_model(device, train_loader, val_loader, test_loader, model, epochs=10):
    """Train the neural network model"""
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Enable mixed precision training if available
    precision_scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    for epoch in tqdm(range(epochs)):
        model.train()
        epoch_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for inputs, labels in tqdm(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Mixed precision forward pass
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
            
            # Backpropagation with precision scaling
            precision_scaler.scale(loss).backward()
            precision_scaler.step(optimizer)
            precision_scaler.update()

            # Track performance metrics
            with torch.no_grad():
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)
                epoch_loss += loss.item() * inputs.size(0)
        
        # Calculate epoch statistics
        avg_loss = epoch_loss / len(train_loader.dataset)
        accuracy = correct_predictions / total_samples
        print(f"Epoch [{epoch+1}/{epochs}], Accuracy: {accuracy * 100:.2f}%, Loss: {avg_loss:.4f}")
        wandb.log({'accuracy': accuracy, 'loss': avg_loss})

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_correct = 0
            val_total = 0
            val_loss = 0.0
            
            for val_inputs, val_labels in tqdm(val_loader):
                val_inputs = val_inputs.to(device, non_blocking=True)
                val_labels = val_labels.to(device, non_blocking=True)
                
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    val_outputs = model(val_inputs)
                    current_loss = loss_function(val_outputs, val_labels)

                _, val_predicted = torch.max(val_outputs, 1)
                val_correct += (val_predicted == val_labels).sum().item()
                val_total += val_labels.size(0)
                val_loss += current_loss.item() * val_inputs.size(0)

            val_loss /= len(val_loader.dataset)
            val_accuracy = val_correct / val_total
            print(f"Validation Accuracy: {val_accuracy * 100:.2f}%, Loss: {val_loss:.4f}")
            wandb.log({'val_accuracy': val_accuracy, 'val_loss': val_loss})

        # Clean up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        # Final evaluation on test set
        if epoch == epochs-1:
            model.eval()
            with torch.no_grad():
                test_correct = 0
                test_total = 0
                test_loss = 0.0
                
                for test_inputs, test_labels in tqdm(test_loader):
                    test_inputs = test_inputs.to(device, non_blocking=True)
                    test_labels = test_labels.to(device, non_blocking=True)
                    
                    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                        test_outputs = model(test_inputs)
                        current_test_loss = loss_function(test_outputs, test_labels)
    
                    _, test_predicted = torch.max(test_outputs, 1)
                    test_correct += (test_predicted == test_labels).sum().item()
                    test_total += test_labels.size(0)
                    test_loss += current_test_loss.item() * test_inputs.size(0)
    
                test_loss /= len(test_loader.dataset)
                test_accuracy = test_correct / test_total
                print(f"Test Accuracy: {test_accuracy * 100:.2f}%, Loss: {test_loss:.4f}")

def freeze_parameters(model):
    """Disable gradient computation for all parameters"""
    for param in model.parameters():
        param.requires_grad = False

def freeze_partial_model(model, freeze_count):
    """Freeze specified number of layers"""
    frozen = 0
    for param in model.parameters():
        if frozen < freeze_count:
            param.requires_grad = False
            frozen += 1
        else:
            break

def enable_all_parameters(model):
    """Enable gradient computation for all parameters"""
    for param in model.parameters():
        param.requires_grad = True

def execute_pipeline():
    """Main execution function"""
    data_path = '/kaggle/input/nature/inaturalist_12K/'  
    augmentation_enabled = True
    batch_size = 64
    num_classes = 10
    fine_tune_option = 2
    layers_to_freeze = 12

    def run_training():
        with wandb.init(project="Testing_3") as experiment:
            config = wandb.config
            experiment_name = f"aug_{augmentation_enabled}_bs_{batch_size}_fine_tune_{fine_tune_option}"
            if fine_tune_option == 1:
                experiment_name += "_freeze_all"
            elif fine_tune_option == 3:
                experiment_name += "_freeze_none"
            else:
                experiment_name += f"_freeze_{layers_to_freeze}"

            wandb.run.name = experiment_name
            
            train_loader, val_loader, test_loader, class_names = prepare_data_loaders(
                data_path, 
                num_classes=num_classes, 
                augment_data=augmentation_enabled, 
                batch_size=batch_size
            )
            
            print(f"Training batches: {len(train_loader)}")
            print(f"Validation batches: {len(val_loader)}")
            print(f"Test batches: {len(test_loader)}")
    
            # Set up computation device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {device}")
        
            # Initialize model with pretrained weights
            try:
                model = models.googlenet(weights='IMAGENET1K_V1')
            except TypeError:
                # Fallback for older PyTorch versions
                model = models.googlenet(pretrained=True)
            model = model.to(device)
            
            # Enable model compilation if available
            if hasattr(torch, 'compile') and torch.cuda.is_available():
                print("Using model compilation for improved performance")

            # Apply selected fine-tuning strategy
            if fine_tune_option == 1:
                freeze_parameters(model)
                model.fc = nn.Linear(model.fc.in_features, num_classes)
                model = model.to(device)
                train_model(device, train_loader, val_loader, test_loader, model, epochs=5)
            
            elif fine_tune_option == 2:
                freeze_partial_model(model, layers_to_freeze)
                model.fc = nn.Linear(model.fc.in_features, num_classes)
                model = model.to(device)
                train_model(device, train_loader, val_loader, test_loader, model, epochs=5)

            else:
                enable_all_parameters(model)
                model.fc = nn.Linear(model.fc.in_features, num_classes)
                model = model.to(device)
                train_model(device, train_loader, val_loader, test_loader, model, epochs=5)
                
            # Clean up GPU resources
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    run_training()
    wandb.finish()
    
if __name__ == "__main__":
    execute_pipeline()
