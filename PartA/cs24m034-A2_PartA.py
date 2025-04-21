from tqdm.auto import tqdm
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pathlib
import wandb

# Enable cuDNN benchmarking for faster convolution operations
torch.backends.cudnn.benchmark = True

wandb.login(key="7f46816d45e3df192c3053bab59032e9d710fef4")

def show_images(class_names, images, labels):
    num_images = len(images)
    cols = 6
    rows = (num_images + cols - 1) // cols  # Auto-calculate required rows

    fig, axes = plt.subplots(rows, cols, figsize=(12, 2 * rows))
    axes = axes.flatten()

    for i in range(len(axes)):
        ax = axes[i]
        if i < num_images:
            img = np.transpose(images[i], (1, 2, 0))  # CHW → HWC
            ax.imshow(img)
            ax.set_title(class_names[labels[i]])
        ax.axis('off')

    plt.tight_layout()
    plt.show()




def show_images_and_labels(device, model, test_loader, class_names):
    model.eval()
    images_to_log = {}
    
    # Track number of images shown per class
    images_per_class = {class_name: 0 for class_name in class_names}

    # Prepare a 10x3 subplot grid (10 classes, 3 images each)
    fig, axes = plt.subplots(10, 3, figsize=(15, 30))
    axes = axes.reshape(10, 3)

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            for img, label, pred in zip(images, labels, predicted):
                class_idx = label.item()
                class_name = class_names[class_idx]
                
                if images_per_class[class_name] < 3:
                    col_idx = images_per_class[class_name]
                    ax = axes[class_idx][col_idx]

                    img_np = img.permute(1, 2, 0).cpu().numpy()
                    ax.imshow(img_np)
                    ax.set_title(f"Predicted: {class_names[pred.item()]}\nOriginal: {class_name}")
                    ax.axis('off')

                    # Log to wandb
                    wandb_image = wandb.Image(img_np, caption=f"Predicted: {class_names[pred.item()]}, Original: {class_name}")
                    wandb.log({f"Image: {class_name}": wandb_image})
                    images_to_log[f"Predicted: {class_names[pred.item()]}, Original: {class_name}"] = wandb_image
                    
                    # Optional print log
                    print({
                        f"Image_{class_name}": wandb.Image(img_np),
                        f"Predicted_{class_name}": class_names[pred.item()],
                        f"Original_{class_name}": class_name
                    })

                    images_per_class[class_name] += 1

            if all(count == 3 for count in images_per_class.values()):
                break

    plt.tight_layout()
    plt.show()


def data_generation(dataset_path, num_classes=10, data_augmentation=False, batch_size=32):
    from torchvision import transforms, datasets
    import torch
    from torch.utils.data import DataLoader, Subset, ConcatDataset
    import random
    import pathlib

    # Mean and std values from get_mean_and_std
    mean = [0.4708, 0.4596, 0.3891]
    std = [0.1951, 0.1892, 0.1859]

    # Define transformations
    base_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    augment_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Load datasets
    train_dataset = datasets.ImageFolder(root=f"{dataset_path}train", transform=base_transform)
    test_dataset = datasets.ImageFolder(root=f"{dataset_path}val", transform=base_transform)

    # Stratified validation split
    train_data_class = {c: [] for c in range(num_classes)}
    for idx, label in enumerate(train_dataset.targets):
        train_data_class[label].append(idx)

    val_indices = []
    for indices in train_data_class.values():
        val_count = int(len(indices) * 0.2)
        val_indices.extend(random.sample(indices, val_count))

    train_indices = [i for i in range(len(train_dataset)) if i not in val_indices]

    train_data = Subset(train_dataset, train_indices)
    val_data = Subset(train_dataset, val_indices)

    # Prepare dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Augment data if required
    if data_augmentation:
        augmented_dataset = datasets.ImageFolder(root=f"{dataset_path}train", transform=augment_transform)
        augmented_data = Subset(augmented_dataset, train_indices)
        combined_data = ConcatDataset([train_data, augmented_data])
        train_loader = DataLoader(combined_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Extract class names
    class_dir = pathlib.Path(f"{dataset_path}train")
    class_names = sorted([folder.name for folder in class_dir.iterdir() if folder.is_dir() and folder.name != ".DS_Store"])

    return train_loader, val_loader, test_loader, class_names


class ClassCNN(nn.Module):
    def __init__(self, num_filters, activation_function, filter_multiplier, 
                 filter_sizes, dropout, batch_norm, dense_size, num_classes, 
                 image_size=256):
        super().__init__()
        
        # Network configuration
        self.activation = getattr(nn, activation_function)()
        self.layers = nn.ModuleList()
        current_channels = 3  # Input channels for RGB images
        output_size = image_size
        
        # Build convolutional blocks dynamically
        for i, kernel_size in enumerate(filter_sizes):
            out_channels = max(1, int(num_filters * (filter_multiplier ** i)))
            
            conv_layer = nn.Conv2d(
                in_channels=current_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size//2  # Add padding to maintain spatial dimensions
            )
            self.layers.append(conv_layer)
            
            if batch_norm:
                self.layers.append(nn.BatchNorm2d(out_channels))
                
            self.layers.append(self.activation)
            self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            
            current_channels = out_channels
            output_size = output_size // 2  # Account for pooling
        
        # Calculate flattened dimensions
        fc_input_size = current_channels * output_size * output_size
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_input_size, dense_size),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(dense_size, num_classes)
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.classifier(x)


def train_cnn_model(device, train_loader, val_loader, test_loader, 
                   model, num_epochs=10, optimizer_type="Adam"):
    """Optimized training procedure for CNN models"""
    
    # Setup loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images, labels = images.to(device), labels.to(device)
            
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            epoch_loss += loss.item() * images.size(0)

        # Calculate metrics
        train_loss = epoch_loss / len(train_loader.dataset)
        train_acc = correct / total
        print(f"Train | Accuracy: {train_acc*100:.2f}% | Loss: {train_loss:.4f}")
        wandb.log({'train_acc': train_acc, 'train_loss': train_loss})

        # Validation phase
        model.eval()
        val_loss, val_acc = evaluate_model(
            model, val_loader, criterion, device, "Validation"
        )
        wandb.log({'val_acc': val_acc, 'val_loss': val_loss})
        scheduler.step(val_acc)  # Adjust learning rate

        # Final test evaluation
        if epoch == num_epochs - 1:
            test_loss, test_acc = evaluate_model(
                model, test_loader, criterion, device, "Test"
            )
            wandb.log({'test_acc': test_acc, 'test_loss': test_loss})


def evaluate_model(model, data_loader, criterion, device, phase_name):
    """Shared evaluation procedure for validation and testing"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc=phase_name):
            images, labels = images.to(device), labels.to(device)
            
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            running_loss += loss.item() * images.size(0)
    
    avg_loss = running_loss / len(data_loader.dataset)
    accuracy = correct / total
    print(f"{phase_name} | Accuracy: {accuracy*100:.2f}% | Loss: {avg_loss:.4f}")
    return avg_loss, accuracy



def get_sweep_config():
    return {
        'method': 'bayes',
        'project': 'Testing_2',
        'metric': {
            'name': 'accuracy',
            'goal': 'maximize'
        },
        'parameters': {
            'data_augmentation': {
                'values': [True, False]
            },
            'batch_size': {
                'values': [32, 64]
            },
            'batch_norm': {
                'values': [True]
            },
            'dropout': {
                'values': [0.2, 0.3, 0.4]
            },
            'dense_size': {
                'values': [256, 512]
            },
            'num_filters': {
                'values': [16, 32, 64]
            },
            'filter_size': {
                'values': [3, 5]
            },
            'activation_function': {
                'values': ['ReLU', 'LeakyReLU', 'GELU']
            },
            'filter_multiplier': {
                'values': [2, 4]
            }
        }
    }



def train_model():
    with wandb.init(project="Testing_2") as run:
        config = wandb.config

        run_name = (
            f"aug_{config.data_augmentation}_bs_{config.batch_size}_norm_{config.batch_norm}_"
            f"dropout_{config.dropout}_fc_{config.dense_size}_nfilters_{config.num_filters}_"
            f"ac_{config.activation_function}_fmul_{config.filter_multiplier}"
        )
        wandb.run.name = run_name

        dataset_path = '/kaggle/input/nature/inaturalist_12K/'
        num_classes = 10
        image_size = 256
        filter_sizes = [config.filter_size] * 5

        train_loader, val_loader, test_loader, class_names = data_generation(
            dataset_path,
            num_classes=num_classes,
            data_augmentation=config.data_augmentation,
            batch_size=config.batch_size
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device in use:", device)

        model = ClassCNN(
            num_filters=config.num_filters,
            activation_function=config.activation_function,
            filter_multiplier=config.filter_multiplier,
            filter_sizes=filter_sizes,
            dropout=config.dropout,
            batch_norm=config.batch_norm,
            dense_size=config.dense_size,
            num_classes=num_classes,
            image_size=image_size
        ).to(device)

        trainCNN(device, train_loader, val_loader, test_loader, model, num_epochs=10, optimizer="Adam")
        
        # Show the images and the labels (pred and true)
        #show_images_and_labels(device, model, test_loader, class_names)


def main():
    sweep_config = get_sweep_config()
    sweep_id = wandb.sweep(sweep=sweep_config)
    wandb.agent(sweep_id, project="Testing_2", function=train_model, count=50)

if __name__ == "__main__":
    main()
