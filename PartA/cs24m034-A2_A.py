from tqdm.auto import tqdm
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import wandb

# Enable cuDNN benchmarking for faster convolution operations
torch.backends.cudnn.benchmark = True

wandb.login(key="7f46816d45e3df192c3053bab59032e9d710fef4")

def show_images(class_names, images, labels):
    plt.figure(figsize=(10, 10))
    for i in range(len(images)):
        plt.subplot(6, 6, i + 1)

        # Transpose the image tensor to (height, width, channels) for displaying
        plt.imshow(np.transpose(images[i], (1, 2, 0)))
        plt.title(f"{class_names[labels[i]]}")
        plt.axis('off')
    plt.subplots_adjust(wspace=0.4, hspace=0.5)
    plt.show()

def show_images_and_labels(device, model, test_loader, class_names):
    model.eval()
    images_to_log = {} 
    with torch.no_grad():  # Disable gradient tracking
        images_per_class = {class_name: 0 for class_name in class_names}
        fig, axes = plt.subplots(10, 3, figsize=(15, 30))  # 10x3 grid
        
        for images, labels in test_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            for image, label, pred in zip(images, labels, predicted):
                class_name = class_names[label.item()]
                if images_per_class[class_name] < 3:
                    ax = axes[label.item(), images_per_class[class_name]]
                    img = image.permute(1, 2, 0).cpu().numpy()
                    ax.imshow(img)
                    ax.set_title(f"Predicted: {class_names[pred.item()]}\nOriginal: {class_name}")
                    ax.axis('off')
                    images_per_class[class_name] += 1

                    # wandb.log({f"Image: {class_name}": wandb.Image(img, caption=f"Predicted: {class_names[pred.item()]}, Original: {class_name}")})
                    images_to_log[f"Predicted: {class_names[pred.item()]}, Original: {class_name}"] = wandb.Image(img)        
                    # print({f"Image_{class_name}": wandb.Image(img), 
                    #            f"Predicted_{class_name}": class_names[pred.item()],
                    #            f"Original_{class_name}": class_name})

            
            if all(count == 3 for count in images_per_class.values()):
                break
                
        # Prevent overlap
        plt.tight_layout()
        plt.show()

def data_generation(dataset_path, num_classes=10, data_augmentation=False, batch_size=32):
    
    # Mean and standard deviation values calculated from function get_mean_and_std
    mean = [0.4708, 0.4596, 0.3891]
    std = [0.1951, 0.1892, 0.1859]

    # Define transformations for training and testing data
    augment_transform = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.RandomHorizontalFlip(), 
        transforms.RandomRotation(30), 
        transforms.ToTensor(), 
        transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
    ])

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
        ])
    
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
    ])

    # Data augmentation (if data_augmentation = True) 
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

    # Create data loaders with optimized parameters for GPU
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    if data_augmentation:
      augmented_dataset = datasets.ImageFolder(root = dataset_path + "train", transform=augment_transform)
      augmented_loader = DataLoader(augmented_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
      train_loader = torch.utils.data.ConcatDataset([train_loader.dataset, augmented_loader.dataset])
      train_loader = DataLoader(train_loader, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Get class names
    classpath = pathlib.Path(dataset_path + "train")
    class_names = sorted([j.name.split('/')[-1] for j in classpath.iterdir() if j.name != ".DS_Store"])

    return train_loader, val_loader, test_loader, class_names

class ClassCNN(nn.Module):
  def __init__(self, num_filters, activation_function, filter_multiplier, filter_sizes, 
               dropout, batch_norm, dense_size, num_classes, image_size=256):
    super(ClassCNN, self).__init__()
        
    # Defining convolution layers
    layers = []
    params = 0
    self.activation = getattr(nn, activation_function)()
    initial_num_filters = num_filters
    
    for i, filter_size in enumerate(filter_sizes):
        
        if i == 0:
            num_filters = max(num_filters, 1)
            layers.append(nn.Conv2d(in_channels=3, out_channels=initial_num_filters, kernel_size=filter_size))
        
        else:
            num_filters = int(initial_num_filters * (filter_multiplier))     
            num_filters = max(num_filters, 1)   
            layers.append(nn.Conv2d(in_channels=initial_num_filters, out_channels=num_filters, kernel_size=filter_size))
            initial_num_filters = num_filters
            
            
        if batch_norm == True:
            layers.append(nn.BatchNorm2d(num_filters))
            
        layers.append(self.activation)
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    
    self.features = nn.Sequential(*layers)
    
    #Calculate the size of the feature maps after convolution and pooling
    layer_output = image_size
    for filter_size in filter_sizes:
        layer_output = (layer_output - filter_size + 1) // 2
        
    fc1_input_size = num_filters * layer_output * layer_output
    print("fc1_input: ", fc1_input_size)       
    
    # Defining maxpooling layer
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    # Defining fully connected layer
    self.flatten = nn.Flatten()
    self.fc1 = nn.Linear(fc1_input_size, dense_size)
    self.dropout = nn.Dropout(dropout)
    self.fc2 = nn.Linear(dense_size, num_classes)
    
  def forward(self, x):
    x = self.features(x)  # Apply convolutional and pooling layers
    x = self.flatten(x)   # Flatten the feature maps into a 1D tensor
    x = self.activation(self.fc1(x))  # Apply activation function to the first fully connected layer
    x = self.dropout(x)   # Apply dropout regularization
    x = self.fc2(x)
    # x = nn.functional.softmax(self.fc2(x), dim=1)  # Apply softmax activation to the output layer
    return x
  

def trainCNN(device, train_loader, val_loader, test_loader, model, num_epochs=10, optimizer="Adam"):    
    criterion = nn.CrossEntropyLoss()
    if optimizer == "Adam":
        opt_func = optim.Adam(model.parameters(), lr=0.001)
    
    # Initialize the scaler for AMP
    scaler = torch.cuda.amp.GradScaler()

    for epoch in tqdm(range(num_epochs)):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            opt_func.zero_grad(set_to_none=True)  # More efficient gradient zeroing
            
            # Use autocast for mixed precision training
            with torch.cuda.amp.autocast():
                outputs = model(inputs)  # Forward pass
                loss = criterion(outputs, labels)  # Compute the loss
            
            # Scale the loss, perform backward pass, and update parameters
            scaler.scale(loss).backward()
            scaler.step(opt_func)
            scaler.update()

            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            running_loss += loss.item() * inputs.size(0)
            
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
                val_inputs, val_labels = val_inputs.to(device, non_blocking=True), val_labels.to(device, non_blocking=True)
                
                # Use autocast for mixed precision validation
                with torch.cuda.amp.autocast():
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
    
        if epoch==num_epochs-1:
            model.eval()
            with torch.no_grad():
                test_total_correct = 0
                test_total_samples = 0
                test_running_loss = 0.0
                for test_inputs, test_labels in tqdm(test_loader):
                    test_inputs, test_labels = test_inputs.to(device, non_blocking=True), test_labels.to(device, non_blocking=True)
                    
                    # Use autocast for mixed precision evaluation
                    with torch.cuda.amp.autocast():
                        test_outputs = model(test_inputs)
                        test_loss = criterion(test_outputs, test_labels)
    
                    _, test_predicted = torch.max(test_outputs, 1)
                    test_total_correct += (test_predicted == test_labels).sum().item()
                    test_total_samples += test_labels.size(0)
    
                    test_running_loss += test_loss.item() * test_inputs.size(0)
    
                test_loss = test_running_loss / len(test_loader.dataset)
                test_accuracy = test_total_correct / test_total_samples
                print(f"Test Accuracy: {test_accuracy * 100:.2f}%, Test Loss: {test_loss:.4f}")

def main():
    dataset_path = '/kaggle/input/nature/inaturalist_12K/'  

    sweep_config = {
        'method' : 'bayes',                    #('random', 'grid', 'bayes')
        'project' : 'Testing_2',
        'metric' : {                           # Metric to optimize
            'name' : 'accuracy', 
            'goal' : 'maximize'
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

    def train():
        with wandb.init(project="Testing_2") as run:
            config = wandb.config
            run_name = "aug_" + str(config.data_augmentation) + "_bs_" + str(config.batch_size) + "_norm_" + str(config.batch_norm) + "_dropout_" + str(config.dropout) + "_fc_" + str(config.dense_size) + "_nfilters_" + str(config.num_filters) +"_ac_" + config.activation_function + "_fmul_" + str(config.filter_multiplier)
            wandb.run.name = run_name


            # Data generation and class names
            train_loader, val_loader, test_loader, class_names = data_generation(dataset_path, 
                                                                             num_classes=10, 
                                                                             data_augmentation=config.data_augmentation, 
                                                                             batch_size=config.batch_size)

            # Creates a length of 5 filter of values filter_size
            filter_sizes = []
            for i in range(5):
                filter_sizes.append(config.filter_size)
            
            
            # Torch function to switch between CPU and GPU
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            #device="cpu"
            print("Device: ", device)
            
            
            model = ClassCNN(num_filters=config.num_filters, 
                                activation_function=config.activation_function, 
                                filter_multiplier=config.filter_multiplier,
                                filter_sizes=filter_sizes, 
                                dropout=config.dropout, 
                                batch_norm=config.batch_norm,
                                dense_size=config.dense_size, 
                                num_classes=10, 
                                image_size=256)
            model.to(device)

            # Train the model
            trainCNN(device, train_loader, val_loader, test_loader, model, num_epochs=10, optimizer="Adam")
            run.finish                #bracket
            
            # Show the images and the labels (pred and true)
            #show_images_and_labels(device, model, test_loader, class_names)

    sweep_id = wandb.sweep(sweep=sweep_config)
    wandb.agent(sweep_id,project="Testing_2" , function=train, count=50)

if __name__ == "__main__":
    main()