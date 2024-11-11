import torch
import torch.nn as nn
from torchvision import transforms
import json
import os
import torch.optim as optim
from model_bagging import FusionModel
import logging
from dataset_bagging import DualImageDataset

train_date = '1111'
train_person = 'yueqi'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logging.basicConfig(filename=f'resnet50/train_logs/fusion_{train_date}_training_log.log', 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def dataset_processing(batch_size):
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])}
    
    data_root = os.getcwd()
    dataset_root = os.path.join(data_root, "resnet50", "handwritten_dataset")

    meander_train_path = os.path.join(dataset_root, "meander", "train")
    meander_val_path = os.path.join(dataset_root, "meander", "val")
    spiral_train_path = os.path.join(dataset_root, "spiral", "train")
    spiral_val_path = os.path.join(dataset_root, "spiral", "val")
    
    train_dataset = DualImageDataset(meander_train_path, spiral_train_path, transform=data_transform["train"])
    val_dataset = DualImageDataset(meander_val_path, spiral_val_path, transform=data_transform["val"])
    
    train_num = len(train_dataset)
    val_num = len(val_dataset)
    
    classifier_list = train_dataset.class_to_idx
    cla_dict = {val: key for key, val in classifier_list.items()}

    json_str = json.dumps(cla_dict, indent=4)
    with open(os.path.join(data_root, 'class_indices.json'), 'w') as json_file:
        json_file.write(json_str)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    validate_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, validate_loader, train_num, val_num

def train(model, optimizer, train_loader, validate_loader, epochs, train_num, val_num, save_path):
    meander_weight_path = "resnet50/models/resnet50_meander_1104.pth"
    spiral_weight_path = "resnet50/models/resnet50_spiral_1103.pth"

    meander_pre_weights = torch.load(meander_weight_path, map_location=device)
    meander_pre_dict = {k: v for k, v in meander_pre_weights.items() if "fc" not in k}
    model.resnet_meander.load_state_dict(meander_pre_dict, strict=False)

    spiral_pre_weights = torch.load(spiral_weight_path, map_location=device)
    spiral_pre_dict = {k: v for k, v in spiral_pre_weights.items() if "fc" not in k}
    model.resnet_spiral.load_state_dict(spiral_pre_dict, strict=False)

    loss_function = nn.CrossEntropyLoss()
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for step, data in enumerate(train_loader, start=0):
            meander_images, spiral_images, labels = data
            meander_images, spiral_images, labels = meander_images.to(device), spiral_images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(meander_images, spiral_images)
            loss = loss_function(outputs, labels)

            loss.backward()
            optimizer.step()
    
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        acc = 0.0  
        with torch.no_grad():
            for val_data in validate_loader:
                meander_images, spiral_images, labels = val_data
                meander_images, spiral_images, labels = meander_images.to(device), spiral_images.to(device), labels.to(device)
                
                outputs = model(meander_images, spiral_images)  
                predict_y = torch.max(outputs, dim=1)[1]
                acc += (predict_y == labels).sum().item()

            val_accurate = acc / val_num
            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(model.state_dict(), save_path)

        log_msg = f'Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Validation Accuracy: {val_accurate:.4f}'
        print(log_msg)
        logging.info(log_msg)

    print('Finished Training')

if __name__ == '__main__':
    save_path = f'resnet50/models/resnet50_fusion_{train_date}.pth'

    batch_size = 16
    model = FusionModel(num_classes=2).to(device)

    learning_rate = 1e-4 
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    epochs = 5  

    train_loader, validate_loader, train_num, val_num = dataset_processing(batch_size)

    log_msg = f'Train Date: {train_date}, Trained by: {train_person}, Training fusion model, Epochs: {epochs}, Batch size: {batch_size}'
    logging.info(log_msg)

    train(model, optimizer, train_loader, validate_loader, epochs, train_num, val_num, save_path)
