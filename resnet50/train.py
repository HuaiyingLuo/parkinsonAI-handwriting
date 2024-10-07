import torch
import torch.nn as nn
from torchvision import transforms, datasets
import json
import os
import torch.optim as optim
from model import Resnet50
import logging

train_date = '1007'
train_person = 'yueqi'
data_type = 'meander'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logging.basicConfig(filename=f'resnet50/{data_type}_{train_date}_training_log.log', 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def dataset_processing(batch_size):
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    
    data_root = os.getcwd() 

    image_path = os.path.join(data_root, f"resnet50/handwritten_dataset/{data_type}/")

    train_dataset = datasets.ImageFolder(root=image_path+"train", transform=data_transform["train"])
    train_num = len(train_dataset)

    classifier_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in classifier_list.items())

    json_str = json.dumps(cla_dict, indent=4)
    with open(os.path.join(image_path, 'class_indices.json'), 'w') as json_file:
        json_file.write(json_str)
 
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
 
    validate_dataset = datasets.ImageFolder(root=image_path + "val",
                                        transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, validate_loader, train_num, val_num


def train(model, optimizer, train_loader, validate_loader, epochs, train_num, val_num, save_path):
    model_weight_path = "resnet50/resnet50-pretrained_weight.pth"
    pre_weights = torch.load(model_weight_path)

    pre_dict = {k: v for k, v in pre_weights.items() if "classifier" not in k}
    model.load_state_dict(pre_dict, strict=False)

    loss_function = nn.CrossEntropyLoss()
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for step, data in enumerate(train_loader, start=0):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            logits = model(images)
            loss = loss_function(logits, labels)

            loss.backward()
            optimizer.step()
    
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
 
        model.eval()
        acc = 0.0  
        with torch.no_grad():
            for val_data in validate_loader:
                val_images, val_labels = val_data
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                
                outputs = model(val_images)  
                predict_y = torch.max(outputs, dim=1)[1]
                acc += (predict_y == val_labels).sum().item()

            val_accurate = acc / val_num
            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(model.state_dict(), save_path)

        log_msg = f'Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Validation Accuracy: {val_accurate:.4f}'
        print(log_msg)
        logging.info(log_msg)

 
    print('Finished Training')


if __name__ == '__main__':
    save_path = f'resnet50/models/resnet50_{data_type}_{train_date}.pth'

    batch_size = 16
    model = Resnet50(num_classes=2).to(device)

    learning_rate = 1e-6
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    epochs = 10

    train_loader, validate_loader, train_num, val_num = dataset_processing(batch_size)

    log_msg = f'Train Date: {train_date}, Trained by: {train_person}, Training with {data_type}, Epochs: {epochs}, Batch size: {batch_size}'
    logging.info(log_msg)

    train(model, optimizer, train_loader, validate_loader, epochs, train_num, val_num, save_path)



