import torch
from torchvision import transforms, models, datasets
from typing import Tuple
import torch.nn as nn
import torch.optim as optim
import numpy as np

Dataloader = torch.utils.data.DataLoader
Dataloaders = Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]

def get_dataloaders(train_dir: str, valid_dir: str, test_dir: str) -> Dataloaders:
    """Returns train, valid and test dataloaders

    Arguments:
        train_dir {str} -- Relative or absolute directory to training data. Each class has a separate folder.
        valid_dir {str} -- Relative or absolute directory to training data. Each class has a separate folder.
        test_dir {str} -- Relative or absolute directory to training data. Each class has a separate folder.

    Returns:
        Dataloaders -- A 3 value tuple containing train, valid and test dataloaders of type torch.utils.data.DataLoader
    """    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485,0.456,0.406],[0.229, 0.224, 0.225])
                                          ])


    validation_transforms = transforms.Compose([transforms.Resize(255),  
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485,0.456,0.406], [0.229, 0.224, 0.225])
                                               ])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=validation_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    validationloader = torch.utils.data.DataLoader(validation_data, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

    return (trainloader, validationloader, testloader)

def get_model_vgg16()->models.vgg.VGG:
    """[summary]

    Returns:
        models.vgg.VGG -- returns a pretrained VGG16 model with a costum classifier (fully connected) layer
    """    
    nbr_output_classes = 3
    vgg16 = models.vgg16(pretrained=True)
    #No gradient computation needed for these parameters (pretrained)
    for param in vgg16.parameters():
        param.requires_grad = False
    
    #define costum classifier
    classifier = nn.Sequential(
        nn.Linear(25088, 4096, bias=True),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(4096, 1024, bias=True),
        nn.ReLU(), 
        nn.Dropout(p=0.2),
        nn.Linear(1024,nbr_output_classes)
    )
    vgg16.classifier = classifier    
    return vgg16

def _validate_model(model:nn.Module, dataloader:Dataloader, criterion:nn.CrossEntropyLoss)->float:
    model.eval()
    valid_loss = 0
    for images, labels in dataloader:
        output = model(images)
        loss = criterion(output, labels)
        valid_loss += loss.item()
        break
    return valid_loss

        

def train_model(model:nn.Module, trainloader:Dataloader, validationloader:Dataloader, optim:optim.Optimizer, epochs:int, save_loc:str)->None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = nn.CrossEntropyLoss()
    min_valid_loss = np.Inf
    for epoch in range(epochs):
        model.to(device)
        train_loss = 0
        valid_loss = 0
        model.train()
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optim.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optim.step()
            train_loss += loss.item()
            break
        with torch.no_grad():
            valid_loss = _validate_model(model, validationloader, criterion)
        
        print("Epoch {}: train loss->{0:.3f}, valid loss->{0:.3f}".format(train_loss, valid_loss).format(epoch, train_loss, valid_loss))
        if valid_loss < min_valid_loss:
            print("Model saved at {}".format(save_loc))
            torch.save(model, save_loc)
    return model
        



            


            




    





    


