import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

import dataset
import cnn_models


def loss_fn():
    return nn.CrossEntropyLoss()

def train_loop(device = 'cuda'):
    ### model, train_loader, val_loader, loss, optimizer, num_epochs
    model = cnn_models.FirstModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = loss_fn()
    
    
    ###  Import Dataset ###
    train_set, valid_set = dataset.get_train_valid_dataset()
    train_loader = DataLoader(train_set, batch_size=10, shuffle=True) 

    # for i, train_batch in enumerate(train_loader):
    #     print(train_batch['image'].shape)
    #     break
    
    for epoch in range(5):
        running_loss = 0.0
        
        for idx, training_batch in enumerate(train_loader):
            '''### data in form sample = {'image': image, 'month': month[idx], 'hour': hour[idx], 'location': location[idx], 'label': label} ###'''
            image = training_batch['image']
            month = training_batch['month']
            hour = training_batch['hour']
            location = training_batch['location']
            labels = training_batch['label']
            
            
            image = image.to(device)
            month = month.to(device)
            hour = hour.to(device)
            location = location.to(device)
            labels = labels.to(device)
            model = model.to(device)
            
            features = torch.cat((month, hour, location), dim = 1)

            ### Zero the parameter gradients
            optimizer.zero_grad()
            
            ### Make predictions based on models
            preds = model(image.type(torch.float32), features.type(torch.float32))
            
            ### Compute loss based on y_hat and y
            loss = criterion(preds, labels)
            
            ### Back Propagation
            loss.backward()
            
            ### Update Parameter Weights
            optimizer.step()
            
            running_loss += loss.item()
            





def train():
    
    ### model, train_loader, val_loader, loss, optimizer, num_epochs
    
    ###  Import Dataset ###
    # train_set, valid_set = dataset.get_train_valid_dataset()
    # train_loader = DataLoader(train_set, batch_size=10, shuffle=True) 

    # for i, train_batch in enumerate(train_loader):
    #     print(train_batch['image'].shape)
    #     break
    
    
    train_loop()
    

    

    
    
    
    
    
'''




class SimpleConv(nn.Module):
    def __init__(self, num_categories, len_dense, weighs):
        super(SimpleConv, self).__init__()
        self.model_conv = models.resnet152(pretrained=False)
        if weighs:
            self.model_conv.load_state_dict(torch.load(weighs))
        self.model_conv.fc = nn.Linear(self.model_conv.fc.in_features, num_categories)
        self.model_dense = nn.Linear(len_dense, num_categories)
        self.model = nn.Linear(2*num_categories, num_categories)
    
    def forward(self, x, y):
        x1 = self.model_conv(x)
        x2 = self.model_dense(y)
        x = F.relu(torch.cat((x1, x2), 1))
        return self.model(x)





device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
simple_conv = SimpleConv(23, 12+24, '../input/resnet152/resnet152.pth')
simple_conv = simple_conv.to(device)


def train_model(model, train_loader, val_loader, loss, optimizer, num_epochs):    
    loss_history = []
    train_history = []
    val_history = []
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    for epoch in range(num_epochs):
        model.train() # Enter train mode
        
        loss_accum = 0
        correct_samples = 0
        total_samples = 0

        for i_step, (x1, x2, y, _) in enumerate(train_loader):
          x1 ==> image
          x2 ==> time
            x1_gpu = x1.to(device)
            x2_gpu = x2.to(device)
            y_gpu = y.to(device)
            
            prediction = model(x1_gpu, x2_gpu)    
            loss_value = loss(prediction, y_gpu)
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            _, indices = torch.max(prediction, 1)
            correct_samples += torch.sum(indices == y_gpu)
            total_samples += y.shape[0]
            
            loss_accum += loss_value
#             print('{}/{}'.format(i_step, len(train_loader)))

        ave_loss = loss_accum / i_step
        train_accuracy = float(correct_samples) / total_samples
        val_f1, val_accuracy = compute_accuracy(model, val_loader)
        
        loss_history.append(float(ave_loss))
        train_history.append(train_accuracy)
        val_history.append(val_accuracy)
        
        print("Average loss: %f, Train accuracy: %f, Val accuracy: %f val F1: %f" % (ave_loss, train_accuracy, val_accuracy, val_f1))
        if val_f1 > best_acc:
            best_acc = val_f1
            best_model_wts = copy.deepcopy(model.state_dict())
            
    return loss_history, train_history, val_history, best_model_wts

'''


if __name__ == '__main__':
    train()