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

def train_loop(model, optimizer, criterion, train_loader, valid_loader, device, epochs):
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, training_batch in enumerate(train_loader):
            '''### data in form sample = {'image': image, 'features': features, 'label': label} ###'''
            image = training_batch['image']
            features = training_batch['features']
            labels = training_batch['label']
            
            image = image.to(device)
            features = features.to(device)
            labels = labels.to(device)
            model = model.to(device)
                        
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
            
            # print(loss.item())
            _, predicted = torch.max(preds.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        training_loss_epoch = running_loss / batch_idx
        print(f'training_loss_epoch = {training_loss_epoch}, validation accuracy = {100*correct/total}')
        
        eval_loop(model = model, 
                  valid_loader = valid_loader,
                  device = device) 

    torch.save(model.state_dict(), 'models/FirstModel.pth')    

    

def eval_loop(model, valid_loader, device):
    
    model.eval()  
    '''model.eval() will notify all your layers that you are in eval mode, that way, batchnorm or dropout layers will work in eval mode instead of training mode.'''
    correct = 0
    total = 0
        
    # Validation loop
    for batch_idx, validation_batch in enumerate(valid_loader): 
    # ''' ### data in form sample = {'image': image, 'features': features, 'label': label} ### ''' 
   
        
        with torch.no_grad():
        # '''
        # torch.no_grad() impacts the autograd engine and deactivate it. 
        # It will reduce memory usage and speed up computations but you won’t be able to backprop (which you don’t want in an eval script)
        # '''
            
            image = validation_batch['image']
            features = validation_batch['features']
            labels = validation_batch['label']
            
            image = image.to(device)
            features = features.to(device)
            labels = labels.to(device)
            model = model.to(device)
            
            preds = model(image, features)
            
            _, predicted = torch.max(preds.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # val_pred = np.append(val_pred, self.get_vector(y_pred.detach().cpu().numpy()))
            # loss = loss_fn(preds, self.get_vector(y_batch))
            # avg_val_loss += loss.item() / len(val_loader)
            
    print(f'validation accuracy = {100*correct/total}')

    ##### Model Checkpoint for best validation f1
    # val_f1 = self.calculate_metrics(train_targets[val_index], val_pred, f1_only=True)
    # if val_f1 > best_val_f1:
    #     prev_best_val_f1 = best_val_f1
    #     best_val_f1 = val_f1
    #     torch.save(model.state_dict(), self.PATHS['xlm'])
    #     evaluated_epoch = epoch

    # # Calc the metrics
    # self.save_metrics(train_targets[train_index], train_pred, avg_loss, 'train')
    # self.save_metrics(train_targets[val_index], val_pred, avg_val_loss, 'val')
    
    
def train():    
    
    ###  Import Dataset ###
    train_set, valid_set = dataset.get_train_valid_dataset()
    train_loader = DataLoader(train_set, batch_size=40, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=100, shuffle=True) 
    
    
    model = cnn_models.FirstModel(features_size = 175, weights = 'models/resnet152-b121ed2d.pth')
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = loss_fn()
    
    
    train_loop(
            model = model, 
            optimizer = optimizer, 
            criterion = criterion, 
            train_loader = train_loader, 
            valid_loader = valid_loader, 
            device = 'cuda', 
            epochs = 5
            )
    




'''


def compute_accuracy(model, loader):
    model.eval() 
    correct = 0
    total = 0
    predictions = np.empty(shape=len(val_sampler)).astype(int)
    ground_truth = np.empty(shape=len(val_sampler)).astype(int)
    
    with torch.no_grad():
        for i,(x1, x2, y, _) in enumerate(loader):
            begin = i*batch_size
            x1_gpu = x1.to(device)
            x2_gpu = x2.to(device)
            y_gpu = y.to(device)
            
            outputs = model(x1_gpu, x2_gpu)
            _, predicted = torch.max(outputs.data, 1)
            total += y_gpu.size(0)
            correct += (predicted == y_gpu).sum().item()
#             print(predictions.shape, np.array(predicted.cpu()).shape)
#             print(begin, len(predictions), min(begin+batch_size, len(predictions)))
            predictions[begin : min(begin+batch_size, len(predictions))] = np.array(predicted.cpu())
            ground_truth[begin : min(begin+batch_size, len(ground_truth))] = np.array(y_gpu.cpu())
        val_f1 = f1_score(predictions, ground_truth, average='macro')
    return val_f1, correct / total

'''


if __name__ == '__main__':
    train()