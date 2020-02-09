import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

import dataset
import cnn_models



def predict():

    ###  Import Dataset ###
    test_set = dataset.get_test_dataset()
    test_loader = DataLoader(test_set, batch_size=360, shuffle=True, num_workers=12)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = cnn_models.FirstModel(features_size = 175, weights = 'models/resnet18-5c106cde.pth')
    state_dict = torch.load('target/FirstModel-resnet18-1.pth')
    model.load_state_dict(state_dict, strict=False)
    model = nn.DataParallel(model)
    model.eval()

    image_ids = list()
    predictions = list()

    with torch.no_grad():
    # '''
    # torch.no_grad() impacts the autograd engine and deactivate it. 
    # It will reduce memory usage and speed up computations but you won’t be able to backprop (which you don’t want in an eval script)
    # '''

        for batch_idx, test_batch in enumerate(test_loader):
        # ''' ### data in form sample = {'image': image, 'features': features, 'label': label, 'image_id': image_id} ### ''' 
            
            image = test_batch['image'].to(device)
            features = test_batch['features'].to(device)
            image_id = test_batch['image_id']
            image_ids += image_id

            preds = model(image.type(torch.float), features.type(torch.float))
            _, predicted = torch.max(preds.data, 1)
            predictions += predicted.tolist()


    predictions_df = pd.DataFrame()
    predictions_df['Id'] = image_ids
    predictions_df['Prediction'] = predictions

    return predictions_df


def submission():    
    predictions_df = predict()
    predictions_df.to_csv('target/submissions.csv', index = None)




if __name__ == '__main__':
    submission()

