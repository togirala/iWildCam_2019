import os
# import cv2
# import math

# from PIL import Image

import pandas as pd
import numpy as np


from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import torch
import os
import io
from PIL import Image
from imread import imread

train_dir = 'data/'
def get_data():
    try:
        data_dir = 'data/'
        train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
        test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    except Exception as e:
        print(f'Exception: {e}')  
          
    return train_df, test_df


def get_time_df(df):        
    try:
        df['date_time'] = pd.to_datetime(df['date_captured'], errors='coerce')
        df["month"] = df['date_time'].dt.month - 1
        df["hour"] = df['date_time'].dt.hour
    except Exception as e:
        print(f'Exception: {e}')
        
    df.loc[np.isfinite(df['hour']) == False, ['month', 'hour']] = 0
    df['hour'] = df['hour'].astype(int)
    df['month'] = df['month'].astype(int)
    
    return df



train_df, test_df = get_data()

train_df = get_time_df(train_df)
test_df = get_time_df(test_df)






# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# simple_conv = SimpleConv(23, 12+24, '../input/resnet152/resnet152.pth')
# simple_conv = simple_conv.to(device)



##########################################

from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader

class SimpleDataset(Dataset):
    def __init__(self, folder, df, n_category, transform=None):
        self.transform = transform
        self.root_dir = folder
        self.df = df
        self.y = np.array(df.get('category_id', []))
#         self.y = np.eye(n_category)[category_ids]
        month = np.eye(12)[df.month.tolist()]

        hours = np.eye(24)[df.hour.tolist()]
        
        
        print(month, hours)
        print('-------------')
        self.time = np.concatenate((month, hours), axis=1)
        print(self.time)
        print('------')
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_name = os.path.join(self.root_dir, self.df.file_name[index])
        image = Image.open(img_name)
        if len(self.y):
            label = self.y[index]
        else:
            label = 0
        image = Image.open(img_name).convert('RGB')
        time = torch.from_numpy(self.time[index]).float()
        print(time)
        
        if self.transform:
            image = self.transform(image)
        return image, time, label, self.df.id[index]









###################



from torchvision import transforms

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.RandomResizedCrop((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


####################

train_ds = SimpleDataset(os.path.join(train_dir, 'train'), train_df, n_category=23)
test_ds = SimpleDataset(os.path.join(train_dir, 'test'), test_df, n_category=23, transform=data_transforms['test'])

for i in range(len(train_ds)):
    sample = train_ds[i]
    # print(sample)
    # print(sample.shape)

    

    # print(i, sample['image'].shape, sample['time'].shape, sample['label'].shape)  
    break    
