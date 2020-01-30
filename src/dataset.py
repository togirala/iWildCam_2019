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
from sklearn import preprocessing


## Based on guidelines from: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class PlaceTimeDataset(Dataset):
    ''' Location and Time with image dataset '''

    def __init__(self, csv_file, root_dir, transform=None):
        '''
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        '''
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir    ## 'data/'
        self.transform = transform  
        # self.time = None 
        # self.location = None
            
        
    def __len__(self):
        return len(self.df)
    

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        self.df['date_time'] = pd.to_datetime(self.df['date_captured'], errors='coerce')
        self.df["month"] = self.df['date_time'].dt.month - 1
        self.df["hour"] = self.df['date_time'].dt.hour

        self.df.loc[np.isfinite(self.df['hour']) == False, ['month', 'hour']] = 0
        self.df['hour'] = self.df['hour'].astype(int)
        self.df['month'] = self.df['month'].astype(int)  

        label = self.df.category_id[idx]
        month = self.df.month[idx]
        hour = self.df.hour[idx]
        location = self.df.location[idx]
        # print(label)
        
        
        
        
        # month = np.eye(12)[self.df.month.tolist()]
        # hours = np.eye(24)[self.df.hour.tolist()]
        # time = np.concatenate((month, hours), axis=1)
        
        
        img_name = os.path.join(self.root_dir, self.df.file_name[idx])
        image = imread(img_name)
        
        
        # image = Image.open(img_name).convert('RGB')
        # time = torch.from_numpy(self.time[index]).float()
        
        
        
        # landmarks = self.df.iloc[idx, 1:]
        
        
        
        # landmarks = np.array([landmarks])
        # landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'month': month, 'hour': hour, 'location': location, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample
        
        
        
train_dataset = PlaceTimeDataset(csv_file='data/train.csv', root_dir='data/train/')     
        
for i in range(len(train_dataset)):
    sample = train_dataset[i]
    # time = sample['time']
    location = sample['location']
    label = sample['label']

    print(i, sample['image'].shape, sample['month'], sample['hour'], sample['location'], sample['label'])  
    
    # print(label)
    
    
    
    
    break     






        
'''
        
    
    ###################

    def __init__(self, folder, df, n_category, transform=None):
        self.transform = transform
        self.root_dir = folder
        self.df = df
        self.y = np.array(df.get('category_id', []))
#         self.y = np.eye(n_category)[category_ids]
        month = np.eye(12)[df.month.tolist()]
        hours = np.eye(24)[df.hour.tolist()]
        self.time = np.concatenate((month, hours), axis=1)
    
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

train_ds = SimpleDataset(os.path.join(train_dir, 'train_images'), train_df, n_category=23, transform=data_transforms['train'])
test_ds = SimpleDataset(os.path.join(train_dir, 'test_images'), test_df, n_category=23, transform=data_transforms['test'])



#############


batch_size = 256

data_size = len(train_ds)
validation_fraction = .2

indices = list(range(data_size))
data_size = len(indices)
val_split = int(np.floor((validation_fraction) * data_size))

np.random.seed(42)
np.random.shuffle(indices)

val_indices, train_indices = indices[:val_split], indices[val_split:]

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, 
                                           sampler=train_sampler)
val_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size,
                                         sampler=val_sampler)
# Notice that we create test data loader in a different way. We don't have the labels
train_sampler = SubsetRandomSampler(train_indices)
# test_loader = torch.utils.data.DataLoader(test_ds, batch_size=512)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=512)




##################################


'''