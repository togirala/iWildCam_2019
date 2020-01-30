from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
import numpy as np
import torch
import os
# from imread import imread
from sklearn import preprocessing
from skimage import io, transform
from skimage.color import gray2rgb
from PIL import Image


import warnings
warnings.filterwarnings("ignore")


'''Based on guidelines from: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html'''
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
        self.root_dir = root_dir    
        self.transform = transform             
        
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
        img_name = os.path.join(self.root_dir, self.df.file_name[idx])
        image = io.imread(img_name) 
        # image = Image.open(img_name).convert('RGB')  

        
        '''
        ### One hot encoding for categorical variables###
        # month = self.df.month[idx]
        # hour = self.df.hour[idx]
        # location = self.df.location[idx]
        
        ohe = preprocessing.OneHotEncoder()
        
        locations = np.array(list(set(self.df.location.values))).reshape(-1,1)   
        ohe.fit(locations)
        location = np.squeeze(np.asarray(ohe.transform(locations).todense()[location - 1,:]))
        
        months = np.array(list(set(self.df.month.values))).reshape(-1,1)   
        ohe.fit(months)
        month = np.squeeze(np.asarray(ohe.transform(months).todense()[month - 1,:]))
        
        hours = np.array(list(set(self.df.hour.values))).reshape(-1,1)    
        ohe.fit(hours) 
        hour = np.squeeze(np.asarray(ohe.transform(hours).todense()[hour - 1,:]))
        '''
        
        ### shortcut implementation for ohe  -- next 3 lines
        location = np.eye(139)[self.df.location.tolist()]
        month = np.eye(12)[self.df.month.tolist()]
        hour = np.eye(24)[self.df.hour.tolist()]
            
        try:
            if self.transform:
                if len(image.shape) == 2:  ## validation on number of channels
                    '''To convert grayscale img to rgb '''
                    image = gray2rgb(image)
                    
                image = self.transform(image)
        
        except Exception as e:
            print(f'Exception raised during image transformation due to: {e}')        
        
        sample = {'image': image, 'month': month[idx], 'hour': hour[idx], 'location': location[idx], 'label': label}

        return sample           

           
class Rescale(object):
    '''Not particularly useful in the current problem'''
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        # self.output_size = 747
        

    def __call__(self, sample):
        image, location, month, hour = sample['image'], sample['location'], sample['month'], sample['hour']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        imgage = transform.resize(image, (new_h, new_w))

        ### no landmarks to resize
        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        # landmarks = landmarks * [new_w / w, new_h / h]


        return {'image': image, 'month': month, 'hour': hour, 'location': location}          
           
     
class ToTensor(object):
    '''Not particularly useful in the current problem'''
    """Convert ndarrays in sample to Tensors."""
    
    def __call__(self, sample):
        image, location, month, hour = sample['image'], sample['location'], sample['month'], sample['hour']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'location': torch.from_numpy(location),
                'month': torch.from_numpy(month),
                'hour': torch.from_numpy(hour)
                }



def get_dataset():
        
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToPILImage(mode = 'RGB'),
            transforms.Resize((600, 822)),
            # transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }


    train_dataset = PlaceTimeDataset(csv_file='data/train.csv', root_dir='data/train/', transform=data_transforms['train'])

    # for i_batch, sample_batched in enumerate(train_dataset):
    #     print(i_batch, sample_batched['image'].shape, sample_batched['location'].shape, sample_batched['month'].shape, sample_batched['hour'].shape, sample_batched['label'])
    #     if i_batch >= 1000:
    #         break
        
        
    dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=4)   

    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].shape, sample_batched['location'].shape, sample_batched['month'].shape, sample_batched['hour'].shape, sample_batched['label'])
        if i_batch == 100:
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





if __name__ == '__main__':
    get_dataset()