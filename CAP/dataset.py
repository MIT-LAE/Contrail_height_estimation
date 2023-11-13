import torch 
import pandas as pd 
import numpy as np 
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import StandardScaler

import torchvision.transforms.functional as TF 
import torchvision.transforms as transforms

import random

AVAILABLE_CHANNELS = ["CMI_C07",
    "CMI_C08",
    "CMI_C09",
    "CMI_C10",
    "CMI_C11",
    "CMI_C12",
    "CMI_C13",
    "CMI_C14",
    "CMI_C15",
    "CMI_C16",
    "longitude",
    "latitude",
    "T_surf",
    "lsm",
    "VZA",
    "cos_doy",
    "sin_doy"
]

def scale_vars(df, columns, mapper=None):
    """
    Scales specified columns in a pandas DataFrame
    """
    if mapper is None:
        map_f = [([n], StandardScaler()) for n in columns]
        mapper = DataFrameMapper(map_f).fit(df[columns])
    df[mapper.transformed_names_] = mapper.transform(df)
    return mapper

class TabularDataset(torch.utils.data.Dataset):
    """
    Class that represents a dataset based on a pandas DataFrame
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame holding the data
    categorical_columns : list
        List of columns containing categorical variables that are used
    continuous_columns : list 
        List of columns containing continuous variables that are used
    target_column : list/string
        Column that contains the regression/classification target
    regression : bool
        Whether the dataset will be used for regression or classification 
    """
    def __init__(self, df, categorical_columns, continuous_columns,
                target_column, regression=True, mapper=None, scale_target=False,
                temperature_noise=None):
        
        self.categorical_columns = categorical_columns
        self.continuous_columns = continuous_columns 
        self.target_column = target_column
        self.regression = regression
        
        # Scale continuous variables 
        if scale_target:
            self.cont_mapper = scale_vars(df, continuous_columns + target_column, mapper=mapper)
        else:
            self.cont_mapper = scale_vars(df, continuous_columns, mapper=mapper)
        df_cat = df[categorical_columns]
        df_cont = df[continuous_columns]
        self.X = np.hstack((df_cat.values, df_cont.values))
        self.y = df[target_column].values 
 
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return [self.X[idx,:], self.y[idx]]


class ImageDataset(torch.utils.data.Dataset):
    """
    Class that represents a dataset based on images and labels stored
    as numpy arrays
    
    Parameters
    ----------
    image_files : List[str]
        List of the input numpy files
    label_files : list[str]
        List of the label numpy files
    means : np.array
        Mean for the channels used in the images
    std_devs: np.array
        Std devs for the channels used in the images
    augmentation : bool (optional)
        Whether or not to augment the data using flips
    overfit : bool (optional)
        Whether or not to use a small subset of the data to overfit
    """
    def __init__(self, image_files, label_files, means, std_devs, channels=None,
                augmentation=False, overfit=False, use_mask=False, cirrus=True,
                temperature_noise=None, gfs=True, random_cropping=False):
        self.means = means
        self.std_devs = std_devs
        self.augmentation = augmentation
        self.cirrus = cirrus
        self.use_mask = use_mask
        self.temperature_noise = temperature_noise
        self.gfs = gfs
        self.random_cropping = random_cropping
        
        if channels is not None:
            self.channel_indices = np.array([np.where(np.array(AVAILABLE_CHANNELS) == ch)[0][0] for ch in channels])
        else:
            self.channel_indices = np.arange(len(AVAILABLE_CHANNELS)).astype(np.int64)

        if self.use_mask:
            self.means = np.hstack((means[self.channel_indices], np.zeros(1)))
            self.std_devs = np.hstack((std_devs[self.channel_indices], np.ones(1)))

        else:
            self.means = means[self.channel_indices]
            self.std_devs = std_devs[self.channel_indices]

        self.normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=self.means, std=self.std_devs)])

        self.input_list = image_files
        self.label_list = label_files

        if overfit:
            self.input_list = self.input_list[:10]
            self.label_list = self.label_list[:10]
        
        print(f"Initialized dataset with {len(self)} entries")
    
    def __len__(self):
        return len(self.input_list)

    def transform(self, image, mask):
        """
        This function performs random horizontal and 
        vertical flipping. The motivation for doing this here
        is that current out-of-the-box PyTorch methods do not work 
        for our images (since we use 3 channels but not RGB).
        """ 

        # TODO: Check if new version of PyTorch allows us to do this 
        
        to_PIL = transforms.ToPILImage()
        channels = [torch.Tensor(image[:,:,i].astype(np.float64)) for i in range(image.shape[2])]
        label = torch.Tensor(mask.astype(np.float64))

        # Random horizontal flipping
        if random.random() > 0.5:
            for i in range(len(channels)):
                channels[i] = TF.hflip(channels[i])
            label = TF.hflip(label)

            
        # Random vertical flipping
        if random.random() > 0.5:
            for i in range(len(channels)):
                channels[i] = TF.vflip(channels[i])
            label = TF.vflip(label)
       
        image = np.dstack([np.array(c) for c in channels])
        
        return image, label.numpy()
    
    def __getitem__(self, idx):

        inp_path = self.input_list[idx]
        
        inputs = np.load(inp_path).transpose((1, 2, 0))[:,:,self.channel_indices[self.channel_indices<10]]
        label = np.load(inp_path.replace("images", "labels"))
        aux = np.load(inp_path.replace("images", "aux"))[self.channel_indices[self.channel_indices>=10] - 10,:,:]
        
        if self.gfs:
            aux[1,:,:] = np.load(inp_path.replace("images", "gfs_tiles"))
                                                

        inputs = np.dstack((inputs, aux.transpose((1,2,0))))

        # Add noise to T_surf
        if self.temperature_noise is not None:
            T_surf = inputs[:,:,-5]

            T_surf += np.random.normal(0, self.temperature_noise)
            inputs[:,:,-5] = T_surf

        if self.use_mask:
            #if not self.cirrus:
            try:
                mask = np.load(inp_path.replace("images", "masks"))
            except FileNotFoundError:
                mask = np.zeros_like(label, dtype=inputs.dtype)
            # else:
            #     mask = np.zeros_like(label, dtype=inputs.dtype)
            inputs = np.dstack((inputs, mask[:,:,np.newaxis]))

        if not self.cirrus:
            if self.random_cropping:
                start_row = np.random.randint(0, 32)
                start_col = np.random.randint(0, 32)
            else:
                start_row = 16
                start_col = 16
            
            inputs = inputs[start_row:start_row+32, start_col:start_col+32,:]
            label = label[start_row:start_row+32, start_col:start_col+32]

        if self.augmentation:
            inputs, label = self.transform(inputs, label)

        to_tensor = transforms.ToTensor()
        return self.normalize(inputs).float(), to_tensor(label).float()