import torch 
import pandas as pd 
import numpy as np 
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import StandardScaler

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
    continnuous_columns : list 
        List of columns containing continuous variables that are used
    target_column : list/string
        Column that contains the regression/classification target
    regression : bool
        Whether the dataset will be used for regression or classification 
    """
    def __init__(self, df, categorical_columns, continuous_columns, target_column, regression=True, mapper=None):
        
        self.categorical_columns = categorical_columns
        self.continuous_columns = continuous_columns 
        self.target_column = target_column
        self.regression = regression
        
        # Scale continuous variables 
        self.cont_mapper = scale_vars(df, continuous_columns, mapper=mapper)
        df_cat = df[categorical_columns]
        df_cont = df[continuous_columns]
        self.X = np.hstack((df_cat.values,df_cont.values))
        self.y = df[target_column].values 
 
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return [self.X[idx,:], self.y[idx]]

