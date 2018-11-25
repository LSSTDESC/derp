import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Dataset

class DerpData(Dataset):
    """Derp dataset."""

    def __init__(self, X_filepath, y_filepath, ignore_null_rows=True, is_train=True):
        """
        Parameters
        ----------
        X_filepath : string
            Path to csv file containing the input X
        y_filepath : string
            Path to csv file containing the label y
        ignore_null_rows : Bool
            Whether rows with null values will be ignored
        is_train : Bool
            Whether to load train or test data
        """
        self.is_train = is_train
        self.is_test = not is_train
        self.mask_val = 9999
        self.ignore_null_rows = ignore_null_rows
        # TODO: eventually, central ra, dec and radius will be for the LSST sky area
        self.central_ra, self.central_dec = 55.8, -28.8
        self.radius = 0.2 # degrees

        self.X = pd.read_csv(X_filepath)
        self.y = pd.read_csv(y_filepath)
        
        if self.ignore_null_rows:
            self.delete_null_rows()
        else:
            self.mask_null_values()
        
        # Engineer X features
        self.engineer_X(normalize_magnitudes=False)
        
        # Engineer y features
        self.engineer_y(binary_features=['extendedness'])
        
        # Save processed data to disk
        save_path_X = os.path.join('data', 'processed_x.csv')
        save_path_y = os.path.join('data', 'processed_y.csv')
        self.X.to_csv(save_path_X, index=False)
        self.y.to_csv(save_path_y, index=False)
        self.X_columns = list(self.X.columns)
        self.y_reg_columns = list(self.y.columns)
        self.y_binary_columns = list(self.y_binary.columns)
        self.y_columns = self.y_reg_columns + self.y_binary_columns
        
        # Mask for setting some loss values to zero
        # Note: Some versions of torch require int64 rather than int32
        self.X_mask = self.X.isnull().values.astype(np.int64) 
        self.y_mask = self.y.isnull().values.astype(np.int64)
        self.y_binary_mask = self.y_binary.isnull().values.astype(np.int64)

        # Convert into numpy array
        self.X = self.X.values.astype(np.float32)
        self.y = self.y.values.astype(np.float32) 
        self.y_binary = self.y_binary.values.astype(np.int64)
        
        # Save metadata: number of examples, input dim, output dim
        self.num_total, self.X_dim = self.X.shape
        _, self.y_dim_regression = self.y.shape
        _, self.y_dim_binary = self.y_binary.shape
        self.y_dim = self.y_dim_regression + self.y_dim_binary
        
        # Split train and test
        self.num_train = int(0.75*self.num_total)
        for sub in ['X', 'X_mask', 'y', 'y_binary', 'y_mask', 'y_binary_mask',]: 
            if self.is_train:
                original = getattr(self, sub)[:self.num_train, :]
                setattr(self, sub, original)
            else:
                original = getattr(self, sub)[self.num_train:, :]
                setattr(self, sub, original)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        sample_X = self.X[idx, :]
        sample_y  = self.y[idx, :]
        sample_y_binary  = self.y_binary[idx, :]
        mask_X = self.X_mask[idx, :]
        mask_y = self.y_mask[idx, :]
        mask_y_binary = self.y_binary_mask[idx, :]
        #return_tuple = tuple([torch.Tensor(v) for v in [sample_X, sample_y, mask_X, mask_y]])
        return sample_X, sample_y, sample_y_binary, mask_X, mask_y, mask_y_binary
    
    def engineer_X(self, normalize_magnitudes):
        """Engineer features in input X for easy optimization
        
        Parameters
        ----------
        normalize_magnitudes : Bool
            Whether to feature-normalize the 6 magnitudes
        
        """
        # Normalize magnitudes (feature normalization)
        if normalize_magnitudes:
            mags = self.X[['u', 'g', 'r', 'i', 'z', 'y']].values
            #print(mags.shape)
            mean_row = np.mean(mags, axis=1)
            std_row = np.std(mags, axis=1)
            #print(mean_row.shape)
            for bp in 'ugrizy':
                self.X[bp] = (self.X[bp] - mean_row)/std_row    
        
        # Negate star feature to approximate "extendedness"
        self.X['not_star'] = 1.0 - self.X['star']
        
        # Normalize redshift
        self.X['redshift'] = (self.X['redshift'] - 0.5)/2.0
        
        # Add normalized ra, dec
        self.X['ra_diff'] = (self.X['ra'].values - self.central_ra)/self.radius
        self.X['dec_diff'] = (self.X['dec'].values - self.central_dec)/self.radius
        
        # Drop irrelevant or duplicate columns
        self.X.drop(['healpix_2048', 'object_id', 'star'], axis=1, inplace=True)
    
    def engineer_y(self, binary_features=[]):
        """Engineer the target label y for easy optimization
        
        Parameters
        ----------
        binary_features : list
            list of binary features
            
        """
        # Set aside binary extendedness feature
        # TODO: split not necessary once continuous values before thresholding are obtained
        if len(binary_features)==0:
            self.y_binary = None    
        else:
            self.y_binary = self.y[binary_features]
        
        # Define target ra, dec as the difference between observed and true times scalar
        #self.y['ra'] = 3600.0*24*(self.y['ra'].values - self.X['ra'].values)
        #self.y['dec'] = 3600.0*24*(self.y['dec'].values - self.X['dec'].values)
        self.y['ra'] = (self.y['ra'].values - self.central_ra)/self.radius
        self.y['dec'] = (self.y['dec'].values - self.central_dec)/self.radius
        
        # Drop irrelevant or duplicate columns
        y_cols_to_delete = ['objectId', 'ext_shapeHSM_HsmShapeRegauss_e1', 'ext_shapeHSM_HsmShapeRegauss_e2', 'ext_shapeHSM_HsmShapeRegauss_sigma'] 
        y_cols_to_delete += binary_features
        y_cols_to_delete += [col for col in self.y.columns if 'fracDev' in col]
        y_cols_to_delete += [col for col in self.y.columns if 'flux' in col]
        self.y.drop(y_cols_to_delete, axis=1, inplace=True)
    
    def delete_null_rows(self):
        """Deletes rows with any null value
        Note
        ----
        This method assumes self.X has no null value.
        
        """
        y_nonull_idx = self.y.dropna(axis=0, how='any').index
        self.X = self.X[self.X.index.isin(y_nonull_idx)]
        self.y.dropna(axis=0, how='any', inplace=True)
    
    def mask_null_values(self):
        """Replaces null values with a token, self.mask_val
        
        """
        self.X.fillna(self.mask_val, inplace=True)
        self.y.fillna(self.mask_val, inplace=True)
    
    def normalize(self, val):
        means = np.mean(val, axis=0, keepdims=True)
        std = np.std(val, axis=0, keepdims=True)
        print(val)
        return (val - means) / std

if __name__ == "__main__":
    from torch.utils.data import DataLoader

    X_path = os.path.join('data', 'X.csv')
    y_path = os.path.join('data', 'y.csv')
    
    # Test constructor
    train_data = DerpData(X_filepath=X_path, y_filepath=y_path, is_train=True)
    test_data = DerpData(X_filepath=X_path, y_filepath=y_path, is_train=False)
    
    # Test __getitem__
    X, y, y_bin, mask_X, mask_y, mask_y_bin = train_data[0]
    print("X shape of train data: ", train_data.X.shape)
    print("X shape of test data: ", test_data.X.shape)
    print("X columns: ", train_data.X_columns)
    print("Y columns: ", train_data.y_columns)
    
    # Test loader instantiated with DerpData instance
    train_loader = DataLoader(train_data, batch_size=7, shuffle=False)
    for batch_idx, (X, y, y_bin, mask_X, mask_y, mask_y_bin) in enumerate(train_loader):
        pass
