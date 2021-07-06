import numpy as np
import keras
from functions_GN_input import make_input, standarization, concat_bulk_surf

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, bulk_list_IDs, surf_list_IDs, E_mean, E_std, batch_size=32, shuffle=False):
        'Initialization'
        self.bulk_list_IDs = bulk_list_IDs
        self.surf_list_IDs = surf_list_IDs
        self.E_mean = E_mean
        self.E_std = E_std
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.bulk_list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        bulk_list_IDs_temp = [self.bulk_list_IDs[k] for k in indexes]
        surf_list_IDs_temp = [self.surf_list_IDs[k] for k in indexes]

        # Generate data
        bulk_inputs, bulk_E = make_input(bulk_list_IDs_temp)
        surf_inputs, surf_E = make_input(surf_list_IDs_temp)
        X_batch, y_batch = concat_bulk_surf(bulk_inputs, bulk_E, surf_inputs, surf_E, self.batch_size, self.E_mean, self.E_std)

        return X_batch, y_batch

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.bulk_list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
