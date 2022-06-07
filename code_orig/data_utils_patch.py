###
### This file contains the Dataset-class, which can be used to load the 
### MatLab data files into Python and perform pre-processing steps.
### Last updated: 2022/05/04 9:15 AM 
###

# Import libraries
import numpy as np
import h5py
from scipy.io import loadmat


### Define the Dataset class
class PatchedDataset:
    def __init__(self, filenames, single_variable_name="", ref_variable_name="", patch_size=None,  
                    normalize=True, svd_denoise=[], flippedaxes=False):
        # Set variables in `self`

        self.filenames         = filenames
        if type(self.filenames) == str: 
            self.filenames = [self.filenames]
        
        self.single_var_name  = single_variable_name
        self.ref_var_name     = ref_variable_name
        self.patch_size       = patch_size
        self.normalize        = normalize
        self.svd_denoise      = svd_denoise  # should contain 'ref' and/or 'single'
        self.model_results    = None # reserve variable for later
        self.flippedaxes      = flippedaxes

        # Load data and set data shape
        self.patches, self.patches_ref = self.load_data()
        
        # Add channel axis to be able to run this model
        self.patches     = self.patches[..., np.newaxis]
        self.patches_ref = self.patches_ref[..., np.newaxis]


    def load_data(self):
        """
        Load the data from the .mat file and do some processing steps.
        Still to be updated!
        """
        # Load the data from the .mat file
        data, data_ref = self.load_mats_as_np()
        
        # Normalize data (Gaussian normalization per frame)
        if self.normalize:
            xmin = np.min(data, axis=(1,2))[..., np.newaxis, np.newaxis]
            xmax = np.max(data, axis=(1,2))[..., np.newaxis, np.newaxis]
            data = (data-xmin) / (xmax-xmin)

            xmin = np.min(data_ref, axis=(1,2))[..., np.newaxis, np.newaxis]
            xmax = np.max(data_ref, axis=(1,2))[..., np.newaxis, np.newaxis]
            data_ref = (data_ref-xmin) / (xmax-xmin)
        
        if 'ref' in self.svd_denoise:
            data_ref = np.array([self.svd_denoise_image(img) for img in data_ref])
        if 'single' in self.svd_denoise:
            data     = np.array([self.svd_denoise_image(img) for img in data])

        return data, data_ref


    def load_mats_as_np(self):
        """
        Reads a .mat file and writes it to a NumPy Array.
        The .mat file should have the following structure:
        - < update when data is available >
        - < >
        """
        
        for i, filename in enumerate(self.filenames):
            try: # In case it is an hdf5-based .mat file
                f        = h5py.File(filename, 'r')
                data     = f.get(self.single_var_name)
                data_ref = f.get(self.ref_var_name)
                # Convert to NumPy array
                data     = np.array(data)
                data_ref = np.array(data_ref)
                if self.flippedaxes:
                    data = np.transpose(data, axes=[1,0,2])   # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<< check whether axes are still correct; 
                                                        # make sure that the format becomes (frame, element, timepoint)
                    data_ref = np.transpose(data_ref, axes=[1,0,2])
                else:
                    data = np.transpose(data, axes=[1,2,0])   # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<< check whether axes are still correct; 
                                                        # make sure that the format becomes (frame, element, timepoint)
                    data_ref = np.transpose(data_ref, axes=[1,2,0])
                

            except OSError: # In case it is a MatLab v7 based file
                mat      = loadmat(filename)
                data     = np.array(mat[self.single_var_name])
                data_ref = np.array(mat[self.ref_var_name])
                
                if self.flippedaxes:
                    data = np.transpose(data, axes=[1,0,2])   # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<< check whether axes are still correct; 
                                                        # make sure that the format becomes (frame, element, timepoint)
                    data_ref = np.transpose(data_ref, axes=[1,0,2])
                else:
                    data = np.transpose(data, axes=[1,2,0])   # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<< check whether axes are still correct; 
                                                        # make sure that the format becomes (frame, element, timepoint)
                    data_ref = np.transpose(data_ref, axes=[1,2,0])

            if i==0:
                patched     = data
                patched_ref = data_ref
            else:
                patched     = np.concatenate((patched, data), axis=0)
                patched_ref = np.concatenate((patched_ref, data_ref), axis=0)

        self.data_size = (patched.shape[1], patched.shape[2])
        assert (self.data_size == self.patch_size, f"Incorrect data shape! Was {self.data_size}, expected {self.patch_size}")
        return patched, patched_ref


    def auto_patch_detect_weights(self):
        # Set empty array
        classes = np.zeros(self.patches.shape[0])

        for i in range(self.data.shape[0]): #loop over every FULL image
            # Get the full image in patches
            patch_batch_ref = self.patches_ref[i*self.patches_per_img:(i+1)*self.patches_per_img]
            
            # Select the pixels/patches with the 15 highest values
            indxs_max = np.array(np.where(patch_batch_ref>=0.9*patch_batch_ref.max()))
            signal_patches = indxs_max[0]
            # Select most occuring patch
            signal_patch = np.bincount(signal_patches).argmax()
            
            # Set class label
            classes[i*self.patches_per_img + signal_patch] = 1

            # Get closest patch in z-direction in case it is in the lower third of the patch
            zs = indxs_max[:, indxs_max[0,:] == signal_patch][2,:]
            for z in zs:
                if (z > 2*self.patch_size[0]//3) and (signal_patch < self.patches_per_img-1):
                    classes[i*self.patches_per_img + signal_patch] = 1
        return classes

    def svd_denoise_image(self, img, k=15):
        # Calculate U (u), Σ (s) and V (vh)
        u, s, vh = np.linalg.svd(img, full_matrices=False)
        # Remove all but the k highest sigma values 
        ind = np.argpartition(s, -k)[-k:] # k highest values for sigma
        s_cleaned = np.zeros(s.shape)
        s_cleaned[ind] = s[ind]
        #s_cleaned = np.array([si if si > 250000 else 0 for si in s])
        # Calculate A' = U * Σ (cleaned) * V
        img_denoised = np.array(np.dot(u * s_cleaned, vh))
        return img_denoised


    def data_augmentation(self, brightness=False, extract_from_1000=False, ef1000_new_t=896, flip=False):
        """
        Perform data augmentation on the data.
        Options:
        - Extract_from_1000: Extract multiple images from 1000 time points, by sampling by (default) 896 time points.
        """
        if extract_from_1000:
            nr_of_new_imgs = 1000-ef1000_new_t
            data = np.zeros((self.data.shape[0]*nr_of_new_imgs, self.data.shape[1], ef1000_new_t))
            data_ref = np.zeros((self.data_ref.shape[0]*nr_of_new_imgs, self.data_ref.shape[1], ef1000_new_t))
            for i in range(nr_of_new_imgs):
                data[i*self.data.shape[0]:(i+1)*self.data.shape[0], :, :] = self.data[:, :, i:i+ef1000_new_t, 0]
                data_ref[i*self.data_ref.shape[0]:(i+1)*self.data_ref.shape[0], :, :] = self.data_ref[:, :, i:i+ef1000_new_t, 0]
            self.data = data
        if flip:
            pass

