"""
Independent component analysis of resting-state fMRI
=====================================================

An example applying ICA to resting-state data.
"""
import numpy as np


### Load nyu_rest dataset #####################################################
from nilearn import datasets
# Here we use only 3 subjects to get faster-running code. For better
# results, simply increase this number
# XXX: must get the code to run for more than 1 subject
nyu_dataset = datasets.fetch_nyu_rest(n_subjects=3)
func_filenames = nyu_dataset.func

### Preprocess ################################################################
from nilearn.input_data import NiftiMasker
mem = 'nilearn_cache'

# This is resting-state data: the background has not been removed yet,
# thus we need to use mask_strategy='epi' to compute the mask from the
# EPI images
masker = NiftiMasker(smoothing_fwhm=8, memory_level=1,
                     mask_strategy='epi', standardize=False,
                     memory=mem, verbose=10)
masker.fit(func_filenames[0])
data_masked = [masker.transform(fn) for fn in func_filenames]
img_masked = [masker.inverse_transform(d) for d in data_masked]


### Apply ICA #################################################################

from nilearn.decomposition import CanICA
n_components = 20
ica = CanICA(n_components=n_components, random_state=42, verbose=10, memory=mem)
components_masked = ica.fit_transform(img_masked)

# Normalize estimated components, for thresholding to make sense
from scipy.stats.mstats import zscore
components_masked = zscore(components_masked)

# Threshold
components_masked[components_masked < .8] = 0

# Now invert the masking operation, going back to a full 3D
# representation
component_img = masker.inverse_transform(components_masked)

### Visualize the results #####################################################
# Show some interesting components
import pylab as plt
from nilearn import image
from nilearn.plotting import plot_stat_map

# Use the mean as a background
mean_img = image.mean_img(func_filename)

plot_stat_map(image.index_img(component_img, 5), mean_img)

plot_stat_map(image.index_img(component_img, 12), mean_img)

plt.show()
