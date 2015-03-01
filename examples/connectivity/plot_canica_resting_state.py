"""
Group analysis of resting-state fMRI with ICA: CanICA
=====================================================

An example applying CanICA to resting-state data. This example applies it
to 40 subjects of the ADHD200 datasets.

CanICA is an ICA method for group-level analysis of fMRI data. Compared
to other strategies, it brings a well-controlled group model, as well as a
thresholding algorithm controlling for specificity and sensitivity with
an explicit model of the signal. The reference papers are:

    * G. Varoquaux et al. "A group model for stable multi-subject ICA on
      fMRI datasets", NeuroImage Vol 51 (2010), p. 288-299

    * G. Varoquaux et al. "ICA-based sparse features recovery from fMRI
      datasets", IEEE ISBI 2010, p. 1177

Pre-prints for both papers are available on hal
(http://hal.archives-ouvertes.fr)
"""
n_components = 42  # Number of CanICA components
n_subjects = 40  # ?

### Load ADHD rest dataset ####################################################
from nilearn import datasets

adhd_dataset = datasets.fetch_adhd(n_subjects)
func_filenames = adhd_dataset.func  # list of 4D nifti files for each subject

### Apply CanICA ##############################################################
import nibabel
from nilearn.decomposition.canica import CanICA

canica = CanICA(n_components=n_components, smoothing_fwhm=6.,
                memory="nilearn_cache", memory_level=5,
                threshold=3., verbose=10, random_state=0,
                max_iter=200)
print("Loading data...")
img_list = func_filenames#[nibabel.load(fn) for fn in func_filenames]
print("Fitting data...")
canica.fit(img_list)

# Retrieve the independent components in brain space
components_img = canica.masker_.inverse_transform(canica.components_)
# components_img is a Nifti Image object, and can be saved to a file with
# the following line:
components_img.to_filename('canica_resting_state.nii.gz')

### Visualize the results #####################################################
# Show some interesting components
import matplotlib.pyplot as plt
import numpy as np
from nilearn.image import index_img, iter_img
from nilearn.plotting import plot_stat_map, find_xyz_cut_coords


fh = plt.figure(facecolor='w', figsize=(18, 10))
nrows = int(np.floor(np.sqrt(0.75 * n_components)))  # 4:3 aspect
ncols = int(np.ceil(n_components / float(nrows)))

all_cut_coords = [find_xyz_cut_coords(img) for img in iter_img(components_img)]
sort_idx = np.argsort(np.array(all_cut_coords)[:, 2])
for ci in range(n_components):
    ax = fh.add_subplot(nrows, ncols, ci + 1)
    ic_idx = sort_idx[ci]
    cut_coords = all_cut_coords[ic_idx]
    var_explained = canica.variance_[ic_idx]
    plot_stat_map(index_img(components_img, ic_idx),
                  title="IC%02d (v=%.1f)" % (ic_idx, var_explained),
                  axes=ax, colorbar=False,
                  display_mode="z", cut_coords=(cut_coords[2],))
plt.show()
