"""
Computing an ROI mask
=======================

Example showing how a T-test can be performed to compute an ROI
mask, and how simple operations can improve the quality of the mask
obtained.
"""

# ## Prep ###################################################

# for manipulating images
import nibabel
from nilearn.input_data import NiftiMasker

# for visualization
import matplotlib.pyplot as plt
from nilearn.plotting import plot_stat_map, plot_roi

# for caching
from sklearn.externals.joblib import Memory
mem = Memory(cachedir='nilearn_cache')

# Coordinates of the selected slice
coronal = -24
sagittal = -33
axial = -17
cut_coords = (coronal, sagittal, axial)

subject_idx = 1

# ## Load the data ###################################################

print("Fetch the data files from Internet")
from nilearn import datasets
haxby_dataset = datasets.fetch_haxby(n_subjects=subject_idx + 1)

print("Second, load the labels")
import numpy as np
haxby_labels = np.genfromtxt(haxby_dataset.session_target[0],
                             skip_header=1, usecols=[0],
                             dtype=basestring)

print np.unique(haxby_labels)

# ## Find voxels of interest #################################################

print("Load the data.")
anat_filename = haxby_dataset.anat[subject_idx]
anat_img = nibabel.load(anat_filename)
fmri_filename = haxby_dataset.func[subject_idx]
fmri_raw_img = nibabel.load(fmri_filename)
shared_affine = fmri_raw_img.get_affine()

print("Build a mask based on the activations.")
epi_masker = NiftiMasker(mask_strategy='epi', detrend=True, standardize=True)
epi_masker = mem.cache(epi_masker.fit)(fmri_raw_img)
plot_roi(epi_masker.mask_img_,
         title='EPI mask',
         cut_coords=cut_coords)

print("Normalize the (transformed) data")  # zscore per pixel, over examples.
from scipy import stats
fmri_masked_vectors = epi_masker.transform(fmri_raw_img)
import pdb; pdb.set_trace()
fmri_normed_vectors = mem.cache(stats.mstats.zscore)(fmri_masked_vectors, axis=0)
fmri_normed_img = epi_masker.inverse_transform(fmri_normed_vectors)

print("Smooth the (spatial) data.")
from nilearn import image
fmri_smooth_img = mem.cache(image.smooth_img)(fmri_normed_img, fwhm=1)

print("Mask the MRI data.")
masked_fmri_vectors = mem.cache(epi_masker.transform)(fmri_smooth_img)
fmri_masked_img = epi_masker.inverse_transform(masked_fmri_vectors)

print("Plot the mean image.")
fig_id = plt.subplot(2, 1, 1)
mean_img = mem.cache(image.mean_img)(fmri_masked_img)
plot_stat_map(mean_img, anat_img, title='Smoothed mean EPI', cut_coords=cut_coords, axes=fig_id)

print("Run a T-test for face and houses.")
_, p_vector = stats.ttest_ind(masked_fmri_vectors[haxby_labels == 'face', :],
                              masked_fmri_vectors[haxby_labels == 'house', :],
                              axis=0)

print("Use a log scale for p-values.")
log_p_vector = -np.log10(p_vector)
log_p_vector[np.isnan(log_p_vector)] = 0.
log_p_vector[log_p_vector > 10.] = 10.
fig_id = plt.subplot(2, 1, 2)
plot_stat_map(epi_masker.inverse_transform(log_p_vector), anat_img,
              title="p-values", cut_coords=cut_coords, axes=fig_id)
plt.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0)


plt.figure()
fig_id = plt.subplot(3, 1, 1)

print("Thresholding @ p <= 1E-5.")
log_p_vector[log_p_vector < 4] = 0
plot_stat_map(epi_masker.inverse_transform(log_p_vector), anat_img,
              title='Thresholded p-values', annotate=False, colorbar=False,
              cut_coords=cut_coords, axes=fig_id)

print("Binarization and intersection with VT mask.")
# (intersection corresponds to an "AND conjunction")
bin_p_vector = (log_p_vector != 0)
mask_vt_filename = haxby_dataset.mask_vt[subject_idx]
vt_image = nibabel.load(mask_vt_filename)
vt_vector = epi_masker.transform(vt_image, detrend=False,
                                 standardize=False)[0].astype(bool)
assert vt_vector.sum() > 0.
bin_p_and_vt_vector = np.logical_and(bin_p_vector, vt_vector)

fig_id = plt.subplot(3, 1, 2)
plot_roi(epi_masker.inverse_transform(bin_p_and_vt_vector.astype(np.int)),
         anat_img, title='Intersection with ventral temporal mask',
         cut_coords=cut_coords, axes=fig_id)

print("Dilation.")
fig_id = plt.subplot(3, 1, 3)
from scipy import ndimage
bin_p_and_vt_img = epi_masker.inverse_transform(bin_p_and_vt_vector.astype(int))
bin_p_and_vt_volume = bin_p_and_vt_img.get_data().astype(bool)
dil_bin_p_and_vt_volume = bin_p_and_vt_volume#mem.cache(ndimage.binary_dilation)(bin_p_and_vt_volume)
dil_bin_p_and_vt_img = nibabel.Nifti1Image(dil_bin_p_and_vt_volume.astype(int),
                                           shared_affine)
plot_roi(dil_bin_p_and_vt_img, anat_img,
         title='Dilated mask', cut_coords=cut_coords,
         axes=fig_id, annotate=False)
plt.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0)

print("Identification of connected components.")
plt.figure()
labels_volume, n_labels = mem.cache(ndimage.label)(dil_bin_p_and_vt_volume)
assert n_labels >= 2
nvoxels_per_label = np.asarray([(labels_volume == label).sum()
                                for label in np.unique(labels_volume)])
top_labels = np.argsort(nvoxels_per_label)
first_roi_img = nibabel.Nifti1Image(     # -1 will always be 0 (background)
    (labels_volume == top_labels[-2]).astype(np.int),
    shared_affine)
second_roi_img = nibabel.Nifti1Image(
    (labels_volume == top_labels[-3]).astype(np.int),
    shared_affine)
fig_id = plt.subplot(2, 1, 1)
plot_roi(first_roi_img,
         anat_img, title='Connected components: first ROI', axes=fig_id)
fig_id = plt.subplot(2, 1, 2)
plot_roi(second_roi_img,
         anat_img, title='Connected components: second ROI', axes=fig_id)
plt.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0)
plot_roi(first_roi_img,
         anat_img, title='Connected components: first ROI_',
         output_file='snapshot_first_ROI.png')
plot_roi(second_roi_img,
         anat_img, title='Connected components: second ROI',
         output_file='snapshot_second_ROI.png')

plt.show()
