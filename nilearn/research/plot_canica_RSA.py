"""
Group analysis of resting-state fMRI with ICA: CanICA
=====================================================

Similar to plot_canica_resting_state.py, BUT
this code separates into 10 different time-courses for each subject.

It then computes a "group" ICA on the 10 time-courses.  It
then generalizes those across subjects.

I want to test the following:
* sub-slices
* randomly shuffled (but shared!)

I also want to see if cutting at different points for differnet people give
better results (i.e. intrinsic periodicity, like in the ADHD paper that Lourdes
shared with me)

Also, could do group CanICA before time-cut CanICA (if we think there's a
shared timecourse)
"""

n_components = 20  # Number of CanICA components
n_subjects = 6  # ?
max_iter = 200
max_timecourses = 1

import numpy as np
from scipy import stats

import nibabel
from sklearn.externals.joblib import Memory

from nilearn import image
from nilearn.input_data import NiftiMasker
from nilearn.plotting import plot_roi

### Load ADHD rest dataset ####################################################
from nilearn import datasets

print("Fetch the data files from Internet")
haxby_dataset = datasets.fetch_haxby(n_subjects=n_subjects)
func_filenames = haxby_dataset.func

### Apply CanICA to each subject ##############################################################
import nibabel
import numpy as np

mem = Memory(cachedir='nilearn_cache')

all_images = []
for si, func_file in enumerate(func_filenames):

    # ## Load the data ###################################################
    print("Load the data.")
    anat_img = haxby_dataset.anat[si] and nibabel.load(haxby_dataset.anat[si])  # not present for subject 6
    fmri_raw_img = nibabel.load(func_file)
    haxby_labels = np.genfromtxt(haxby_dataset.session_target[si],
                                 skip_header=1, usecols=[0],
                                 dtype=basestring)
    if len(haxby_labels) != fmri_raw_img.shape[-1]:
        print "\n\n\n **** WARNING FOR SUBJECT %d ***** \n\n\n" % si
    if len(np.unique(haxby_labels)) != 9:
        print "\n\n\n **** WARNING FOR SUBJECT %d (2) ***** \n\n\n" % si

    # ## Find voxels of interest ##############################################
    print("Build a mask based on the activations.")
    epi_masker = NiftiMasker(mask_strategy='epi', detrend=True, standardize=True)
    epi_masker = mem.cache(epi_masker.fit)(fmri_raw_img)
    plot_roi(epi_masker.mask_img_,
             bg_img=anat_img,
             title='EPI mask (Subj %d)' % si)

    print("Normalize the (transformed) data")  # zscore per pixel, over examples.
    fmri_masked_vectors = epi_masker.transform(fmri_raw_img)
    fmri_normed_vectors = mem.cache(stats.mstats.zscore)(fmri_masked_vectors, axis=0)
    fmri_normed_img = epi_masker.inverse_transform(fmri_normed_vectors)

    print("Smooth the (spatial) data.")
    fmri_smooth_img = mem.cache(image.smooth_img)(fmri_normed_img, fwhm=6)

    print("Mask the MRI data.")
    masked_fmri_vectors = mem.cache(epi_masker.transform)(fmri_smooth_img)
    fmri_masked_img = epi_masker.inverse_transform(masked_fmri_vectors)

    # ## Use similarity across conditions as the 4th dimension ##########################################
    print("Compute similarity via ttest.")
    condition_names = list(np.unique(haxby_labels))
    n_conds = len(condition_names)
    n_compares = n_conds * (n_conds - 1) / 2

    p_vectors = np.zeros((n_compares, masked_fmri_vectors.shape[1]))
    comparison_text = []
    comparison_img = []
    idx = 0
    for i, cond in enumerate(condition_names):
        for j, cond2 in enumerate(condition_names[i+1:]):
            print("Computing ttest for %s vs. %s." % (cond, cond2))
            _, p_vector = stats.ttest_ind(
                masked_fmri_vectors[haxby_labels == cond, :],
                masked_fmri_vectors[haxby_labels == cond2, :],
                axis=0)

            # Normalize and log-transform
            p_vector /= p_vector.max()  # normalize
            p_vector = -np.log10(p_vector)
            p_vector[np.isnan(p_vector)] = 0.
            p_vector[p_vector > 10.] = 10.

            p_img = epi_masker.inverse_transform(p_vector)
            comparison_img.append(p_img)
            comparison_text.append('%s vs. %s' % (cond, cond2))
            p_vectors[idx, :] = p_vector
            idx += 1

    # ## Convert similarities into a single subject image (like a time-course) ################
    p_img = epi_masker.inverse_transform(p_vectors)
    all_images.append(p_img)

### Apply CanICA across subjects ##############################################################
from nilearn.decomposition.canica import CanICA

print("Fitting data over all subjects (%d images)..." % len(all_images))
canica = CanICA(n_components=n_components, smoothing_fwhm=6.,
                memory="nilearn_cache", memory_level=5,
                max_iter=max_iter,
                threshold=3., verbose=10, random_state=0)
canica.fit(all_images)
components_img = canica.masker_.inverse_transform(canica.components_)
nibabel.save(components_img, 'components_img_RSA.nii')

### Visualize the results #####################################################
# Show some interesting components
import matplotlib.pyplot as plt
from nilearn.plotting import plot_stat_map, find_xyz_cut_coords
from nilearn.image import index_img, iter_img

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
