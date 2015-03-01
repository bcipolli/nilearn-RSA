import matplotlib.pyplot as plt
import numpy as np
import time
from scipy import stats

import nibabel
from sklearn.cluster import WardAgglomeration
from sklearn.externals.joblib import Memory
from sklearn.feature_extraction import image as sk_image

from nilearn import datasets, image
from nilearn.input_data import NiftiMasker
from nilearn.plotting import plot_stat_map, plot_roi


def boo(subject_idx=0, cut_coords=None, n_components=20, n_clusters=2000, memory='nilearn_cache'):

    mem = Memory(cachedir='nilearn_cache')

    # ## Load the data ###################################################

    print("Fetch the data files from Internet")
    haxby_dataset = datasets.fetch_haxby(n_subjects=subject_idx + 1)

    print("Second, load the labels")
    haxby_labels = np.genfromtxt(haxby_dataset.session_target[0],
                                 skip_header=1, usecols=[0],
                                 dtype=basestring)

    # ## Find voxels of interest ##############################################

    print("Load the data.")
    anat_filename = haxby_dataset.anat[subject_idx]
    anat_img = nibabel.load(anat_filename)
    fmri_filename = haxby_dataset.func[subject_idx]
    fmri_raw_img = nibabel.load(fmri_filename)

    print("Build a mask based on the activations.")
    epi_masker = NiftiMasker(mask_strategy='epi', detrend=True, standardize=True)
    epi_masker = mem.cache(epi_masker.fit)(fmri_raw_img)
    plot_roi(epi_masker.mask_img_,
             bg_img=anat_img,
             title='EPI mask',
             cut_coords=cut_coords)

    print("Normalize the (transformed) data")  # zscore per pixel, over examples.
    fmri_masked_vectors = epi_masker.transform(fmri_raw_img)
    fmri_normed_vectors = mem.cache(stats.mstats.zscore)(fmri_masked_vectors, axis=0)
    fmri_normed_img = epi_masker.inverse_transform(fmri_normed_vectors)

    print("Smooth the (spatial) data.")
    fmri_smooth_img = mem.cache(image.smooth_img)(fmri_normed_img, fwhm=1)

    print("Mask the MRI data.")
    masked_fmri_vectors = mem.cache(epi_masker.transform)(fmri_smooth_img)
    fmri_masked_img = epi_masker.inverse_transform(masked_fmri_vectors)

    # ## Compute mean values based on condition matrix ##########################################
    condition_names = list(np.unique(haxby_labels))
    n_conditions = len(condition_names)
    n_good_voxels = masked_fmri_vectors.shape[1]

    mean_vectors = np.empty((n_conditions, n_good_voxels))
    for ci, condition in enumerate(condition_names):
        condition_vectors = masked_fmri_vectors[haxby_labels == condition, :]
        mean_vectors[ci, :] = condition_vectors.mean(axis=0)

    # ## Use similarity across conditions as the 4th dimension ##########################################
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

            p_vector /= p_vector.max()  # normalize
            p_vector = -np.log10(p_vector)
            p_vector[np.isnan(p_vector)] = 0.
            p_vector[p_vector > 10.] = 10.

            p_img = epi_masker.inverse_transform(p_vector)
            comparison_img.append(p_img)
            comparison_text.append('%s vs. %s' % (cond, cond2))
            p_vectors[idx, :] = p_vector
            idx += 1

    #n_comparisons = n_conditions * (n_conditions-1) / 2
    #similarity_vectors = np.empty((n_good_voxels, n_comparisons))
    #for vi in np.arange(n_good_voxels):
    #    similarity_vectors[vi, :] = pdist(mean_vectors[:, vi])



    # Compute a connectivity matrix (for constraining the clustering)
    mask_data = epi_masker.mask_img_.get_data().astype(np.bool)
    connectivity = sk_image.grid_to_graph(n_x=mask_data.shape[0], n_y=mask_data.shape[1],
                                          n_z=mask_data.shape[2], mask=mask_data)

    # Cluster (#2)

    start = time.time()
    ward = WardAgglomeration(n_clusters=n_clusters, connectivity=connectivity, memory=memory)
    ward.fit(p_vectors)

    print("Ward agglomeration %d clusters: %.2fs" % (
        n_clusters, time.time() - start))

    # Compute an image with one ROI per label, and save to disk
    labels = ward.labels_ + 1    # Avoid 0 label - 0 means mask.
    labels_img = epi_masker.inverse_transform(labels)
    labels_img.to_filename('parcellation.nii')

    # Plot image with len(labels) ROIs, and store
    #   the cut coordinates to reuse for all plots
    #   and the figure for plotting all to a common axis
    first_plot = plot_roi(labels_img, title="Ward parcellation", bg_img=anat_img)
    plt.show()


if __name__ == '__main__':

    # Coordinates of the selected slice
    coronal = -24
    sagittal = -33
    axial = -17
    cut_coords = (coronal, sagittal, axial)

    boo(subject_idx=0, n_clusters=300)
    boo(subject_idx=1, n_clusters=300)
