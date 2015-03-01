import matplotlib.pyplot as plt
import numpy as np

import nibabel
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.externals.joblib import Memory

from nilearn import datasets, image
from nilearn.input_data import NiftiMasker
from nilearn.plotting import plot_stat_map, plot_roi, plot_glass_brain


def plot_two_maps(plot_fn, img1, img2, **kwargs):
    fig_id = plt.subplot(2, 1, 1)
    plot_fn(img1[0], title=img1[1], axes=fig_id, **kwargs)
    fig_id = plt.subplot(2, 1, 2)
    plot_fn(img2[0], title=img2[1], axes=fig_id, **kwargs)


def boo(subject_idx=0, cut_coords=None):

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
    shared_affine = fmri_raw_img.get_affine()

    print("Build a mask based on the activations.")
    epi_masker = NiftiMasker(mask_strategy='epi', detrend=True, standardize=True)
    epi_masker = mem.cache(epi_masker.fit)(fmri_raw_img)
    plot_roi(epi_masker.mask_img_,
             title='EPI mask',
             cut_coords=cut_coords)

    from nipy.labs.viz import plot_map
#    plot_map(epi_masker.mask_img_.get_data(), epi_masker.mask_img_.get_affine())
#    plt.show()
#    exit()

    #print("Normalize the (transformed) data")  # zscore per pixel, over examples.
    #fmri_masked_vectors = epi_masker.transform(fmri_raw_img)
    #fmri_normed_vectors = mem.cache(stats.mstats.zscore)(fmri_masked_vectors, axis=0)
    fmri_normed_img = fmri_raw_img #epi_masker.inverse_transform(fmri_normed_vectors)

    print("Smooth the (spatial) data.")
    fmri_smooth_img = mem.cache(image.smooth_img)(fmri_normed_img, fwhm=1)

    print("Mask the MRI data.")
    masked_fmri_vectors = mem.cache(epi_masker.transform)(fmri_smooth_img)
    fmri_masked_img = epi_masker.inverse_transform(masked_fmri_vectors)

    # ## Compute a similarity matrix ##########################################

    condition_names = list(np.unique(haxby_labels))
    n_cond_img = (haxby_labels == condition_names[0]).sum()
    n_conds = len(condition_names)
    n_compares = n_conds * (n_conds - 1) / 2
    p_vectors = np.zeros((n_compares, masked_fmri_vectors.shape[1]))

    idx = 0
    for i, cond in enumerate(condition_names):
        for j, cond2 in enumerate(condition_names[i+1:]):
            print("Computing ttest for %s vs. %s." % (cond, cond2))
            _, p_vectors[idx, :] = stats.ttest_ind(
                masked_fmri_vectors[haxby_labels == cond, :],
                masked_fmri_vectors[haxby_labels == cond2, :],
                axis=0)
            idx += 1

    p_vectors_normd = p_vectors / p_vectors.max(axis=0)
    log_p_vectors = -np.log10(p_vectors)
    log_p_vectors[np.isnan(log_p_vectors)] = 0.
    log_p_vectors[log_p_vectors > 10.] = 10.
    #log_p_normd_vectors = log_p_vectors / log_p_vectors.sum(axis=0)
    plt.figure(); plt.hist(p_vectors_normd.max(axis=0), 100); plt.show()

    idx = 0
    for i, cond in enumerate(condition_names):
        for j, cond2 in enumerate(condition_names[i+1:]):
            if cond != 'face' and cond2 != 'face': continue

            print("Plotting compares for %s vs. %s." % (cond, cond2))
            log_p_img = epi_masker.inverse_transform(1/p_vectors[idx, :])
            log_p_normd_img = epi_masker.inverse_transform(1. - p_vectors_normd[idx, :])
            plot_two_maps(plot_stat_map,
                          (log_p_img, "%s vs. %s." % (cond, cond2)),
                          (log_p_normd_img, "%s vs. %s. (norm'd)"
                              % (cond, cond2)), bg_img=anat_img)
            import pdb; pdb.set_trace()
            plt.show()
            idx += 1

if __name__ == '__main__':

    # Coordinates of the selected slice
    coronal = -24
    sagittal = -33
    axial = -17
    cut_coords = (coronal, sagittal, axial)

    boo(subject_idx=0, cut_coords=cut_coords)
