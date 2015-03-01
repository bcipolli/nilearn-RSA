import contextlib
import cPickle
import os
import matplotlib.pyplot as plt

import nibabel
from sklearn.externals.joblib import Memory

from nilearn import image
from nilearn.input_data import NiftiMasker, NiftiLabelsMasker
from nilearn.plotting import plot_stat_map, plot_roi

saved_filepath = 'output/a.pkl'


if not os.path.exists(saved_filepath):
    # Run and save.
    from plot_roi_extraction import *

    print("Saving data to disk.")
    nibabel.save(fmri_masked_img, 'output/masked_fmri.nii')
    nibabel.save(anat_img, 'output/anat.nii')
    nibabel.save(first_roi_img, 'output/roi1.nii')
    nibabel.save(second_roi_img, 'output/roi2.nii')
    nibabel.save(epi_masker.mask_img_, 'output/mask.nii')
    with contextlib.closing(open(saved_filepath, 'wb')) as fp:
        cPickle.dump(cut_coords, fp)
        cPickle.dump(haxby_labels, fp)
        cPickle.dump(top_labels, fp)

else:
    print("Loading data from disk.")
    fmri_masked_img = nibabel.load('output/masked_fmri.nii')
    anat_img = nibabel.load('output/anat.nii')
    first_roi_img = nibabel.load('output/roi1.nii')
    second_roi_img = nibabel.load('output/roi2.nii')
    mask_img = nibabel.load('output/mask.nii')
    shared_affine = fmri_masked_img.get_affine()

    # Load
    with contextlib.closing(open(saved_filepath, 'rb')) as fp:
        cut_coords = cPickle.load(fp)
        haxby_labels = cPickle.load(fp)
        top_labels = cPickle.load(fp)

    # Compute
    mem = Memory(cachedir='nilearn_cache')
    epi_masker = NiftiMasker(mask_img=mask_img, detrend=False,
                             standardize=False)
    epi_masker.fit()
    masked_fmri_vectors = epi_masker.transform(fmri_masked_img)


def plot_two_maps(plot_fn, img1, img2, bg_img=anat_img, **kwargs):
    fig_id = plt.subplot(2, 1, 1)
    plot_fn(img1[0], bg_img,
            title=img1[1], axes=fig_id, **kwargs)
    fig_id = plt.subplot(2, 1, 2)
    plot_fn(img2[0], bg_img,
            title=img2[1], axes=fig_id, **kwargs)

print("Plot mean activation (per condition) over the whole brain.")
face_image = image.index_img(fmri_masked_img, haxby_labels == 'face')
house_image = image.index_img(fmri_masked_img, haxby_labels == 'house')
mean_face_img = mem.cache(image.mean_img)(face_image)
mean_house_img = mem.cache(image.mean_img)(house_image)
plot_two_maps(plot_stat_map,
              (mean_face_img, 'Mean face image'),
              (mean_house_img, 'Mean house image'))
plt.show()

print("Plot mean activation (per condition) for the first two ROIs.")

print("Use the new ROIs to extract data maps in both ROIs.")
first_roi_masker = NiftiLabelsMasker(labels_img=first_roi_img,
                                     resampling_target=None,
                                     standardize=False, detrend=False)
first_roi_masker.fit()
second_roi_masker = NiftiLabelsMasker(labels_img=second_roi_img,
                                      resampling_target=None,
                                      standardize=False, detrend=False)
second_roi_masker.fit()
condition_names = list(set(haxby_labels))
n_cond_img = masked_fmri_vectors[haxby_labels == 'house', :].shape[0]
n_conds = len(condition_names)
for i, cond in enumerate(condition_names):
    cond_mask = haxby_labels == cond
    cond_img = image.index_img(fmri_masked_img, cond_mask)
    X1_vectors = mem.cache(first_roi_masker.transform)(cond_img).T
    X2_vectors = mem.cache(second_roi_masker.transform)(cond_img).T
    X1_image = first_roi_masker.inverse_transform(X1_vectors)
    X2_image = second_roi_masker.inverse_transform(X2_vectors)
    X1_mean_img = mem.cache(image.mean_img)(X1_image)
    X2_mean_img = mem.cache(image.mean_img)(X2_image)
    plot_two_maps(plot_stat_map,
                  (X1_mean_img, 'ROI #1 mean %s image' % cond),
                  (X2_mean_img, 'ROI #2 mean %s image' % cond))
    plt.show()

print("Compare activation profiles for face vs. house, for the first two ROIs.")
