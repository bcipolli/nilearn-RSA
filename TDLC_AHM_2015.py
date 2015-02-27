import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import pearsonr
from scipy.spatial.distance import squareform

import nibabel
from nilearn.plotting import plot_mosaic_stat_map

from RSA_balls import SearchlightAnalysis


def compute_best_detector(label, all_labels):
    # Compute the optimal detector

    n_labels = len(all_labels)
    best_detector = np.zeros((n_labels * (n_labels - 1) / 2.,))
    idx = 0
    li = np.nonzero(np.asarray(all_labels) == label)[0]
    print label, li
    for li1 in range(n_labels):
        for li2 in range(li1 + 1, n_labels):
            if li1 == li or li2 == li:
                best_detector[idx] = 1.
            idx += 1
    return best_detector


def examine_detector_correlations(subj_idx=0, force=True):
    # Compute RSA within VT
    RSA_img_filename = 'haxby_RSA_searchlight_subj%02d.nii' % subj_idx
    corr_img_filename = 'haxby_RSA_corr2perfect_subj%02d.nii' % subj_idx
    analysis = SearchlightAnalysis('haxby', subj_idx=subj_idx)
    analysis.fit()

    if not force and os.path.exists(corr_img_filename):
        corr_img = nibabel.load(corr_img_filename)

    else:
        analysis.transform(seeds_img=analysis.vt_mask_img)
        analysis.save(RSA_img_filename)
        analysis.visualize()

        # Compare the result to the optimally object-selective DSM
        n_labels = len(analysis.labels)
        n_seeds = len(analysis.searchlight.sphere_masker.seeds_)
        voxelwise_corr = np.empty((n_labels, n_seeds))
        voxelwise_pval = np.empty((n_labels, n_seeds))

        fh1 = plt.figure(figsize=(18, 10))
        fh2 = plt.figure(figsize=(18, 10))
        fh3 = plt.figure(figsize=(18, 10))
        for li, label in enumerate(analysis.labels):
            best_detector = compute_best_detector(label, analysis.labels)

            ax1 = fh1.add_subplot(3, 3, li + 1)
            sq = squareform(best_detector)  # + np.eye(n_labels)
            ax1.imshow(sq, interpolation='nearest')
            ax1.set_title('Best detector: %s' % label)

            # Compare it to every voxel.
            RSA_data = analysis.searchlight.RSA_data
            for si in range(n_seeds):
                pr, pv = pearsonr(best_detector, RSA_data[si].T)
                voxelwise_corr[li, si] = 0. if np.isnan(pr) else pr
                voxelwise_pval[li, si] = 0. if np.isnan(pv) else pv

            ax2 = fh2.add_subplot(3, 3, li + 1)
            ax2.hist(voxelwise_corr[li], bins=25, normed=True)
            ax2.set_title('Correlation values: %s' % label)

            ax3 = fh3.add_subplot(3, 3, li + 1)
            ax3.hist(voxelwise_pval[li], 25, normed=True)
            ax3.set_title('Significance values: %s' % label)

        # Save the result
        sphere_masker = analysis.searchlight.sphere_masker
        corr_img = sphere_masker.inverse_transform(voxelwise_corr)
        nibabel.save(corr_img, corr_img_filename)

    # Plot the result
    fh = plt.figure(figsize=(18, 10))
    titles = ['Vs. perfect %s detector' % l for l in analysis.labels]
    plot_mosaic_stat_map(corr_img, colorbar=True, figure=fh,
                         bg_img=analysis.anat_img,
                         display_mode='z', cut_coords=1,
                         title=titles, shape=(3, 3))

    # Return the computation
    if 'voxelwise_corr' in locals():
        return voxelwise_corr, voxelwise_pval


if __name__ == '__main__':
    print examine_detector_correlations(subj_idx=2, force=True)
    plt.show()
