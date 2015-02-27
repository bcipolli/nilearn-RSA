import matplotlib.pyplot as plt
import numpy as np
import os
import shelve
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
    for li1 in range(n_labels):
        for li2 in range(li1 + 1, n_labels):
            if li1 == li or li2 == li:
                best_detector[idx] = 1.
            idx += 1
    return best_detector


def examine_detector_correlations(subj_idx=0, force=False, visualize=True):
    # Compute RSA within VT
    RSA_img_filename = 'haxby_RSA_searchlight_subj%02d.nii' % subj_idx
    corr_img_filename = 'haxby_RSA_corr2perfect_subj%02d.nii' % subj_idx
    analysis_filename = 'haxby_RSA_analysis_subj%02d.db' % subj_idx

    if not force and os.path.exists(analysis_filename):
        analysis = nibabel.load(analysis_filename)

    else:
        analysis = SearchlightAnalysis('haxby', subj_idx=subj_idx)
        analysis.fit()
        analysis.transform(seeds_img=analysis.vt_mask_img)
        analysis.save(RSA_img_filename)

        # Save the result
        # shelf = shelve.open(analysis_filename)
        # shelf[subj_idx] = analysis

    # Compare the result to the optimally object-selective DSM
    n_labels = len(analysis.labels)
    n_seeds = len(analysis.searchlight.sphere_masker.seeds_)
    voxelwise_corr = np.empty((n_labels, n_seeds))
    voxelwise_pval = np.empty((n_labels, n_seeds))

    for li, label in enumerate(analysis.labels):
        best_detector = compute_best_detector(label, analysis.labels)
        # Compare it to every voxel.
        RSA_data = analysis.searchlight.RSA_data
        for si in range(n_seeds):
            pr, pv = pearsonr(best_detector, RSA_data[si].T)
            voxelwise_corr[li, si] = 0. if np.isnan(pr) else pr
            voxelwise_pval[li, si] = 0. if np.isnan(pv) else pv

    # Save the result
    sphere_masker = analysis.searchlight.sphere_masker
    corr_img = sphere_masker.inverse_transform(voxelwise_corr)
    nibabel.save(corr_img, corr_img_filename)

    # Plot the result
    if visualize:
        # analysis.visualize()

        # Plot detector
        # fh1 = plt.figure(figsize=(18, 10))
        # ax1 = fh1.add_subplot(3, 3, li + 1)
        # sq = squareform(best_detector)  # + np.eye(n_labels)
        # ax1.imshow(sq, interpolation='nearest')
        # ax1.set_title('Best detector: %s' % label)

        fh2 = plt.figure(figsize=(18, 10))
        fh3 = plt.figure(figsize=(18, 10))
        for li, label in enumerate(analysis.labels):

            ax2 = fh2.add_subplot(3, 3, li + 1)
            ax2.hist(voxelwise_corr[li], bins=25, normed=True)
            ax2.set_title('Correlation values: %s' % label)

            ax3 = fh3.add_subplot(3, 3, li + 1)
            ax3.hist(voxelwise_pval[li], 25, normed=True)
            ax3.set_title('Significance values: %s' % label)

        fh4 = plt.figure(figsize=(18, 10))
        titles = ['Vs. perfect %s detector' % l for l in analysis.labels]
        plot_mosaic_stat_map(corr_img, colorbar=True, figure=fh4,
                             bg_img=analysis.anat_img,
                             display_mode='z', cut_coords=1,
                             title=titles, shape=(3, 3))

    # Return the computation
    return voxelwise_corr, voxelwise_pval, analysis.labels


def group_examine_detector_correlations(visualize=True):
    n_bins = 25
    n_subj = 6
    n_labels = 9
    corr_hists = np.empty((n_subj, n_labels, n_bins))
    pval_hists = np.empty(corr_hists.shape)

    corr_bins = np.linspace(-0.75, 0.75, n_bins + 1).tolist()
    pval_bins = np.linspace(0., 1., n_bins + 1).tolist()
    for subj_idx in range(n_subj):
        corr, pval, labels = examine_detector_correlations(subj_idx=subj_idx,
                                                           force=True,
                                                           visualize=visualize)
        for li in range(n_labels):
            corr_hists[subj_idx, li], _ = np.histogram(corr[li], corr_bins)
            pval_hists[subj_idx, li], _ = np.histogram(pval[li], pval_bins)

    fh1 = plt.figure(figsize=(18, 10))
    fh2 = plt.figure(figsize=(18, 10))
    bar_bins = lambda bins: [(bins[bi - 1] + bins[bi]) / 2.
                             for bi in range(1, len(bins))]
    bar_width = lambda bins: bins[1] - bins[0]
    for li in range(n_labels):

        mean_corr_hist = corr_hists[:, li].mean(axis=0).flatten()
        std_corr_hist = corr_hists[:, li].std(axis=0).flatten()
        ax1 = fh1.add_subplot(3, 3, li + 1)
        ax1.bar(bar_bins(corr_bins), mean_corr_hist, yerr=std_corr_hist,
                width=bar_width(corr_bins))
        ax1.set_title('Correlation (%s)' % labels[li])

        mean_pval_hist = pval_hists[:, li].mean(axis=0).flatten()
        std_pval_hist = pval_hists[:, li].std(axis=0).flatten()
        ax2 = fh2.add_subplot(3, 3, li + 1)
        ax2.bar(bar_bins(pval_bins), mean_pval_hist, yerr=std_pval_hist,
                width=bar_width(pval_bins))
        ax2.set_title('p-values (%s)' % labels[li])

    plt.show()

if __name__ == '__main__':
    group_examine_detector_correlations(visualize=True)
