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


def compute_detector(label, all_labels):
    # Compute the optimal detector

    n_labels = len(all_labels)
    best_detector = np.nan * np.empty((n_labels * (n_labels - 1) / 2.,))
    idx = 0
    li = np.nonzero(np.asarray(all_labels) == label)[0][0]
    for li1 in range(n_labels):
        for li2 in range(li1 + 1, n_labels):
            if li1 == li or li2 == li:
                best_detector[idx] = 1.
            idx += 1
    return best_detector


def examine_correlations(detector_fn, subj_idx=0, radius=10.,
                         smoothing_fwhm=None,
                         force=False, visualize=True):
    # Compute RSA within VT
    RSA_img_filename = 'haxby_RSA_searchlight_subj%02d.nii' % subj_idx
    corr_img_filename = 'haxby_RSA_corr2perfect_subj%02d.nii' % subj_idx
    analysis_filename = 'haxby_RSA_analysis_subj%02d.db' % subj_idx
    shelf_key = 'r%.2f s%.2f subj%02d' % (radius, smoothing_fwhm or 0., subj_idx)

    if not force:
        print("Loading from shelf...")
        shelf = shelve.open(analysis_filename)
        try:
            analysis = shelf[shelf_key]
            analysis.loaded = True
        except Exception as e:
            print "Load error: %s" % e
        finally:
            shelf.close()
            del shelf

    if 'analysis' not in locals():
        analysis = SearchlightAnalysis('haxby', subj_idx=subj_idx,
                                       radius=radius,
                                       smoothing_fwhm=smoothing_fwhm)
        analysis.fit()
        analysis.transform(seeds_img=analysis.vt_mask_img)
        analysis.save(RSA_img_filename)

    # Compare the result to the optimally object-selective DSM
    n_labels = len(analysis.labels)
    n_seeds = len(analysis.searchlight.sphere_masker.seeds_)
    voxelwise_corr = np.empty((n_labels, n_seeds))
    voxelwise_pval = np.empty((n_labels, n_seeds))

    RSA_data = analysis.searchlight.RSA_data
    for li, label in enumerate(analysis.labels):
        best_detector = 2 * detector_fn(label, analysis.labels)
        idx = np.logical_not(np.isnan(best_detector))
        # Compare it to every voxel.
        for si in range(n_seeds):
            pr, pv = pearsonr(best_detector[idx], RSA_data[si, idx].T)
            # pr = np.dot(best_detector[idx], RSA_data[si, idx].T) / (4.**2)
            voxelwise_corr[li, si] = 0. if np.isnan(pr) else pr
            voxelwise_pval[li, si] = 1. if np.isnan(pv) else pv
    good_seeds = np.logical_not(np.isnan(RSA_data.mean(axis=1)))
    mean_RSA_data = RSA_data[good_seeds].mean(axis=0).copy()

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

    # Save the result last as we need to modify the object
    #   in order to save.
    labels = analysis.labels
    if not getattr(analysis, 'loaded', False):
        print("Saving to shelf...")
        shelf = shelve.open(analysis_filename, writeback=True)
        try:
            analysis.searchlight.sphere_masker.xform_fn = None
            shelf[shelf_key] = analysis
            shelf.sync()
            shelf.close()
        except Exception as e:
            print e
        finally:
            del shelf
            del analysis

    # Return the computation
    return voxelwise_corr, voxelwise_pval, labels, mean_RSA_data


def group_examine_correlations(detector_fn,
                               visualize=True,
                               force=False,
                               radius=10.,
                               smoothing_fwhm=None):
    n_bins = 25
    n_subj = 6
    sorted_labels = ['face', 'house', 'cat', 'bottle', 'scissors',
                     'shoe', 'chair', 'scrambledpix', 'rest']
    n_labels = 9

    # Data to save off
    corr_hists = np.empty((n_subj, n_labels, n_bins))
    pval_hists = np.empty(corr_hists.shape)
    RSA_data = np.empty((n_subj, n_labels, n_labels))

    # Get all subject data; save into histograms and means.
    corr_bins = np.linspace(-0.75, 0.75, n_bins + 1).tolist()
    pval_bins = np.linspace(0., 1., n_bins + 1).tolist()
    for subj_idx in range(n_subj):
        corr, pval, labels, RSA_compares = examine_correlations(
            detector_fn=detector_fn,
            subj_idx=subj_idx,
            radius=radius,
            smoothing_fwhm=smoothing_fwhm,
            force=force,
            visualize=visualize)

        # Find desired label indices
        lbl_idx = np.empty((n_labels,))
        for li in range(n_labels):
            lbl_idx[li] = np.nonzero(np.asarray(labels) == sorted_labels[li])[0][0]

        # Unwind data (summarize & sort)
        for li in range(n_labels):
            corr_hists[subj_idx, lbl_idx[li]], _ = np.histogram(corr[li], corr_bins, density=True)
            pval_hists[subj_idx, lbl_idx[li]], _ = np.histogram(pval[li], pval_bins, density=True)
        RSA_data[subj_idx, :, :] = squareform(RSA_compares)
        RSA_data[subj_idx] = RSA_data[subj_idx, lbl_idx.tolist()]  # reorder
        RSA_data[subj_idx] = RSA_data[subj_idx, :, lbl_idx.tolist()]  # reorder
        labels = sorted_labels

    # Eliminate resting state data
    corr_hists = corr_hists[:-1]
    pval_hists = pval_hists[:-1]
    RSA_data = RSA_data[:, :-1, :-1]
    labels = labels[:-1]
    n_labels = len(labels)

    # Plot mean correlation and p-value histograms
    fh1 = plt.figure(figsize=(18, 10))
    fh2 = plt.figure(figsize=(18, 10))
    bar_bins = lambda bins: np.asarray([(bins[bi - 1] + bins[bi]) / 2.
                                        for bi in range(1, len(bins))])
    bar_width = lambda bins: bins[1] - bins[0]
    for li in range(n_labels):

        mean_corr_hist = corr_hists[:, li].mean(axis=0).flatten()
        std_corr_hist = corr_hists[:, li].std(axis=0).flatten()
        ax1 = fh1.add_subplot(3, 3, li + 1)
        ax1.bar(bar_bins(corr_bins) - bar_width(corr_bins) / 2.,
                mean_corr_hist * bar_width(corr_bins),
                yerr=std_corr_hist * bar_width(corr_bins),
                width=bar_width(corr_bins))
        ax1.set_title('Correlation (%s)' % labels[li])
        ax1.set_ylim([0., 0.4])
        ax1.set_xlim([-0.6, 0.6])

        mean_pval_hist = pval_hists[:, li].mean(axis=0).flatten()
        std_pval_hist = pval_hists[:, li].std(axis=0).flatten()
        ax2 = fh2.add_subplot(3, 3, li + 1)
        ax2.bar(bar_bins(pval_bins) - bar_width(pval_bins) / 2.,
                mean_pval_hist * bar_width(pval_bins),
                yerr=std_pval_hist * bar_width(corr_bins),
                width=bar_width(pval_bins))
        ax2.set_title('p-values (%s)' % labels[li])
        ax2.set_ylim([0., 0.2])
        ax2.set_xlim([0., 1.0])

    # Plot the mean correlation matrix
    fh3 = plt.figure(figsize=(14, 10))
    for subj_idx in range(n_subj + 1):
        if subj_idx < n_subj:
            mat = RSA_data[subj_idx]
            subj_id = str(subj_idx)
        else:
            mat = RSA_data.mean(axis=0)
            subj_id = 'mean'
        ax3 = fh3.add_subplot(3, 3, subj_idx + 1)
        ax3.imshow(1. - mat - np.eye(n_labels),
                   interpolation='nearest',
                   vmin=-1., vmax=1.)
        ax3.set_title('Subject %s similarity' % subj_id)

        ax3.set_yticks(list(range(n_labels)))
        ax3.set_yticklabels(labels)
        ax3.set_xticks([])  # remove x-ticks

    # Plot haxby figure (ish)
    confusion_mat = 1. - RSA_data.mean(axis=0)
    confusion_mat = confusion_mat
    fh4 = plt.figure(figsize=(14, 10))
    for li, label in enumerate(labels):
        ax4 = fh4.add_subplot(4, 2, li + 1)
        ax4.bar(range(n_labels), confusion_mat[li])
        ax4.set_title(label)
        ax4.set_xticks(list(range(n_labels)))
        ax4.set_xticklabels(labels[:9])
        ax4.set_ylim([-0.2, 1.0])
    fh4.subplots_adjust(hspace=0.4)
    plt.show()


if __name__ == '__main__':
    # compute_best_detector
    # compute_detector
    group_examine_correlations(detector_fn=compute_best_detector,
                               visualize=False,
                               force=False,
                               radius=5.,
                               smoothing_fwhm=None)
