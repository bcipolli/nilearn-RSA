"""
"""

# Massage paths
import os
import sys
script_dir = os.path.abspath(os.path.dirname(__file__))
sys.path = [os.path.join(script_dir, 'nilearn')] + sys.path

import matplotlib.pyplot as plt
import numpy as np
import shelve
from scipy.stats import pearsonr
from scipy.spatial.distance import squareform

import nibabel
from nilearn.image import index_img, concat_imgs, mean_img
from nilearn.plotting import plot_mosaic_stat_map
from sklearn.externals.joblib import Memory

from RSA_balls import HaxbySearchlightAnalysis

memory = Memory(cachedir='nilearn_cache', verbose=10)


def compute_best_detector(li, img_labels):
    # Compute the distance matrix for the optimal detector

    n_imgs = len(img_labels)
    best_detector = np.zeros((n_imgs * (n_imgs - 1) / 2.,))
    idx = 0
    for li1 in range(n_imgs):
        for li2 in range(li1 + 1, n_imgs):
            if li1 == li or li2 == li:
                best_detector[idx] = 1.
            idx += 1
    return best_detector


def compute_detector(li, img_labels):
    # Compute the optimal detector
    n_imgs = len(img_labels)
    best_detector = np.nan * np.empty((n_imgs * (n_imgs - 1) / 2.,))
    idx = 0
    for li1 in range(n_imgs):
        for li2 in range(li1 + 1, n_imgs):
            if li1 == li or li2 == li:
                best_detector[idx] = 1.
            idx += 1
    return best_detector


@memory.cache
def compute_stats(RSA_data, img_labels, detector_fn):
    n_imgs = len(img_labels)
    n_seeds = RSA_data.shape[0]
    voxelwise_corr = np.empty((n_imgs, n_seeds))
    voxelwise_pval = np.empty((n_imgs, n_seeds))

    for li, img_label in enumerate(img_labels):
        # Retrieve the cur_count'th img_label in img_labels
        # e.g. the 3rd 'face' in img_labels.
        best_detector = 2 * detector_fn(li, img_labels)

        idx = np.logical_not(np.isnan(best_detector))
        # Compare it to every voxel.
        for si in range(n_seeds):
            pr, pv = pearsonr(best_detector[idx], RSA_data[si, idx].T)
            # pr = np.dot(best_detector[idx], RSA_data[si, idx].T) / (4.**2)
            voxelwise_corr[li, si] = 0. if np.isnan(pr) else pr
            voxelwise_pval[li, si] = 1. if np.isnan(pv) else pv

    return voxelwise_corr, voxelwise_pval


def get_indices(sorted_class_labels, class_labels, img_labels):
    n_classes = len(class_labels)

    # Find desired label indices
    class_class_idx = np.empty((n_classes,), dtype=int)
    class_img_idx = []  # can vary in length
    for ci, sorted_class_label in enumerate(sorted_class_labels):
        class_mask = class_labels == sorted_class_label
        class_class_idx[ci] = np.nonzero(class_mask)[0]
        img_mask = img_labels == sorted_class_label
        class_img_idx.append(np.nonzero(img_mask)[0])
    return class_class_idx, class_img_idx


def examine_correlations(detector_fn, subj_idx=0, radius=10.,
                         grouping='class', smoothing_fwhm=None,
                         force=False, visualize=True, standardize=True):
    # Compute RSA within VT
    RSA_img_filename = 'haxby_RSA_searchlight_subj%02d.nii' % subj_idx
    corr_img_filename = 'haxby_RSA_corr2perfect_subj%02d.nii' % subj_idx
    analysis_filename = 'haxby_RSA_analysis_subj%02d.db' % subj_idx
    shelf_key = 'r%.2f s%.2f g%s subj%02d' % (radius, smoothing_fwhm or 0., grouping, subj_idx)

    if not force:
        print("Loading from shelf...")
        shelf = shelve.open(analysis_filename)
        try:
            analysis = shelf[shelf_key]
            for prop in ['subj_idx', 'radius', 'grouping',
                         'smoothing_fwhm', 'standardize']:
                assert (getattr(analysis, prop) == locals[prop],
                        "analysis value didn't match for %s." % prop)
            analysis.loaded = True
        except Exception as e:
            print "Load error: %s" % e
        finally:
            shelf.close()
            del shelf

    if 'analysis' not in locals():
        analysis = HaxbySearchlightAnalysis(subj_idx=subj_idx,
                                            radius=radius,
                                            smoothing_fwhm=smoothing_fwhm,
                                            grouping=grouping,
                                            standardize=standardize)
        analysis.fit()
        analysis.transform(seeds_img=analysis.vt_mask_img)
        # analysis.save(RSA_img_filename)
    print("Mean # voxels: %.2f" % np.asarray(analysis.n_voxels).mean())

    # Compare the result to the optimally object-selective DSM
    print("Computing stats...")
    n_classes = len(analysis.class_labels)
    n_imgs = len(analysis.img_labels)
    n_seeds = len(analysis.searchlight.sphere_masker.seeds_)

    RSA_data = analysis.similarity_comparisons
    voxelwise_corr, voxelwise_pval = compute_stats(RSA_data=RSA_data,
                                                   img_labels=analysis.img_labels,
                                                   detector_fn=detector_fn)
    good_seeds = np.logical_not(np.isnan(RSA_data.mean(axis=1)))
    mean_RSA_data = RSA_data[good_seeds].mean(axis=0).copy()

    # Save the result
    sphere_masker = analysis.searchlight.sphere_masker
    corr_img = sphere_masker.inverse_transform(voxelwise_corr)
    # nibabel.save(corr_img, corr_img_filename)

    # Plot the result
    if visualize:
        analysis.visualize_func_img()
        analysis.searchlight.visualize_comparisons(
            similarity_comparisons=analysis.similarity_comparisons,
            labels=analysis.img_labels, anat_img=analysis.anat_img)
        analysis.searchlight.visualize_comparisons_std(
            similarity_std=analysis.similarity_std,
            anat_img=analysis.anat_img)

        # Plot detector
        fh1 = plt.figure(figsize=(18, 10))
        class_imgs = []
        n_rows = int(np.round(0.75 * np.sqrt(n_classes)))
        n_cols = int(np.ceil(n_classes / float(n_rows)))
        for ci, class_label in enumerate(analysis.class_labels):
            ax1 = fh1.add_subplot(n_rows, n_cols, ci + 1)
            li = np.nonzero(analysis.img_labels == class_label)[0][0]
            sq = squareform(detector_fn(li, analysis.img_labels))
            ax1.imshow(sq, interpolation='nearest')
            ax1.set_title('Best detector: %s' % class_label)

        # Plot correlation, p-value distributions over classes 
        fh2 = plt.figure(figsize=(18, 10))
        fh3 = plt.figure(figsize=(18, 10))
        class_imgs = []
        for ci, class_label in enumerate(analysis.class_labels):
            idx = np.nonzero(analysis.img_labels == class_label)[0]
            class_imgs.append(mean_img(index_img(corr_img, idx)))

            class_corr = voxelwise_corr[idx].mean(axis=0)
            class_pval = voxelwise_pval[idx].mean(axis=0)

            ax2 = fh2.add_subplot(3, 3, ci + 1)
            ax2.hist(class_corr, bins=25, normed=True)
            ax2.set_title('Correlation values: %s' % class_label)

            ax3 = fh3.add_subplot(3, 3, ci + 1)
            ax3.hist(class_pval, 25, normed=True)
            ax3.set_title('Significance values: %s' % class_label)

        fh4 = plt.figure(figsize=(18, 10))
        titles = ['Vs. perfect %s detector' % l for l in analysis.class_labels]
        class_img = concat_imgs(class_imgs)
        plot_mosaic_stat_map(class_img, colorbar=True, figure=fh4,
                             bg_img=analysis.anat_img,
                             display_mode='z', cut_coords=1,
                             title=titles, shape=(3, 3))

    # Save the result last as we need to modify the object
    #   in order to save.
    class_labels = np.asarray(analysis.class_labels).copy()
    img_labels = np.asarray(analysis.img_labels).copy()
    if not getattr(analysis, 'loaded', False):
        print("Saving to shelf...")
        shelf = shelve.open(analysis_filename, writeback=True)
        try:
            analysis.searchlight.sphere_masker.xform_fn = None
            shelf[shelf_key] = analysis
            shelf.sync()
            shelf.close()
        except Exception as e:
            print "Error saving to shelve: %s." % e
        finally:
            del shelf
            del analysis

    # Return the computation
    return voxelwise_corr, voxelwise_pval, img_labels, class_labels, mean_RSA_data


def group_examine_correlations(detector_fn,
                               visualize=True,
                               force=False,
                               remove_rest=False,
                               grouping='class',
                               radius=10.,
                               smoothing_fwhm=None,
                               standardize=True,
                               resort_stims=False):
    n_bins = 25
    n_subj = 6
    sorted_class_labels = ['face', 'house', 'cat', 'bottle', 'scissors',
                           'shoe', 'chair', 'scrambledpix', 'rest']
    n_classes = 9
    n_imgs = n_classes if grouping == 'class' else 121
    n_imgperclass = n_imgs / n_classes

    # Data to save off
    corr_hists = np.empty((n_subj, n_classes, n_bins))
    pval_hists = np.empty(corr_hists.shape)
    RSA_data = np.empty((n_subj, n_imgs, n_imgs))

    # Get all subject data; save into histograms and means.
    corr_bins = np.linspace(-0.75, 0.75, n_bins + 1).tolist()
    pval_bins = np.linspace(0., 1., n_bins + 1).tolist()

    for subj_idx in range(n_subj):
        # Compute the RSA, correlation to the selected detector.
        corr, pval, img_labels, class_labels, RSA_compares = examine_correlations(
            detector_fn=detector_fn,
            subj_idx=subj_idx,
            radius=radius,
            smoothing_fwhm=smoothing_fwhm,
            grouping=grouping,
            force=force,
            visualize=visualize,
            standardize=standardize)
        class_labels = np.asarray(class_labels)
        img_labels = np.asarray(img_labels)

        print "Corr: ", corr
        print "RSA: ", RSA_compares
        print "PVal: ", pval

        # Compute indices; no sorting!
        class_class_idx, class_img_idx = get_indices(class_labels, class_labels, img_labels)

        # Summarize data
        RSA_data[subj_idx, :, :] = squareform(RSA_compares)
        for ci, class_label in enumerate(class_labels):
            idx = class_img_idx[ci]
            corr_hists[subj_idx, class_class_idx[ci]], _ = np.histogram(corr[idx].flatten(), corr_bins, density=True)
            pval_hists[subj_idx, class_class_idx[ci]], _ = np.histogram(pval[idx].flatten(), pval_bins, density=True)

    # Reorder data
    if resort_stims:
        class_class_idx, class_img_idx = get_indices(sorted_class_labels, class_labels, img_labels)
        flat_class_img_idx = [ii for ci in class_img_idx for ii in ci]
        corr_hists = corr_hists[:, class_class_idx]
        pval_hists = pval_hists[:, class_class_idx]
        RSA_data = RSA_data[:, flat_class_img_idx]  # reorder
        RSA_data = RSA_data[:, :, flat_class_img_idx]
        class_labels = class_labels[class_class_idx]
        img_labels = img_labels[flat_class_img_idx]

        # Refresh the index values
        del class_class_idx
    _, class_img_idx = get_indices(class_labels, class_labels, img_labels)
    flat_class_img_idx = [ii for ci in class_img_idx for ii in ci]

    # Eliminate resting state data
    if remove_rest:
        non_rest_idx = list(set(flat_class_img_idx) - set(class_img_idx[-1]))
        corr_hists = corr_hists[:-1]
        pval_hists = pval_hists[:-1]
        RSA_data = RSA_data[:, non_rest_idx]
        RSA_data = RSA_data[:, :, non_rest_idx]
        class_labels = class_labels[:-1]
        img_labels = img_labels[non_rest_idx]
        n_classes = len(class_labels)
        n_imgs = len(img_labels)

    # Plot mean (over subjects) correlation and p-value histograms
    fh1 = plt.figure(figsize=(18, 10))
    fh2 = plt.figure(figsize=(18, 10))
    bar_bins = lambda bins: np.asarray([(bins[bi - 1] + bins[bi]) / 2.
                                        for bi in range(1, len(bins))])
    bar_width = lambda bins: bins[1] - bins[0]
    for ci, class_label in enumerate(class_labels):
        idx = class_labels == class_label

        # FIGURE 1: Histogram of correlation values
        mean_corr_hist = corr_hists[:, idx].mean(axis=0).flatten()
        std_corr_hist = corr_hists[:, idx].std(axis=0).flatten()
        ax1 = fh1.add_subplot(3, 3, ci + 1)
        ax1.bar(bar_bins(corr_bins) - bar_width(corr_bins) / 2.,
                mean_corr_hist * bar_width(corr_bins),
                yerr=std_corr_hist * bar_width(corr_bins),
                width=bar_width(corr_bins))
        ax1.set_title('Correlation (%s)' % class_label)
        ax1.set_ylim([0., 0.4])
        ax1.set_xlim([-0.6, 0.6])

        # FIGURE 2: Histogram of p-values
        mean_pval_hist = pval_hists[:, idx].mean(axis=0).flatten()
        std_pval_hist = pval_hists[:, idx].std(axis=0).flatten()
        ax2 = fh2.add_subplot(3, 3, ci + 1)
        ax2.bar(bar_bins(pval_bins) - bar_width(pval_bins) / 2.,
                mean_pval_hist * bar_width(pval_bins),
                yerr=std_pval_hist * bar_width(corr_bins),
                width=bar_width(pval_bins))
        ax2.set_title('p-values (%s)' % class_label)
        ax2.set_ylim([0., 0.2])
        ax2.set_xlim([0., 1.0])

    # FIGURE 3: Plot the mean correlation matrix
    fh3 = plt.figure(figsize=(14, 10))
    for subj_idx in range(n_subj + 1):
        if subj_idx < n_subj:
            mat = RSA_data[subj_idx]
            subj_id = str(subj_idx)
        else:
            mat = RSA_data.mean(axis=0)
            subj_id = 'mean'
        ax3 = fh3.add_subplot(3, 3, subj_idx + 1)
        ax3.imshow(1. - mat - np.eye(n_imgs),
                   interpolation='nearest',
                   vmin=-1., vmax=1.)
        ax3.set_title('Subject %s dissimilarity' % subj_id)

        ax3.set_yticks(np.arange(0, n_imgs, n_imgperclass) + n_imgperclass / 2.)
        ax3.set_yticklabels(class_labels)
        ax3.set_xticks([])  # remove x-ticks

    # FIGURE 4: Plot haxby figure (ish)
    confusion_mat = 1. - RSA_data.mean(axis=0)
    confusion_mat = confusion_mat
    short_class_labels = [lbl[:5] for lbl in class_labels]
    fh4 = plt.figure(figsize=(12, 10))
    plt_order = [1, 3, 5, 7, 2, 4, 6, 8, 9]
    for ci, class_label in enumerate(class_labels):
        idx1 = img_labels == class_label

        bars_mean = np.empty(n_classes)
        bars_std = np.empty(n_classes)
        for ci2, class_label2 in enumerate(class_labels):
            idx2 = img_labels == class_label2
            cur_mat = confusion_mat[idx1]
            cur_mat = cur_mat[:, idx2]
            bars_mean[ci2] = cur_mat.mean()
            bars_std[ci2] = cur_mat.std()

        n_rows = 4 if remove_rest else 3
        n_cols = 2 if remove_rest else 3
        ax4 = fh4.add_subplot(n_rows, n_cols, plt_order[ci])
        ax4.bar(np.arange(n_classes) - 0.5, bars_mean, yerr=bars_std)
        ax4.set_title(class_label)
        ax4.set_xticks(list(range(n_classes)))
        ax4.set_xticklabels(short_class_labels)
        ax4.set_ylim([-0.2, 1.0])
    fh4.subplots_adjust(hspace=0.4)
    plt.show()


if __name__ == '__main__':
    # compute_best_detector
    # compute_detector
    group_examine_correlations(detector_fn=compute_best_detector,
                               visualize=True,
                               force=False,
                               radius=10.,
                               smoothing_fwhm=None,
                               grouping='img',
                               standardize=True,
                               remove_rest=True,
                               resort_stims=True)
