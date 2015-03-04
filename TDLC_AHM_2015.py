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

from RSA_searchlight import HaxbySearchlightAnalysis, get_class_indices

memory = Memory(cachedir='nilearn_cache', verbose=10)


def compute_best_detector(class_label, img_labels):
    # Compute the dissimilarity matrix for the optimal detector (correlation)
    # Everything looks the same, except if it's compared to
    #   this current class.  Then, it looks completely different.
    n_imgs = len(img_labels)
    best_detector = np.zeros((n_imgs * (n_imgs - 1) / 2.,))
    idx = 0
    for li1 in range(n_imgs):
        matches1 = img_labels[li1] == class_label
        for li2 in range(li1 + 1, n_imgs):
            matches2 = img_labels[li2] == class_label
            if np.logical_xor(matches1, matches2):
                best_detector[idx] = 1.
            idx += 1
    return best_detector


def compute_detector(class_label, img_labels):
    # Compute the optimal dissimilarity detector (correlation)
    n_imgs = len(img_labels)
    best_detector = np.nan * np.empty((n_imgs * (n_imgs - 1) / 2.,))
    idx = 0
    for li1 in range(n_imgs):
        matches1 = img_labels[li1] == class_label
        for li2 in range(li1 + 1, n_imgs):
            matches2 = img_labels[li2] == class_label
            if np.logical_xor(matches1, matches2):
                best_detector[idx] = 1.
            idx += 1
    return best_detector


def low_rank_regression(X, Y):
    from scipy.linalg import lstsq
    residuals = lstsq(X, Y)[1]
    return residuals.mean()

@memory.cache
def compute_matrix_similarity(RDM_data, img_labels, class_labels, detector_fn, method='corr'):
    # RDM data: vector of pairwise dissimilarities
    # method: corr or regress
    n_imgs = len(img_labels)
    n_classes = len(class_labels)
    n_seeds = RDM_data.shape[0]
    voxelwise_corr = np.empty((n_classes, n_seeds))
    voxelwise_pval = np.empty((n_classes, n_seeds))

    if method == 'corr':
        method_fn = pearsonr
    elif method == 'regress':
        method_fn = lambda x, y: (low_rank_regression(squareform(x), squareform(y)), np.nan)
    else:
        raise Exception("Unknown method: %s" % method)

    # This wil
    for ci, class_label in enumerate(class_labels):
        # Retrieve the cur_count'th img_label in img_labels
        # e.g. the 3rd 'face' in img_labels.
        best_detector = detector_fn(class_label, img_labels)  # diag=0
        idx = np.logical_not(np.isnan(best_detector))

        # Compare it to every voxel.
        for si in range(n_seeds):
            if np.any(np.isnan(RDM_data[si, idx])):
                print "nan"
                pr = pv = np.nan
            else:
                pr, pv = method_fn(best_detector[idx], RDM_data[si, idx].T)
            voxelwise_corr[ci, si] = 0. if np.isnan(pr) else pr
            voxelwise_pval[ci, si] = 1. if np.isnan(pv) else pv

    return voxelwise_corr, voxelwise_pval


def examine_correlations(detector_fn, subj_idx=0, radius=10.,
                         grouping='img', smoothing_fwhm=None,
                         seeds_mask='vt',
                         force=False, visualize=list(range(5)),
                         standardize=True, detrend=False):

    # Compute filenames for loading / saving.
    shelve_filename = 'db/haxby_RSA_analysis_subj%02d.db' % subj_idx
    shelve_key = 'r%.2f g%s s%.2f m%s subj%02d' % (radius, grouping,
                                                   smoothing_fwhm or 0.,
                                                   seeds_mask, subj_idx)

    if not force:
        print("Loading subject %s from %s..." % (subj_idx, shelve_filename))
        shelf = shelve.open(shelve_filename)
        try:
            analysis = shelf[shelve_key]
            for prop in ['subj_idx', 'radius', 'grouping',
                         'smoothing_fwhm', 'standardize']:
                assert getattr(analysis, prop) == locals()[prop], \
                       "analysis value didn't match for %s." % prop
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

        if seeds_mask == 'all':
            analysis.transform(seeds_img=analysis.mask_img)
        elif seeds_mask == 'vt':
            analysis.transform(seeds_img=analysis.vt_mask_img)
        else:
            raise ValueError("Unknown mask name: %s" % seeds_mask)

    # Compare the result to the optimally object-selective DSM
    print("Computing stats...")
    print("\tMean # voxels: %.2f" % np.asarray(analysis.n_voxels).mean())
    n_classes = len(analysis.class_labels)
    n_imgs = len(analysis.img_labels)
    n_seeds = len(analysis.searchlight.sphere_masker.seeds_)

    RDM_data = analysis.similarity_comparisons
    voxelwise_corr, voxelwise_pval = \
        compute_matrix_similarity(RDM_data=RDM_data, img_labels=analysis.img_labels,
                                  class_labels=analysis.class_labels,
                                  detector_fn=detector_fn, method='corr')
    good_seeds = np.logical_not(np.isnan(RDM_data.mean(axis=1)))
    mean_RDM_data = RDM_data[good_seeds].mean(axis=0)

    # Plot the result
    if 0 in visualize:
        analysis.searchlight.visualize_comparisons_std(
            similarity_std=analysis.similarity_std,
            anat_img=analysis.anat_img)

    # Plot detector
    if 1 in visualize:
        fh1 = plt.figure(figsize=(18, 10))
        n_rows = int(np.round(0.75 * np.sqrt(n_classes)))
        n_cols = int(np.ceil(n_classes / float(n_rows)))
        for ci, class_label in enumerate(analysis.class_labels):
            ax1 = fh1.add_subplot(n_rows, n_cols, ci + 1)
            li = np.nonzero(analysis.img_labels == class_label)[0][0]
            sq = squareform(detector_fn(li, analysis.img_labels))
            ax1.imshow(sq, interpolation='nearest')
            ax1.set_title('Best detector: %s' % class_label)

    # Plot correlation, p-value distributions over classes
    if 2 in visualize or 3 in visualize:
        fh2 = plt.figure(figsize=(18, 10))
        fh3 = plt.figure(figsize=(18, 10))
        for ci, class_label in enumerate(analysis.class_labels):
            ax2 = fh2.add_subplot(3, 3, ci + 1)
            ax2.hist(voxelwise_corr[ci], bins=25, normed=True)
            ax2.set_title('Correlation values: %s' % class_label)

            ax3 = fh3.add_subplot(3, 3, ci + 1)
            ax3.hist(voxelwise_pval[ci], bins=25, normed=True)
            ax3.set_title('Significance values: %s' % class_label)

    if 4 in visualize:
        sphere_masker = analysis.searchlight.sphere_masker
        class_img = sphere_masker.inverse_transform(voxelwise_corr)

        fh4 = plt.figure(figsize=(18, 10))
        titles = ['Vs. perfect %s detector' % l for l in analysis.class_labels]
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
        shelf = shelve.open(shelve_filename, writeback=True)
        try:
            analysis.searchlight.sphere_masker.xform_fn = None
            shelf[shelve_key] = analysis
            shelf.sync()
            shelf.close()
        except Exception as e:
            print "Error saving to shelve: %s." % e
        finally:
            del shelf
            del analysis

    # Return the computation
    return voxelwise_corr, voxelwise_pval, img_labels, class_labels, mean_RDM_data


def group_examine_correlations(detector_fn,
                               n_subj=6,
                               visualize=range(10),
                               force=False,
                               remove_rest=True,
                               grouping='img',
                               radius=10.,
                               seeds_mask='vt',
                               smoothing_fwhm=None,
                               standardize=True):
    n_bins = 25
    n_classes = 9
    n_imgperclass = np.asarray([9] * 8 + [49])  # rest has 49
    n_imgs = n_classes if grouping == 'class' else np.sum(n_imgperclass)

    # Data to save off
    corr_hists = np.empty((n_subj, n_classes, n_bins))
    pval_hists = np.empty(corr_hists.shape)
    RDM_data = np.nan * np.zeros((n_subj, n_imgs, n_imgs))

    # Get all subject data; save into histograms and means.
    if grouping == 'img':
        corr_bins = np.linspace(-0.10, 0.10, n_bins + 1).tolist()
    else:
        corr_bins = np.linspace(-0.50, 0.50, n_bins + 1).tolist()
    pval_bins = np.linspace(0., 1., n_bins + 1).tolist()

    for subj_idx in range(n_subj):
        # Compute the RSA, correlation to the selected detector.
        corr, pval, img_labels, class_labels, RDM_compares = examine_correlations(
            detector_fn=detector_fn,
            subj_idx=subj_idx,
            radius=radius,
            smoothing_fwhm=smoothing_fwhm,
            grouping=grouping,
            seeds_mask=seeds_mask,
            force=force,
            visualize=visualize,
            standardize=standardize)
        class_labels = np.asarray(class_labels)
        img_labels = np.asarray(img_labels)

        print "Corr: ", corr
        print "RDM: ", RDM_compares
        print "PVal: ", pval

        # Compute indices; no sorting!
        img_class_idx = get_class_indices(class_labels, img_labels)

        # Summarize data
        RDM_data[subj_idx, :, :] = squareform(RDM_compares)
        for ci, class_label in enumerate(class_labels):
            corr_hists[subj_idx, ci], _ = np.histogram(corr[ci], corr_bins, density=True)
            pval_hists[subj_idx, ci], _ = np.histogram(pval[ci], pval_bins, density=True)

    flat_class_img_idx = [ii for ci in img_class_idx for ii in ci]

    # Eliminate resting state data
    if remove_rest:
        non_rest_class_idx = range(0, 8)
        non_rest_img_idx = list(set(flat_class_img_idx) - set(img_class_idx[-1]))
        corr_hists = corr_hists[:, non_rest_class_idx]
        pval_hists = pval_hists[:, non_rest_class_idx]
        RDM_data = RDM_data[:, non_rest_img_idx]
        RDM_data = RDM_data[:, :, non_rest_img_idx]
        class_labels = class_labels[non_rest_class_idx]
        img_labels = img_labels[non_rest_img_idx]
        n_classes = len(class_labels)
        n_imgperclass = n_imgperclass[non_rest_class_idx]
        n_imgs = len(img_labels)

    # Plot mean (over subjects) correlation and p-value histograms
    if 5 in visualize or 6 in visualize:
        fh5 = plt.figure(figsize=(18, 10))
        fh6 = plt.figure(figsize=(18, 10))
        bar_bins = lambda bins: np.asarray([(bins[bi - 1] + bins[bi]) / 2.
                                            for bi in range(1, len(bins))])
        bar_width = lambda bins: bins[1] - bins[0]
        bar_edges = lambda bins: bar_bins(bins) - bar_width(bins) / 2.

        for ci, class_label in enumerate(class_labels):
            # FIGURE 1: Histogram of correlation values
            mean_corr_hist = corr_hists[:, ci].mean(axis=0).flatten()
            std_corr_hist = corr_hists[:, ci].std(axis=0).flatten()
            ax5 = fh5.add_subplot(3, 3, ci + 1)
            ax5.bar(bar_edges(corr_bins),
                    mean_corr_hist * bar_width(corr_bins),
                    yerr=std_corr_hist * bar_width(corr_bins),
                    width=bar_width(corr_bins))
            ax5.set_title('Correlation (%s)' % class_label)
            y_max5 = bar_width(corr_bins) * (corr_hists.mean(0) +
                                             corr_hists.std(0)).max()
            ax5.set_ylim([0., y_max5 * 1.02])
            ax5.set_xlim(np.array([-1, 1]) * np.abs(bar_edges(corr_bins)[0]))

            # FIGURE 2: Histogram of p-values
            mean_pval_hist = pval_hists[:, ci].mean(axis=0).flatten()
            std_pval_hist = pval_hists[:, ci].std(axis=0).flatten()
            ax6 = fh6.add_subplot(3, 3, ci + 1)
            ax6.bar(bar_edges(pval_bins),
                    mean_pval_hist * bar_width(pval_bins),
                    yerr=std_pval_hist * bar_width(pval_bins),
                    width=bar_width(pval_bins))
            ax6.set_title('p-values (%s)' % class_label)
            y_max6 = bar_width(pval_bins) * (pval_hists.mean(0) +
                                             pval_hists.std(0)).max()
            ax6.set_ylim([0., y_max6 * 1.02])
            ax6.set_xlim([0., 1.0])

    # FIGURE 7: Plot the mean correlation matrix as an image.
    if 7 in visualize:
        fh7 = plt.figure(figsize=(14, 10))
        for subj_idx in range(n_subj + 1):
            if subj_idx < n_subj:
                mat = RDM_data[subj_idx]
                subj_id = str(subj_idx)
            else:
                mat = RDM_data.mean(axis=0)
                subj_id = 'mean'
            ax7 = fh7.add_subplot(3, 3, subj_idx + 1)
            ax7.imshow(1. - mat,
                       interpolation='nearest',
                       vmin=-1., vmax=1.)
            ax7.set_title('Subject %s dissimilarity' % subj_id)

            ax7.set_yticks(np.cumsum(n_imgperclass) - n_imgperclass / 2.)
            ax7.set_yticklabels(class_labels)
            ax7.set_xticks([])  # remove x-ticks

    # FIGURE 8: Plot haxby figure (ish)
    if 8 in visualize:
        confusion_mat = 1. - RDM_data.mean(axis=0)
        confusion_mat = confusion_mat
        short_class_labels = [lbl[:5] for lbl in class_labels]
        fh8 = plt.figure(figsize=(12, 10))
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
            ax8 = fh8.add_subplot(n_rows, n_cols, plt_order[ci])
            ax8.bar(np.arange(n_classes) - 0.5, bars_mean, yerr=bars_std)
            ax8.set_title(class_label)
            ax8.set_xticks(list(range(n_classes)))
            ax8.set_xticklabels(short_class_labels)
            ax8.set_ylim([-0.2, 1.0])
        fh8.subplots_adjust(hspace=0.4)


if __name__ == '__main__':
    # Directories for output images and shelve db's
    for dir_name in ['db']:
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

    # compute_best_detector
    # compute_detector
    group_examine_correlations(detector_fn=compute_best_detector,
                               n_subj=6,         # up to 6
                               visualize=[5, 6],
                               force=False,
                               radius=5.,
                               grouping='img',   # 'img' or 'class'
                               seeds_mask='vt')  # 'vt' or 'all'

    plt.show()
