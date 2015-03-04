"""
RSA on haxby dataset.
"""

import matplotlib.pyplot as plt
import numpy as np
from functools import partial

import nibabel
from scipy.spatial.distance import pdist

from nilearn import datasets
from nilearn._utils.cache_mixin import cache
from nilearn.image import concat_imgs, index_img, mean_img
from nilearn.input_data import NiftiMasker, NiftiSpheresMasker
from nilearn.plotting import plot_roi, plot_stat_map, plot_mosaic_stat_map


def get_class_indices(class_labels, img_labels):
    # Find desired label indices
    img_class_idx = []  # can vary in length
    for ci, class_label in enumerate(class_labels):
        img_mask = img_labels == class_label
        img_class_idx.append(np.nonzero(img_mask)[0])
    return img_class_idx


def average_data(grouping, func_img, ordered_class_labels, stim_labels, sessions=None):
    if sessions is not None:
        sess_ids = np.unique(sessions)

    img_list = []
    if grouping == 'class':
        for class_label in ordered_class_labels:
            stim_idx = stim_labels == class_label
            stim_img = index_img(func_img, stim_idx)
            img_list.append(mean_img(stim_img))
        img_labels = ordered_class_labels

    elif grouping == 'img':
        n_sess = len(sess_ids)
        n_exemplars = stim_labels.shape[0] / n_sess

        idx = np.empty((n_exemplars, n_sess), dtype=int)
        for sidx, sess_id in enumerate(sess_ids):
            idx[:, sidx] = np.nonzero(sessions == sess_id)[0]
            idx[:, sidx] = idx[np.argsort(stim_labels[idx[:, sidx]]), sidx]
        img_labels = []
        for lidx in range(n_exemplars):
            #assert len(np.unique(stim_labels[idx[lidx]])) == 1, \
            #    ("All stimuli at the same position across sessions should be "
            #     "of the same stimulus class.")
            stim_img = index_img(func_img, idx[lidx].astype(int))
            img_list.append(mean_img(stim_img))
            img_labels.append(stim_labels[idx[lidx]][0])  # just one label needed.

    else:
        raise ValueError('Unrecognized grouping: %s' % grouping)

    return concat_imgs(img_list), np.asarray(img_labels)


def reorder_haxby_labels(stim_labels, sessions):
    # output sorted_idx is a relative index (relative to all images
    #    from within a session)
    sorted_class_labels = np.array(['face', 'house', 'cat', 'bottle', 'scissors',
                                    'shoe', 'chair', 'scrambledpix', 'rest'])
    sorted_stim_labels = []
    sorted_idx = []
    for sess_idx in np.unique(sessions):
        img_labels = stim_labels[sessions == sess_idx]
        sorted_idx.append(get_class_indices(sorted_class_labels, img_labels))
        for idx in sorted_idx[-1]:
            sorted_stim_labels += img_labels[idx]

    return sorted_class_labels, np.asarray(sorted_stim_labels), sorted_idx


class RsaSearchlight(object):

    def __init__(self, mask_img, seeds_img, radius=10., distance_method='correlation',
                 memory_params=None):
        # Defs
        self.memory_params = memory_params or dict()
        self.seeds_img = seeds_img
        self.mask_img = mask_img
        self.radius = radius
        self.distance_method = distance_method

    def rsa_on_ball_axis_1(self, sphere_data):
        """
        Data: axis=1: [nvoxels, nslices]
        """

        # sphere_data could be a single voxel; in this case, we'll get
        # nan
        similarity_comparisons = pdist(sphere_data.T, self.distance_method)
        self.similarity_comparisons[self.si, :] = similarity_comparisons
        self.n_voxels[self.si] = sphere_data.shape[0]
        self.si += 1

        if self.memory_params.get('verbose', 0) > 1 and self.si % 100 == 99:
            print 'Processed %s of %s...' % (self.si + 1, self.n_seeds)
        return similarity_comparisons.std()  # output value for all slices

    def fit(self):
        # Create mask
        print("Fit the SphereMasker...")

        self.n_seeds = int(self.seeds_img.get_data().sum())

        # Pass our xform_fn for a callback on each seed.
        self.sphere_masker = NiftiSpheresMasker(
            seeds=self.seeds_img, mask_img=self.seeds_img, radius=self.radius,
            xform_fn=self.rsa_on_ball_axis_1, standardize=False)  # no mem
        self.sphere_masker.fit()

    def transform(self, func_img):
        print("Transforming the image...")

        n_images = func_img.shape[3]
        n_compares = n_images * (n_images - 1) / 2

        # These are computed within the callback.
        self.si = 0
        self.n_voxels = np.empty((self.n_seeds))
        self.similarity_comparisons = np.empty((self.n_seeds, n_compares))

        similarity_std = self.sphere_masker.transform(func_img)

        # Pull the values off of self, set locally.
        n_voxels = self.n_voxels
        similarity_comparisons = self.similarity_comparisons
        delattr(self, 'si')
        delattr(self, 'n_voxels')
        delattr(self, 'similarity_comparisons')

        return similarity_comparisons, similarity_std, n_voxels

    def visualize(self, similarity_comparisons, similarity_std=None,
                  anat_img=None, labels=None):
        print("Plotting the results...")

        self.visualize_seeds(anat_img=anat_img)
        self.visualize_mask(anat_img=anat_img)
        self.visualize_comparisons(similarity_comparisons=similarity_comparisons,
                                   labels=labels, anat_img=anat_img)
        self.visualize_comparisons_std(similarity_std=similarity_std,
                                       anat_img=anat_img)

    def visualize_seeds(self, anat_img=None):
        plot_roi(self.sphere_masker.seeds_img_, bg_img=anat_img, title='seed img')

    def visualize_mask(self, anat_img=None):
        plot_roi(self.sphere_masker.mask_img_, bg_img=anat_img, title='mask img')

    def visualize_comparisons(self, similarity_comparisons, labels=None,
                              anat_img=None):
        # Plot (up to) twenty comparisons.
        plotted_similarity = similarity_comparisons[:, 0]
        plotted_img = self.sphere_masker.inverse_transform(plotted_similarity.T)
        plot_stat_map(plotted_img, bg_img=anat_img,
                      title='RSA comparison %s vs. %s' % tuple(labels[:2]))

        # Plot mosaic of up to 20

        # Choose the comparisons
        idx = np.linspace(0, similarity_comparisons.shape[1] - 1, 20)
        idx = np.unique(np.round(idx).astype(int))  # if there's less than 20

        # Make (and filter) titles
        if labels is None:
            titles = None
        else:
            titles = []
            for ai, label1 in enumerate(labels):
                for bi, label2 in enumerate(labels[(ai + 1):]):
                    titles.append('%s vs. %s' % (label1, label2))
            titles = np.asarray(titles)[idx]

        # Create the image
        plotted_similarity = similarity_comparisons[:, idx]
        plotted_img = self.sphere_masker.inverse_transform(plotted_similarity.T)

        fh = plt.figure(figsize=(18, 10))
        plot_mosaic_stat_map(plotted_img, colorbar=False, display_mode='z',
                             bg_img=anat_img, cut_coords=1, figure=fh,
                             title=titles)

    def visualize_comparisons_std(self, similarity_std, anat_img=None):
        if similarity_std is not None:
            RSA_std_img = self.sphere_masker.inverse_transform(similarity_std[0])
            plot_stat_map(RSA_std_img, bg_img=anat_img, title='RSA std')

        # faces compares
        # contains_faces = np.asarray(['face' in t for t in titles])
        # fh = plt.figure(figsize=(18, 10))
        # plot_mosaic_stat_map(index_img(RSA_img, contains_faces),
        #                      colorbar=False, figure=fh,
        #                      display_mode='z', bg_img=anat_img, cut_coords=1,
        #                      title=np.asarray(titles)[contains_faces],
        #                      shape=(4, 2))


class HaxbySearchlightAnalysis(object):
    def __init__(self, dataset='haxby', subj_idx=0, memory_params=None,
                 radius=10., smoothing_fwhm=None, standardize=True,
                 grouping='class'):
        self.dataset = dataset
        self.subj_idx = subj_idx
        self.radius = radius
        self.grouping = grouping

        # Masking params
        self.smoothing_fwhm = smoothing_fwhm
        self.standardize = standardize

        # Caching stuff
        self.memory_params = memory_params or dict(memory='nilearn_cache',
                                                   memory_level=10,
                                                   verbose=10)
        self.my_cache = partial(cache, func_memory_level=0,
                                **self.memory_params)

    def fit(self):
        # Get data
        print("Loading data...")
        dataset_fn = getattr(datasets, 'fetch_%s' % self.dataset)
        dataset_files = dataset_fn(n_subjects=self.subj_idx + 1)
        metadata = np.recfromcsv(dataset_files.session_target[self.subj_idx],
                                 delimiter=" ")
        self.func_img = nibabel.load(dataset_files.func[self.subj_idx])
        self.vt_mask_img = nibabel.load(dataset_files.mask_vt[self.subj_idx])
        self.anat_img = (dataset_files.anat[self.subj_idx] and
                         nibabel.load(dataset_files.anat[self.subj_idx]))
        self.stim_labels = np.asarray(metadata['labels'])
        self.class_labels = np.unique(self.stim_labels)
        self.sessions = np.asarray(metadata['chunks'])
        self.func_img.get_data()  # force load

        # Delete bad images.
        if self.subj_idx == 4:
            # Note: data for the run 9 (chunk 8) of subject 5 was corrupted
            # (from http://dev.pymvpa.org/datadb/haxby2001.html)
            bad_idx = self.sessions == 8
            self.func_img = index_img(self.func_img, np.logical_not(bad_idx))
            self.stim_labels = self.stim_labels[np.logical_not(bad_idx)]
            self.sessions = self.sessions[np.logical_not(bad_idx)]

        # Compute mask
        print("Computing mask...")
        self.masker = NiftiMasker(mask_strategy='epi', detrend=False,
                                  smoothing_fwhm=self.smoothing_fwhm,
                                  standardize=self.standardize,
                                  **self.memory_params)
        self.masker.fit(self.func_img)
        self.mask_img = self.masker.mask_img_

    def transform(self, seeds_img=None):
        seeds_img = seeds_img or self.vt_mask_img

        X = self.masker.transform(self.func_img)
        self.func_img = self.masker.inverse_transform(X)

        # Reordering data
        print("Sorting data...")
        self.class_labels, self.stim_labels, sorted_idx \
            = reorder_haxby_labels(self.stim_labels, self.sessions)
        imgs = []
        for sess_idx, sess_id in zip(sorted_idx, np.unique(self.sessions)):
            for img_idx in sess_idx:
                # Convert the relative img_idx (to the session start) into
                # absolute stim_idx (over all sessions)
                stim_idx = np.nonzero(self.sessions == sess_id)[0][img_idx]
                imgs += nibabel.four_to_three(index_img(self.func_img, stim_idx))
        self.func_img = concat_imgs(imgs)

        # Average across sessions
        print("Averaging data...")
        self.func_img, self.img_labels = self.my_cache(average_data)(
            grouping=self.grouping,
            func_img=self.func_img,
            ordered_class_labels=self.class_labels,
            stim_labels=self.stim_labels,
            sessions=self.sessions)
        self.searchlight = RsaSearchlight(mask_img=self.mask_img,
                                          seeds_img=seeds_img,
                                          memory_params=self.memory_params,
                                          radius=self.radius)
        self.searchlight.fit()
        self.similarity_comparisons, self.similarity_std, self.n_voxels = \
            self.searchlight.transform(func_img=self.func_img)

    def save(self, outfile):
        # Dump output file
        sphere_masker = self.searchlight.sphere_masker
        RSA_img = sphere_masker.inverse_transform(self.searchlight.RSA_data.T)
        nibabel.save(RSA_img, outfile)

    def visualize(self):
        self.visualize_func_img()

        # Plot results
        self.searchlight.visualize(self.similarity_comparisons,
                                   self.similarity_std,
                                   anat_img=self.anat_img,
                                   labels=self.img_labels)

    def visualize_func_img(self):
        return
        # Functional image
        fh = plt.figure(figsize=(18, 10))
        class_img = self.my_cache(average_data)(grouping='class',
                                                func_img=self.func_img,
                                                ordered_class_labels=self.class_labels,
                                                img_labels=self.img_labels)[0]
        plot_mosaic_stat_map(class_img,
                             bg_img=self.anat_img, title=self.class_labels,
                             figure=fh, shape=(5, 2))


if __name__ == 'main':
    subj_idx = 0
    analysis = HaxbySearchlightAnalysis(subj_idx=subj_idx)
    analysis.fit()
    analysis.transform()
    analysis.save('haxby_RSA_searchlight_subj%02d.nii' % subj_idx)
    analysis.visualize()
    plt.show()
