"""
RSA on haxby dataset.
"""

import matplotlib.pyplot as plt
import numpy as np
from functools import partial

import nibabel
from scipy.spatial.distance import pdist

from nilearn import datasets
from nilearn._utils import concat_niimgs
from nilearn._utils.cache_mixin import cache
from nilearn.image import index_img, mean_img
from nilearn.input_data import NiftiMasker, NiftiSpheresMasker
from nilearn.plotting import plot_roi, plot_stat_map, plot_mosaic_stat_map


def average_data(func_img, stim_labels, labels):
    img_list = []
    for stim_label in labels:
        stim_img = index_img(func_img, stim_labels == stim_label)
        img_list.append(mean_img(stim_img))
    return concat_niimgs(img_list)


class RsaSearchlight(object):

    def __init__(self, mask_img, seeds_img, memory_params=None):
        # Defs
        self.memory_params = memory_params or dict()
        self.seeds_img = seeds_img
        self.mask_img = mask_img

    def rsa_on_ball_axis_1(self, sphere_data):
        """
        Data: axis=1: [nvoxels, nslices]
        """
        similarity_data = pdist(sphere_data.T, 'correlation')
        self.RSA_data[self.si, :] = similarity_data
        self.si += 1

        if self.memory_params.get('verbose', 0) > 1 and self.si % 100 == 99:
            print 'Processed %s of %s...' % (self.si + 1, self.n_seeds)
        return similarity_data.std()  # output value for all slices

    def fit(self):
        # Create mask
        print("Fit the SphereMasker...")

        self.n_seeds = int(self.seeds_img.get_data().sum())

        self.sphere_masker = NiftiSpheresMasker(
            seeds=self.seeds_img, mask_img=self.seeds_img, radius=5,
            xform_fn=self.rsa_on_ball_axis_1, standardize=False)  # no mem
        self.sphere_masker.fit()

    def transform(self, func_img, anat_img=None, labels=None):
        print("Transforming the image...")

        # Store them off as associated with the RSA_* data
        self.func_img = func_img
        self.anat_img = anat_img
        self.labels = labels

        n_images = self.func_img.shape[3]
        n_compares = n_images * (n_images - 1) / 2

        self.si = 0
        self.RSA_data = np.empty((self.n_seeds, n_compares))

        # Compute the RSA vector.
        self.RSA_std = self.sphere_masker.transform(func_img)

        return self.RSA_data

    def visualize(self, func_img=None, anat_img=None, labels=None,
                  RSA_data=None, RSA_std=None):
        print("Plotting the results...")
        func_img = func_img or self.func_img
        anat_img = anat_img or self.anat_img
        labels = labels or self.labels
        RSA_data = RSA_data or self.RSA_data
        RSA_std = RSA_std or self.RSA_std

        plot_roi(self.sphere_masker.seeds_img_, bg_img=anat_img, title='seed img')
        plot_roi(self.sphere_masker.mask_img_, bg_img=anat_img, title='mask img')

        RSA_std_img = self.sphere_masker.inverse_transform(self.RSA_std[0])
        RSA_img = self.sphere_masker.inverse_transform(self.RSA_data.T)

        # Plot results
        plot_stat_map(RSA_std_img, bg_img=anat_img, title='RSA std')
        plot_stat_map(index_img(RSA_img, 0), bg_img=anat_img,
                      title='RSA comparison %s vs. %s' % tuple(labels[:2]))

        titles = []
        for ai, label1 in enumerate(labels):
            for bi, label2 in enumerate(labels[(ai + 1):]):
                titles.append('%s vs. %s' % (label1, label2))

        fh = plt.figure(figsize=(18, 10))
        plot_mosaic_stat_map(RSA_img, colorbar=False, display_mode='z',
                             bg_img=anat_img, cut_coords=1, figure=fh,
                             title=titles)

        # faces compares
        # contains_faces = np.asarray(['face' in t for t in titles])
        # fh = plt.figure(figsize=(18, 10))
        # plot_mosaic_stat_map(index_img(RSA_img, contains_faces),
        #                      colorbar=False, figure=fh,
        #                      display_mode='z', bg_img=anat_img, cut_coords=1,
        #                      title=np.asarray(titles)[contains_faces],
        #                      shape=(4, 2))


class SearchlightAnalysis(object):
    def __init__(self, dataset, subj_idx=0, memory_params=None):
        self.dataset = dataset
        self.subj_idx = subj_idx
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
        self.func_img = nibabel.load(dataset_files.func[self.subj_idx])
        self.vt_mask_img = nibabel.load(dataset_files.mask_vt[self.subj_idx])
        self.anat_img = (dataset_files.anat[self.subj_idx] and
                         nibabel.load(dataset_files.anat[self.subj_idx]))
        self.metadata = np.recfromcsv(dataset_files.session_target[self.subj_idx],
                                      delimiter=" ")
        self.stim_labels = self.metadata['labels']
        self.labels = np.unique(self.stim_labels).tolist()
        # sessions = metadata['chunks']

        # Compute mask
        print("Computing mask...")
        self.masker = NiftiMasker(mask_strategy='epi', detrend=False,
                                  smoothing_fwhm=None, standardize=True,
                                  **self.memory_params)
        self.masker.fit(self.func_img)
        self.mask_img = self.masker.mask_img_

    def transform(self, seeds_img=None):
        seeds_img = seeds_img or self.vt_mask_img

        X = self.masker.transform(self.func_img)
        self.func_img = self.masker.inverse_transform(X)

        # Average across sessions
        print("Averaging data...")
        self.func_img = self.my_cache(average_data)(self.func_img,
                                                    self.stim_labels,
                                                    self.labels)
        self.searchlight = RsaSearchlight(mask_img=self.mask_img,
                                          seeds_img=seeds_img,
                                          memory_params=self.memory_params)
        self.searchlight.fit()
        self.searchlight.transform(func_img=self.func_img)

    def save(self, outfile):
        # Dump output file
        sphere_masker = self.searchlight.sphere_masker
        RSA_img = sphere_masker.inverse_transform(self.searchlight.RSA_data.T)
        nibabel.save(RSA_img, outfile)

    def visualize(self):

        # Functional image
        fh = plt.figure(figsize=(18, 10))
        plot_mosaic_stat_map(self.func_img,
                             bg_img=self.anat_img, title=self.labels,
                             figure=fh, shape=(5, 2))

        # Plot results
        self.searchlight.visualize(func_img=self.func_img,
                                   anat_img=self.anat_img,
                                   labels=self.labels)


if __name__ == 'main':
    subj_idx = 0
    analysis = SearchlightAnalysis('haxby', subj_idx=subj_idx)
    analysis.fit()
    analysis.transform()
    analysis.save('haxby_RSA_searchlight_subj%02d.nii' % subj_idx)
    analysis.visualize()
    plt.show()
