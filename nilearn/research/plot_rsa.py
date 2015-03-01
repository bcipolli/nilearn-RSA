"""
Simple example of NiftiMasker use
==================================

Here is a simple example of automatic mask computation using the nifti masker.
The mask is computed and visualized.

Things to do:
1. Cluster by 3D spatial position (or with resting state data PLUS spatial position)
2. Downsample, compute similarity matrix, cluster based on similarity
   * Downsampling by pixel, or donsample based on #1's clustering
3. Compute & plot a single pixel's similarity map
4. Get each of the clustering algorithms to work
"""

import matplotlib.pyplot as plt
import numpy as np
import time
from matplotlib.colors import LogNorm

import nibabel
import scipy
from scipy.misc import comb
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import WardAgglomeration, SpectralClustering
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.externals.joblib import Memory
from sklearn.feature_extraction import image as sk_image

from nilearn import input_data
from nilearn import datasets
from nilearn._utils.cache import cache
from nilearn.image import mean_img
from nilearn.plotting import plot_epi, plot_roi, plot_stat_map, cm


# Indexing function into pdist (will get you squareform)
idx_fn = lambda i, j, n: int(round(comb(n, 2) - comb(n-min(i, j), 2) +
                             (max(i, j) - min(i, j)-1))) if i != j else None
squareform_dynamic = lambda arr, i, j, orig_shape: \
    arr[idx_fn(i, j, orig_shape[0])] if i != j else 0.


def compute_label_values(X, labels, n_labels=None, memory=None, verbose=1):
    """

        Parameters
        -----------
        X: ndarray
            raw data
            
        labels: array
            labels for each raw data point
        n_labels: int, optional
            total # of labels

        Outputs
        -----------
        array, len == n_labels, containing the mean X value
            for each label.
    """
    if n_labels is None:
        n_labels = len(np.unique(labels))

    if len(X.shape) == 2:
        # We received a time-series; compute a value at each time point
        if verbose > 0:
            print("Compute labels at each time point (%d total)." % X.shape[0])
        n_samples = X.shape[0]
        vals = np.zeros((n_samples, n_labels))
        for si in np.arange(n_samples):
            if verbose > 0 and si % 100 == 0:
                print ("Processing timepoint # %d of %d ..." % (si + 1, n_samples))
            vals[si, :] = compute_label_values(X[si, :], labels=labels, n_labels=n_labels, memory=memory, verbose=verbose)
        return vals

    # X is a single time-slice
    def compute_label_values_single(X, labels, n_labels):
        assert len(X) == len(labels)
        vals = np.zeros((n_labels,))
        counts = np.zeros((n_labels,))
        for xi, label in enumerate(labels):
            counts[label] += 1
            # Online computation of the mean.
            vals[label] *= (counts[label] - 1.) / counts[label]
            vals[label] += float(X[xi])/counts[label]
        return vals
    return cache(compute_label_values_single, memory)(X, labels, n_labels)

def compute_voxel_values(X, labels, memory=None):
    """ Contour a 3D map in all the views.

        Parameters
        -----------
        X: ndarray
            
        labels: array
            
        Outputs
        -----------
        array, len == n_labels, containing the mean X value 
        for each label
    """
    if len(X.shape) == 2:
        n_samples = X.shape[0]
        vals = np.zeros((n_samples, len(labels)))
        for si in np.arange(n_samples):
            fn = cache(compute_voxel_values, memory) if memory else compute_voxel_values
            vals[si, :] = fn(X[si, :], labels)
        return

    n_labels = len(X)
    vals = np.zeros(labels.shape)
    for vi, label in enumerate(labels):
        vals[vi] = X[label]
    return vals


def compute_centroids(masker, labels, n_labels=None):
    """ Contour a 3D map in all the views.

        Parameters
        -----------
        masker: NiftiMasker
            
        labels: array
            
        n_labels: int, optional

        Outputs
        -----------
        array, len == n_labels, containing the mean X value 
        for each label
    """
    if n_labels is None:
        n_labels = len(np.unique(labels))
    counts = np.zeros((n_labels,))
    centroids = np.zeros((n_labels, 3))
    labels_img = masker.inverse_transform(labels + 1)

    for label in range(n_labels):
        img = (labels_img.get_data().astype(int) == label + 1)
        counts[label] = img.sum()
        idx = np.transpose(np.nonzero(img))
        centroids[label, :] = np.mean(idx, axis=0)
    return centroids


def compute_distance_from_centroids(masker, labels, centroids,
                                    n_labels=None, normalize=False):
    """ Contour a 3D map in all the views.

        Parameters
        -----------
        X: ndarray
            
        labels: array
            
        n_labels: int, optional

        Outputs
        -----------
        array, len == n_labels, containing the mean X value 
        for each label
    """
    if n_labels is None:
        n_labels = len(np.unique(labels))
    labels_img = masker.inverse_transform(labels + 1)
    dist_img = np.zeros(labels_img.get_data().shape)
    for label in range(n_labels):
        img = (labels_img.get_data().astype(int) == label + 1)
        idx = np.transpose(np.nonzero(img))
        dist = np.linalg.norm(idx - centroids[label], axis=1)
        if normalize and np.any(dist > 0.):
            dist = dist / dist.max()
        for pi in range(idx.shape[0]):
            dist_img[idx[pi, 0], idx[pi, 1], idx[pi, 2]] = dist[pi]
    return nibabel.Nifti1Image(dist_img, labels_img.get_affine())


def plot_compressed_samples(compressed_vol, labels, n_voxels,
                            masker, title="", memory=None):

    # Now convert those similarities back to uncompressed images
    uncompressed_images = []
    n_samples = compressed_vol.shape[0]
    for si in np.arange(n_samples):
        voxel_values = compute_voxel_values(compressed_vol[si, :],
                                            labels=labels,
                                            memory=memory)
        uncompressed_image = masker.inverse_transform(voxel_values)
        uncompressed_images.append(uncompressed_image)

    # Plot the final figure
    figure1 = plt.figure(figsize=(18, 9))
    rows_cols = len(uncompressed_images)
    for si, uncompressed_image in enumerate(uncompressed_images):

        # Stat map
        axes1 = figure1.add_subplot(rows_cols, 1, si + 1)
        data = uncompressed_image.get_data()
        stretched_data = data / np.abs(data).max()
        img = nibabel.Nifti1Image(stretched_data,
                                  affine=uncompressed_image.get_affine())
        plot_stat_map(img, display_mode='y',  # bg_img=mean_func_img,
                      title="%s %d (stats)" % (title, si+1),
                      figure=figure1, axes=axes1)

    figure2 = plt.figure(figsize=(18, 9))
    for si, uncompressed_image in enumerate(uncompressed_images):
        # EPI map
        plot_epi(uncompressed_image, figure=figure2,
                 axes=figure2.add_subplot(2, 2, si + 1),
                 title="%s %d (raw)" % (title, si), display_mode='xz',
                 colorbar=True,
                 vmin=-np.abs(uncompressed_image.get_data().max()),
                 vmax=np.abs(uncompressed_image.get_data().max()),
                 cmap=cm.cold_hot)


if __name__ == "__main__":

    # Running parameters
    subject_idx = 0  # in (0, 13)
    n_clusters = 250  # less than ~40,000
    example_time_idx = 196  # in (0, 196)
    n_display_samples = 4
    memory = 'nilearn_cache'
    plots = [1, 2, 3]
    # dataset_fn = datasets.fetch_nyu_rest
    dataset_fn = datasets.fetch_haxby

    # Load data
    dataset = dataset_fn(n_subjects=subject_idx + 1)
    func_filename = dataset.func[subject_idx]

    # This is resting-state data: the background has not been removed yet,
    # thus we need to use mask_strategy='epi' to compute the mask from the
    # EPI images
    nifti_masker = input_data.NiftiMasker(memory=memory, memory_level=1, verbose=10,
                                          mask_strategy='epi',
                                          standardize=False)
    # Mask the data
    fmri_masked = nifti_masker.fit_transform(func_filename)
    mask = nifti_masker.mask_img_.get_data().astype(np.bool)

    # Augment the features with spatial information.
    X = np.asarray(np.where(mask)).T.astype(float)
    factors = (0., 0., 0.)
    types = (float, float, float)
    for ai, axis in enumerate(range(3)):
        data_axis = X[:, axis]
        data_range = float(data_axis.max() - data_axis.min())
        hemi = (data_axis - np.median(data_axis)) / float(data_range)
        if types[ai] == bool:
            hemi = (hemi <= 0.).astype(float) - 0.5
        hemi = factors[ai] * hemi
        X = np.append(X.T, [hemi], axis=0).T
    X = np.append(X[:, 3:].T, fmri_masked, axis=0).T
    print(X.shape)
    # Compute a connectivity matrix (for constraining the clustering)
    connectivity = sk_image.grid_to_graph(n_x=mask.shape[0], n_y=mask.shape[1],
                                          n_z=mask.shape[2], mask=mask)

    # Cluster (#2)
    start = time.time()
    ward = WardAgglomeration(n_clusters=n_clusters, connectivity=connectivity, memory=memory)
    ward.fit(X.T)

    print("Ward agglomeration %d clusters: %.2fs" % (
        n_clusters, time.time() - start))

    # Compute an image with one ROI per label, and save to disk
    labels = ward.labels_ + 1    # Avoid 0 label - 0 means mask.
    labels_img = nifti_masker.inverse_transform(labels)
    labels_img.to_filename('parcellation.nii')

    # Plot image with len(labels) ROIs, and store
    #   the cut coordinates to reuse for all plots
    #   and the figure for plotting all to a common axis
    if 1 in plots:
        figure1 = plt.figure(figsize=(18, 9))
        mean_func_img = mean_img(func_filename, verbose=10)
        first_plot = plot_roi(labels_img, bg_img=mean_func_img,
                              axes=figure1.add_subplot(2, 2, 3),
                              title="Ward parcellation", display_mode='xz')
        cut_coords = first_plot.cut_coords

        # Plot an image of the original functional data, at
        #   the example_time_idx timepoint.
        plot_epi(nifti_masker.inverse_transform(fmri_masked[example_time_idx]),
                 cut_coords=cut_coords, axes=figure1.add_subplot(2, 2, 1),
                 title='Original (%i voxels)' % fmri_masked.shape[1],
                 display_mode='xz', colorbar=True)

    # A reduced data can be create by taking the parcel-level average:
    #   Note that, as many objects in the scikit-learn, the ward object exposes
    #   a transform method that modifies input features. Here it reduces their
    #   dimension by averaging data within a cluster.
    fmri_reduced = ward.transform(fmri_masked)

    # Plot an image of the data compressed using the parcellation
    fmri_clustered = ward.inverse_transform(fmri_reduced)
    clustered_img = nifti_masker.inverse_transform(fmri_clustered[0])

    if 1 in plots:
        plot_epi(clustered_img, cut_coords=cut_coords,
                 axes=figure1.add_subplot(2, 2, 2),
                 title='Clustered representation (%d clusters)' % n_clusters,
                 display_mode='xz', colorbar=True)

    # To visualize the centroids, compute an image of the distance
    #   from each voxel's associated centroid to that voxel's position.
    # Now, do true compression--reduce the number of samples.
    centroids = compute_centroids(nifti_masker, labels=ward.labels_,
                                  n_labels=n_clusters)
    dist_img = compute_distance_from_centroids(nifti_masker,
                                               labels=ward.labels_,
                                               centroids=centroids)

    # Plot an image of the distance to centroids
    if 1 in plots:
        plot_epi(dist_img, cut_coords=cut_coords,
                 axes=figure1.add_subplot(2, 2, 4), colorbar=True,
                 title='Distances from center', display_mode='xz')
    plt.show()
    # Compute the similarity matrix across the compressed
    fmri_compressed = compute_label_values(fmri_masked,
                                           labels=ward.labels_,
                                           n_labels=n_clusters,
                                           memory=memory,)

    # Compute the distance and similarity matrices
    X = fmri_compressed
    dist = pdist(X.T, metric='correlation')

    # Select 20 random samples (voxels, or clusters) to display
    n_samples = X.shape[1]
    sample_indices = np.random.choice(
        np.arange(n_samples), size=(n_display_samples,), replace=False)

    # Compute the full volume similarities just for the 20 random samples
    sample_similarities = np.zeros((n_display_samples, n_samples))
    for si, sample_idx in enumerate(sample_indices):
        for comparator_idx in np.arange(n_samples):
            # For example sample_idx, compare to every other sample.
            cur_dist = squareform_dynamic(dist, sample_idx,
                                          comparator_idx, X.T.shape)
            sample_similarities[si, comparator_idx] = 1. - cur_dist

    if 2 in plots:
        plot_compressed_samples(sample_similarities,
                                labels=ward.labels_,
                                n_voxels=fmri_masked.shape[1],
                                memory=memory,
                                masker=nifti_masker,
                                title="Similarities")

    pca = PCA(n_components=4, memory=memory)
    pca = pca.fit(squareform(dist))
    import pdb; pdb.set_trace()
    if 3 in plots:
        plot_compressed_samples(pca.components_,
                                labels=ward.labels_,  # np.arange(pca.n_components),
                                n_voxels=fmri_masked.shape[1],
                                memory=memory,
                                masker=nifti_masker,
                                title="PCA components")

    # Finally, show the plot
    plt.show()
