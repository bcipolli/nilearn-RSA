"""
Transformer for computing seeds signals.
"""

import numpy as np
from functools import partial

import nibabel
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals.joblib import Memory

from .. import _utils
from .. import image
from .. import masking
from .. import region
from .. import signal
from .._utils import logger, CacheMixin
from .._utils.niimg_conversions import check_niimg, is_img
from ..image import iter_img


def _signals_from_seeds(seeds, niimg, xform_fn, radius=None, mask_img=None,
                        loop_axis=True):
    """ Note: this function is sub-optimal for small radius

    Parameters
    ==========
    xform_fn: function handle

    loop_axis: int, optional

    """
    n_seeds = len(seeds)
    niimg = check_niimg(niimg)
    shape = niimg.get_data().shape
    affine = niimg.get_affine()
    if mask_img is not None:
        mask_img = check_niimg(mask_img, ensure_3d=True)
        mask_img = image.resample_img(mask_img, target_affine=affine,
                                      target_shape=shape[:3],
                                      interpolation='nearest')
        mask, _ = masking._load_mask_img(mask_img)
    signals = np.empty((shape[3], n_seeds))
    # Create an array of shape (3, array.shape) containing the i, j, k indices
    # in voxel space
    coords = np.vstack((np.indices(shape[:3]),
                        np.ones((1,) + shape[:3])))
    # Transform the indices into native space
    coords = np.tensordot(affine, coords, axes=[[1], [0]])[:3]

    if loop_axis == 0:
        # Compute square distance to the seed
        seeds = np.asarray(seeds)
        dists = np.zeros((n_seeds,) + coords.shape[1:])
        for si, seed in enumerate(seeds):
            tiled_seed = np.rollaxis(np.tile(seed, coords.shape[1:] + (1,)),
                                     -1, 0)
            dists[si] = ((coords - tiled_seed) ** 2).sum(axis=0)
        del tiled_seed
        del coords
        if radius is None or radius ** 2 < np.min(dists):
            dists_mask = (dists == np.min(dists))
        else:
            dists_mask = (dists <= radius ** 2)
        dists_mask = np.logical_and(mask, dists_mask)

        # Per slice
        for ii, img in enumerate(iter_img(niimg)):
            sphere_data = [img.get_data()[dists_mask[si]] for si in range(n_seeds)]
            signals[ii, :] = xform_fn(sphere_data)

    elif loop_axis == 1:
        for i, seed in enumerate(seeds):
            seed = np.asarray(seed)
            # Compute square distance to the seed
            dist = ((coords - seed[:, None, None, None]) ** 2).sum(axis=0)
            if radius is None or radius ** 2 < np.min(dist):
                dist_mask = (dist == np.min(dist))
            else:
                dist_mask = (dist <= radius ** 2)
            if mask_img is not None:
                dist_mask = np.logical_and(mask, dist_mask)
            if not dist_mask.any():
                raise ValueError('Seed #%i is out of the mask' % i)
            signals[:, i] = xform_fn(niimg.get_data()[dist_mask])

    return signals


def _compose_err_msg(msg, **kwargs):
    """Append key-value pairs to msg, for display.

    Parameters
    ==========
    msg: string
        arbitrary message
    kwargs: dict
        arbitrary dictionary

    Returns
    =======
    updated_msg: string
        msg, with "key: value" appended. Only string values are appended.
    """
    updated_msg = msg
    for k, v in kwargs.iteritems():
        if isinstance(v, basestring):
            updated_msg += "\n" + k + ": " + v

    return updated_msg


class NiftiSpheresMasker(BaseEstimator, TransformerMixin, CacheMixin):
    """Class for masking of Niimg-like objects using seeds.

    NiftiSpheresMasker is useful when data from given seeds should be
    extracted. Use case: Summarize brain signals from seeds that were
    obtained from prior knowledge.

    Parameters
    ==========
    seeds: Binary image, or list of triplet of coordinates in native space
        Seed definitions. Image, or list of coordinates of the seeds must
        be in the same space as the images (typically MNI or TAL).

    radius: float, optional.
        Indicates, in millimeters, the radius for the sphere around the seed.
        Default is None (signal is extracted on a single voxel).

    loop_axis: int, optional.
        Indicates whether the signal will be computed per seed (loop_axis=1)
        or per slice (loop_axis=0)

    xform_fn: function handle, optional.
        Indicates what transformation to compute on the spherical ball of
        voxels selected by each seed.  By default, the mean value is computed.
        The function should receive a 2D matrix of data, and output a 1D vector
        of signals.  If loop_axis=1, input size is
        (n_voxels_per_seed, n_img_slices); output size is n_img_slices.
        If loop_axis=0, input size is (n_voxels_per_seed, n_seeds); output
        size is n_seeds.

    smoothing_fwhm: float, optional
        If smoothing_fwhm is not None, it gives the full-width half maximum in
        millimeters of the spatial smoothing to apply to the signal.

    standardize: boolean, optional
        If standardize is True, the time-series are centered and normed:
        their mean is set to 0 and their variance to 1 in the time dimension.

    detrend: boolean, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    low_pass: False or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    high_pass: False or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    t_r: float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    memory: joblib.Memory or str, optional
        Used to cache the region extraction process.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    memory_level: int, optional
        Aggressiveness of memory caching. The higher the number, the higher
        the number of functions that will be cached. Zero means no caching.

    verbose: integer, optional
        Indicate the level of verbosity. By default, nothing is printed.

    See also
    ========
    nilearn.input_data.NiftiMasker
    """
    # memory and memory_level are used by CacheMixin.

    def __init__(self, seeds, radius=None, mask_img=None,
                 xform_fn=partial(np.mean, axis=0), loop_axis=1,
                 smoothing_fwhm=None, standardize=False, detrend=False,
                 low_pass=None, high_pass=None, t_r=None,
                 memory=Memory(cachedir=None, verbose=0), memory_level=1,
                 verbose=0):
        if is_img(seeds):
            self.seeds_img = check_niimg(seeds, ensure_3d=True)
            self.seeds = None
        else:
            self.seeds = seeds
            self.seeds_img = None  # compute via fit()
        self.mask_img = mask_img
        self.radius = radius
        self.xform_fn = xform_fn
        self.loop_axis = loop_axis

        # Parameters for _smooth_array
        self.smoothing_fwhm = smoothing_fwhm

        # Parameters for clean()
        self.standardize = standardize
        self.detrend = detrend
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.t_r = t_r

        # Parameters for joblib
        self.memory = memory
        self.memory_level = memory_level

        self.verbose = verbose

    def fit(self, X=None, y=None):
        """Prepare signal extraction from regions.

        All parameters are unused, they are for scikit-learn compatibility.
        """
        self.mask_img_ = self.mask_img

        # Got an image, compute the seeds.
        if self.seeds_img:
            self.seeds_img_ = check_niimg(self.seeds_img, ensure_3d=True)
            n_seeds = self.seeds_img_.get_data().sum()
            coords = zip(*(np.nonzero(self.seeds_img.get_data()) +
                           (np.ones((n_seeds,)).tolist(),)))
            coords = np.asarray(coords).T
            seeds_T = np.tensordot(self.seeds_img_.get_affine(), coords,
                                   axes=[[1], [0]])[:3]
            self.seeds_ = seeds_T.T

        # Got the seeds, place them in an image.
        else:
            # This is not elegant but this is the easiest way to test it.
            try:
                for seed in self.seeds:
                    assert(len(seed) == 3)
            except Exception as e:
                if self.verbose > 0:
                    print('Seeds not valid, error' + str(e))
                raise ValueError('Seeds must be a list of triplets of '
                                 'coordinates in native space.')
            self.seeds_ = self.seeds

            seeds_array = np.asarray(self.seeds_)
            if is_img(X):
                vol_shape = X.shape[:3]
                affine = X.get_affine()
                header = X.get_header()
            elif self.mask_img_:
                vol_shape = self.mask_img_.shape[:3]
                affine = self.mask_img_.get_affine()
                header = self.mask_img_.get_header()
            else:
                vol_shape = seeds_array.max(axis=0) + 1  # 0-indexed
                affine = np.eye(4)
                header = None

            seeds_volume = np.zeros(vol_shape, dtype=np.bool)
            seeds_volume[seeds_array.T[0], seeds_array.T[1], seeds_array.T[2]] = True
            assert seeds_volume.sum() == len(self.seeds_)

            self.seeds_img_ = nibabel.Nifti1Image(seeds_volume.astype(int),
                                                  affine=affine, header=header)

        return self

    def fit_transform(self, imgs, confounds=None):
        return self.fit().transform(imgs, confounds=confounds)

    def _check_fitted(self):
        if not hasattr(self, "seeds_"):
            raise ValueError('It seems that %s has not been fitted. '
                             'You must call fit() before calling transform().'
                             % self.__class__.__name__)

    def transform(self, imgs, confounds=None):
        """Extract signals from Nifti-like objects.

        Parameters
        ==========
        imgs: Niimg-like object
            See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
            Images to process. It must boil down to a 4D image with scans
            number as last dimension.

        confounds: array-like, optional
            This parameter is passed to signal.clean. Please see the related
            documentation for details.
            shape: (number of scans, number of confounds)

        Returns
        =======
        signals: 2D numpy.ndarray
            Signal for each region.
            shape: (number of scans, number of regions)

        """
        self._check_fitted()

        logger.log("loading images: %s" %
                   _utils._repr_niimgs(imgs)[:200], verbose=self.verbose)
        imgs = _utils.check_niimgs(imgs)

        if self.smoothing_fwhm is not None:
            logger.log("smoothing images", verbose=self.verbose)
            imgs = self._cache(image.smooth_img, func_memory_level=1)(
                imgs, fwhm=self.smoothing_fwhm)

        logger.log("extracting region signals", verbose=self.verbose)
        signals = self._cache(
            _signals_from_seeds, func_memory_level=1)(
                self.seeds_, imgs, radius=self.radius, mask_img=self.mask_img,
                xform_fn=self.xform_fn, loop_axis=self.loop_axis)

        logger.log("cleaning extracted signals", verbose=self.verbose)
        signals = self._cache(signal.clean, func_memory_level=1
                                     )(signals,
                                       detrend=self.detrend,
                                       standardize=self.standardize,
                                       t_r=self.t_r,
                                       low_pass=self.low_pass,
                                       high_pass=self.high_pass,
                                       confounds=confounds)
        return signals

    def inverse_transform(self, X):
        """Compute voxel signals from sphere-computed data

        Any mask given at initialization is taken into account.

        Parameters
        ==========
        X: 2D numpy.ndarray
            Signal for each region.
            shape: (number of scans, number of regions)

        Returns
        =======
        img: nibabel.Nifti1Image
            Signal for each voxel. shape: that of maps.
        """
        self._check_fitted()

        logger.log("computing image from signals", verbose=self.verbose)
        img = self._cache(masking.unmask, func_memory_level=1,
            )(X, self.seeds_img_)

        # Be robust again memmapping that will create read-only arrays in
        # internal structures of the header: remove the memmaped array
        try:
            img._header._structarr = np.array(img._header._structarr).copy()
        except:
            pass
        return img
