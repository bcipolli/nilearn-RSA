"""
Group analysis of resting-state fMRI with ICA: CanICA
=====================================================

Similar to plot_canica_resting_state.py, BUT
this code separates into 10 different time-courses for each subject.

It then computes a "group" ICA on the 10 time-courses.  It
then generalizes those across subjects.

I want to test the following:
* sub-slices
* randomly shuffled (but shared!)

I also want to see if cutting at different points for differnet people give
better results (i.e. intrinsic periodicity, like in the ADHD paper that Lourdes
shared with me)

Also, could do group CanICA before time-cut CanICA (if we think there's a
shared timecourse)
"""

n_components = 20  # Number of CanICA components
n_subjects = 2  # ?
max_iter = 200

### Load ADHD rest dataset ####################################################
from nilearn import datasets

adhd_dataset = datasets.fetch_adhd(n_subjects)
func_filenames = adhd_dataset.func  # list of 4D nifti files for each subject

### Apply CanICA to each subject ##############################################################
import nibabel
import numpy as np
from nilearn.decomposition.canica import CanICA
from nilearn.image import index_img

loadings_imgs = []

for si, fn in enumerate(func_filenames):
    print("Loading and cutting up data for subject %d..." % si)
    subj_img = nibabel.load(fn)
    n_timesteps = subj_img.shape[-1]
    n_timecourses = np.floor(n_timesteps / float(n_components)).astype(int)
    idx = np.linspace(0, n_timesteps, n_timecourses, dtype=int)
    print "%d time slices, %d seconds per time slice" % (n_timecourses, idx[1] - idx[0])
    subj_img_list = [index_img(subj_img, range(idx[ii-1], idx[ii]))
                     for ii in range(1, n_timecourses)]

    print("Fitting data for subject %d..." % si)
    subj_canica = CanICA(n_components=n_components, smoothing_fwhm=6.,
                         memory="nilearn_cache", memory_level=5,
                         max_iter=max_iter,
                         threshold=3., verbose=10, random_state=0)
    subj_canica.fit(subj_img_list)

    print("Projecting subject data into computed components...")
    subj_loadings = subj_canica.transform(subj_img)
    subj_component_imgs = subj_canica.masker_.inverse_transform(subj_canica.components_)
    subj_loadings_imgs = []
    for time_series, component_volume in zip(subj_loadings.T, subj_component_imgs.get_data()):
        tiled_img_data = np.tile(component_volume[..., np.newaxis],
                                 len(time_series))
        loaded_time_series = tiled_img_data * time_series
        loading_img = nibabel.Nifti1Image(loaded_time_series,
                                          subj_img.get_affine())
        subj_loadings_imgs.append(loading_img)

    loadings_imgs.append(subj_loadings_imgs)
all_loadings_images = [img for subj_images in loadings_imgs
                       for img in subj_loadings_imgs]

### Apply CanICA across subjects ##############################################################
print("Fitting data over all subjects (%d images)..." % len(all_loadings_images))
canica = CanICA(n_components=n_components, smoothing_fwhm=6.,
                memory="nilearn_cache", memory_level=5,
                max_iter=max_iter,
                threshold=3., verbose=10, random_state=0)
canica.fit(all_loadings_images)
components_img = canica.masker_.inverse_transform(canica.components_)
nibabel.save(components_img, 'components_img.nii')

### Visualize the results #####################################################
# Show some interesting components
import matplotlib.pyplot as plt
from nilearn.plotting import plot_roi, plot_stat_map

fh = plt.figure(facecolor='w')
nrows = int(np.floor(np.sqrt(n_components)))
ncols = int(np.ceil(n_components / float(nrows)))

for ci in range(n_components):
    ax = fh.add_subplot(nrows, ncols, ci + 1)
    plot_stat_map(index_img(components_img, ci),
                  display_mode="z", title="IC %d" % ci, cut_coords=1,
                  colorbar=False, axes=ax)

plt.show()

