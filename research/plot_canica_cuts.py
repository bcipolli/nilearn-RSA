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

n_components = 42  # Number of CanICA components
n_subjects = 40  # ?
max_iter = 200
max_timecourses = 1

### Load ADHD rest dataset ####################################################
from nilearn import datasets

adhd_dataset = datasets.fetch_adhd(n_subjects)
func_filenames = adhd_dataset.func  # list of 4D nifti files for each subject

### Apply CanICA to each subject ##############################################################
import nibabel
import numpy as np
from nilearn.decomposition.canica import CanICA
from nilearn.image import index_img, iter_img

all_images = []

for si, fn in enumerate(func_filenames):
    print("Loading and cutting up data for subject %d..." % si)
    subj_img = nibabel.load(fn)
    n_timesteps = subj_img.shape[-1]
    n_timecourses = np.min([max_timecourses or np.inf,
                            int(np.floor(n_timesteps / float(n_components)))]).astype(int)
    idx = np.linspace(0, n_timesteps, n_timecourses + 1, dtype=int)
    print "%d time slices, %d seconds per time slice" % (n_timecourses, idx[1] - idx[0])
    subj_img_list = [index_img(subj_img, range(idx[ii], idx[ii + 1]))
                     for ii in range(n_timecourses)]

    all_images += subj_img_list

### Apply CanICA across subjects ##############################################################
print("Fitting data over all subjects (%d images)..." % len(all_images))
canica = CanICA(n_components=n_components, smoothing_fwhm=6.,
                memory="nilearn_cache", memory_level=5,
                max_iter=max_iter,
                threshold=3., verbose=10, random_state=0)
canica.fit(all_images)
components_img = canica.masker_.inverse_transform(canica.components_)
nibabel.save(components_img, 'components_img_cuts.nii')

### Visualize the results #####################################################
# Show some interesting components
import matplotlib.pyplot as plt
from nilearn.plotting import plot_roi, plot_stat_map, plot_glass_brain, find_xyz_cut_coords

fh = plt.figure(facecolor='w', figsize=(18, 10))
nrows = int(np.floor(np.sqrt(0.75 * n_components)))  # 4:3 aspect
ncols = int(np.ceil(n_components / float(nrows)))

all_cut_coords = [find_xyz_cut_coords(img) for img in iter_img(components_img)]
sort_idx = np.argsort(np.array(all_cut_coords)[:, 2])
for ci in range(n_components):
    ax = fh.add_subplot(nrows, ncols, ci + 1)
    ic_idx = sort_idx[ci]
    cut_coords = all_cut_coords[ic_idx]
    var_explained = canica.variance_[ic_idx]
    plot_stat_map(index_img(components_img, ic_idx),
                  title="IC%02d (v=%.1f)" % (ic_idx, var_explained),
                  axes=ax, colorbar=False,
                  display_mode="z", cut_coords=(cut_coords[2],))

plt.show()
