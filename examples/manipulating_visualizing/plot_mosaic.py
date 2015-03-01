import matplotlib.pyplot as plt

from nilearn import datasets
from nilearn.plotting import (plot_epi, plot_glass_brain, plot_roi,
                              plot_stat_map)
from nilearn.image import index_img

n_maps = 1

msdl_atlas_dataset = datasets.fetch_msdl_atlas()
maps_img = msdl_atlas_dataset.maps


for plot_fn in (plot_stat_map,):
    imgs = index_img(maps_img, slice(n_maps))
    plot_args = dict(figure=plt.figure(figsize=(12, 6)), colorbar=True)

    titles = ['Brain %d' % bi for bi in range(n_maps)]
    fh = plot_fn(imgs, title='', **plot_args)
plt.show()
