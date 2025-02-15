"""
Run to see the distribution of pixel intentities.
We want to clip to "reasonable" values.
"""

import os

import numpy as np
import seaborn as sns
import tifffile
from matplotlib import pyplot as plt


def compute_img_histogram(img_fp, _bins=256, _range=(0, 5000)):
    arr = tifffile.imread(img_fp)
    hist, bins = np.histogram(arr.flatten(), bins=_bins, range=_range, density=True)
    return hist, bins[:-1]


def read_img_histograms(my_dir, _bins=256, _range=(0, 5000)):
    hist_ls = []
    bins_ls = []
    fp_ls = []
    for fp in os.listdir(my_dir):
        print(fp)
        hist, bins = compute_img_histogram(
            os.path.join(my_dir, fp, "registration", "3_trimmed.tif"), _bins=_bins, _range=_range
        )
        hist_ls.append(hist)
        bins_ls.append(bins)
        fp_ls.append(fp)
    return hist_ls, bins_ls, fp_ls


def plot_img_histograms(hist_ls, bins_ls, fp_ls):
    fig, ax = plt.subplots(figsize=(10, 5))
    for hist, bins, fp in zip(hist_ls, bins_ls, fp_ls):
        sns.lineplot(x=bins, y=hist, label=fp, alpha=0.3, ax=ax)
    ax.legend().remove()
    return fig


def run_plot_img_histograms(my_dir):
    hist_ls, bins_ls, fp_ls = read_img_histograms(my_dir)
    fig = plot_img_histograms(hist_ls, bins_ls, fp_ls)
    fig.show()
    return fig
