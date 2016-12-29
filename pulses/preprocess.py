# -*- coding: utf-8 -*-
import os
import numpy as np
from scipy.ndimage.measurements import label
from scipy.ndimage.morphology import generate_binary_structure
from skimage.measure import regionprops
from skimage.morphology import opening
from utils import circular_mean, circular_median, gaussian_filter


class PreProcesser(object):
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self, dddsp, cache_dir=None):
        if cache_dir is None:
            cache_dir = os.getcwd()
        cache_fname = os.path.join(cache_dir,
                                    dddsp.dsp._cache_fname_prefix +
                                    "_preprocess.hdf5")
        dddsp.array = self.func(dddsp.array, *self.args, **self.kwargs)
        return dddsp

    def clear_cache(dddsp, cache_dir):
        pass


# FIXME: ``skimage.filters.median`` use float images with ranges ``[-1, 1]``. I
# can scale original, use ``median`` and then scale back - it is much faster
# then mine
def create_ellipses(tdm_image, disk_size=3, threshold_big_perc=97.5,
                    threshold_perc=None, statistic='mean',
                    opening_selem=np.ones((3, 3)), max_prop_size=25000):
    """
    Function that pre-process de-dispersed plane `t-DM` by filtering out BIG
    regions of high intensity and subsequent filtering, noise cleaning by
    opening and thresholding.
    :param tdm_image:
        2D numpy.ndarray  of `t-DM` plane.
    :param disk_size: (optional)
        Disk size to use when calculating filtered values. (default: ``3``)
    :param threshold_big_perc: (optional)
        Threshold [0. - 100.] to threshold image after filtering to find BIG
        regions that would be filtered out. (default: ``97.5``)
    :param threshold_perc: (optional)
        Threshold [0. - 100.] to threshold image after filtering BIG big regions
        out. (default: ``97.5``)
    :param statistic: (optional)
        Statistic to use when filtering (``mean``, ``median`` or ``gauss``).
        (default: ``mean``)
    :param opening_selem: (optional)
        The neighborhood expressed as a 2-D array of 1’s and 0’s for opening
        step. (default: ``np.ones((4, 4))``)
    :param max_prop_size: (optional)
        Maximum size of region to be filtered out from ``tdm_array``. (default:
        ``25000``)
    :return:
        2D numpy.ndarray of thresholded image of `t - DM` plane.
    """
    statistic_dict = {'mean': circular_mean, 'median': circular_median,
                      'gauss': gaussian_filter}

    if threshold_big_perc is not None:
        image = tdm_image.copy()
        image = statistic_dict[statistic](image, disk_size)
        threshold = np.percentile(image.ravel(), threshold_big_perc)
        image[image < threshold] = 0
        # FIXME: In ubuntu 16.04 this raises ``ValueError: Images of type float
        # must be between -1 and 1.`` Upgrading to newer ``skimage`` solved the
        # problem
        image = opening(image, opening_selem)

        # FInd BIG regions & exclude them in original ``tdm_image``. Then redo
        a = image.copy()
        s = generate_binary_structure(2, 2)
        # Label image
        labeled_array, num_features = label(a, structure=s)
        # Find objects
        props = regionprops(labeled_array, intensity_image=image)
        big_sized_props = list()
        for prop in props:
            if prop.area > max_prop_size:
                big_sized_props.append(prop)
        for prop in big_sized_props:
            bb = prop.bbox
            print "Filtering out region {} with area {}".format(bb, prop.area)
            tdm_image[bb[0]:bb[2], bb[1]:bb[3]] = np.mean(tdm_image)

    image = statistic_dict[statistic](tdm_image, disk_size)
    if threshold_perc is None:
        threshold_perc = threshold_big_perc
    threshold = np.percentile(image.ravel(), threshold_perc)
    image[image < threshold] = 0
    image = opening(image, opening_selem)

    return image
