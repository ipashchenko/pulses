import numpy as np
from scipy.stats import rayleigh
from skimage.measure import regionprops
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.measurements import label
from sklearn.cluster import DBSCAN
from astropy.modeling import models, fitting
import matplotlib
matplotlib.use('Agg')


vint = np.vectorize(int)
vround = np.vectorize(round)


class NoIntensityRegionException(Exception):
    pass


def circular_mean(data, radius):
    """
    :param data:
    :param radius:
    :return:
    """
    from scipy.ndimage.filters import generic_filter as gf
    from skimage.morphology import disk

    kernel = disk(radius)
    return gf(data, np.mean, footprint=kernel)


def gaussian_filter(data, sigma):
    from skimage.filters import gaussian_filter as gf
    return gf(data, sigma)


def circular_median(data, radius):
    """
    :param data:
    :param radius:
    :return:
    """
    from scipy.ndimage.filters import generic_filter as gf
    from skimage.morphology import disk

    kernel = disk(radius)
    return gf(data, np.median, footprint=kernel)


def infer_gaussian(data):
    """
    Return (amplitude, x_0, y_0, width), where width - rough estimate of
    gaussian width
    """
    amplitude = data.max()
    x_0, y_0 = np.unravel_index(np.argmax(data), np.shape(data))
    row = data[x_0, :]
    column = data[:, y_0]
    x_0 = float(x_0)
    y_0 = float(y_0)
    dx = len(np.where(row - amplitude/2 > 0)[0])
    dy = len(np.where(column - amplitude/2 > 0)[0])
    width = np.sqrt(dx ** 2. + dy ** 2.)

    return amplitude, x_0, y_0, width


def get_props(image, threshold=None):
    """
    Return measured properties list of imaged labeled at specified threshold.

    :param image:
        Numpy 2D array with image.
    :param threshold: (optional)
        Threshold to label image. [0.-100.]. If ``None`` then don't threshold.
        (default: ``None``)

    :return:
        List of ``skimage.measure._regionprops._RegionProperties`` instances.
    """
    threshold = np.percentile(image.ravel(), threshold)
    a = image.copy()
    if threshold is not None:
        # Keep only tail of image values distribution with signal
        a[a < threshold] = 0
    s = generate_binary_structure(2, 2)
    # Label image
    labeled_array, num_features = label(a, structure=s)
    return regionprops(labeled_array, intensity_image=image)


def fit_elliplse(prop, plot=False, save_file=None, colorbar_label=None,
                 close=False, show=True):
    """
    Fit 2D ellipses to part of image image represented by
    ``skimage.measure._regionprops._RegionProperties`` instance.

    :param prop:
        ``skimage.measure._regionprops._RegionProperties`` instance
    :return:
        Instance of ``astropy.modelling.functional_models.Gaussian2D`` class
        fitted to `t-DM` image in region of ``prop``.
    """
    data = prop.intensity_image.copy()

    # Remove high-intensity background
    try:
        data -= np.unique(sorted(data.ravel()))[1]
    except IndexError:
        raise NoIntensityRegionException("No intensity in region!")
    data[data < 0] = 0

    # Make some initial guess based on fitting by method of moments.
    amp, x_0, y_0, width = infer_gaussian(data)
    x_lims = [0, data.shape[0]]
    y_lims = [0, data.shape[1]]
    g = models.Gaussian2D(amplitude=amp, x_mean=x_0, y_mean=y_0,
                          x_stddev=0.5 * width, y_stddev=0.5 * width,
                          theta=0, bounds={'x_mean': x_lims, 'y_mean': y_lims})
    fit_g = fitting.LevMarLSQFitter()
    x, y = np.indices(data.shape)
    gg = fit_g(g, x, y, data)

    if plot:
        fig, ax = matplotlib.pyplot.subplots(1, 1)
        ax.hold(True)
        im = ax.matshow(data, cmap=matplotlib.pyplot.cm.jet)
        model = gg.evaluate(x, y, gg.amplitude, gg.x_mean, gg.y_mean,
                            gg.x_stddev, gg.y_stddev, gg.theta)
        try:
            ax.contour(y, x, model, colors='w')
        except ValueError:
            print "Can't plot contours"
        ax.set_xlabel('t steps')
        ax.set_ylabel('DM steps')
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="10%", pad=0.00)
        cb = fig.colorbar(im, cax=cax)
        if colorbar_label is not None:
            cb.set_label(colorbar_label)
        if save_file is not None:
            fig.savefig(save_file, bbox_inches='tight', dpi=200)
        if show:
            fig.show()
        if close:
            matplotlib.pyplot.close()

    return gg


# FIXME: When # amplitudes is small enough, ``eps`` becomes too large...
def find_clusters_ell_amplitudes(amplitudes, min_samples=10, leaf_size=5,
                                 eps=None):
    """

    :param amplitudes:
    :param eps:
        `eps` parameter of `DBSCAN`.
    :param min_samples:
        `min_samples` parameter of `DBSCAN`.
    :param leaf_size:
        `leaf_size` parameter of `DBSCAN`.
    :return:
        Threshold for amplitude. Chosen in a way that fitted elliptical
        gaussians with amplitude higher then the threshold should be outliers
        (ie. represent signals).
    """

    data = np.asarray(amplitudes).copy()
    ldata = np.log(data)
    data_ = data.reshape((data.size, 1))
    ldata_ = ldata.reshape((ldata.size, 1))

    data_range = np.max(data) - np.min(data)
    if eps is None:
        eps = data_range / np.sqrt(len(amplitudes))
        print "eps {}".format(eps)
    db = DBSCAN(eps=eps, min_samples=min_samples,
                leaf_size=leaf_size).fit(data_)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    unique, unique_counts = np.unique(labels, return_counts=True)
    largest_cluster_data = data[labels == unique[np.argmax(unique_counts)]]
    outliers = data[labels == -1]
    params = rayleigh.fit(largest_cluster_data)
    distr = rayleigh(loc=params[0], scale=params[1])
    threshold = distr.ppf(0.999)

    # Need data > threshold to ensure that low power outliers haven't been
    # included
    indx = np.logical_and(labels == -1, data > threshold)
    # return threshold
    return min(data[indx]) - eps


    def plot_2d(array, bbox=None, colorbar_label=None, close=False, save_file=None,
                show=True, xlabel=None, ylabel=None, ):
        """
        Plot [part of the] 2D array.

        :param array:
            2D array to plot.

        :param bbox: (optional)
            Bounding box of region to plot (x1, y1, x2, y2) - ``prop.bbox``. If ``None``
            then plot all.
        """
        fig, ax = matplotlib.pyplot.subplots(1, 1)
        ax.hold(True)
        if bbox is not None:
            data = array[bbox[0]: bbox[2], bbox[1]: bbox[3]]
        else:
            data = array
        im = ax.matshow(data, aspect='auto', cmap=matplotlib.pyplot.cm.jet)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="10%", pad=0.00)
        cb = fig.colorbar(im, cax=cax)
        if colorbar_label is not None:
            cb.set_label(colorbar_label)
        if save_file is not None:
            fig.savefig(save_file, bbox_inches='tight', dpi=200)
        if show:
            fig.show()
        if close:
            matplotlib.pyplot.close()
