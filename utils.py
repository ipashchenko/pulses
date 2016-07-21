import numpy as np


vint = np.vectorize(int)
vround = np.vectorize(round)


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


def get_props(image, threshold):
    """
    Rerurn measured properties list of imaged labeled at specified threshold.
    :param image:
        Numpy 2D array with image.
    :param threshold:
        Threshold to label image. [0.-100.]
    :return:
        List of RegionProperties -
        (``skimage.measure._regionprops._RegionProperties`` instances)
    """
    threshold = np.percentile(image.ravel(), threshold)
    a = image.copy()
    # Keep only tail of image values distribution with signal
    a[a < threshold] = 0
    s = generate_binary_structure(2, 2)
    # Label image
    labeled_array, num_features = label(a, structure=s)
return regionprops(labeled_array, intensity_image=image)


def fit_elliplse(prop, plot=False, save_file=None, colorbar_label=None,
                 close=False, show=True):
    """
    Function that fits 2D ellipses to `t-DM` image.
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
    amp, x_0, y_0, width = infer_gaussian(data)
    x_lims = [0, data.shape[0]]
    y_lims = [0, data.shape[1]]
    g = models.Gaussian2D(amplitude=amp, x_mean=x_0, y_mean=y_0,
                          x_stddev=0.5 * width, y_stddev=0.5 * width,
                          theta=0, bounds={'x_mean': x_lims, 'y_mean': y_lims})
    fit_g = fitting.LevMarLSQFitter()
    x, y = np.indices(data.shape)
    gg = fit_g(g, x, y, data)
    # print gg.x_stddev, gg.y_stddev
    # print abs(gg.x_stddev), abs(gg.y_stddev / gg.x_stddev),\
    #     np.rad2deg(gg.theta) % 180

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

