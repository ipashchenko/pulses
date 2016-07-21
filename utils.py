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
