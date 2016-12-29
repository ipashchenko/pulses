# -*- coding: utf-8 -*-
import numpy as np
from detect_peaks import detect_peaks
from utils import NoIntensityRegionException, fit_elliplse, find_clusters_ell_amplitudes, \
    get_props
from candidate import Candidate
from skimage.transform import warp, AffineTransform
from scipy.signal import medfilt
from astropy.stats import mad_std
from astropy.time import TimeDelta
import matplotlib
matplotlib.use('Agg')


class Searcher(object):
    """
    Class wrapper around search functions that returns instances of
    ``Candidate`` class.
    """
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self, dddsp, plot_candidates=False):
        found_dmt = self.func(dddsp.array, *self.args, **self.kwargs)
        candidates = list()
        for ix_dm, ix_t in found_dmt:
            candidate = Candidate(dddsp.t_0 +
                                  ix_t * TimeDelta(dddsp.d_t, format='sec'),
                                  float(dddsp.dm_values[ix_dm]))
            candidates.append(candidate)
        if plot_candidates:
            for candidate in candidates:
                candidate.plot()
        return candidates


def search_shear(image, mph=3.5, mpd=50, shear=0.4):
    """
    Rotate image and bring dedispersed signal to vertical lines. Average them
    and found locations of peaks.
    :param image:
    :param mph:
    :param mpd:
    :param shear:
    :return:
    """
    tform = AffineTransform(shear=shear)
    warped_image = warp(image, tform)
    warped = np.sum(warped_image, axis=0)
    smoothed = medfilt(warped, 101)
    warped = medfilt(warped, 5)
    warped = (warped - smoothed) / mad_std(warped)
    ixs_t = detect_peaks(warped, mph=mph, mpd=mpd)
    ixs_dm = list()
    for ix_t in ixs_t:
        ixs_dm.append(np.argmax(warped_image[:, ix_t]))
    return zip(ixs_dm, ixs_t)


def search_clf(image, pclf, save_fig=True):
    """
    Search FRB in de-dispersed and pre-processed dynamical spectra using
    instance of trained ``PulseClassifier`` instance.
    :param image:
        2D numpy.ndarray of de-dispersed and pre-processed dynamical
        spectra.
    :param pclf:
        Instance of ``PulseClassifier``.
    :return:
        List of ``Candidate`` instances.
    """
    out_features_dict, out_responses_dict = pclf.classify(image)

    # Select only positively classified regions
    positive_props = list()
    for i, (prop, response) in enumerate(out_responses_dict.items()):
        if response:
            positive_props.append([i, prop])
    ixs = list()
    # Fit them with ellipse and create ``Candidate`` instances
    for i, prop in positive_props:
        try:
            gg = fit_elliplse(prop, plot=save_fig, show=False, close=True,
                              save_file="search_clf_{}.png".format(i))
        except NoIntensityRegionException:
            continue

        max_pos = (gg.x_mean + prop.bbox[0], gg.y_mean + prop.bbox[1])
        ixs.append(max_pos)

    return ixs


def search_ell(image, x_stddev, x_cos_theta, y_to_x_stddev, theta_lims,
               save_fig=False, amplitude=None):
    props = get_props(image)
    ixs = list()
    if amplitude is None:
        amplitudes = list()
        for i, prop in enumerate(props):
            try:
                gg = fit_elliplse(prop, plot=False)
                if gg.amplitude.value:
                    amplitudes.append(gg.amplitude.value)
            except NoIntensityRegionException:
                continue
        amplitude = find_clusters_ell_amplitudes(amplitudes)

    print "amplitude threshold {}".format(amplitude)
    print "log amplitude threshold {}".format(np.log(amplitude))
    fig, ax = matplotlib.pyplot.subplots(1, 1)
    ax.hist(amplitudes, bins=300)
    ax.axvline(amplitude)
    ax.set_xlabel('Gaussian amplitude')
    ax.set_ylabel('N')
    fig.savefig('amps_hist.png', bbox_inches='tight', dpi=200)
    fig.show()
    matplotlib.pyplot.close()
    fig, ax = matplotlib.pyplot.subplots(1, 1)
    ax.hist(np.log(amplitudes), bins=300)
    ax.axvline(np.log(amplitude))
    ax.set_xlabel('Gaussian amplitude, log')
    ax.set_ylabel('N')
    fig.savefig('amps_hist_log.png', bbox_inches='tight', dpi=200)
    fig.show()
    matplotlib.pyplot.close()

    for i, prop in enumerate(props):
        try:
            gg = fit_elliplse(prop, plot=False)
        except NoIntensityRegionException:
            continue
        if ((abs(gg.x_stddev) > abs(x_stddev)) and
                (abs(gg.x_stddev * np.cos(gg.theta)) > x_cos_theta) and
                (abs(gg.y_stddev / gg.x_stddev) < y_to_x_stddev) and
                (gg.amplitude > amplitude) and
                (theta_lims[0] < np.rad2deg(gg.theta) % 180 < theta_lims[1])):
            gg = fit_elliplse(prop, plot=save_fig, show=False, close=True,
                              save_file="search_ell_{}.png".format(i))
            max_pos = (gg.x_mean + prop.bbox[0], gg.y_mean + prop.bbox[1])
            ixs.append(max_pos)

    return ixs
