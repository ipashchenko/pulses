from skimage.transform import warp, AffineTransform
from scipy.signal import medfilt
from astropy.stats import mad_std


# FIXME: Should work with any class instances - `dsp` or `dddsp`
class Searcher(object):
    def __init__(func, db_file, *args, **kwargs):
        pass

    def __call__(self, dddsp, plot_candidates=True):
        found_dmt = self.function(dd_dsp, *self.args, **self.kwargs)
        candidates = list()
        for ix_dm, ix_t in found_dmt:
            candidate = Candidate(dddsp.t_0 + ix_t * TimeDelta(dddsp.d_t, format='sec'),
                                  float(dddsp.dm_values[ix_dm))
            candidates.append(candidate)
        if plot_candidates:
            for candidate in candidates:
                candidate.plot()
        return candidates


def search_candidates_shear(image, t_0, d_t, d_dm, mph=3.5, mpd=50,
                            original_dsp=None, shear=0.4):
    tform = AffineTransform(shear=shear)
    warped_image = warp(image, tform)
    warped = np.sum(warped_image, axis=0)
    smoothed = medfilt(warped, 101)
    warped = medfilt(warped, 5)
    warped = (warped - smoothed) / mad_std(warped)
    indxs = detect_peaks(warped, mph=mph, mpd=mpd)
    dm_indxs = list()
    for indx in indxs:
        dm_indxs.append(np.argmax(warped_image[:, indx]))
    candidates = list()
    for i, (t_indx, dm_indx) in enumerate(zip(indxs, dm_indxs)):
        candidate = Candidate(t_0 + t_indx * TimeDelta(d_t, format='sec'),
                              dm_indx * float(d_dm))
        candidates.append(candidate)
        if original_dsp is not None:
            plot_rect_original_dsp(t_indx, 50,
                                   original_dsp=original_dsp, show=False,
                                   close=True,
                                   save_file="search_shear_dsp_{}.png".format(i))

return candidates


# TODO: All search functions must returns instances of ``Candidate`` class
def search_candidates_clf(image, pclf, t_0, d_t, d_dm, save_fig=False,
                          original_dsp=None):
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
    out_features_dict, out_responses_dict = pclf.classify_data(image)

    # Select only positively classified regions
    positive_props = list()
    for i, (prop, response) in enumerate(out_responses_dict.items()):
        if response:
            positive_props.append([i, prop])
    candidates = list()
    # Fit them with ellipse and create ``Candidate`` instances
    for i, prop in positive_props:
        try:
            gg = fit_elliplse(prop, plot=save_fig, show=False, close=True,
                              save_file="search_clf_{}.png".format(i))
        except NoIntensityRegionException:
            continue


        if original_dsp is not None:
            plot_prop_original_dsp(prop, original_dsp=original_dsp,
                                    show=False, close=True,
                                    save_file="search_clf_dsp_{}.png".format(i))


        max_pos = (gg.x_mean + prop.bbox[0], gg.y_mean + prop.bbox[1])
        candidate = Candidate(t_0 + max_pos[1] * TimeDelta(d_t, format='sec'),
                              max_pos[0] * float(d_dm))
        candidates.append(candidate)

return candidates


def search_candidates_ell(image, x_stddev, x_cos_theta,
                          y_to_x_stddev, theta_lims, t_0, d_t, d_dm,
                          save_fig=False, amplitude=None,
                          original_dsp=None):
    a = image.copy()
    s = generate_binary_structure(2, 2)
    # Label image
    labeled_array, num_features = label(a, structure=s)
    # Find objects
    props = regionprops(labeled_array, intensity_image=image)
    candidates = list()
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
            if original_dsp is not None:
                plot_prop_original_dsp(prop, original_dsp=original_dsp,
                                       show=False, close=True,
                                       save_file="search_ell_dsp_{}.png".format(i))
            max_pos = (gg.x_mean + prop.bbox[0], gg.y_mean + prop.bbox[1])
            candidate = Candidate(t_0 + max_pos[1] * TimeDelta(d_t,
                                                               format='sec'),
                                  max_pos[0] * float(d_dm))
            candidates.append(candidate)

return candidates
