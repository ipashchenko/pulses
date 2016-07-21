from skimage.transform import warp, AffineTransform
from scipy.signal import medfilt
from astropy.stats import mad_std


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
