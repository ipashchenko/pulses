import os
import numpy as np
import multiprocessing
from utils import vint, vround
from dsp import DDDSP


# MHz ** 2 * cm ** 3 * s / pc
k = 1. / (2.410331 * 10 ** (-4))


# It is a one step for next function
def _de_disperse_by_value_freq_average(params):
    """
    De-disperse using specified value of DM and average in frequency.

    :param array:
        Numpy 2D array (#freq, #t) with dynamical spectra.
    :param dm:
        Dispersion measure to use in de-dispersion [cm^3 / pc].
    :param nu:
        Numpy array (#freq) with frequencies [MHz].
    :param nu_max:
        Maximum frequency [MHz].
    :param d_t:
        Value of time step [s].

    :notes:
        This method avoids creating ``(n_nu, n_t)`` arrays and must be
        faster for data with big sizes. But it returns already frequency
        averaged de-dispersed dyn. spectra.
    """
    array, dm, nu, nu_max, d_t = params
    n_nu, n_t = array.shape

    # Calculate shift of time caused by de-dispersion for all channels
    dt_all = k * dm * (1. / nu ** 2. - 1. / nu_max ** 2.)
    # Find what number of time bins corresponds to this shifts
    nt_all = vint(vround(dt_all / d_t))
    # Container for summing de-dispersed frequency channels
    values = np.zeros(n_t)
    # Roll each axis (freq. channel) to each own number of time steps.
    for i in range(n_nu):
        values += np.roll(array[i], -nt_all[i])

    return values / n_nu


def noncoherent_dedispersion(array, dm_grid, nu_max, d_nu, d_t, threads=1):
    """
    Method that de-disperse dynamical spectra with range values of dispersion
    measures and average them in frequency to obtain image in (t, DM)-plane.

    :param array:
        Numpy 2D array (#freq, #t) with dynamical spectra.
    :param dm_grid:
        Array-like of values of DM on which to de-disperse [cm^3/pc].
    :param nu_max:
        Maximum frequency [MHz].
    :param d_nu:
        Value of frequency step [MHz].
    :param d_t:
        Value of time step [s].
    :param threads: (optional)
        Number of threads used for parallelization with ``multiprocessing``
        module. If ``1`` then it isn't used. (default: 1)
    """

    n_nu, n_t = array.shape
    nu = np.arange(n_nu, dtype=float)
    nu = (nu_max - nu * d_nu)[::-1]

    pool = None
    if threads > 1:
        pool = multiprocessing.Pool(threads, maxtasksperchild=1000)

    if pool:
        m = pool.map
    else:
        m = map

    params = [(array, dm, nu, nu_max, d_t) for dm in dm_grid]

    # Accumulator of de-dispersed frequency averaged frames
    result = list(m(_de_disperse_by_value_freq_average, params))
    result = np.array(result)

    if pool:
        # Close pool
        pool.close()
        pool.join()

    return result


class DeDisperser(object):

    def __init__(self, func, dm_values, *args, **kwargs):
        self.func = func
        self.dm_values = dm_values
        self.args = args
        self.kwargs = kwargs
        self.info = None

    def __call__(self, dsp, cache_dir=None):
        """
        :param dsp:
            Instance of ``DSP`` class.
        :param cache_dir:
            Directory to store cached results.
        :return:
            Instance of ``DDDSP`` calss
        """
        if cache_dir is None:
            cache_dir = os.getcwd()
        de_disp_cache_fname = os.path.join(cache_dir,
                                           dsp._cache_fname_prefix +
                                           "_dedisp.hdf5")

        dddsp = DDDSP(dsp, self.dm_values)
        dddsp.array = self.func(dsp.values, self.dm_values, *self.args,
                                **self.kwargs)
        return dddsp

    def clear_cache(self, dsp, cache_dir):
        pass

