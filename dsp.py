# -*- coding: utf-8 -*-
import multiprocessing
import ctypes
import numpy as np
from astropy.time import Time, TimeDelta
from utils import plot_2d

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot
except ImportError:
    plt = None


class MetaData(dict):
    """
    Class that describes RA experiment metadata.
    See http://stackoverflow.com/questions/2060972/subclassing-python-dictionary-to-override-setitem
    """

    meta_keys = ['exp_code', 'antenna', 'freq', 'band', 'pol']
    required_keys = meta_keys
    meta_values = {'freq': ('K', 'C', 'L', 'P'),
                   'band': ('U', 'L', 'UL', 'LU', 'UULL', 'LLUU'),
                   'pol': ('L', 'R', 'LR', 'LLRR', 'RRLL', 'RLRL')}

    def __init__(self, *args, **kwargs):
        super(MetaData, self).__init__()
        self.update(*args, **kwargs)
        for key in self.required_keys:
            if key not in self:
                raise Exception("Absent key {} in metadata".format(key))

    def __setitem__(self, key, value):
        # optional processing here
        if key not in self.meta_keys:
            raise Exception("Not allowed key in metadata: {}".format(key))
        if key in self.meta_values and value not in self.meta_values[key]:
            raise Exception("Not allowed value: {} in"
                            " metadata for key: {}".format(value, key))
        super(MetaData, self).__setitem__(key, value)

    def update(self, *args, **kwargs):
        if args:
            if len(args) > 1:
                raise TypeError("update expected at most 1 arguments, "
                                "got {}".format(len(args)))
            other = dict(args[0])
            for key in other:
                self[key] = other[key]
        for key in kwargs:
            self[key] = kwargs[key]

    def setdefault(self, key, value=None):
        if key not in self:
            self[key] = value
        return self[key]


class DSP(object):
    """
    Basic class that represents a set of regularly spaced frequency channels
    with regularly measured values (time sequence of autospectra).
    :param n_nu:
        Number of spectral channels.
    :param n_t:
        Number of time steps.
    :param nu_0:
        Frequency of highest frequency channel [MHz].
    :param dnu:
        Width of spectral channel [MHz].
    :param d_t:
        Time step [s].
    :param meta_data:
        Dictionary with metadata describing current dynamical spectra. It must
        include ``exp_code`` [string], ``antenna`` [string], ``freq`` [string],
        ``band`` [string], ``pol`` [string] keys.
        Eg. ``{'exp_code': 'raks03ra', 'antenna': 'AR', 'freq': 'l',
        'band': 'u', 'pol': 'r'}``
    :param t_0: (optional)
        Time of first measurement. Instance of ``astropy.time.Time`` class. If
        ``None`` then use time of initialization. (default: ``None``)
    """
    def __init__(self, n_nu, n_t, nu_0, d_nu, d_t, meta_data=None, t_0=None):
        self.n_nu = n_nu
        self.n_t = n_t
        self.nu_0 = nu_0
        self.t_0 = t_0 or Time.now()
        # Using shared array (http://stackoverflow.com/questions/5549190 by pv.)
        shared_array_base = multiprocessing.Array(ctypes.c_float, n_nu * n_t)
        self.values =\
            np.ctypeslib.as_array(shared_array_base.get_obj()).reshape((n_nu,
                                                                        n_t,))

        nu = np.arange(n_nu)
        t = np.arange(n_t)
        self.nu = (nu_0 - nu * d_nu)[::-1]
        self.d_t = TimeDelta(d_t, format='sec')
        self.t = self.t_0 + t * self.d_t
        self.t_end = self.t[-1]
        self.d_nu = d_nu
        self.meta_data = MetaData(meta_data)
        
    @property
    def _cache_fname_prefix(self):
        date_0, time_0 = str(self.dsp.t_0.utc.datetime).split(' ')
        date_1, time_1 = str(self.dsp.t_end.utc.datetime).split(' ')
        return "{}_{}_{}_{}_{}_{}_{}".format(self.meta_data['exp_code'],
                                             self.meta_data['antenna'],
                                             self.meta_data['freq'], date_0,
                                             time_0, date_1, time_1)

    def __repr__(self):
        outprint = "# channels: {}\n".format(self.n_nu)
        outprint += "# times: {}\n".format(self.n_t)
        outprint += "Max. freq. [MHz]: {}\n".format(self.nu_0)
        outprint += "Freq. spacing [MHz]: {}\n".format(self.d_nu)
        outprint += "Time spacing [s]: {}\n".format(self.d_t.sec)
        outprint += "Start time: {}\n".format(str(self.t_0))
        return outprint

    @property
    def shape(self):
        """
        Length of time [s] and frequency [MHz] dimensions.
        """
        return self.n_t * self.d_t.sec, self.n_nu * self.d_nu

    def add_values(self, array):
        """
        Add dyn. spectra in form of numpy array (#ch, #t,) to instance.
        :param array:
            Array-like of dynamical spectra (#ch, #t,).
        """
        array = np.atleast_2d(array)
        assert self.values.shape == array.shape
        self.values += array

    # FIXME: Handle start time in slices somehow
    def slice(self, t_start, t_stop):
        """
        Slice frame using specified fractions of time interval.
        :param t_start:
            Number [0, 1] - fraction of total time interval.
        :param t_stop:
            Number [0, 1] - fraction of total time interval.
        :return:
            Instance of ``DSP`` class.
        """
        assert t_start < t_stop
        frame = DSP(self.n_nu, int(round(self.n_t * (t_stop - t_start))),
                    self.nu_0, self.d_nu, self.d_t,
                    meta_data=self.meta_data, t_0=self.t_0)
        frame.add_values(self.values[:, int(t_start * self.n_t): int(t_stop *
                                                                     self.n_t)])
        return frame

    def plot(self, bbox=None, colorbar_label=None, close=False, save_file=None,
             show=True):
        """
        Plot dynamical spectra.
        
        :param bbox: (optional)
            Bounding box of region to plot (x1, y1, x2, y2) - ``prop.bbox``. If ``None``
            then plot all.
        """
        plot_2d(self.values, bbox=bbox, colorbar_label=colorbar_label, close=close,
                save_file=save_file, show=show, xlabel='Time',
                ylabel='Dynamical spectra')


    # TODO: if one choose what channels to plot - use ``extent`` kwarg.
    def plot(self, plot_indexes=True, savefig=None):
        if plt is not None:
            matplotlib.pyplot.figure()
            matplotlib.pyplot.matshow(self.values, aspect='auto')
            matplotlib.pyplot.colorbar()
            if not plot_indexes:
                raise NotImplementedError("Ticks haven't implemented yet")
                # plt.xticks(np.linspace(0, 999, 10, dtype=int),
                # frame.t[np.linspace(0, 999, 10, dtype=int)])
            matplotlib.pyplot.xlabel("time steps")
            matplotlib.pyplot.ylabel("frequency ch. #")
            matplotlib.pyplot.title('Dynamical spectra')
            if savefig is not None:
                matplotlib.pyplot.savefig(savefig, bbox_inches='tight')
            matplotlib.pyplot.show()

    def add_pulse(self, t_0, amp, width, dm):
        """
        Add pulse to frame.
        :param t_0:
            Arrival time of pulse at highest frequency channel [s]. Counted
            from start time of ``DSP`` instance.
        :param amp:
            Amplitude of pulse.
        :param width:
            Width of gaussian pulse [s] (in time domain).
        :param dm: (optional)
            Dispersion measure of pulse [cm^3 / pc].
        """
        t_0 = TimeDelta(t_0, format='sec')

        # MHz ** 2 * cm ** 3 * s / pc
        k = 1. / (2.410331 * 10 ** (-4))

        # Calculate arrival times for all channels
        t0_all = (t_0.sec * np.ones(self.n_nu)[:, np.newaxis] +
                  k * dm * (1. / self.nu ** 2. -
                            1. / self.nu_0 ** 2.))[0]
        pulse = amp * np.exp(-0.5 * ((self.t - self.t_0).sec -
                                     t0_all[:, np.newaxis]) ** 2 / width ** 2.)
        self.values += pulse

    def rm_pulse(self, t_0, amp, width, dm):
        """
        Remove pulse to frame.
        :param t_0:
            Arrival time of pulse at highest frequency channel [s]. Counted
            from start time of ``DSP`` instance.
        :param amp:
            Amplitude of pulse.
        :param width:
            Width of gaussian pulse [s] (in time domain).
        :param dm:
            Dispersion measure of pulse [cm^3 / pc].
        """
        self.add_pulse(t_0, -amp, width, dm)

    def add_noise(self, std):
        """
        Add noise to frame using specified rayleigh-distributed noise.
        :param std:
            Std of rayleigh-distributed uncorrelated noise.
        """
        noise =\
            np.random.rayleigh(std,
                               size=(self.n_t *
                                     self.n_nu)).reshape(np.shape(self.values))
        self.values += noise


class DDDSP(object):
    def __init__(self, dm_values, dsp=None):
        """
        :param d_t:
            Time step [s].
        :param t_0: (optional)
            Time of first measurement. Instance of ``astropy.time.Time`` class. If
            ``None`` then use time of initialization. (default: ``None``)
        """

        self.dsp = dsp
        self.n_t = dsp.n_t
        self.n_dm = len(dm_values)
        self.t_0 = self.dsp.t_0
        self.d_t = self.dsp.d_t
        self.array = np.zeros((self.n_dm, self.n_t), float)
        self.dm_values = dm_values
        
    def plot(self, bbox=None, colorbar_label=None, close=False, save_file=None,
             show=True):
        """
        Plot de-dispersed dynamical spectra.
        
        :param bbox: (optional)
            Bounding box of region to plot (x1, y1, x2, y2) - ``prop.bbox``. If ``None``
            then plot all.
        """
        plot_2d(self.array, bbox=bbox, colorbar_label=colorbar_label, close=close,
                save_file=save_file, show=show, xlabel='Time',
                ylabel='Freq. averaged de-dispersed spectra')
