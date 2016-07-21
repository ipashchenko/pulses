import os
import h5py
import numpy as np
from dsp import DSP
from astropy.time import Time, TimeDelta


def create_from_txt(fn, t0, dt, nchan, metadata):
    pass


def create_from_hdf5(fn, t0, dt, nchan, d_t, metadata):
    pass


def create_from_mk5(fn, fmt, t0, dt, nchan, d_t, metadata):
    pass


def create_from_fits(fn, t0, dt, nchan, d_t, metadata):
    pass


def read_hdf5(fname, name):
    """
    Read data from HDF5 format.

    :param fname:
        File to read data.
    :param name:
        Name of dataset to use.

    :return:
        Numpy array with data & dictionary with metadata.

    :note:
        HDF5 hasn't time formats. Use ``unicode(datetime)`` to create strings
        with microseconds.
    """
    f = h5py.File(fname, "r")
    dset = f[name]
    meta_data = dict()
    for key, value in dset.attrs.items():
        meta_data.update({str(key): value})
    data = dset.value
    f.close()
    return data, meta_data


def save_to_hdf5(dsp, fname, name='dsp'):
    """
    Save data to HDF5 format.

    :param dsp:
        Instance of ``DSP`` class to save.
    :param fname:
        File to save data.
    :param name: (optional)
        Name of dataset to use. (default: ``dsp``)
    :note:
        HDF5 hasn't time formats. Using ``str(datetime)`` to create strings
        with microseconds.
    """
    import h5py
    f = h5py.File(fname, "w")
    dset = f.create_dataset(name, data=dsp.values, chunks=True,
                            compression='gzip')
    meta_data = dsp.meta_data.copy()
    meta_data.update({'n_nu': dsp.n_nu, 'n_t': dsp.n_t, 'nu_0': dsp.nu_0,
                      'd_nu': dsp.d_nu, 'd_t': dsp.d_t.sec,
                      't_0': str(dsp.t_0)})
    for key, value in meta_data.items():
        dset.attrs[key] = value
    f.flush()
    f.close()


def create_from_hdf5(fname, name='dsp', n_nu_discard=0):
    """
    Function that creates instance of ``DSP`` class from HDF5-file.

    :param fname:
        Name of HDF5-file with ``dsp`` data set that is 2D numpy.ndarray with rows
        representing frequency channels and columns - 1d-time series of data for
        each frequency channel and meta-data.
    :param name: (optional)
        Name of dataset to use. (default: ``dsp``)
    :param n_nu_discard: (optional)
        NUmber of spectral channels to discard symmetrically from both low and
         high frequency.
    :return:
        Instance of ``DSP`` class.
    """
    data, meta_data = read_hdf5(fname, name)
    n_nu = meta_data.pop('n_nu')
    n_t = meta_data.pop('n_t')
    nu_0 = meta_data.pop('nu_0')
    d_nu = meta_data.pop('d_nu')
    d_t = meta_data.pop('d_t')
    t_0 = Time(meta_data.pop('t_0'))
    dsp = DSP(n_nu - n_nu_discard, n_t, nu_0 - n_nu_discard * d_nu / 2.,
              d_nu, d_t, meta_data, t_0=t_0)
    dsp.add_values(data)
    return dsp


def create_from_txt(fname, nu_0, d_nu, d_t, meta_data, t_0=None,
                    n_nu_discard=0):
    """
    Function that creates instance of ``DSP`` class from text file.
    :param fname:
        Name of txt-file with rows representing frequency channels and columns -
        1d-time series of data for each frequency channel.
    :param nu_0:
        Frequency of highest frequency channel [MHz].
    :param d_nu:
        Width of spectral channel [MHz].
    :param d_t:
        Time step [s].
    :param meta_data:
        Dictionary with metadata describing current dynamical spectra. It must
        include ``exp_name`` [string], ``antenna`` [string], ``freq`` [string],
        ``band`` [string], ``pol`` [string] keys.
    :param t_0: (optional)
        Time of first measurement. Instance of ``astropy.time.Time`` class. If
        ``None`` then use time of initialization. (default: ``None``)
    :param n_nu_discard: (optional)
        NUmber of spectral channels to discard symmetrically from both low and
         high frequency.
    :return:
        Instance of ``DSP`` class.
    """
    assert not int(n_nu_discard) % 2

    try:
        values = np.load(fname).T
    except IOError:
        values = np.loadtxt(fname, unpack=True)
    n_nu, n_t = np.shape(values)
    dsp = DSP(n_nu - n_nu_discard, n_t, nu_0 - n_nu_discard * d_nu / 2.,
              d_nu, d_t, meta_data=meta_data, t_0=t_0)
    if n_nu_discard:
        dsp.values += values[n_nu_discard / 2: -n_nu_discard / 2, :]
    else:
        dsp.values += values

    return dsp


class DSPIterator(object):
    """
    http://stackoverflow.com/a/11690539
    """
    def __init__(self):
        pass

    def __iter__(self):
        pass


class DSPIterator(object):
    """
       Generator that returns instances of ``DSP`` class.
       :param m5_file:
           Raw data file in M5 format.
       :param m5_params:
           Dictionary with meta data.
       :param chunk_size:
           Size (in s) of chunks to process raw data.
    """
    def __init__(self, m5_file, m5_fmt, freq_band_pol, chunk_size):
        """
        :param freq_band_pol:
            Iterable of bands specification, eg. ['4828.00-L-U', '4828.00-R-U', '4828.00-L-L', '4828.00-R-L']
        """
        self.m5_file = m5_file
        self.m5_fmt = m5_fmt
        self.chunk_size = chunk_size
        self.freq_band_pol = freq_band_pol

    def __iter__(self):
        m5 = M5(m5_file, m5_fmt)
        offset = 0

        while offset * 32e6 < m5.size:
            ds = m5.create_dspec(n_nu, d_t, offset, dur, outfile, dspec_path=None)

            # NOTE: all 4 channels are stacked forming dsarr:
            dsarr = dspec_cat(os.path.basename(ds['Dspec_file']),
                              self.freq_band_pol)
            metadata = ds
            t_0 = m5.start_time + TimeDelta(offset, format='sec')
            print "t_0 : ", t_0.datetime

            metadata.pop('Dspec_file')
            metadata.update({'antenna': m5_params['antenna'],
                             'freq': self.cfx.freq,
                             'band': "".join(m5_params['band']),
                             'pol': "".join(m5_params['pol']),
                             'exp_code': m5_params['exp_code']})
            # FIXME: ``2`` means combining U&L bands.
            dsp = DSP(2 * n_nu, n_t, nu_0, d_nu, d_t, meta_data, t_0)
            dsp.add_values(dsarr.T)
            offset += chunk_size

        yield dsp

