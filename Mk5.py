# -*- coding: utf-8 -*-
import os
import numpy as np
import subprocess
import time
import re
from astropy.time import Time, TimeDelta
from dsp import DSP


my5spec = "../my5spec/./my5spec"


class M5(object):
    """ working with raw data """
    def __init__(self, m5_file, m5_fmt=None):
        """

        :param m5_file:
            Mk5 filename.
        :param m5_fmt:
            Mark5access data format in form <FORMAT>-<Mbps>-<nchan>-<nbit>
        """
        self.m5_file = m5_file
        self.m5_fmt = m5_fmt
        self.my5spec = my5spec
        self.m5dir = os.path.dirname(os.path.abspath(self.m5_file))
        self.size = os.path.getsize(self.m5_file)
        self.starttime = self.start_time

    @property
    def start_time(self):
        """
        Determine start time for the m5_file.
        """
        cmd = "m5time " + self.m5_file + " " + self.m5_fmt
        res = subprocess.check_output(cmd.split())
        res = re.search('\d{5}/\d{2}:\d{2}:\d{2}\.\d{2}', res).group()
        m5_mjd = float(res.split('/')[0])
        m5_hms = res.split('/')[1].split(':')
        m5_time = float(m5_hms[0])/24.0 + float(m5_hms[1])/1440.0 \
            + float(m5_hms[2])/86400.0
        res = m5_mjd + m5_time
        return Time(res, format='mjd')

    def __repr__(self):
        """ Show some info about the m5_file """
        outprint = "File: %s\n" % self.m5_file
        outprint += "Format: %s\n" % self.m5_fmt
        outprint += "File size: %s\n" % self.size
        outprint += "File start MJD/time: %s\n" % self.starttime
        outprint += "Last modified: %s\n" % \
                    time.ctime(os.path.getmtime(self.m5_file))
        return outprint

    def create_dspec(self, n_nu=64, d_t=1, offset=0, dur=None, outfile=None,
                     dspec_path=None, **kwargs):
        """
        Create 4 DS files for selected M5datafile with nchan, dt[ms], ...
        The input options are the same as for ``my5spec``.
        """
        if dspec_path is None:
            dspec_path = os.getcwd()

        # my5spec options:
        opt1 = "-a %s " % d_t
        opt2 = "-n %s " % n_nu

        if dur is not None:
            opt3 = "-l %s " % dur
        else:
            opt3 = ""

        if offset != 0.0:
            opt4 = "-o %s " % offset
        else:
            opt4 = ""

        opts = opt1 + opt2 + opt3 + opt4

        if not outfile:
            opts2 = re.sub("-", "", "".join(opts.split()))
            outfile = os.path.join(dspec_path,
                                   os.path.basename(self.m5_file).split('.')[0] +
                                   '_' + opts2 + "_dspec")

        cmd = self.my5spec + " " + opts + "%s %s %s" \
                                          % (self.m5_file, self.m5_fmt, outfile)
        subprocess.check_call(cmd.split())
        res = {'Dspec_file': outfile}

        return res


# extra manipulations with dspec files
def get_cfx_format(fname, cfx_data):
    return cfx_data[fname][-1]


def dspec_cat(fname, cfx_fmt, dspec_path=None):
    """
    Concatenate dynamical spectra files, returning array.

    :param fname:
        Base filename pattern of DS-files.
    :param cfx_fmt:
        Format `freq-pol-band` eg. ``[4828.00-L-U, 4828.00-R-U, 4828.00-L-L,
        4828.00-R-L]``
    pol - sum polarizations\n
    uplow - concat UPPER and LOWER bands\n
    OUTPUT: np.array
    """
    if dspec_path is None:
        dspec_path = os.getcwd()
    from utils import find_file
    flist = find_file(fname + '*_0?', dspec_path)
    if flist is None:
        raise Exception("dspec_cat: Can't find files matching %s" % fname)
    if len(flist) > len(cfx_fmt):
        raise Exception("WARNING! dspec_cat: There are difference in files"
                        " number and CFX-format length")
    # FIXME: improve the above checkings
    flist = sorted(flist, key=lambda x: int(x[-2:]))
    ashape = np.loadtxt(flist[0]).shape
    arr = np.zeros((ashape[0], ashape[1]*2))
    for fmt, fil in zip(cfx_fmt, flist):
        if fmt.split('-')[2] == 'L':
            arr[:, :ashape[1]] = arr[:, :ashape[1]] + np.loadtxt(fil)[:, ::-1]
        else:
            arr[:, ashape[1]:] = arr[:, ashape[1]:] + np.loadtxt(fil)
    return arr/2


class DSPIterator(object):
    def __init__(self, m5_file, m5_fmt, freq_band_pol, chunk_size, n_nu, d_t,
                 nu_0, d_nu, meta_data):
        """
       Generator that returns instances of ``DSP`` class.

       :param m5_file:
           Raw data file in M5 format.
       :param m5_fmt:
           Mark5access data format in form <FORMAT>-<Mbps>-<nchan>-<nbit>
       :param chunk_size:
           Size (in s) of chunks to process raw data.
        :param freq_band_pol:
            Iterable of bands specification, eg. ['4828.00-L-U', '4828.00-R-U',
            '4828.00-L-L', '4828.00-R-L']
        :param chunk_size:
            Time duration of one chunk [s].
        :param n_nu:
            Number of spectral channels.
        :param d_t:
            Time step [s].
        :param nu_0:
            Frequency of highest frequency channel [MHz].
        :param d_nu:
            Width of spectral channel [MHz].
        :param meta_data:
            Dictionary with metadata describing current dynamical spectra. It
            must include ``exp_code`` [string], ``antenna`` [string], ``freq``
            [string], ``band`` [string], ``pol`` [string] keys. Eg.
            ``{'exp_code': 'raks03ra', 'antenna': 'AR', 'freq': 'l',
            'band': 'u', 'pol': 'r'}``.
        """
        self.m5_file = m5_file
        self.m5_fmt = m5_fmt
        self.chunk_size = chunk_size
        self.freq_band_pol = freq_band_pol
        self.n_nu = n_nu
        self.d_t = d_t
        self.n_t = int(chunk_size / d_t)
        self.nu_0 = nu_0
        self.d_nu = d_nu
        self.meta_data = meta_data
        self.m5 = M5(self.m5_file, self.m5_fmt)

    def get_dsp(self, chunk_size=None, offset=0.):
        if chunk_size is None:
            chunk_size = self.chunk_size
        ds = self.m5.create_dspec(self.n_nu, self.d_t, offset,
                                  dur=self.chunk_size, outfile=None,
                                  dspec_path=None)

        # NOTE: all 4 channels are stacked forming dsarr:
        dsarr = dspec_cat(os.path.basename(ds['Dspec_file']),
                          self.freq_band_pol)
        t_0 = self.m5.start_time + TimeDelta(offset, format='sec')
        print "t_0 : ", t_0.datetime

        # FIXME: ``2`` means combining U&L bands.
        dsp = DSP(2 * self.n_nu, self.n_t, self.nu_0, self.d_nu, self.d_t,
                  self.meta_data, t_0)
        dsp.add_values(dsarr.T)

        return dsp

    def __iter__(self):
        offset = 0

        while offset * 32e6 < self.m5.size:
            dsp = self.get_dsp(offset=offset)
            offset += self.chunk_size

        yield dsp

