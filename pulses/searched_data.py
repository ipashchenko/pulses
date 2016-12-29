# -*- coding: utf-8 -*-
from sqlalchemy import (Column, Integer, Float, String, DateTime)
from base import Base


class SearchedData(Base):
    """
    Class that describes dynamical spectra and it's metadata.
    """
    __tablename__ = "searched_data"

    id = Column(Integer, primary_key=True)
    antenna = Column(String)
    freq = Column(String)
    band = Column(String)
    pol = Column(String)
    exp_code = Column(String)
    algo = Column(String)
    t_0 = Column(DateTime)
    t_end = Column(DateTime)
    d_t = Column(Float)
    d_nu = Column(Float)
    nu_max = Column(Float)

    def __init__(self, dsp, algo=None, t_0=None, t_end=None, d_t=None,
                 d_nu=None, nu_max=None):
        self.dsp = dsp
        self.freq = dsp.meta_data['freq']
        self.band = dsp.meta_data['band']
        self.pol = dsp.meta_data['pol']
        self.exp_code = dsp.meta_data['exp_code']
        self.algo = algo
        self.t_0 = t_0
        self.t_end = t_end
        self.d_t = d_t
        self.d_nu = d_nu
        self.nu_max = nu_max

    def __repr__(self):
        return "Experiment: {}, antenna: {}, time begin: {}, time end: {}," \
               "freq: {}, band: {}, polarization: {}," \
               " algo: {}".format(self.exp_code, self.antenna, self.t_0,
                                  self.t_end, self.freq, self.band, self.pol,
                                  self.algo)
