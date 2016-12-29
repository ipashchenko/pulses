# -*- coding: utf-8 -*-
from sqlalchemy import (Column, Integer, Float, ForeignKey, DateTime)
from base import Base
from searched_data import SearchedData
from sqlalchemy.orm import (backref, relation)


class Candidate(Base):
    """
    Class that describes FRB candidates related to dynamical spectra searched.
    """
    __tablename__ = "candidates"

    id = Column(Integer, primary_key=True)
    t = Column(DateTime)
    dm = Column(Float)
    searched_data_id = Column(Integer, ForeignKey('searched_data.id'))

    candidate = relation(SearchedData, backref=backref('candidates',
                                                       order_by=id))

    def __init__(self, t, dm):
        """
        :param t:
            Instance of ``astropy.time.Time``.
        :param dm:
            Dispersion measure of pulse.
        """
        self.t = t.utc.datetime
        self.dm = dm

    def __repr__(self):
        # return "FRB candidate. t0: {}, DM: {}".format(self.t, self.dm)
        return "FRB candidate. t0: " \
               "{:%Y-%m-%d %H:%M:%S.%f}".format(self.t)[:-3] + \
               " DM: {:.0f}".format(self.dm)
