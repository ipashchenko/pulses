from unittest import (TestCase, skipIf)

import numpy as np
from astropy.time import Time, TimeDelta

from pulses.dsp import DSP
from pulses.dedispersion import DeDisperser, noncoherent_dedispersion
from pulses.preprocess import PreProcesser, create_ellipses
from pulses.search import Searcher, search_shear, search_ell
from pulses.pipeline import Pipeline
from pulses.candidate import Candidate


# Test shapes, test direction of increasing DM
class TestAll(TestCase):
    def setUp(self):
        self.cache_dir = None
        self.meta_data = {'exp_code': 'test', 'freq': 'K',
                                  'band': 'U', 'pol': 'LLRR', 'antenna': 'EF'}
        self.n_nu = 64
        self.n_t = 1000
        self.nu_0 = 1668.
        self.d_nu = 0.5
        self.d_t = 0.001
        self.t_0 = Time.now()
        self.dsp = DSP(self.n_nu, self.n_t, self.nu_0, self.d_nu, self.d_t,
                       meta_data=self.meta_data, t_0=self.t_0)
        self.std = 1.
        self.dm = 400.
        self.width = 0.003
        self.amp = 1.5
        self.t0 = 0.5
        self.dsp.add_noise(self.std)
        self.dsp.add_pulse(self.t0, self.amp, self.width, self.dm)
        self.dm_grid = np.arange(0, 1000, 20, dtype=float)
        ddsp = DeDisperser(noncoherent_dedispersion, self.dm_grid,
                           nu_max=1668, d_nu=0.5, d_t=0.001, threads=4)
        self.ddsp = ddsp(self.dsp)

    def test_dsp_shape(self):
        self.assertEqual(self.dsp.values.shape, (self.n_nu, self.n_t))

    def test_dsp_d_t(self):
        self.assertEqual(self.dsp.d_t, TimeDelta(self.d_t, format='sec'))

    def test_dsp_tfull(self):
        self.assertEqual(self.dsp.t[-1],
                         self.t_0 + (self.n_t-1) * TimeDelta(self.d_t,
                                                             format='sec'))
    def test_shear(self):
        searcher = Searcher(search_shear, mph=3.5, mpd=50, shear=0.4)
        candidates = searcher(self.ddsp)

        self.assertGreaterEqual(len(candidates), 1)
        if len(candidates) == 1:
            candidate = candidates[0]
        self.assertAlmostEqual(candidate.dm, self.dm, delta=100.)

    @skipIf(True, 'Passing')
    def test_ell(self):
        preprocesser = PreProcesser(create_ellipses, disk_size=3,
                                    threshold_big_perc=90., threshold_perc=97.5,
                                    statistic='mean')
        ddsp = preprocesser(self.ddsp)
        searcher = Searcher(search_ell, x_stddev=10., y_to_x_stddev=0.3,
                            theta_lims=[130., 180.], x_cos_theta=3.)
        candidates = searcher(ddsp)

        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertAlmostEqual(candidate.dm, self.dm, delta=100.)


class TestDB(TestCase):
    def setUp(self):
        pipeline = Pipeline(None, None, [None], [None], 'db.sqlite')
        candidates = list()
        found_dmt = list()
        t_0 = 0.
        d_t = 0.001
        for dm, ix_t in found_dmt:
            candidate = Candidate(t_0 +
                                  ix_t * TimeDelta(d_t, format='sec'),
                                  dm)
            candidates.append(candidate)
        pipeline.save_to_db(candidates)



