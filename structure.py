# -*- coding: utf-8 -*-
import os
from collections import defaultdict
from searched_data import SearchedData
from db import connect_to_db
from io import DSPIterator
from cfx import CFX


# FIXME: Split file to RA-independent part & RA-part
class Pipeline(object):
    def __init__(self, dsp_iterator, de_disperser, pre_processers, searchers,
                 db_file, cache_dir=None):
        """
        :param dsp_iterator:
            Iterator that returns instances of ``DSP`` class.
        :param de_disperser:
            Instance of ``DeDisperion`` class used to de-disperse data.
        :param pre_processers:
            Iterable of ``PreProcesser`` class instances used to pre-process
            de-dispersed data.
        :param searchers:
            Iterable of ``Searcher`` class instances used to search pulses in
            de-dispersed & optionally preprocessed data.
        """
        self.dsp_iterator = dsp_iterator
        self.de_disperser = de_disperser
        self.pre_processers = pre_processers
        self.searchers = searchers
        self.db_file = db_file
        self.cache_dir = cache_dir
        assert len(pre_processers) == len(searchers)

    def run(self):
        for dsp in self.dsp_iterator:
            for pre_processer, searcher in zip(self.pre_processers,
                                               self.searchers):
                dddsp = self.de_disperser(dsp, cache_dir=self.cache_dir)

                try:
                    dddsp = pre_processer(dddsp, cache_dir=self.cache_dir)
                except TypeError:
                    pass

                candidates = searcher(dddsp)

                algo = 'de_disp_{}_{}_{} pre_process_{}_{}_{}' \
                       ' search_{}_{}'.format(de_disp_func.__name__, de_disp_args,
                                              de_disp_kwargs,
                                              preprocess_func_name,
                                              preprocess_args, preprocess_kwargs,
                                              search_func.__name__, search_kwargs)

                searched_data = SearchedData(algo=algo, **self.meta_data)
                searched_data.candidates = candidates
                # Saving searched meta-data and found candidates to DB
                if self.db_file is not None:
                    session = connect_to_db(self.db_file)
                    session.add(searched_data)
                    session.commit()


class RAPipeline(object):
    def __init__(self, exp_code, cfx_file, raw_data_dir, db_file, cache_dir):
        self.exp_code = exp_code
        self.raw_data_dir = raw_data_dir
        self.db_file = db_file
        self.cache_dir = cache_dir
        self.cfx = CFX(cfx_file)
        self._dedisperser = None
        self._preprocessers = None
        self._searchers = None

    def add_dedispersion(self, dedisperser):
        self._dedisperser = dedisperser

    def add_preprocessers(self, preprocessers):
        self._preprocessers = preprocessers

    def add_searchers(self, searchers):
        self._searchers = searchers

    @property
    def exp_params(self):
        """
        Returns dictionary with key - raw data file name & value - instance of
        MetaData.
        """
        return self.cfx.parse_cfx(self.exp_code)

    def run(self, freq_band_pol, chunk_size, n_nu, d_t, nu_0, d_nu, meta_data):
        exp_candidates = defaultdict(list)
        for m5_file, m5_params in self.exp_params.items():
            m5_file = os.path.join(self.raw_data_dir, m5_file)
            iterator = DSPIterator(m5_file, m5_params, chunk_size=chunk_size,
                                   n_nu=n_nu, d_t=d_t, nu_0=nu_0, d_nu=d_nu,
                                   meta_data=meta_data)
            pipeline = Pipeline(iterator, self._dedisperser,
                                self._preprocessers, self._searchers,
                                self.db_file, self.cache_dir)
            pipeline.run()


if __name__ == '__main__':
    db_file = None
    exp_code = None
    cache_dir = None
    raw_data_dir = None
    cfx_file = None

    from dedispersion import DeDisperser, noncoherent_dedispersion
    from preprocess import PreProcesser, create_ellipses
    from search import Searcher, search_clf, search_ell, search_shear
    dm_values = range(0, 1000, 30)
    dedisperser = DeDisperser(noncoherent_dedispersion, [dm_values],
                              {'threads': 4})
    preprocesser = PreProcesser(create_ellipses, [], {'disk_size': 3,
                                                      'threshold_big_perc': 90.,
                                                      'threshold_perc': 97.5,
                                                      'statistic': 'mean'})
    preprocessers = [None, preprocesser, preprocesser]

    # Training instance of ``PulseClassifier``
    from classifier import PulseClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    param_grid = {'learning_rate': [0.3, 0.1, 0.05, 0.01],
                  'max_depth': [2, 3, 4, 5],
                  'min_samples_leaf': [2, 3, 6, 10],
                  'max_features': [1.0, 0.5, 0.2, 0.1]}
    clf_kwargs = {'n_estimators': 3000}
    pclf = PulseClassifier(GradientBoostingClassifier, preprocesser, param_grid,
                           clf_kwargs)
    from Mk5 import DSPIterator
    dsp_train = DSPIterator(m5_file, m5_fmt, freq_band_pol, chunk_size, n_nu,
                            d_t, nu_0, d_nu, meta_data).get_dsp()
    features_dict, responses_dict = pclf.create_samples(dsp_train, pls_params)
    pclf.train(features_dict, responses_dict)


    searchers = [Searcher(search_shear, {'mph': 3.5, 'mpd': 50,
                                         'shear': 0.4}),
                 Searcher(search_ell, {'x_stddev': 10., 'y_to_x_stddev': 0.3,
                                        'theta_lims': [130., 180.],
                                        'x_cos_theta': 3., 'save_fig': True}),
                 Searcher(search_clf, {'save_fig': True})]
    ra_pipeline = RAPipeline(exp_code, cfx_file, raw_data_dir, db_file,
                             cache_dir)
    ra_pipeline.add_dedisperser(dedisperser)
    ra_pipeline.add_preprocessers(preprocessers)
    ra_pipeline.add_searchers(searchers)
    ra_pipeline.run()

