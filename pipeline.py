# -*- coding: utf-8 -*-
from searched_data import SearchedData
from db import connect_to_db


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
