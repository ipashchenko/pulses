class DeDisperser(object):
    def __init__(self, func, *args, **kwargs):
        pass
        
    def __call__(self, dsp, cache_dir):
        pass
        
    def clear_cache(dsp, cache_dir):
        pass
        
class MetaData(dict):
    def __init__(self):
        pass
                
class DynSpectra(object):
    def __init__(self):
        pass
        
def create_from_txt(fn, t0, dt, nchan, metadata):
    pass
    
def create_from_hdf5(fn, t0, dt, nchan, d_t, metadata):
    pass
    
def create_from_mk5(fn, fmt, t0, dt, nchan, d_t, metadata):
    pass
    
def create_from_fits(fn, t0, dt, nchan, d_t, metadata):
    pass

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
       Generator that returns instances of ``DynSpectra`` class.
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
            dsp = DynSpectra(2 * n_nu, n_t, nu_0, d_nu, d_t, meta_data, t_0)
            dsp.add_values(dsarr.T)
            offset += chunk_size

yield dsp
        yield dsp
        
class DedispersedDynSpectra(object):
    def __init__(self):
        self.array
        self.dm
        self.t
        self.dsp
        self.dd_params
        
class PreProcesser(object):
    def __init__(self, func, *args, **kwargs):
        pass
        
    def __call__(self, dd_dsp, cache_dir):
        pass
        
    def clear_cache(dd_dsp, cache_dir):
        pass

# FIXME: Should work with any class instances - `dsp` or `dddsp`        
class Searcher(object):
    def __init__(func, db_file, *args, **kwargs):
        pass
        
    def __call__(self, dd_dsp):
        pass
        
class PulseClassifier(object):
    def __init__(clf, dd, prep, *dd_args, *pre_args,
                 **pre_kwargs):
        pass
        
    def train(dsp, n_pulses):
        pass

class Pipeline(object):
    def __init__(self, iterator, de_disperser, pre_processers, searchers, db_file, cache_dir=None):
        """
        :param iterator:
            Iterator that returns instances of ``DynSpectra`` class.
        :param de_disperser:
            Instance of ``DeDisperion`` class used to de-disperse data.
        :param pre_processers:
            Iterable of ``PreProcesser`` class instances used to pre-process
            de-dispersed data.
        :param searchers:
            Iterable of ``Searcher`` class instances used to search pulses in
            de-dispersed & optionally preprocessed data.
        """
        self.iterator = iterator
        self.de_disperser = de_disperser
        self.pre_processers = pre_processers
        self.searchers = searchers
        assert len(pre_processers) == len(searchers)
        
    def run(self):
        for dsp in self.iterator:         
            for pre_processer, searcher in zip(self.pre_processers, self.searchers):
                dddsp = de_disperser(dsp, cache_dir=cache_dir)
                try:
                    dddsp = pre_processor(dddsp, cache_dir=cache_dir)
                except TypeError:
                    pass
                candidates = searcher(dddsp)

class CFX(object):
    pass
 
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
        
    def add_dedispersion(dedisperser):
        self._dedisperser = dedisperser
        
    def add_preprocessers(preprocessers):
        self._preprocessers = preprocessers
        
    def add_searchers(searchers):
        self._searchers = searchers
        
   @property
    def exp_params(self):
        """
        Returns dictionary with key - raw data file name & value - instance of
        MetaData.
        """
        return self.cfx.parse_cfx(self.exp_code)

    def run(self):
        exp_candidates = defaultdict(list)
        for m5_file, m5_params in self.exp_params.items():
            m5_file = os.path.join(self.raw_data_dir, m5_file)
            iterator = DSPIterator(m5_file, m5_params)
            pipeline = Pipeline(iterator, self._dedisperser, self._preprocessers, self._searchers,
                                self.db_file, self.cache_dir)
            pipeline.run()
            
if __name__ = '__main__':
    dedisperser = DeDisperser(non_coherent_dedispersion, [dm_values], {'threads': 4})
    preprocessers = [None, PreProcesser(create_ellipses, [],
                                        {'disk_size': 3,
                                         'threshold_big_perc': 90.,
                                         'threshold_perc': 97.5,
                                         'statistic': 'mean'})]
    searchers = [Searcher(search_shear, {'mph': 3.5, 'mpd': 50,
                                         'shear': 0.4, 'd_dm': d_dm}),
                 Searcher(search_ell, {'x_stddev': 10., 'y_to_x_stddev': 0.3,
                                        'theta_lims': [130., 180.],
                                        'x_cos_theta': 3., 'd_dm': d_dm,
                                         'amplitude': None, 'save_fig': True})]
    ra_pipeline = RAPipeline(exp_code, cfx_file, raw_data_dir, db_file, cache_dir)
    ra_pipeline.add_dedisperser(dedisperser)
    ra_pipeline.add_preprocessers(preprocessers)
    ra_pipeline.add_searchers(searchers)
    ra_pipeline.run()
                                            
