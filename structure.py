class DeDisperser(object):
    def __init__(self, func, do_caching, cache_dir):
        pass
        
    def __call__(self, dsp, *args, **kwargs):
        pass
        
    def reset_cache(dsp, cache_dir):
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
        
class DedispersedDynSpectra(object):
    def __init__(self):
        self.array
        self.dm
        self.t
        self.dsp
        self.dd_params
        
class PreProcesser(object):
    def __init__(self, func, do_caching, cache_dir, *args, **kwargs):
        pass
        
    def __call__(self, dd_dsp):
        pass
        
    def reset_cache(dd_dsp, cache_dir):
        pass

# FIXME: Should work with any class instances - `dsp` or `dddsp`        
class Searcher(object):
    def __init__(func, *args, **kwargs):
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
    def __init__(self, iterator, de_dispersers, pre_processers, searchers, db_file):
        """
        :param iterator:
            Iterator that returns instances of ``DynSpectra`` class.
        :param de_dispersers:
            Instance of ``DeDisperion`` class used to de-disperse data.
        :param pre_processers:
            Iterable of ``PreProcesser`` class instances used to pre-process
            de-dispersed data.
        :param searchers:
            Iterable of ``Searcher`` class instances used to search pulses in
            de-dispersed & optionally preprocessed data.
        """
        self.iterator = iterator
        
    def run(self):
        for dsp in self.iterator:         
            dd = Dedispersion(de_disp_params['func'])
            prep = PreProcessing(pre_process_params['func'])
            dd_dsp = dd(dsp, dm_values, threads)
            search_1 = Search(search_params[0]['func'])
            candidates_1 = search_1(dd_dsp, *args, **kwargs)

            dd_dsp = prep(dd_dsp, *pre_args, **pre_kwargs)

            search_2 = Search(search_ell)
            search_3 = Search(search_clf)

            candidates_2 = search_2(dd_dsp, *args, **kwargs)
            candidates_3 = search_3(dd_dsp, *args, **kwargs)
            t0 += dt


 
class RAPipeline(object):
    def __init__(self, exp_code, cfx_file, dsp_params, raw_data_dir, db_file):
        pass
        
   @property
    def exp_params(self):
        """
        Returns dictionary with key - raw data file name & value - instance of
        MetaData.
        """
        return self.cfx.parse_cfx(self.exp_code)
В 
    def dsp_generator(self, m5_file, m5_params, chunk_size):
        pass
        
    for m5_file, m5_params in self.exp_params.items():
            m5_file = os.path.join(self.raw_
       
t0 = 0    
while True:         
    dsp = create_from_mk5(fn, fmt, t0, dt, nchan, d_t, metadata)
    dd = Dedispersion(non_coher_dd)
    prep = PreProcessing(create_ellipses)
    dd_dsp = dd(dsp, dm_values, threads)
    search_1 = Search(search_shear)
    candidates_1 = search_1(dd_dsp, *args, **kwargs)

    dd_dsp = prep(dd_dsp, *pre_args, **pre_kwargs)

    search_2 = Search(search_ell)
    search_3 = Search(search_clf)

    candidates_2 = search_2(dd_dsp, *args, **kwargs)
    candidates_3 = search_3(dd_dsp, *args, **kwargs)
    t0 += dt
