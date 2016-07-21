class PreProcesser(object):
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self, dddsp, cache_dir):
        cache_fname = os.path.join(cache_dir,
                                    dddsp.dsp._cache_fname_prefix +
                                    "_preprocess.hdf5")
        dddsp.array = self.func(dddsp.array, *self.args, **self.kwargs)

    def clear_cache(dddsp, cache_dir):
        pass
