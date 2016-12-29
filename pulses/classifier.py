# -*- coding: utf-8 -*-
import numpy as np
from utils import get_ellipse_features_for_classification


class PulseClassifier(object):
    def __init__(self, clf, preprocessor, param_grid=None, **clf_kwargs):
        self.clf = clf
        self.preprocessor = preprocessor
        self.param_grid = param_grid
        self.clf_kwargs = clf_kwargs

    def create_samples(self, dsp, pls_params=None):
        pass

    # TODO: Use CV only if ``param_grid`` is not ``None``.
    def train(self):
        pass

    def classify(self, image):
        """
        Classify objects in ``image``.

        :param image:
            2D image [eg. de-dispersed and pre-processed dynamical spectra].

        :return:
            2 dictionaries. Keys - instances of
            ``skimage.measure._regionprops._RegionProperties``, values - arrays
            of feature values & array of responses.
        """
        features_dict = get_ellipse_features_for_classification(image)

        # Remove regions with ``nan`` features
        for prop in sorted(features_dict):
            features = np.array(features_dict[prop])
            if np.any(np.isnan(features)):
                del features_dict[prop]

        X = list()
        for prop in sorted(features_dict):
            features = np.array(features_dict[prop])
            X.append(features)
        print "Sample consists of {} samples".format(len(X))
        X_scaled = self.scaler.transform(X)
        y = self._clf.predict(X_scaled)
        y_arr = np.array(y)
        positive_indx = y_arr == 1
        print "Predicted probabilities of being fake/real FRBs for found" \
              " candidates :"
        print self._clf.predict_proba(X_scaled[positive_indx])
        responces_dict = dict()
        for i, prop in enumerate(sorted(features_dict)):
            responces_dict[prop] = y[i]

        return features_dict, responces_dict

