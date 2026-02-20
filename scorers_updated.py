from __future__ import division
import numpy as np

from sklearn.metrics import make_scorer
from sklearn.metrics._regression import _check_reg_targets

from mcmodels.core import Mask
from mcmodels.utils import squared_norm


class HybridScorer(object):

    DEFAULT_STRUCTURE_SET_ID = 687527945

    @staticmethod
    def voxel_scorer():
        return make_scorer(mean_squared_relative_error, greater_is_better=False)


    @staticmethod
    def regional_scorer(**kwargs):
        return make_scorer(regional_mean_squared_relative_error, greater_is_better=False, **kwargs)

    @property
    def _default_structure_ids(self):
        structure_tree = self.cache.get_structure_tree()
        structures = structure_tree.get_structures_by_set_id([self.DEFAULT_STRUCTURE_SET_ID])

        return [s['id'] for s in structures if s['id'] not in (934, 1009)]

    def __init__(self, cache, structure_ids=None):
        self.cache = cache
        self.structure_ids = structure_ids

        if self.structure_ids is None:
            self.structure_ids = self._default_structure_ids

    @property
    def scoring_dict(self):

        def get_nnz_assigned(key):
            assigned = np.unique(key)
            if assigned[0] == 0:
                return assigned[1:]
            return assigned

        # target is whole brain
        target_mask = Mask.from_cache(cache=self.cache, hemisphere_id=3)

        ipsi_key = target_mask.get_key(structure_ids=self.structure_ids, hemisphere_id=2)
        contra_key = target_mask.get_key(structure_ids=self.structure_ids, hemisphere_id=1)

        reg_kwargs = dict(ipsi_key=ipsi_key,
                          contra_key=contra_key,
                          ipsi_regions=get_nnz_assigned(ipsi_key),
                          contra_regions=get_nnz_assigned(contra_key))

        return dict(voxel=self.voxel_scorer(), regional=self.regional_scorer(**reg_kwargs))

def log_mean_squared_relative_error(y_true, y_pred):
    """Compute mean squared relative error after log10 transformation.

    Args:
        y_true: Array of true values.
        y_pred: Array of predicted values.

    Returns:
        float: Mean squared relative error between log10(y_true) and log10(y_pred),
            where values are shifted by 1e-8 before taking the log.
    """
    log = lambda x: np.log10(x + 1e-8)
    return mean_squared_relative_error(log(y_true), log(y_pred))

def log_regional_mean_squared_relative_error(y_true, y_pred, **kwargs):
    """Compute regional mean squared relative error after log10 transformation.

    Args:
        y_true: Array of true values.
        y_pred: Array of predicted values.
        **kwargs: Passed to regional_mean_squared_relative_error (ipsi_key,
            contra_key, ipsi_regions, contra_regions).

    Returns:
        float: Regional mean squared relative error between log10-transformed inputs.
    """
    log = lambda x: np.log10(x + 1e-8)
    return regional_mean_squared_relative_error(log(y_true), log(y_pred), **kwargs)


class LogHybridScorer(HybridScorer):

    @staticmethod
    def voxel_scorer():
        return make_scorer(log_mean_squared_relative_error, greater_is_better=False)

    @staticmethod
    def regional_scorer(**kwargs):
        return make_scorer(log_regional_mean_squared_relative_error,
                           greater_is_better=False, **kwargs)


def unionize(v, ipsi_key, contra_key, ipsi_regions, contra_regions):
    """Unionize experiment connectivity array v to ipsilateral and contralateral regions.

    Sums columns of v according to the region assignments given by ipsi_key and
    contra_key, producing one aggregated value per region per experiment.

    Args:
        v: Array of shape (n_experiments, n_voxels) to unionize.
        ipsi_key: Array mapping voxels to ipsilateral region IDs.
        contra_key: Array mapping voxels to contralateral region IDs.
        ipsi_regions: Ordered list of ipsilateral region IDs to aggregate.
        contra_regions: Ordered list of contralateral region IDs to aggregate.

    Returns:
        numpy.ndarray: Array of shape (n_experiments, len(ipsi_regions) +
            len(contra_regions)) with summed values per region.

    Raises:
        ValueError: If ipsi_key and contra_key have different shapes, or if
            the number of voxels in v does not match the key size.
    """
    if ipsi_key.shape != contra_key.shape:
        # NOTE: better error message
        raise ValueError("keys are incompatible")

    v = np.atleast_2d(v)
    if v.shape[1] != ipsi_key.size: # or contra, doesnt matter
        raise ValueError("key must be the same size as the n columns in vector!")

    j = 0
    result = np.empty((v.shape[0], len(ipsi_regions) + len(contra_regions)))
    for key, regions in zip((ipsi_key, contra_key), (ipsi_regions, contra_regions)):
        for k in regions:
            result[:, j] = v[:, np.where(key == k)[0]].sum(axis=1)
            j += 1

    return result


def mean_squared_relative_error(y_true, y_pred, multioutput='uniform_average'):
    """Compute mean squared relative error between true and predicted values.

    Args:
        y_true: Array of true target values.
        y_pred: Array of predicted values.
        multioutput (str): Aggregation strategy for multiple outputs.
            Defaults to 'uniform_average'.

    Returns:
        float: Symmetric mean squared relative error:
            2 * ||y_true - y_pred||^2 / (||y_true||^2 + ||y_pred||^2).
    """
    _, y_true, y_pred, _ = _check_reg_targets(y_true, y_pred, multioutput)
    #result = squared_norm(y_true - y_pred) / max(squared_norm(y_true), squared_norm(y_pred))
    result = 2 * squared_norm(y_true - y_pred) / (squared_norm(y_true) + squared_norm(y_pred))
    return result


def regional_mean_squared_relative_error(y_true, y_pred, **kwargs):
    """Compute mean squared relative error at the regional (unionized) level.

    Args:
        y_true: Array of true voxel-level target values.
        y_pred: Array of predicted voxel-level values.
        **kwargs: Must include ipsi_key, contra_key, ipsi_regions, and
            contra_regions for unionization.

    Returns:
        float: Mean squared relative error between the regionalized true and
            predicted arrays.

    Raises:
        ValueError: If required kwargs (ipsi_key, contra_key, ipsi_regions,
            contra_regions) are not provided.
    """
    try:
        ipsi_key = kwargs.pop('ipsi_key')
        contra_key = kwargs.pop('contra_key')
        ipsi_regions = kwargs.pop('ipsi_regions')
        contra_regions = kwargs.pop('contra_regions')
    except KeyError:
        raise ValueError("must be called with 'key' and 'order' kwargs")

    y_true = unionize(y_true, ipsi_key, contra_key, ipsi_regions, contra_regions)
    y_pred = unionize(y_pred, ipsi_key, contra_key, ipsi_regions, contra_regions)

    return mean_squared_relative_error(y_true, y_pred, **kwargs)


def mse_rel():
    """Return a scikit-learn scorer for mean squared relative error.

    Returns:
        sklearn.metrics.scorer._PredictScorer: Scorer wrapping
            mean_squared_relative_error with greater_is_better=False.
    """
    return make_scorer(mean_squared_relative_error, greater_is_better=False)


def regional_mse_rel(**kwargs):
    """Return a scikit-learn scorer for regional mean squared relative error.

    Args:
        **kwargs: Passed to regional_mean_squared_relative_error (ipsi_key,
            contra_key, ipsi_regions, contra_regions).

    Returns:
        sklearn.metrics.scorer._PredictScorer: Scorer wrapping
            regional_mean_squared_relative_error with greater_is_better=False.
    """
    return make_scorer(regional_mean_squared_relative_error, greater_is_better=False, **kwargs)
