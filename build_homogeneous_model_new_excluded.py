from __future__ import division
import os
import logging

import numpy as np
import sys
import pandas as pd
sys.path.insert(0,"./mouse_connectivity_models/paper/figures/model_comparison/")
import allensdk.core.json_utilities as ju

from mcmodels.core import VoxelModelCache
from mcmodels.models import HomogeneousModel
from mcmodels.utils import nonzero_unique

####model_data_updated accounts for Oh regions

from model_data_updated import ModelData
from helpers.utils import get_structure_id, get_ordered_summary_structures


exp_input_dir='./mouse_connectivity_models/paper/'
INPUT_JSON = os.path.join(exp_input_dir, 'input.json')
TOP_DIR=exp_input_dir
ROOT_ID = 997
HIGH_RES = False
OUTPUT_DIR='./mouse_connectivity_models/paper/figures/model_comparison/output/'
THRESHOLD_INJECTION = True

def get_summary_structure_ids(rgn_list_path): ###change this to be consistent with Oh et al., 2014
    """Read structure IDs from a region list CSV file.

    Args:
        rgn_list_path (str): Path to a headerless CSV file whose first column
            contains integer structure IDs.

    Returns:
        pandas.Series: Series of structure IDs.
    """
    structures = pd.read_csv(rgn_list_path, header=None).loc[:,0]
    return structures


def fit(cache, rgn_list_path, eid_set=None, experiments_exclude=[], high_res=False, threshold_injection=True):
    """Fit a homogeneous connectivity model for ipsilateral and contralateral projections.

    Args:
        cache: VoxelModelCache instance providing access to Allen SDK data.
        rgn_list_path (str): Path to a headerless CSV file of region structure IDs.
        eid_set (list, optional): Restrict fitting to these experiment IDs.
            Defaults to None (use all available experiments).
        experiments_exclude (list): Experiment IDs to exclude. Defaults to [].
        high_res (bool): Use high-resolution regional data. Defaults to False.
        threshold_injection (bool): Apply injection thresholding. Defaults to True.

    Returns:
        pandas.DataFrame: Multi-level DataFrame with ipsilateral and contralateral
            weight matrices indexed by injection region and columned by projection region.
    """
    logging.debug('getting data')
    ipsi_data = ModelData(cache, ROOT_ID).get_regional_data(rgn_list_path,
        eid_set=eid_set, experiments_exclude=experiments_exclude, high_res=high_res,
        threshold_injection=threshold_injection, projection_hemisphere_id=2)

    contra_data = ModelData(cache, ROOT_ID).get_regional_data(rgn_list_path,
        eid_set=eid_set, experiments_exclude=experiments_exclude, high_res=high_res,
        projection_hemisphere_id=1, threshold_injection=threshold_injection)

    X = ipsi_data.injections
    y = np.hstack((ipsi_data.projections, contra_data.projections))

    logging.debug('fitting')
    reg = HomogeneousModel(kappa=np.inf)
    reg.fit(X, y)

    # get ids
    ss_ids = get_summary_structure_ids(rgn_list_path)
    injection_key = ipsi_data.injection_mask.get_key(structure_ids=ss_ids, hemisphere_id=2)
    ipsi_key = ipsi_data.projection_mask.get_key(structure_ids=ss_ids, hemisphere_id=2)
    contra_key = contra_data.projection_mask.get_key(structure_ids=ss_ids, hemisphere_id=1)

    injection_regions = nonzero_unique(injection_key)
    ipsi_regions = nonzero_unique(ipsi_key)
    contra_regions = nonzero_unique(contra_key)

    ipsi_w = pd.DataFrame(data=reg.weights[:, :len(ipsi_regions)],
                          index=injection_regions,
                          columns=ipsi_regions)
    contra_w = pd.DataFrame(data=reg.weights[:, len(ipsi_regions):],
                            index=injection_regions,
                            columns=contra_regions)

    return pd.concat((ipsi_w, contra_w), keys=('ipsi', 'contra'), axis=1)


def main():
    """Build and save a homogeneous connectivity model.

    Reads configuration from Allen SDK input.json, fits a HomogeneousModel using
    the specified experiments-to-exclude JSON (sys.argv[1]), region list (sys.argv[2]),
    and output suffix (sys.argv[3]), then writes the resulting weight DataFrame as a CSV.
    """
    input_data = ju.read(INPUT_JSON)

    manifest_file = input_data.get('manifest_file')
    manifest_file = os.path.join(TOP_DIR, manifest_file)

    log_level = input_data.get('log_level', logging.DEBUG)
    logging.getLogger().setLevel(log_level)

    # experiments to exclude
    ###CHANGED FROM ORIGINAL LIST TO ACCOUNT FOR QC FAILURE MODES
    exp_exc_path=sys.argv[1]
    EXPERIMENTS_EXCLUDE_JSON = os.path.join("./mouse_connectivity_models/paper/", exp_exc_path)
    experiments_exclude = ju.read(EXPERIMENTS_EXCLUDE_JSON)

    # load hyperparameter dict
    suffix = 'high_res' if HIGH_RES else 'standard'

    # get caching object
    cache = VoxelModelCache(manifest_file=manifest_file)

    fit_kwargs = dict(high_res=HIGH_RES, threshold_injection=THRESHOLD_INJECTION,
                      experiments_exclude=experiments_exclude)
    
    rgn_list_path=sys.argv[2]
    model = fit(cache, rgn_list_path, **fit_kwargs)

    # write results
    logging.debug('saving')
    outfile_suffix = sys.argv[3]
    output_file = os.path.join(
        OUTPUT_DIR,
        'homogeneous-%s-model_%s.csv' % (suffix, outfile_suffix)
     )


    model.to_csv(output_file)


if __name__ == "__main__":
    main()
