"""Retrieve and organize Allen connectivity experiment data for model building."""

import numpy as np
from mcmodels.core import VoxelData, RegionalData
from mcmodels.utils import unionize
import pandas as pd

class ModelData(object):
    """Wrapper for Allen Mouse Connectivity experiment data retrieval.

    Args:
        cache: VoxelModelCache instance for data access.
        structure_id: Allen brain structure ID for filtering experiments.
    """

    def __init__(self, cache, structure_id):
        self.cache = cache
        self.structure_id = structure_id

    def get_structure_id(self, acronym):
        """Look up structure ID by acronym.

        Args:
            acronym: Brain structure acronym string.

        Returns:
            Integer structure ID.
        """
        structure_tree = self.cache.get_structure_tree()
        return structure_tree.get_structures_by_acronym([acronym])[0]['id']

    def get_experiment_ids(self, eid_set=None, experiments_exclude=None):
        """Get filtered experiment IDs for this structure.

        Args:
            eid_set: Optional set of experiment IDs to restrict to.
            experiments_exclude: Optional list of experiment IDs to exclude.

        Returns:
            Set of experiment IDs after filtering.
        """
        if experiments_exclude is None:
            experiments_exclude = []

        # get experiments
        experiments = self.cache.get_experiments(
            injection_structure_ids=[self.structure_id], cre=False)
        experiment_ids = [e['id'] for e in experiments]

        # exclude bad, restrict to eid_set
        eid_set = experiment_ids if eid_set is None else eid_set
        return set(experiment_ids) & set(eid_set) - set(experiments_exclude)

    def get_voxel_data(self, **kwargs):
        """Retrieve voxel-level injection/projection data.

        Args:
            **kwargs: Passed to get_experiment_ids for filtering.

        Returns:
            VoxelData instance with experiment data loaded.
        """
        experiment_ids = self.get_experiment_ids(**kwargs)

        data = VoxelData(self.cache, injection_structure_ids=[self.structure_id],
                         injection_hemisphere_id=2)
        data.get_experiment_data(experiment_ids)

        return data

    def get_regional_data(self, rgn_list_path, high_res=False, threshold_injection=True, **kwargs):
        """Retrieve regionalized injection/projection data.

        Args:
            rgn_list_path: Path to CSV file listing region IDs.
            high_res: If True, use RegionalData instead of VoxelData.
            threshold_injection: If True, zero out injections below 5th percentile.
            **kwargs: Passed to get_experiment_ids for filtering.

        Returns:
            Data container with injections and projections loaded.
        """
        def get_summary_structure_ids(): ###toggle/change this to be consistent with Oh et al., 2014
            #structure_tree = self.cache.get_structure_tree()
            #structures = structure_tree.get_structures_by_set_id([687527945])
            #return [s['id'] for s in structures if s['id'] not in (934, 1009)]
            structures = pd.read_csv(rgn_list_path, header=None).loc[:,0]
            return structures
        
        def get_injection_regions(region_set):
            """Return regions in region_set if descend from structure_id"""
            st = self.cache.get_structure_tree()
            return [r for r in region_set
                    if st.structure_descends_from(r, self.structure_id)]
        projection_hemisphere_id = kwargs.pop('projection_hemisphere_id', 3)

        # get summary structures
        region_set = get_summary_structure_ids()
        injection_set = get_injection_regions(region_set)

        # get experiments
        experiment_ids = self.get_experiment_ids(**kwargs)

        container = RegionalData if high_res else VoxelData
        container_kwargs = dict(injection_structure_ids=injection_set,
                                projection_structure_ids=region_set,
                                injection_hemisphere_id=2,
                                projection_hemisphere_id=projection_hemisphere_id,
                                normalized_injection=True,
                                normalized_projection=True,
                                flip_experiments=True)

        data = container(self.cache, **container_kwargs)
        data.get_experiment_data(experiment_ids)

        # get model data
        if not high_res:
            data.injections = unionize(
                data.injections, data.injection_mask.get_key(
                    structure_ids=injection_set, hemisphere_id=2))
            data.projections = unionize(
                data.projections, data.projection_mask.get_key(
                    structure_ids=region_set, hemisphere_id=projection_hemisphere_id))

        # threshold injection
        # NOTE
        if threshold_injection:
            pct = np.percentile(data.injections[data.injections.nonzero()], 5)
            data.injections[data.injections < pct] = 0

        return data
