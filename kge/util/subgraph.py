import os
import numba
import numpy as np
import igraph as ig
from kge import Dataset
import yaml
import pandas as pd


class Subgraph:
    """
    Subgraph class.
    """

    RANDOM_SEED = 0
    # set random seed for train-valid-split
    np.random.seed(0)

    def __init__(self, dataset: Dataset):

        self._dataset = dataset
        self._valid_frac = 0.2
        self._valid_max = 5000
        self._subsets = dict()
        self._subset_stats = dict()

        self._train = self._dataset._triples["train"]
        self._valid = self._dataset._triples["valid"]
        self._entities = np.array(self._dataset._meta["entity_ids"])
        self._entity_ids = np.arange(len(self._entities))
        self._relations = np.array(self._dataset._meta["relation_ids"])
        self._relation_ids = np.arange(len(self._relations))

    def get_k_core_stats(self):
        """
        Provides stats about k-core subsets and performs the decomposition if they are not yet available.
        Select all entities that are contained in the k-cores (for all k) of the original graph and keep all
        interrelations. Note that multiple parallel edges only count as 1 due to the implementation.

        :return: subset_stats
        """

        # check if all files are already available. todo: May we assume that files re existent if stats are there?
        try:
            with open(f"{self._dataset.folder}/subsets/subset_stats.yaml", "r") as stream:
                self._subset_stats = yaml.safe_load(stream)
                # todo: write log-file message
        except IOError:

            # todo: write log-file message

            # convert tensor to numpy array
            train_np = self._train.cpu().detach().numpy()

            # perform k-core decomposition
            vertices = np.unique(train_np[:, (0, 2)])
            edges = train_np[:, (0, 2)]

            # create igraph
            graph = ig.Graph()
            graph.add_vertices(vertices)
            graph.add_edges(edges)
            graph.simplify(multiple=True, loops=True)

            # compute core values
            core_numbers = graph.coreness()

            # add whole graph stats
            self._subset_stats[0] = {"entities": len(self._entities), "relations": len(self._relations),
                                     "train": len(self._train), "valid": len(self._valid), "rel_triples": 1.0,
                                     "rel_entities": 1.0, "filename_suffix": ""}

            # compute k-cores
            k = 1
            previous_subset = train_np
            while True:
                # obtain entities of current k-core subgraph
                core_indices = [v_idx for v_idx in range(len(vertices)) if core_numbers[v_idx] >= k]
                k_core_graph = graph.subgraph(core_indices)
                if k_core_graph.vcount() == 0:
                    # exit loop if max k was reached
                    break
                else:
                    # select all triples that are contained in k-core
                    v_selected = k_core_graph.get_vertex_dataframe().name.values

                    # filter the previous subgraph triples with the list of entities
                    subset_core_indices = self.numba_is_in_2d(previous_subset, v_selected)
                    subset_core = previous_subset[subset_core_indices]
                    previous_subset = subset_core.copy()

                    # finalize all files and compute stats
                    self._finalize_and_compute_stats(subset_core, k)
                    k += 1

                    # todo: directly save files and write message into log file

            self._save_files()

        return self._subset_stats

    def _train_valid_split(self, subset):
        """
        Randomly split the subset into train and valid sets w.r.t. max amount

        :param subset: subset of original data to be divided into new train and valid
        :return: train, valid parts
        """

        number_valid = round(len(subset)*self._valid_frac)
        if number_valid > self._valid_max:
            number_valid = self._valid_max

        np.random.shuffle(subset)
        valid, train = subset[:number_valid, :], subset[number_valid:, :]

        return train, valid

    def _filter_entities_relations(self, subset):
        """
        Filter entities and relations and only keep those that appear in the subset. ALso reindex entities and
        relations for required density.

        :param subset: subset of original data to use for filtering
        :return: entities, relations, subset
        """

        selected_entity_ids = np.unique(subset[:, (0, 2)])
        selected_relation_ids = np.unique(subset[:, 1])

        # only select entities and relations that appear in subset
        entities_indices = self.numba_is_in_1d(self._entity_ids, selected_entity_ids)
        entities = self._entities[entities_indices]
        relation_indices = self.numba_is_in_1d(self._relation_ids, selected_relation_ids)
        relations = self._relations[relation_indices]

        # reindex the entity and relation ids
        new_entity_ids = np.arange(len(entities))
        new_relation_ids = np.arange(len(relations))

        entity_mapper = np.empty(len(self._entity_ids), dtype=np.long)
        relation_mapper = np.empty(len(self._relation_ids), dtype=np.long)

        entity_mapper[selected_entity_ids] = new_entity_ids
        relation_mapper[selected_relation_ids] = new_relation_ids

        for (i, mapper) in [(0, entity_mapper), (1, relation_mapper), (2, entity_mapper)]:
            subset[:, i] = mapper[subset[:, i]]

        entities_new = np.vstack((new_entity_ids, entities)).transpose()
        relations_new = np.vstack((new_relation_ids, relations)).transpose()

        return entities_new, relations_new, subset

    def _finalize_and_compute_stats(self, subset, core_number: int):
        """
        Finalize the subset creation and compute subset statistics.

        :param subset: subset of original data
        :param core_number: k-value of k-core
        :return: None
        """

        # filter and reindex files
        entities, relations, subset = self._filter_entities_relations(subset)

        # perform train-valid-split
        train, valid = self._train_valid_split(subset)

        # compute relative triples and entities compared to original graph
        rel_triples = len(train) / len(self._train)
        rel_entities = len(entities) / len(self._entities)

        # add subset statistics to dict
        self._subset_stats[core_number] = {"entities": len(entities), "relations": len(relations), "train": len(train),
                                           "valid": len(valid), "rel_triples": rel_triples, "rel_entities":
                                           rel_entities, "filename_suffix": f"_{core_number}_core"}

        # add subset and filtered files to dict
        self._subsets[core_number] = [entities, relations, train, valid]

    def _save_files(self):
        """
        Save stats in yaml and subsets in del format.
        """

        # check if sub folder was already created
        if not os.path.exists(f"{self._dataset.folder}/subsets"):
            os.mkdir(f"{self._dataset.folder}/subsets")

        # export subset stats
        f = open(f"{self._dataset.folder}/subsets/subset_stats.yaml", 'w+')
        yaml.dump(self._subset_stats, f, allow_unicode=True)

        # check if sub folder was already created
        if not os.path.exists(f"{self._dataset.folder}/subsets/k-core"):
            os.mkdir(f"{self._dataset.folder}/subsets/k-core")

        # save subset files
        for k, subset in self._subsets.items():

            pd.DataFrame(subset[0]).to_csv(f"{self._dataset.folder}/subsets/k-core/entity_ids_{k}_core.del", sep="\t",
                                           header=False, index=False)
            pd.DataFrame(subset[1]).to_csv(f"{self._dataset.folder}/subsets/k-core/relation_ids_{k}_core.del", sep="\t",
                                           header=False, index=False)
            pd.DataFrame(subset[2]).to_csv(f"{self._dataset.folder}/subsets/k-core/train_{k}_core.del", sep="\t",
                                           header=False, index=False)
            pd.DataFrame(subset[3]).to_csv(f"{self._dataset.folder}/subsets/k-core/valid_{k}_core.del", sep="\t",
                                           header=False, index=False)

    @staticmethod
    @numba.njit(parallel=True)
    def numba_is_in_2d(arr, vec2):
        """
        Return filtering mask for values that occur in vec2.
        """

        out = np.empty(arr.shape[0], dtype=numba.boolean)
        vec2_set = set(vec2)

        for i in numba.prange(arr.shape[0]):
            if arr[i][0] in vec2_set and arr[i][2] in vec2_set:
                out[i] = True
            else:
                out[i] = False

        return out

    @staticmethod
    @numba.njit(parallel=True)
    def numba_is_in_1d(arr, vec2):
        """
        Return filtering mask for values that occur in vec2.
        """

        out = np.empty(arr.shape[0], dtype=numba.boolean)
        vec2_set = set(vec2)

        for i in numba.prange(arr.shape[0]):
            if arr[i] in vec2_set:
                out[i] = True
            else:
                out[i] = False

        return out

