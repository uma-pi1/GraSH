import os
import pandas as pd
import igraph as ig
from sklearn.model_selection import train_test_split
from kge import Dataset
import yaml


class Subgraph:
    """
    Stores graph data (train, valid, test) and provides methods to perform compression
    utilizing various strategies.
    """

    TRIPLE_COLUMNS = ['subj', 'rel', 'obj']
    TUPLE_COLUMN = ['name']
    # todo: use hyperband seed?
    RANDOM_SEED = 0

    def __init__(self, dataset: Dataset):

        self._dataset = dataset
        self._valid_frac = 0.2
        self._valid_max = 5000
        self._subsets = dict()
        self._subset_stats = dict()

        """

        # prepare relevant knowledge graph files for subset creation
        self._train = self._dataset._triples["train"]
        self._train_df = pd.DataFrame(self._train, columns=self.TRIPLE_COLUMNS).astype("int")
        self._valid = self._dataset._triples["valid"]
        self._valid_df = pd.DataFrame(self._valid, columns=self.TRIPLE_COLUMNS).astype("int")
        self._entities_df = pd.DataFrame(self._dataset._meta["entity_ids"], columns=self.TUPLE_COLUMN)
        self._entities_df.index.name = "id"
        self._relations_df = pd.DataFrame(self._dataset._meta["relation_ids"], columns=self.TUPLE_COLUMN)
        self._relations_df.index.name = "id"

        """

    def get_k_core_stats(self):
        """
        Select all entities that are contained in the k-cores (for all k) of the original graph and keep all
        interrelations. Note that multiple parallel edges only count as 1 due to the implementation.

        :return: None
        """

        # check if all files are already available. todo: May we assume that files re existent if stats are there?
        try:
            with open(f"{self._dataset.folder}/subsets/subset_stats.yaml", "r") as stream:
                self._subset_stats = yaml.safe_load(stream)
        except IOError:
            # perform k-core decomposition
            vertices = self._train[:, (0, 2)].unique(sorted=True)
            edges = self._train[:, (0, 2)]

            # create igraph
            graph = ig.Graph()
            graph.add_vertices(vertices.tolist())
            graph.add_edges(edges.tolist())
            graph.simplify(multiple=True, loops=True)

            # compute core values
            core_numbers = graph.coreness()

            # add whole graph stats
            self._subset_stats[0] = {"entities": len(self._entities_df), "relations": len(self._relations_df),
                                     "train": len(self._train_df), "valid": len(self._valid_df), "rel_triples": 1.0,
                                     "rel_entities": 1.0, "filename_suffix": ""}

            # compute k-cores
            k = 1
            while True:
                core_indices = [v_idx for v_idx in range(len(vertices)) if core_numbers[v_idx] >= k]
                k_core_graph = graph.subgraph(core_indices)
                if k_core_graph.vcount() == 0:
                    # exit loop if max k was reached
                    break
                else:
                    # select all triples that are contained in k-core and save file
                    v_selected = k_core_graph.get_vertex_dataframe().name.values
                    subset_core = self._train_df[self._train_df.subj.isin(v_selected) & self._train_df.obj.isin(v_selected)]
                    self._finalize_and_compute_stats(subset_core, k)
                    k += 1

            self._save_files()

        return self._subset_stats

    def _train_valid_split(self, subset: pd.DataFrame):
        """
        Randomly split the subset into train and valid sets

        :param subset: subset of original data to be divided into new train and valid
        :return: train, valid Dataframes
        """

        if len(subset)*self._valid_frac < self._valid_max:
            train, valid = train_test_split(subset, test_size=self._valid_frac, random_state=self.RANDOM_SEED)
        else:
            train, valid = train_test_split(subset, test_size=self._valid_max, random_state=self.RANDOM_SEED)

        return train, valid

    def _filter_entities_relations(self, subset: pd.DataFrame):
        """
        Filter entities and relations file and only keep those that appear in the subset. ALso reindex entities and
        relations for required density.

        :param subset: subset of original data to use for filtering
        :return: entities, relations, subset Dataframes
        """

        # only select entities and relations that appear in subset
        entities = self._entities_df[(self._entities_df.index.isin(subset.subj)) |
                                     (self._entities_df.index.isin(subset.obj))]
        relations = self._relations_df[self._relations_df.index.isin(subset.rel)]

        # add new dense index column
        entities.insert(0, 'id_new', range(0, len(entities)))
        relations.insert(0, 'id_new', range(0, len(relations)))

        # merge with subset to update ids
        subset = subset.merge(entities.rename(columns={'id_new': 'subj_new'}), left_on='subj', right_index=True)
        subset = subset.merge(entities.rename(columns={'id_new': 'obj_new'}), left_on='obj', right_index=True)
        subset = subset.merge(relations.rename(columns={'id_new': 'rel_new'}), left_on='rel', right_index=True)

        # only keep new columns and rename them
        subset = subset[['subj_new', 'rel_new', 'obj_new']]
        subset.rename(columns={'subj_new': 'subj', 'rel_new': 'rel', 'obj_new': 'obj'}, inplace=True)
        entities = entities[['id_new', 'name']]
        entities.rename(columns={'id_new': 'id'}, inplace=True)
        relations = relations[['id_new', 'name']]
        relations.rename(columns={'id_new': 'id'}, inplace=True)

        return entities, relations, subset

    def _finalize_and_compute_stats(self, subset: pd.DataFrame, core_number: int):
        """
        Reindex subset, relations, and entities. Save all files.

        :param subset: subset of original data
        :param core_number: k-value of k-core
        :return: None
        """

        # filter and reindex files
        entities, relations, subset = self._filter_entities_relations(subset)

        # perform train-valid-split
        train, valid = self._train_valid_split(subset)

        # compute relative computational savings compared to original dataset with and without scaling negative samples
        rel_triples = len(train) / len(self._train_df)
        rel_entities = len(entities) / len(self._entities_df)

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
            subset[0].to_csv(f"{self._dataset.folder}/subsets/k-core/entity_ids_{k}_core.del", sep='\t', header=False,
                             index=False)
            subset[1].to_csv(f"{self._dataset.folder}/subsets/k-core/relation_ids_{k}_core.del", sep='\t', header=False,
                             index=False)
            subset[2].to_csv(f"{self._dataset.folder}/subsets/k-core/train_{k}_core.del", sep='\t', header=False,
                             index=False)
            subset[3].to_csv(f"{self._dataset.folder}/subsets/k-core/valid_{k}_core.del", sep='\t', header=False,
                             index=False)




