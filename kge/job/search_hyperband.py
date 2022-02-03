import concurrent.futures
import logging
import numpy as np
from kge.job import AutoSearchJob
from kge import Config
from kge import Dataset
import kge.job.search
from kge.util.package import package_model
from kge.config import _process_deprecated_options
import copy
import math
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import hpbandster.core.nameserver as hpns
from hpbandster.optimizers import HyperBand as HyperBand
from hpbandster.core.worker import Worker
from argparse import Namespace
import os
import yaml
import shutil


class HyperBandSearchJob(AutoSearchJob):
    """
        Job for hyperparameter search using HyperBand (Li et al. 2017)
        Source: https://github.com/automl/HpBandSter
    """

    def __init__(self, config: Config, dataset, parent_job=None):
        super().__init__(config, dataset, parent_job)
        self.name_server = None  # Server address to run the job on
        self.workers = []       # Workers that will run in parallel

    def init_search(self):
        # Assigning the port
        port = None \
            if self.config.get("hyperband_search.port") == "None" \
            else self.config.get("hyperband_search.port")

        # Assigning the address
        self.name_server = hpns.NameServer(run_id=self.config.get("hyperband_search.run_id"),
                                           host=self.config.get("hyperband_search.host"),
                                           port=port)
        # Start the server
        self.name_server.start()

        # Create workers (dummy logger to avoid output overhead from HPBandSter)
        for i in range(self.config.get("hyperband_search.num_workers")):
            w = HyperBandWorker(
                nameserver=self.config.get("hyperband_search.host"),
                logger=logging.getLogger('dummy'),
                run_id=self.config.get("hyperband_search.run_id"),
                job_config=self.config,
                parent_job=self,
                id=i
            )
            w.run(background=True)
            self.workers.append(w)

        if self.config.get("hyperband_search.variant") != "epochs":
            # compute relative sizes and costs for the available subsets
            self.subset_stats = self.config.get("hyperband_search.subsets")
            for i in range(len(self.subset_stats)):
                self.subset_stats[i]["relative_entities"] = \
                    self.subset_stats[i]["num_entities"] / self.subset_stats[0]["num_entities"]
                self.subset_stats[i]["relative_train"] = \
                    self.subset_stats[i]["num_train_triples"] / self.subset_stats[0]["num_train_triples"]
                # use custom power estimate if available, else compute it
                if "estimated_power_usage" in self.subset_stats[i]:
                    self.subset_stats[i]["relative_costs"] = self.subset_stats[i]["estimated_power_usage"]
                else:
                    self.subset_stats[i]["relative_costs"] = \
                        self.subset_stats[i]["relative_entities"] * self.subset_stats[i]["relative_train"]

            # load subgraph datasets
            self.subsets = list()
            for i in range(len(self.subset_stats)):
                config_custom_data = copy.deepcopy(self.config)
                config_custom_data = self.modify_dataset_config(i, config_custom_data)
                custom_dataset = Dataset.create(config_custom_data)
                self.subsets.append(custom_dataset)

        # create empty dict for id generation
        self.id_dict = dict()


    def modify_dataset_config(self, subset_index, config):
        """
        Modify the dataset part of a given config by replacing with the subset data.
        :return: modified config
        """
        config.set("dataset.num_entities",
                               self.subset_stats[subset_index]["num_entities"])
        config.set("dataset.num_relations",
                               self.subset_stats[subset_index]["num_relations"])
        config.set("dataset.files.entity_ids.filename",
                               f"entity_ids{self.subset_stats[subset_index]['name']}.del")
        config.set("dataset.files.entity_strings.filename",
                               f"entity_ids{self.subset_stats[subset_index]['name']}.del")
        config.set("dataset.files.relation_ids.filename",
                               f"relation_ids{self.subset_stats[subset_index]['name']}.del")
        config.set("dataset.files.relation_strings.filename",
                               f"relation_ids{self.subset_stats[subset_index]['name']}.del")
        config.set("dataset.files.train.filename",
                               f"train{self.subset_stats[subset_index]['name']}.del")
        config.set("dataset.files.valid.filename",
                               f"valid{self.subset_stats[subset_index]['name']}.del")
        # only set test set for original dataset. Use the valid sets for all others.
        if self.subset_stats[subset_index]['name'] == '':
            config.set("dataset.files.test.filename", "test.del")
        else:
            config.set("dataset.files.test.filename",
                       f"valid{self.subset_stats[subset_index]['name']}.del")
        return config

    def register_trial(self, parameters=None):
        # HyperBand does this itself
        pass

    def register_trial_result(self, trial_id, parameters, trace_entry):
        # HyperBand does this itself
        pass

    def get_best_parameters(self):
        # HyperBand does this itself
        pass

    def run(self):
        """
        Runs the hyper-parameter optimization program.
        :return:
        """
        self.init_search()

        # Configure the job
        hpb = HyperBand(
            configspace=self.workers[0].get_configspace(self.config, self.config.get("hyperband_search.seed")),
            run_id=self.config.get("hyperband_search.run_id"),
            nameserver=self.config.get("hyperband_search.host"),
            eta=self.config.get("hyperband_search.eta"),
            min_budget= 1 / math.pow(self.config.get("hyperband_search.eta"),
                                     self.config.get("hyperband_search.max_sh_rounds") - 1),
            max_budget= 1
        )
        # Run it
        res = hpb.run(n_iterations=self.config.get("hyperband_search.num_hpb_iter"),
                      min_n_workers=self.config.get("hyperband_search.num_workers"))

        # Shut it down
        hpb.shutdown(shutdown_workers=True)
        self.name_server.shutdown()


class HyperBandWorker(Worker):
    """
    Class of a worker for the HyperBand hyper-parameter optimization algorithm.
    """

    def __init__(self, *args, **kwargs):
        self.job_config = kwargs.pop('job_config')
        self.parent_job = kwargs.pop('parent_job')
        super().__init__(*args, **kwargs)
        self.next_trial_no = 0

    def compute(self, config_id, config, budget, **kwargs):
        """
        Creates a trial of the hyper-parameter optimization job and returns the best configuration of the trial.
        :param config_id: a triplet of ints that uniquely identifies a configuration. the convention is id =
        (iteration, budget index, running index)
        :param config: dictionary containing the sampled configurations by the optimizer
        :param budget: (float) amount of time/epochs/etc. the model can use to train
        :param kwargs:
        :return: dictionary containing the best hyper-parameter configuration of the trial
        """

        parameters = _process_deprecated_options(copy.deepcopy(config))

        # use first and thrid value of config_id to create the basis of the foldername
        hpb_iter = str('{:02d}'.format(config_id[0]))
        config_no = str('{:04d}'.format(config_id[2]))

        if (hpb_iter, config_no) in self.parent_job.id_dict:
            self.parent_job.id_dict[(hpb_iter, config_no)] += 1
        else:
            self.parent_job.id_dict[(hpb_iter, config_no)] = 0

        sh_iter = str('{:02d}'.format(self.parent_job.id_dict[(hpb_iter, config_no)]))

        # compute new result
        trial_no = f"{hpb_iter}{sh_iter}{config_no}"

        # create job for trial
        conf = self.job_config.clone(trial_no)
        conf.set("job.type", "train")
        conf.set_all(parameters)

        # determine the subset and epoch budget for this trial
        if self.parent_job.config.get("hyperband_search.variant") == 'dataset':
            size_budget = budget
            epochs = self.parent_job.config.get("hyperband_search.max_epoch_budget")
        elif self.parent_job.config.get("hyperband_search.variant") == 'both':
            size_budget = budget* math.pow(2, self.parent_job.config.get("hyperband_search.max_sh_rounds") -
                                           int(sh_iter) - 1)
            epoch_budget = budget * math.pow(2, self.parent_job.config.get("hyperband_search.max_sh_rounds") -
                                            int(sh_iter) - 1)
            epochs = math.floor(self.parent_job.config.get("hyperband_search.max_epoch_budget") * epoch_budget)
            epochs += self.parent_job.config.get("hyperband_search.epoch_budget_tolerance")[int(sh_iter)]
        elif self.parent_job.config.get("hyperband_search.variant") == 'epochs':
            epoch_budget = budget
            epochs = math.floor(self.parent_job.config.get("hyperband_search.max_epoch_budget") * epoch_budget)
            epochs += self.parent_job.config.get("hyperband_search.epoch_budget_tolerance")[int(sh_iter)]

        if self.parent_job.config.get("hyperband_search.variant") != 'epochs':
            # determine and set the dataset to use based on the budget
            for i in range(len(self.parent_job.subset_stats)):
                if self.parent_job.subset_stats[i]['relative_costs'] <= size_budget:
                    subset_index = i
                    break
            self.parent_job.dataset = self.parent_job.subsets[subset_index]
            conf = self.parent_job.modify_dataset_config(subset_index, conf)

            # downscale number of negatives
            number_samples_s = parameters.get("negative_sampling.num_samples.s")
            conf.set("negative_sampling.num_samples.s",
                     math.ceil(number_samples_s * self.parent_job.subset_stats[subset_index]["relative_entities"]))
            number_samples_o = parameters.get("negative_sampling.num_samples.o")
            conf.set("negative_sampling.num_samples.o",
                     math.ceil(number_samples_o * self.parent_job.subset_stats[subset_index]["relative_entities"]))

            # reuse the predecessor model checkpoint if available to keep initialization
            if sh_iter != '00':
                predecessor_trial_id = f"{hpb_iter}{str('{:02d}'.format(int(sh_iter) - 1))}{config_no}"
                path_to_model = f"{os.path.dirname(conf.folder)}/{predecessor_trial_id}/model_00000.pt"
                conf.set("lookup_embedder.pretrain.model_filename", path_to_model)

        # set the number of epochs
        conf.set("train.max_epochs", epochs)

        # set valid.every to train.max_epochs if its modulo != 0
        if epochs % self.parent_job.config.get("valid.every") != 0:
            conf.set("valid.every", epochs)

        # todo: compute total number of trials in init
        num_train_trials = 'x'

        # save config.yaml
        conf.init_folder()

        # copy last checkpoint from previous sh round to new folder for epoch only variant
        if self.parent_job.config.get("hyperband_search.variant") == 'epochs' and sh_iter != '00':
            predecessor_trial_id = f"{hpb_iter}{str('{:02d}'.format(int(sh_iter) - 1))}{config_no}"
            for filename in os.listdir(f"{os.path.dirname(conf.folder)}/{predecessor_trial_id}/"):
                if filename.endswith(".pt") and filename != 'checkpoint_best.pt':
                    shutil.copy(f"{os.path.dirname(conf.folder)}/{predecessor_trial_id}/{filename}",
                                f"{conf.folder}/{filename}")

        # run trial
        best = kge.job.search._run_train_job((
            self.parent_job,
            int(trial_no),
            conf,
            num_train_trials,
            list(parameters.keys()),
        ))
        self.parent_job.wait_task(concurrent.futures.ALL_COMPLETED)

        # save package checkpoint
        args = Namespace()
        args.checkpoint = f"{conf.folder}/checkpoint_00000.pt"
        args.file = None
        package_model(args)
        best_score = best[1]['metric_value']

        return {'loss': 1 - best_score, 'info': {}}

    @staticmethod
    def get_configspace(config, seed):
        """
        Reads the config file and produces the necessary variables. Returns a configuration space with
        all variables and their definition.
        :param config: dictionary containing the variables and their possible values
        :return: ConfigurationSpace containing all variables.
        """
        config_space = CS.ConfigurationSpace(
            seed=seed
        )

        parameters = config.get("hyperband_search.parameters")
        for p in parameters:
            v_name = p['name']
            v_type = p['type']

            if v_type == 'choice':
                config_space.add_hyperparameter(CSH.CategoricalHyperparameter(
                    v_name,
                    choices=p['values']
                ))
            elif v_type == 'range':
                log_scale = False
                if "log_scale" in p.keys():
                    log_scale = p['log_scale']
                if type(p['bounds'][0]) is int and type(p['bounds'][1]) is int:
                    config_space.add_hyperparameter(CSH.UniformIntegerHyperparameter(
                        name=v_name,
                        lower=p['bounds'][0],
                        upper=p['bounds'][1],
                        default_value=p['bounds'][1],
                        log=log_scale
                    ))
                else:
                    config_space.add_hyperparameter(CSH.UniformFloatHyperparameter(
                        name=v_name,
                        lower=p['bounds'][0],
                        upper=p['bounds'][1],
                        default_value=p['bounds'][1],
                        log=log_scale
                    ))
            elif v_type == 'fixed':
                config_space.add_hyperparameter(CSH.Constant(
                    name=v_name,
                    value=p['value']
                ))
            else:
                raise ValueError("Unknown variable type")
        return config_space