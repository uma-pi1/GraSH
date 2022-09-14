import concurrent.futures
import logging
import numpy as np
import torch.cuda
import copy
import math
import ConfigSpace as CS
import kge.job.search
import ConfigSpace.hyperparameters as CSH
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
import os
import sys
import shutil
import time
import gc
import torch.multiprocessing as mp

from kge.job import AutoSearchJob
from kge import Config, Dataset
from kge.util.package import package_model
from kge.config import _process_deprecated_options
from hpbandster.optimizers import HyperBand
from hpbandster.core.worker import Worker
from argparse import Namespace
from collections import defaultdict
from multiprocessing import Manager
from kge.util import Subgraph


class HyperBandPatch(HyperBand):
    def __init__(
        self,
        free_devices,
        trial_dict,
        id_dict,
        result_dict,
        workers_per_round=None,
        configspace=None,
        eta=3,
        min_budget=0.01,
        max_budget=1,
        **kwargs,
    ):
        self.free_devices = free_devices
        self.trial_dict = trial_dict
        self.id_dict = id_dict
        self.result_dict = result_dict
        self.assigned_devices = defaultdict(lambda: None)
        if workers_per_round is None:
            self.workers_per_round = defaultdict(lambda: 10000)
        else:
            self.workers_per_round = workers_per_round
        # todo: add option to block search workers
        super(HyperBandPatch, self).__init__(
            configspace, eta, min_budget, max_budget, **kwargs
        )

    def _submit_job(self, config_id, config, budget):
        hpb_iter = str("{:02d}".format(config_id[0]))
        config_no = str("{:04d}".format(config_id[2]))
        sh_iter = 0
        if (hpb_iter, config_no) in self.id_dict:
            sh_iter = self.id_dict[(hpb_iter, config_no)]
        # todo: we should not do busy waiting here
        # block search workers if workers are reduces for specific sh round
        while self.num_running_jobs >= self.workers_per_round[sh_iter]:
            time.sleep(1)
        with self.thread_cond:
            self.assigned_devices[config_id] = self.free_devices.pop()
            config["job.device"] = self.assigned_devices[config_id]
        return super(HyperBandPatch, self)._submit_job(config_id, config, budget)

    def job_callback(self, job):
        with self.thread_cond:
            self.free_devices.append(self.assigned_devices[job.id])
            del self.assigned_devices[job.id]
        return super(HyperBandPatch, self).job_callback(job)


class GraSHSearchJob(AutoSearchJob):
    """
    Job for hyperparameter search using GraSH (Kochsiek et al. 2022)
    Source: todo: add github link
    """

    def __init__(self, config: Config, dataset, parent_job=None):
        super().__init__(config, dataset, parent_job)
        self.name_server = None  # Server address to run the job on
        self.workers = []  # Workers that will run in parallel
        # create empty dict for id generation
        manager = Manager()
        self.trial_dict = manager.dict()
        self.id_dict = manager.dict()
        self.result_dict = manager.dict()
        self.processes = []
        self.sh_rounds = 0
        self.eta = 0
        self.num_trials = 0
        self.subset_stats = dict()
        self.subsets = dict()

    def init_search(self):
        # Assigning the port
        port = (
            None
            if self.config.get("grash_search.port") == "None"
            else self.config.get("grash_search.port")
        )

        # Assigning the address
        self.name_server = hpns.NameServer(
            run_id=self.config.get("grash_search.run_id"),
            host=self.config.get("grash_search.host"),
            port=port,
        )
        # Start the server
        self.name_server.start()

        # Load prior results if available
        if self.results:
            for k, v in self.results[0].items():
                self.trial_dict[k] = v

        # Determine corresponding Hyperband configuration for GraSH configuration
        self.num_trials = self.config.get("grash_search.num_trials")
        self.eta = self.config.get("grash_search.eta")
        sh_rounds = math.log(self.num_trials, self.eta)
        if not sh_rounds.is_integer():
            if self.config.get("job.auto_correct"):
                sh_rounds = math.floor(sh_rounds)
                self.num_trials = self.eta ** sh_rounds
                self.config.log(
                    "Setting grash_search.num_trials to {}, was set to {} and needs to "
                    "equal a positive integer power of eta.".format(self.num_trials,
                                                                    self.config.get("grash_search.num_trials"))
                )
            else:
                raise Exception(
                    "grash_search.num_trials was set to {}, "
                    "needs to equal a positive integer power of eta.".format(self.num_trials)
                )
        self.sh_rounds = int(sh_rounds)

        # Perform k-core decomposition if not done yet
        if self.config.get("grash_search.variant") != "epoch":
            # add full dataset to subset dict and get k_core_stats
            self.subsets[0] = self.dataset
            subgraph = Subgraph(self.dataset)
            self.subset_stats = subgraph.get_k_core_stats()

            # compute relative sizes and costs for the available subsets
            cost_metric = self.config.get("grash_search.cost_metric")
            for i in range(len(self.subset_stats)):
                if cost_metric in ["triples_and_entities"]:
                    self.subset_stats[i]["rel_costs"] = (
                        self.subset_stats[i]["rel_entities"]
                        * self.subset_stats[i]["rel_triples"]
                    )
                elif cost_metric == "triples":
                    self.subset_stats[i]["rel_costs"] = self.subset_stats[i][
                        "rel_triples"
                    ]
                else:
                    raise ValueError(
                        f"GraSH cost metric {cost_metric} is not supported."
                    )

        # Create workers (dummy logger to avoid output overhead from HPBandSter)
        worker_logger = logging.getLogger("dummy")
        worker_logger.setLevel(logging.DEBUG)
        for i in range(self.config.get("search.num_workers")):
            w = GraSHWorker(
                nameserver=self.config.get("grash_search.host"),
                # logger=logging.getLogger('dummy'),
                logger=worker_logger,
                run_id=self.config.get("grash_search.run_id"),
                job_config=self.config,
                parent_job=self,
                trial_dict=self.trial_dict,
                id_dict=self.id_dict,
                result_dict=self.result_dict,
                id=i,
            )
            # todo: figure out why the process pool is not starting the jobs
            print(f"starting process hpb-worker-process {i}")

            # w.run(background=True)
            p = mp.Process(target=w.run, args=(False,))
            self.processes.append(p)
            p.start()

            # future = self.process_pool.submit(w.run, w, False)
            # worker_futures.append(future)
            self.workers.append(w)

    def modify_dataset_config(self, subset_index, config):
        """
        Modify the dataset part of a given config by replacing with the subset data.
        :return: modified config
        """
        subset_stats = self.subset_stats[subset_index]
        if subset_stats['filename_suffix'] == "":
            config.set("dataset.files.test.filename", "test.del")
            return config
        path_to_subsets = os.path.join("subsets", "k-core")
        config.set("dataset.num_entities", subset_stats["entities"])
        config.set("dataset.num_relations", subset_stats["relations"])
        config.set(
            "dataset.files.entity_ids.filename",
            os.path.join(path_to_subsets, f"entity_ids{subset_stats['filename_suffix']}.del"),
        )
        config.set(
            "dataset.files.entity_strings.filename",
            os.path.join(path_to_subsets, f"entity_ids{subset_stats['filename_suffix']}.del"),
        )
        config.set(
            "dataset.files.relation_ids.filename",
            os.path.join(path_to_subsets, f"relation_ids{subset_stats['filename_suffix']}.del"),
        )
        config.set(
            "dataset.files.relation_strings.filename",
            os.path.join(path_to_subsets, f"relation_ids{subset_stats['filename_suffix']}.del"),
        )
        config.set(
            "dataset.files.train.filename",
            os.path.join(path_to_subsets, f"train{subset_stats['filename_suffix']}.del"),
        )
        valid_split = config.get("valid.split")
        config.set(
            f"dataset.files.{valid_split}.filename",
            os.path.join(path_to_subsets, f"valid{subset_stats['filename_suffix']}.del"),
        )
        # also set default valid set as it may be used for filtering
        config.set(
            f"dataset.files.valid.filename",
            os.path.join(path_to_subsets, f"valid{subset_stats['filename_suffix']}.del"),
        )
        # only set test set for original dataset. Use the valid sets for all others.
        config.set(
            "dataset.files.test.filename",
            os.path.join(path_to_subsets, f"valid{subset_stats['filename_suffix']}.del"),
        )
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
        # previous_run = hpres.logged_results_to_HBS_result("test-hpo")
        result_logger = hpres.json_result_logger(
            directory=self.config.folder, overwrite=True
        )

        # high number as upper limit is handled by hpb
        workers_per_round = defaultdict(lambda: 10000)
        workers_per_round.update(
            self.config.get("grash_search.num_search_worker_per_round")
        )

        # Configure the job
        hpb = HyperBandPatch(
            free_devices=self.free_devices,
            configspace=self.workers[0].get_configspace(
                self.config, self.config.get("grash_search.seed")
            ),
            run_id=self.config.get("grash_search.run_id"),
            nameserver=self.config.get("grash_search.host"),
            result_logger=result_logger,
            eta=self.eta,
            min_budget=1 / self.num_trials,
            max_budget=1,
            trial_dict=self.trial_dict,
            id_dict=self.id_dict,
            result_dict=self.result_dict,
            workers_per_round=workers_per_round,
        )

        # Run it
        print("run hpo search")
        # set n_iterations to 1 to get a Successive Halving search
        hpb.run(
            n_iterations=1,
            min_n_workers=self.config.get("search.num_workers"),
        )

        # Shut it down
        hpb.shutdown(shutdown_workers=True)
        self.name_server.shutdown()
        for p in self.processes:
            p.terminate()


class GraSHWorker(Worker):
    """
    Class of a worker for the GraSH hyper-parameter optimization algorithm.
    """

    def __init__(self, *args, **kwargs):
        self.job_config = kwargs.pop("job_config")
        self.parent_job = kwargs.pop("parent_job")
        self.trial_dict = kwargs.pop("trial_dict")
        self.id_dict = kwargs.pop("id_dict")
        self.result_dict = kwargs.pop("result_dict")
        self.search_worker_id = kwargs.get("id")
        super().__init__(*args, **kwargs)
        self.next_trial_no = 0
        self.search_budget = self.parent_job.config.get("grash_search.search_budget")

    def compute(self, config_id, config, budget, **kwargs):
        try:
            return self._compute(config_id, config, budget, **kwargs)
        except Exception as e:
            print("error:", e, file=sys.stderr)
            if self.parent_job.config.get("search.on_error") == "abort":
                os._exit(1)
            else:
                raise e

    def _compute(self, config_id, config, budget, **kwargs):
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

        # use first and third value of config_id to create the basis of the foldername
        hpb_iter = config_id[0]
        config_no = config_id[2]

        if (hpb_iter, config_no) in self.parent_job.id_dict:
            self.id_dict[(hpb_iter, config_no)] += 1
        else:
            self.id_dict[(hpb_iter, config_no)] = 0

        sh_iter = self.id_dict[(hpb_iter, config_no)]

        # put together the trial number
        trial_no = str("{:02d}".format(hpb_iter)) + str("{:02d}".format(sh_iter)) + str("{:04d}".format(config_no))

        # create job for trial
        conf = self.job_config.clone(trial_no)
        conf.set("job.type", "train")
        conf.set_all(parameters)

        # check if trial result is already available for the given parameters
        if trial_no in self.trial_dict:
            # and parameters == self.trial_dict[trial_no][1]:
            set_new = set(parameters.items())
            set_old = set(self.trial_dict[trial_no][1].items())
            difference = set_old - set_new
            if not difference:
                valid_metric = conf.get('valid.metric')
                best_score = self.trial_dict[trial_no][0]
                conf.log(f"Trial {conf.folder} registered with {valid_metric} {best_score}")
                return {"loss": 1 - best_score, "info": {}}
        #  else:
            # todo: delete checkpoint if parameters have changed

        # scale given budget based on the max budget in terms of train runs
        # -1 since we need one complete run in the last round so distribute rest over
        # other rounds
        if sh_iter < self.parent_job.sh_rounds:
            budget = budget * (self.parent_job.config.get(
                "grash_search.search_budget") / self.parent_job.sh_rounds)

        # determine the epochs for this trial
        if self.parent_job.config.get("grash_search.variant") == "graph":
            epochs = self.parent_job.config.get("train.max_epochs")
        elif self.parent_job.config.get("grash_search.variant") == "combined":
            # share savings equally between graph and epochs by taking the square root
            budget = math.sqrt(budget)
            epochs = math.floor(
                self.parent_job.config.get("train.max_epochs")
                * budget
            )
        elif self.parent_job.config.get("grash_search.variant") == "epoch":
            epochs = (
                self.parent_job.config.get("train.max_epochs")
                * budget
            )
            print("num epochs", epochs)
            if epochs < 1:
                max_batches = (
                    len(self.parent_job.dataset.split("train"))
                    / conf.get("train.batch_size")
                    * epochs
                )
                if "distributed" in conf.get("model"):
                    max_batches /= conf.get("job.distributed.num_partitions")
                max_batches = max(1, math.floor(max_batches))
                conf.set("train.max_batches", max_batches)
                epochs = 1
            epochs = math.floor(epochs)
            print("num epochs", epochs)

        # determine the subset for this trial
        if self.parent_job.config.get("grash_search.variant") != "epoch":
            # determine and set the dataset to use based on the budget
            subset_index = -1
            for i in range(len(self.parent_job.subset_stats)):
                if self.parent_job.subset_stats[i]["rel_costs"] <= budget:
                    subset_index = i
                    break
            if subset_index == -1:
                raise ValueError(
                    f"no fitting subgraph for size_budget {budget} found"
                )

            # check if dataset was already loaded and do so if not
            if subset_index not in self.parent_job.subsets:
                config_custom_data = copy.deepcopy(self.parent_job.config)
                config_custom_data = self.parent_job.modify_dataset_config(subset_index, config_custom_data)
                custom_dataset = Dataset.create(config_custom_data)
                self.parent_job.subsets[subset_index] = custom_dataset

            self.parent_job.dataset = self.parent_job.subsets[subset_index]
            conf = self.parent_job.modify_dataset_config(subset_index, conf)

            # downscale number of negatives
            number_samples_s = conf.get("negative_sampling.num_samples.s")
            negatives_scaler = max(
                self.parent_job.subset_stats[subset_index]["rel_entities"],
                self.parent_job.config.get("grash_search.min_negatives_percentage"),
            )
            conf.set(
                "negative_sampling.num_samples.s",
                math.ceil(number_samples_s * negatives_scaler),
            )
            number_samples_o = conf.get("negative_sampling.num_samples.o")
            conf.set(
                "negative_sampling.num_samples.o",
                math.ceil(number_samples_o * negatives_scaler),
            )

            # reuse the predecessor model checkpoint if available to keep initialization
            if sh_iter != 0:
                predecessor_trial_id = str("{:02d}".format(hpb_iter)) + str("{:02d}".format(sh_iter - 1)) + str(
                    "{:04d}".format(config_no))
                path_to_model = ""
                if conf.get("grash_search.keep_initialization"):
                    path_to_model = os.path.join(
                        f"{os.path.dirname(conf.folder)}",
                        f"{predecessor_trial_id}",
                        f"model_00000.pt",
                    )
                if conf.get("grash_search.keep_pretrained"):
                    path_to_model = os.path.join(
                        f"{os.path.dirname(conf.folder)}",
                        f"{predecessor_trial_id}",
                        f"model_best.pt",
                    )
                conf.set("lookup_embedder.pretrain.model_filename", path_to_model)

        # set the number of epochs
        conf.set("train.max_epochs", epochs)

        # set valid.every to train.max_epochs if its modulo != 0
        if epochs % self.parent_job.config.get("valid.every") != 0:
            conf.set("valid.every", epochs)

        # define distributed setup
        distributed_workers_per_round = self.parent_job.config.get("grash_search.distributed_worker_per_round")
        if str(sh_iter) in distributed_workers_per_round.keys():
            # change to distributed model if not yet defined
            if "distributed" not in conf.get("model"):
                conf.set("distributed_model.base_model.type", conf.get("model"))
                conf.set("model", "distributed_model")
            # set right number of workers and partitions
            # for now we only assume random partitioning with shared ps
            num_workers = distributed_workers_per_round[str(sh_iter)]
            conf.set("job.distributed.parameter_server", "shared")
            conf.set("job.distributed.num_workers", num_workers)
            conf.set("job.distributed.num_partitions", num_workers)
            # choose a distributed train job
            conf.set("train.type", f"distributed_{conf.get('train.type')}")

            current_optim = conf.get("train.optimizer.default.type")
            if "dist" not in current_optim:
                dist_optim_dict = {
                    "Adagrad": "dist_adagrad",
                    "RowAdagrad": "dist_rowadagrad",
                }
                conf.set("train.optimizer.default.type", dist_optim_dict[current_optim])

            # we need to define the device pool
            # just rotate the overall device pool so that it starts with given device
            # but sort first so that we use up one device fully first
            device_pool = np.array(self.parent_job.device_pool)
            device_pool.sort()
            device_position = np.argwhere(device_pool == conf.get("job.device"))[0][0]
            device_pool = np.concatenate((device_pool[device_position:], device_pool[:device_position]))
            device_pool = device_pool.tolist()
            conf.set("job.device_pool", device_pool)

        # todo: compute total number of trials in init
        num_train_trials = "x"

        # save config.yaml
        conf.init_folder()

        # copy last checkpoint from previous sh round to new folder for epoch only variant
        if (
            self.parent_job.config.get("grash_search.variant") == "epoch"
            and sh_iter != 0
        ):
            copied = False
            predecessor_trial_id = str("{:02d}".format(hpb_iter)) + str("{:02d}".format(sh_iter - 1)) + str(
                "{:04d}".format(config_no))
            for filename in os.listdir(
                f"{os.path.dirname(conf.folder)}/{predecessor_trial_id}/"
            ):
                if filename.endswith(".pt") and filename != "checkpoint_best.pt":
                    shutil.copy(
                        f"{os.path.dirname(conf.folder)}/{predecessor_trial_id}/{filename}",
                        f"{conf.folder}/{filename}",
                    )
                    copied = True
            if not copied:
                conf.log(
                    "Could not copy predecessor checkpoint. Starting new round from scratch"
                )

        # change port for distributed training
        if "distributed" in conf.get("model"):
            print("worker id", self.search_worker_id)
            conf.set(
                "job.distributed.master_port",
                conf.get("job.distributed.master_port") + self.search_worker_id,
            )

        # run trial
        best = kge.job.search._run_train_job(
            (
                self.parent_job,
                int(trial_no),
                conf,
                num_train_trials,
                list(parameters.keys()),
            )
        )
        # self.parent_job.wait_task(concurrent.futures.ALL_COMPLETED)

        # save package checkpoint
        args = Namespace()
        args.checkpoint = None
        if conf.get("grash_search.keep_initialization"):
            args.checkpoint = f"{conf.folder}/checkpoint_00000.pt"
        if conf.get("grash_search.keep_pretrained"):
            args.checkpoint = f"{conf.folder}/checkpoint_best.pt"
        if args.checkpoint is not None:
            args.file = None
            package_model(args)

        best_score = best[1]["metric_value"]
        del best

        def _kill_active_cuda_tensors():
            """
            Returns all tensors initialized on cuda devices
            """
            for obj in gc.get_objects():
                try:
                    if torch.is_tensor(obj) and obj.device.type == "cuda":
                        #yield obj
                        del obj
                except:
                    pass
        gc.collect()
        _kill_active_cuda_tensors()
        with torch.cuda.device(conf.get("job.device")):
            torch.cuda.empty_cache()
        gc.collect()

        # remove parameters that can be ignored when resuming
        parameters.pop('job.device', None)
        # add score and parameters to trial dict
        self.trial_dict[trial_no] = [best_score, parameters]

        # save search checkpoint - todo: move to parent job
        filename = f"{os.path.dirname(conf.folder)}//checkpoint_00001.pt"
        self.parent_job.config.log("Saving checkpoint to {}...".format(filename))
        torch.save(
            {
                "type": "search_grash",
                "parameters": [],
                "results": [self.trial_dict._getvalue()],
                "job_id": self.parent_job.job_id,
            },
            filename,
        )

        return {"loss": 1 - best_score, "info": {"metric_value": best_score}}

    @staticmethod
    def get_configspace(config, seed):
        """
        Reads the config file and produces the necessary variables. Returns a configuration space with
        all variables and their definition.
        :param config: dictionary containing the variables and their possible values
        :return: ConfigurationSpace containing all variables.
        """
        config_space = CS.ConfigurationSpace(seed=seed)

        parameters = config.get("grash_search.parameters")
        for p in parameters:
            v_name = p["name"]
            v_type = p["type"]

            if v_type == "choice":
                config_space.add_hyperparameter(
                    CSH.CategoricalHyperparameter(v_name, choices=p["values"])
                )
            elif v_type == "range":
                log_scale = False
                if "log_scale" in p.keys():
                    log_scale = p["log_scale"]
                if type(p["bounds"][0]) is int and type(p["bounds"][1]) is int:
                    config_space.add_hyperparameter(
                        CSH.UniformIntegerHyperparameter(
                            name=v_name,
                            lower=p["bounds"][0],
                            upper=p["bounds"][1],
                            default_value=p["bounds"][1],
                            log=log_scale,
                        )
                    )
                else:
                    config_space.add_hyperparameter(
                        CSH.UniformFloatHyperparameter(
                            name=v_name,
                            lower=p["bounds"][0],
                            upper=p["bounds"][1],
                            default_value=p["bounds"][1],
                            log=log_scale,
                        )
                    )
            elif v_type == "fixed":
                config_space.add_hyperparameter(
                    CSH.Constant(name=v_name, value=p["value"])
                )
            else:
                raise ValueError("Unknown variable type")
        return config_space
