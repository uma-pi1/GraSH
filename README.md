# GraSH: Knowledge Graph Successive Halving


This is the code and configuration accompanying the paper ["Start Small, Think Big: On Hyperparameter Optimization for Large-Scale Knowledge Graph Embeddings"](tbd.).
The code extends the knowledge graph embedding library for distributed training [Dist-KGE](https://github.com/uma-pi1/dist-kge).
For documentation on Dist-KGE refer to the Dist-KGE repository.
We provide the hyperparameter settings for the searches and finally selected trials in /examples/experiments/.


## Table of contents

1. [Quick start](#quick-start)
2. [Dataset preparation for parallel training](#dataset-preparation-for-parallel-training)
3. [Single Machine Multi-GPU Training](#single-machine-multi-gpu-training)
4. [Multi-GPU Multi-Machine Training](#multi-gpu-multi-machine-training)
5. [Folder structure of experiment results](#folder-structure-of-experiment-results)
6. [Results and Configurations](#results-and-configurations)
7. [How to cite](#how-to-cite)

## Quick start

```sh
# retrieve and install project in development mode
git clone https://github.com/uma-pi1/grash.git
cd GraSH
pip install -e .

# download and preprocess datasets
cd data
sh download_all.sh
cd ..

# train an example model on toy dataset (you can omit '--job.device cpu' when you have a gpu)
kge start examples/toy-complex-train.yaml --job.device cpu
todo: provide toy example for GraSH

```
This example will train on a toy dataset in a sequential setup on CPU


## Configuration of GraSH Search

The most important configuration options for a hyperparameter search with GraSH are:
````yaml
dataset:
  name: yago3-10
grash_search:
  eta: 4
  num_trials: 64
  search_budget: 3
  variant: combined
  parameters: # define your search space here
job:
  type: search
model: complex
train:
  max_epochs: 400
````

## Run a GraSH hyperparameter search
Run the default search on yago3-10 with the following command:
```sh
kge start examples/experiments/search_configs/yago3-10/search-complex-yago-combined.yaml
```
The k-core subgraphs will automatically be generated and saved to data/yago3-10/subsets/k-core/.
By default, each experiment will create a new folder in `local/experiments/<timestamp>-<config-name>` where the results can be found.


## Results

#### Yago3-10
**ComplEx**

Variant		| B		|  	η  		|   MRR 	|  Hits@1  	|   Hits@10 |   Hits@100    | config
----    	| ----- |   ----:   |   ----:   |   ----:   |   ----:   |   ----:   	|   ----
Epoch    	|   3	|   4		|   0.536   |  0.460	|   0.672   |   0.601   	|  [config](examples/experiments/search_configs/yago3-10/search-complex-yago-epoch.yaml)
Graph    	|   3	|   4		|   0.463   |  0.375	|   0.634   |   0.800   	|  [config](examples/experiments/search_configs/yago3-10/search-complex-yago-graph.yaml)
Combined   	|   3	|   4		|   0.528   |  0.455	|   0.660   |   0.772   	|  [config](examples/experiments/search_configs/yago3-10/search-complex-yago-combined.yaml)


# How to cite
```
@article{kochsiek2022parallel,
  title={Parallel Training of Knowledge Graph Embedding Models: A Comparison of Techniques},
  author={Kochsiek, Adrian and Gemulla, Rainer},
  journal={Proceedings of the VLDB Endowment},
  volume={15},
  number={3},
  year={2022}
}
```