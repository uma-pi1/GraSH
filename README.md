# GraSH: Successive Halving for Knowledge Graphs


This is the code and configuration accompanying the paper ["Start Small, Think Big: On Hyperparameter Optimization for Large-Scale Knowledge Graph Embeddings"](https://arxiv.org/abs/2207.04979).
The code extends the knowledge graph embedding library for distributed training [Dist-KGE](https://github.com/uma-pi1/dist-kge).
For documentation on Dist-KGE refer to the Dist-KGE repository.
We provide the hyperparameter settings for the searches and finally selected trials in /examples/experiments/.

**UPDATE:**
GraSH was recently merged into our main library [LibKGE](https://github.com/uma-pi1/kge). All configs from this repository, except the ones for Freebase that require distributed training, can be executed in LibKGE. Please use LibKGE for your own experiments with GraSH.


## Table of contents

1. [Quick start](#quick-start)
2. [Configuration of GraSH Search](#configuration-of-grash-search)
3. [Run a GraSH hyperparameter search](#run-a-grash-hyperparameter-search)
4. [Results and Configurations](#results-and-configurations)
5. [How to cite](#how-to-cite)

## Quick start

#### Setup
```sh
# retrieve and install project in development mode
git clone https://github.com/uma-pi1/grash.git
cd grash
pip install -e .

# download and preprocess datasets
cd data
sh download_all.sh
cd ..
```

#### Training
```sh

# train an example model on a toy dataset (you can omit '--job.device cpu' when you have a gpu)
python -m kge start examples/toy-complex-train.yaml --job.device cpu
```
This example will train on a toy dataset in a sequential setup on CPU


#### GraSH Hyperparameter Search
```sh
# perform a search with GraSH on a toy dataset (you can omit '--job.device cpu' when you have a gpu)
python -m kge start examples/toy-complex-search-grash.yaml --job.device cpu
```

This example will perform a small GraSH search with 16 trials on a toy dataset in a sequential setup on CPU


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

- `eta` defines the reduction factor during the search. Per round the number of remaining trials is reduced to `1/eta`
- `search_budget` is defined in "number of full training runs". The default choice `search_budget=3`, for example, corresponds to an overall search cost of three full training runs. 
- `variant` controls which reduction technique to use (only epoch, only graph, or combined)

## Run a GraSH hyperparameter search
Run the default search on yago3-10 with the following command:
```sh
python -m kge start examples/experiments/search_configs/yago3-10/search-complex-yago-combined.yaml
```
The k-core subgraphs will automatically be generated and saved to data/yago3-10/subsets/k-core/.
By default, each experiment will create a new folder in `local/experiments/<timestamp>-<config-name>` where the results can be found.


## Results and Configurations
All results were obtained with the GraSH default settings (`num_trials=64, eta=4, search_budget=3, variant=combined`)

#### Yago3-10

Model		|Variant	|   MRR 	|  Hits@1  	|   Hits@10 |   Hits@100    | config
----    	|----    	|   ----:   |   ----:   |   ----:   |   ----:   	|   ----
ComplEx    	|Epoch    	|   0.536   |  0.460	|   0.672   |   0.601   	|  [config](examples/experiments/selected_trials/yago3-10/complex-yago-epoch.yaml)
ComplEx    	|Graph    	|   0.463   |  0.375	|   0.634   |   0.800   	|  [config](examples/experiments/selected_trials/yago3-10/complex-yago-graph.yaml)
ComplEx   	|Combined   |   0.528   |  0.455	|   0.660   |   0.772   	|  [config](examples/experiments/selected_trials/yago3-10/complex-yago-combined.yaml)
RotatE    	|Epoch    	|   0.432   |  0.337	|   0.619   |   0.768   	|  [config](examples/experiments/selected_trials/yago3-10/rotate-yago-epoch.yaml)
RotatE    	|Graph    	|   0.432   |  0.337	|   0.619   |   0.768   	|  [config](examples/experiments/selected_trials/yago3-10/rotate-yago-graph.yaml)
RotatE   	|Combined   |   0.434   |  0.342	|   0.607   |   0.742   	|  [config](examples/experiments/selected_trials/yago3-10/rotate-yago-combined.yaml)
TransE    	|Epoch    	|   0.499   |  0.406	|   0.661   |   0.794   	|  [config](examples/experiments/selected_trials/yago3-10/transe-yago-epoch.yaml)
TransE    	|Graph    	|   0.422   |  0.311	|   0.628   |   0.802   	|  [config](examples/experiments/selected_trials/yago3-10/transe-yago-graph.yaml)
TransE   	|Combined   |   0.499   |  0.406	|   0.661   |   0.794   	|  [config](examples/experiments/selected_trials/yago3-10/transe-yago-combined.yaml)

#### Wikidata5M

Model		|Variant	|   MRR 	|  Hits@1  	|   Hits@10 |   Hits@100    | config
----    	|----    	|   ----:   |   ----:   |   ----:   |   ----:   	|   ----
ComplEx    	|Epoch    	|   0.300   |  0.247	|   0.390   |   0.506   	|  [config](examples/experiments/selected_trials/wikidata5m/complex-wikidata-epoch.yaml)
ComplEx    	|Graph    	|   0.300   |  0.247	|   0.390   |   0.506   	|  [config](examples/experiments/selected_trials/wikidata5m/complex-wikidata-graph.yaml)
ComplEx   	|Combined   |   0.300   |  0.247	|   0.390   |   0.506   	|  [config](examples/experiments/selected_trials/wikidata5m/complex-wikidata-combined.yaml)
RotatE    	|Epoch    	|   0.241   |  0.187	|   0.331   |   0.438   	|  [config](examples/experiments/selected_trials/wikidata5m/rotate-wikidata-epoch.yaml)
RotatE    	|Graph    	|   0.232   |  0.169	|   0.326   |   0.432   	|  [config](examples/experiments/selected_trials/wikidata5m/rotate-wikidata-graph.yaml)
RotatE   	|Combined   |   0.241   |  0.187	|   0.331   |   0.438   	|  [config](examples/experiments/selected_trials/wikidata5m/rotate-wikidata-combined.yaml)
TransE    	|Epoch    	|   0.263   |  0.210	|   0.358   |   0.483   	|  [config](examples/experiments/selected_trials/wikidata5m/transe-wikidata-epoch.yaml)
TransE    	|Graph    	|   0.263   |  0.210	|   0.358   |   0.483   	|  [config](examples/experiments/selected_trials/wikidata5m/transe-wikidata-graph.yaml)
TransE   	|Combined   |   0.268   |  0.213	|   0.363   |   0.480   	|  [config](examples/experiments/selected_trials/wikidata5m/transe-wikidata-combined.yaml)

#### Freebase

Model		|Variant	|   MRR 	|  Hits@1  	|   Hits@10 |   Hits@100    | config
----    	|----    	|   ----:   |   ----:   |   ----:   |   ----:   	|   ----
ComplEx    	|Epoch    	|   0.572   |  0.486	|   0.714   |   0.762   	|  [config](examples/experiments/selected_trials/freebase/complex-freebase-epoch.yaml)
ComplEx    	|Graph    	|   0.594   |  0.511	|   0.726   |   0.767   	|  [config](examples/experiments/selected_trials/freebase/complex-freebase-graph.yaml)
ComplEx   	|Combined   |   0.594   |  0.511	|   0.726   |   0.767   	|  [config](examples/experiments/selected_trials/freebase/complex-freebase-combined.yaml)
RotatE    	|Epoch    	|   0.561   |  0.522	|   0.625   |   0.679   	|  [config](examples/experiments/selected_trials/freebase/rotate-freebase-epoch.yaml)
RotatE    	|Graph    	|   0.613   |  0.578	|   0.669   |   0.719   	|  [config](examples/experiments/selected_trials/freebase/rotate-freebase-graph.yaml)
RotatE   	|Combined   |   0.613   |  0.578	|   0.669   |   0.719   	|  [config](examples/experiments/selected_trials/freebase/rotate-freebase-combined.yaml)
TransE    	|Epoch    	|   0.261   |  0.078	|   0.518   |   0.636   	|  [config](examples/experiments/selected_trials/freebase/transe-freebase-epoch.yaml)
TransE    	|Graph    	|   0.553   |  0.520	|   0.614   |   0.682   	|  [config](examples/experiments/selected_trials/freebase/transe-freebase-graph.yaml)
TransE   	|Combined   |   0.553   |  0.520	|   0.614   |   0.682   	|  [config](examples/experiments/selected_trials/freebase/transe-freebase-combined.yaml)

# How to cite
```
@inproceedings{kochsiek2022start,
  title={Start Small, Think Big: On Hyperparameter Optimization for Large-Scale Knowledge Graph Embeddings},
  author={Kochsiek, Adrian and Niesel, Fritz and Gemulla, Rainer},
  booktitle={Joint European Conference on Machine Learning and Knowledge Discovery in Databases},
  year={2022}
}
```
