# An Efficient Graph Convolutional Network Technique for the Travelling Salesman Problem

>**:rocket: Update:** If you are interested in this work, you may be interested in [**our latest paper**](https://arxiv.org/abs/2006.07054) and [**up-to-date codebase**](https://github.com/chaitjo/learning-tsp) bringing together several architectures and learning paradigms for learning-driven TSP solvers under one pipeline.

This repository contains code for the paper 
[**"An Efficient Graph Convolutional Network Technique for the Travelling Salesman Problem"**](https://arxiv.org/abs/1906.01227)
by Chaitanya K. Joshi, Thomas Laurent and Xavier Bresson.

We introduce a new learning-based approach for approximately solving the
Travelling Salesman Problem on 2D Euclidean graphs. 
We use deep Graph Convolutional Networks to build efficient TSP graph representations 
and output tours in a non-autoregressive manner via highly parallelized beam search. 
Our approach outperforms all recently proposed autoregressive deep learning 
techniques in terms of solution quality, inference speed and sample efficiency 
for problem instances of fixed graph sizes. 

![model-blocks](/res/model-blocks.png)

## Overview

The notebook `main.ipynb` contains top-level methods to reproduce our experiments or train models for TSP from scratch.
Several modes are provided:
- **Notebook Mode**: For debugging as a Jupyter Notebook
- **Visualization Mode**: For visualization and evaluation of saved model checkpoints (in a Jupyter Notebook)
- **Script Mode**: For running full experiments as a python script

Configuration parameters for notebooks and scripts are passed as `.json` files and are documented in `config.py`.

## Pre-requisite Downloads

#### TSP Datasets
Download TSP datasets from [this link](https://drive.google.com/open?id=1-5W-S5e7CKsJ9uY9uVXIyxgbcZZNYBrp): 
Extract the `.tar.gz` file and place each `.txt` file in the `/data` directory. (We provide TSP10, TSP20, TSP30, TSP50 and TSP100.) 

#### Pre-trained Models
Download pre-trained model checkpoints from [this link](https://drive.google.com/open?id=1qmk1_5a8XT_hrOV_i3uHM9tMVnZBFEAF): 
Extract the `.tar.gz` file and place each directory in the `/logs` directory. (We provide TSP20, TSP50 and TSP100 models.)

## Usage

#### Installation
We ran our code on Ubuntu 16.04, using Python 3.6.7, PyTorch 0.4.1 and CUDA 9.0.

> **Note:** This codebase was developed for a rather outdated version of PyTorch. Attempting to run the code with PyTorch 1.x may need further modifications, e.g. see [this issue](https://github.com/chaitjo/graph-convnet-tsp/issues/16).

Step-by-step guide for local installation using a Terminal (Mac/Linux) or Git Bash (Windows) via Anaconda:
```sh
# Install [Anaconda 3](https://www.anaconda.com/) for managing Python packages and environments.
curl -o ~/miniconda.sh -O https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x ~/miniconda.sh
./miniconda.sh
source ~/.bashrc

# Clone the repository. 
git clone https://github.com/chaitjo/graph-convnet-tsp.git
cd graph-convnet-tsp

# Set up a new conda environment and activate it.
conda create -n gcn-tsp-env python=3.6.7
source activate gcn-tsp-env

# Install all dependencies and Jupyter Lab (for using notebooks).
conda install pytorch=0.4.1 cuda90 -c pytorch
conda install numpy==1.15.4 scipy==1.1.0 matplotlib==3.0.2 seaborn==0.9.0 pandas==0.24.2 networkx==2.2 scikit-learn==0.20.2 tensorflow-gpu==1.12.0 tensorboard==1.12.0 Cython
pip3 install tensorboardx==1.5 fastprogress==0.1.18
conda install -c conda-forge jupyterlab
```

#### Running in Notebook/Visualization Mode
Launch Jupyter Lab and execute/modify `main.ipynb` cell-by-cell in Notebook Mode.
```sh
jupyter lab
```

Set `viz_mode = True` in the first cell of `main.ipynb` to toggle Visualization Mode.

#### Running in Script Mode
Set `notebook_mode = False` and `viz_mode = False` in the first cell of `main.ipynb`.
Then convert the notebook from `.ipynb` to `.py` and run the script (pass path of config file as arguement):
```sh
jupyter nbconvert --to python main.ipynb 
python main.py --config <path-to-config.json>
```

#### Splitting datasets into Training and Validation sets
For TSP10, TSP20 and TSP30 datasets, everything is good to go once you download and extract the files.
For TSP50 and TSP100, the 1M training set needs to be split into 10K validation samples and 999K training samples.
Use the `split_train_val.py` script to do so.
For consistency, the script uses the first 10K samples in the 1M file as the validation set and the remaining 999K as the training set.

```sh
cd data
python split_train_val.py --num_nodes <num-nodes>
```

#### Generating new data
New TSP data can be generated using the [Concorde solver](https://github.com/jvkersch/pyconcorde).

```sh
# Install the pyConcorde library in the /data directory
cd data
git clone https://github.com/jvkersch/pyconcorde
cd pyconcorde
pip install -e .
cd ..

# Run the data generation script
python generate_tsp_concorde.py --num_samples <num-sample> --num_nodes <num-nodes>
```

## Resources

- [Optimal TSP Datasets generated with Concorde](https://drive.google.com/open?id=1-5W-S5e7CKsJ9uY9uVXIyxgbcZZNYBrp)
- [Paper on arXiv](https://arxiv.org/abs/1906.01227)
- [Follow-up workshop paper](https://arxiv.org/abs/1910.07210)
