# An Efficient Graph Convolutional Network for the Travelling Salesman Problem

This repository contains code for the paper 
"An Efficient Graph Convolutional Network for the Travelling Salesman Problem" 
by Chaitanya K. Joshi, Thomas Laurent and Xavier Bresson.

We introduce a new learning-based approach for approximately solving the
Travelling Salesman Problem on 2D Euclidean graphs. 
We use deep Graph Convolutional Networks to build efficient TSP graph representations 
and output tours in a non-autoregressive manner via highly parallelized beam search. 
Our approach outperforms all recently proposed autoregressive deep learning 
techniques in terms of solution quality, inference speed and sample efficiency 
for problem instances of fixed graph sizes. 

![model-blocks](/res/model-blocks.png)

# Overview

The notebook `main.ipynb` contains top-level methods to reproduce our experiments or train models for TSP from scratch.
Several modes are provided:
- **Notebook Mode**: For debugging as a Jupyter Notebook
- **Visualization Mode**: For visualization and evaluation of saved model checkpoints (in a Jupyter Notebook)
- **Script Mode**: For running full experiments as a python script

Configuration parameters for notebooks and scripts are passed as `.json` files and are documented in `config.py`.

# Pre-requisite Downloads

1. Download TSP datasets from [this link](https://drive.google.com/open?id=1-5W-S5e7CKsJ9uY9uVXIyxgbcZZNYBrp), extract the `.tar.gz` file, and place each `.txt` file in the `/data` directory. (We provide TSP10, TSP20, TSP30, TSP50 and TSP100.) 
2. Download pre-trained model checkpoints from [this link](https://drive.google.com/open?id=1qmk1_5a8XT_hrOV_i3uHM9tMVnZBFEAF), extract the `.tar.gz` file, and place each directory in the `/logs` directory. (We provide TSP20, TSP50 and TSP100 models.) .

# Usage

## Installation
Step-by-step guide for local installation using a Terminal (Mac/Linux) or Git Bash (Windows):
1. Install [Anaconda 3](https://www.anaconda.com/) for managing Python packages and environments.
```sh
curl -o ~/miniconda.sh -O https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x ~/miniconda.sh
./miniconda.sh
source ~/.bashrc
```
2. Clone the repository. 
```sh
git clone https://github.com/chaitjo/graph-convnet-tsp.git
cd graph-convnet-tsp
```
3. Set up a new conda environment and activate it.
```sh
conda create -n gcn-tsp-env python=3.6.7
source activate gcn-tsp-env
```
4. Install all dependencies and Jupyter Lab (for using notebooks).
```sh
conda install pytorch=0.4.1 cuda90 -c pytorch
conda install numpy==1.15.4 scipy==1.1.0 matplotlib==3.0.2 seaborn==0.9.0 pandas==0.24.2 networkx==2.2 scikit-learn==0.20.2 tensorflow-gpu==1.12.0 tensorboard==1.12.0
pip3 install tensorboardx==1.5 fastprogress==0.1.18
conda install -c conda-forge jupyterlab
```

## Running in Notebook/Visualization Mode
Launch Jupyter Lab and execute/modify `main.ipynb` cell-by-cell in Notebook Mode.
```sh
jupyter lab
```

Set `viz_mode = True` in the first cell of `main.ipynb` to toggle Visualization Mode.

## Running in Script Mode
0. Set `notebook_mode = False` and `viz_mode = False` in the first cell of `main.ipynb`.
1. Convert from .ipynb to .py: `jupyter nbconvert --to python main.ipynb`
2. Run the script (pass path of configuration file as arguement): `python main.py --config <path-to-config.json>`

## Splitting datasets into Training and Validation sets
For TSP10, TSP20 and TSP30 datasets, everything is good to go.
For TSP50 and TSP100, the 1M training set needs to be split into 10K validation samples and 999K training samples.
Use the `split_train_val.ipynb` notebook to do this through Jupyter Lab.
