REPORT link https://wandb.ai/pankuu21-indian-institute-of-technology-madras/DA6401_Assignment1/reports/CS24M029-DA6401-Assignment-1--VmlldzoxMTcxMjIwMg?accessToken=5cy782mzph3ubnvfyl3gdshmfdb3lfm9r9kd0qzsajcfwjexlop43zajbnkqo5b1

# Deep Learning Neural Network Implementation

This repository contains an implementation of feedforward neural networks from scratch using NumPy, with integration for hyperparameter tuning and experiment tracking using Weights & Biases.

## Project Overview

The project implements a flexible neural network architecture that can be configured with various hyperparameters:
- Multiple hidden layers with configurable sizes
- Various activation functions (ReLU, sigmoid, tanh, identity)
- Different optimization algorithms (SGD, Momentum, NAG, RMSProp, Adam, NAdam)
- Various weight initialization methods (random, Xavier/Glorot)
- Support for L2 regularization
- Choice of loss functions (cross-entropy, mean squared error)

## Installation Requirements

```bash
pip install numpy wandb keras tensorflow
```

## Setup

1. Clone this repository:
```bash
git clone https://github.com/pankuu21/DA6401_Assignment1.git
cd DA6401_Assignment1
```

2. Set up Weights & Biases:
```bash
wandb login
```

## File Structure

- **Q1.py**: Data loading utilities for MNIST and Fashion MNIST datasets
- **Q2.py**: Forward propagation and activation functions implementation
- **Q3.py**: Backpropagation and parameter update functions
- **Q4.py**: Hyperparameter sweep configuration and execution
- **Q7.py**: Training with the best hyperparameters found from sweeps
- **Q8.py**: Comparison of different loss functions (cross-entropy vs MSE)
- **train.py**: Main training script with command-line interface
- **optimizers.py**: Implementation of various optimization algorithms
- **sweep_config.yaml**: Configuration for hyperparameter tuning

## Usage

### Basic Training

To train a model with default parameters:

```bash
python train.py
```

### Configuring Training Options

```bash
python train.py -d mnist -e 10 -b 32 -o adam -lr 0.001 -a ReLU -nhl 2 -sz 128
```

### Available Arguments

| Argument | Description |
|----------|-------------|
| `-wp`, `--wandb_project` | WandB project name |
| `-we`, `--wandb_entity` | WandB entity name |
| `-d`, `--dataset` | Dataset choice: mnist or fashion_mnist |
| `-e`, `--epochs` | Number of training epochs |
| `-b`, `--batch_size` | Mini-batch size |
| `-l`, `--loss` | Loss function: mean_squared_error or cross_entropy |
| `-o`, `--optimizer` | Optimizer: sgd, momentum, nag, rmsprop, adam, nadam |
| `-lr`, `--learning_rate` | Learning rate |
| `-m`, `--momentum` | Momentum parameter |
| `-beta`, `--beta` | Beta parameter for rmsprop |
| `-beta1`, `--beta1` | Beta1 for adam/nadam |
| `-beta2`, `--beta2` | Beta2 for adam/nadam |
| `-eps`, `--epsilon` | Epsilon for numerical stability |
| `-w_d`, `--weight_decay` | L2 regularization strength |
| `-w_i`, `--weight_init` | Weight initialization: random or Xavier |
| `-nhl`, `--num_layers` | Number of hidden layers |
| `-sz`, `--hidden_size` | Number of neurons per hidden layer |
| `-a`, `--activation` | Activation function: identity, sigmoid, tanh, or ReLU |

### Run Hyperparameter Sweep

```bash
python Q4.py
```

### Train with Best Parameters

```bash
python Q7.py
```

### Compare Loss Functions

```bash
python Q8.py
```

## Hyperparameter Tuning

This project uses Weights & Biases for hyperparameter tuning. The `sweep_config.yaml` file contains the configuration for the hyperparameter search. You can modify this file to change the search space.

There are three search methods implemented:
- Grid search
- Random search
- Bayesian optimization

## Results Visualization

All results are automatically logged to Weights & Biases. You can view training curves, validation metrics, test accuracy, and confusion matrices in the W&B dashboard.

## Implemented Optimizers

- SGD (Stochastic Gradient Descent)
- Momentum
- NAG (Nesterov Accelerated Gradient)
- RMSProp
- Adam
- NAdam (Nesterov Adam)

## Project Extensions

- Test different optimizers and their effectiveness
- Compare performance across datasets
- Analyze impact of weight initialization techniques
- Study effects of varying network depth and width
- Compare loss functions for classification tasks
