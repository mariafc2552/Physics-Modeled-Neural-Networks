# Physics-Modeled-Neural-Networks (PMNNs)

## Abstract
The aim of this work is to (mathematically) describe and analyze a new type of neural network architecture, which we will call Dynamical Physics-Modeled Neural Networks (DynPMNN), that uses ordinary differential equations in its construction and which is consistent with the idea of what a neural network is. Its performance is evaluated in comparison with Ordinary Differential Neural Networks (NODEs) and also some illustrative examples are shown.

The original article can be found at the following link: -----

## The model proposal
While Neural Ordinary Differential Equations (NODEs) transform the forward propagation process into a continuous model governed by an ODE, and Physics-Informed Neural Networks (PINNs) incorporate physical laws directly into the loss function, Dynamical Physics-Modeled Neural Networks (DynPMNNs) extend this by modeling the hidden layers of the neural network using ODEs, where each hidden layer is treated as a dynamic system whose evolution is governed by a differential equation.

The PMNN (Physics-Modeled Neural Network) framework in the paper integrates systems of ODEs to model the dynamics of the hidden layer. Specifically, the model uses a system based on the FitzHugh-Nagumo or Hodgkin-Huxley models to describe neuronal activation, representing each hidden layer as a set of ODEs with trainable parameters. The solution to these ODEs determines the layer’s output, effectively introducing dynamic behavior within the network. The PMNN framework canbe trained using numerical methods such as the Euler method to approximate the solution of the ODE governing each hidden layer's dynamics. This approach enhances the predictive power of the network by incorporating continuous-time dynamics.

<p align="center">
 <img width="636" height="338" alt="image" src="https://github.com/user-attachments/assets/d4544ea8-1bce-4316-9151-b81c42856e9a" />
</p>

## Files
The repository consists of four working folders, each studying a different model:

1. FitzHugh-Nugamo-based PMNN.
2. Liquid-time continous neural network (LTC).
3. Closed-form continous time neural network (CfC).
4. Deep neural network (fully-connected layers).

In each folder, the implementation of each model for the regression task concerning the California Housing dataset is provided.

## Cite

