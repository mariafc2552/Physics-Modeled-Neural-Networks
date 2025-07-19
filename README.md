# Physics-Modeled-Neural-Networks

## Abstract
The aim of this work is to (mathematically) describe and analyze a new type of neural network architecture, which we will call Dynamical Physics-Modeled Neural Networks (DynPMNN), that uses ordinary differential equations in its construction and which is consistent with the idea of what a neural network is. Its performance is evaluated in comparison with Ordinary Differential Neural Networks (NODEs) and also some illustrative examples are shown.

The original article can be found at the following link: -----

## The model proposal



## Files
The repository consists of four working folders, each studying a different scenario for how the data can be presented:

1. Regular mesh data.
2. Noisy data.
3. Randomly selected data.
4. Abundant data at the beginning of the evolution.

In each folder, both the implementation of a Physics-Informed Neural Network (PINN) and a classical method to solve the inverse problem of fitting the parameter \(\alpha_L\) to the observed data for each compartment are provided.
