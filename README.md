# Physics-Modeled-Neural-Networks (PMNNs)

Raul Felipe-Sosa, Angel Martin del Rey, Maria Flores Ceballos

## Abstract
The aim of this work is to (mathematically) describe and analyze a new type of neural network architecture, which we will call Dynamical Physics-Modeled Neural Networks (DynPMNN), that uses ordinary differential equations in its construction and which is consistent with the idea of what a neural network is. Its performance is evaluated in comparison with Ordinary Differential Neural Networks (NODEs) and also some illustrative examples are shown.

The original article can be found at the following [link](https://arxiv.org/abs/2605.08176).

## The model proposal
While Neural Ordinary Differential Equations (NODEs) transform the forward propagation process into a continuous model governed by an ODE, and Physics-Informed Neural Networks (PINNs) incorporate physical laws directly into the loss function, Dynamical Physics-Modeled Neural Networks (DynPMNNs) extend this by modeling the hidden layers of the neural network using ODEs, where each hidden layer is treated as a dynamic system whose evolution is governed by a differential equation.

The PMNN (Physics-Modeled Neural Network) framework in the paper integrates systems of ODEs to model the dynamics of the hidden layer. Specifically, the model uses a system based on the FitzHugh-Nagumo or Hodgkin-Huxley models to describe neuronal activation, representing each hidden layer as a set of ODEs with trainable parameters. The solution to these ODEs determines the layer’s output, effectively introducing dynamic behavior within the network. The PMNN framework can be trained using numerical methods such as the Euler method to approximate the solution of the ODE governing each hidden layer's dynamics. This approach enhances the predictive power of the network by incorporating continuous-time dynamics.

Despite having substantially fewer trainable parameters, the proposed model achieves competitive performance, illustrating the expressive power and efficiency that arise from embedding physically meaningful dynamics into neural architectures.

Representation of a PMNN with a two-layer Euler block:

<p align="center">
 <img width="636" height="338" alt="image" src="https://github.com/user-attachments/assets/d4544ea8-1bce-4316-9151-b81c42856e9a" />
</p>

## Experiments
We considered six datasets (Banknote, Breast Cancer, Califournia House Pricing, Diabetes, Energy efficiency and Iris) and four baseline models (NODEs, Closed-form Continous time models -CfCs-, Decision tree and Deep Neural Network -DNN-) to compare against our proposal in Section 5. Each dataset was selected to evaluate the PMNN under different classification and regression scenarios, thereby providing a broader assessment of its performance. The aim of our proposal is to provide a preliminary neural architecture in which physical dynamics are embedded into the hidden representation of the network. This kind of models may offer a more realistic framework for modeling brain-inspired processes and information processing. For this reason, we considered that the comparison would be more meaningful if it included baseline models that also incorporate continuous-time or physics-related modeling principles. Accordingly, NODEs and CfCs were included to provide a fairer comparison with architectures that share this dynamical modeling perspective, while standard DNN and Decision Tree models were also added to complete the empirical evaluation.

## Repository structure
Each dataset follows the same general organization. The folder name corresponds to the dataset, and inside it there are the source files, the configuration file, the execution script, and the folders generated during training.

```text
[Dataset_name]/
├── src/
│   ├── [model_name].py
│   ├── data.py
│   └── auxiliar.py
├── config.yaml
├── main.py
├── runs_[model_name]
│   └── best_hparams/
│       ├── best_summary.txt
│       └── [model_hparams]_loss_graph.png
```

## How to use it

To run an experiment, go to the folder corresponding to the dataset and model of interest. Each experiment must be executed in two steps. First, run `data.py` to download, preprocess and split the dataset into training, validation and test sets:
```bash
python src/data.py
```

This script creates the processed files inside the directory specified in `config.yaml`, usually through the `paths.datadir` field.

After the dataset has been prepared, run the main script:
```bash
python main.py
```

The `main.py` file reads the same `config.yaml` file, builds the selected model, trains it, evaluates it on the validation and test sets, and saves the results automatically.

After execution, the results will be saved in a folder named according to the model. Inside these folders, the best hyperparameter configuration is copied to:
```text
best_hparams/
```

## Cite
If you use this repository, please cite:

```bibtex
@misc{felipesosa2026physicsmodeledneuralnetworks,
      title={Physics-Modeled Neural Networks}, 
      author={Raul Felipe-Sosa and Angel Martin del Rey and Maria Flores Ceballos},
      year={2026},
      eprint={2605.08176},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.08176}, 
}
```
