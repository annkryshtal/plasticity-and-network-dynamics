# Gamma Oscillations: STDP-Induced Plasticity in Hippocampal Networks

This repository contains simulation code and results from the project:

> Investigating STDP-Induced Plasticity Effects on Hippocampal Gamma Oscillations  
> By [Anna Kryshtal](https://github.com/annkryshtal) 
> Supervisor: Dr. Caroline Geisler, LMU Munich

The study explores how spike-timing-dependent plasticity (STDP) shapes the dynamics of gamma oscillations (30–80 Hz) in simplified hippocampal neural networks composed of excitatory and inhibitory units. All simulations were implemented using the `Brian2` simulator.

## Overview

The simulations use spiking neural networks built with Brian2, incorporating STDP learning rules. The project examines how plasticity influences connectivity and the emergence of gamma oscillations across different synaptic pathways: EE, EI, IE, and II.

## Repository Structure

### 🧠 Learning Phase Notebooks

These notebooks execute STDP-based learning, simulating how synaptic weights evolve and affect network dynamics:

- `NN_test_#2_learning_EE.ipynb` – Excitatory→Excitatory
- `NN_test_#2_learning_EI.ipynb` – Excitatory→Inhibitory
- `NN_test_#2_learning_IE.ipynb` – Inhibitory→Excitatory
- `NN_test_#2_learning_II.ipynb` – Inhibitory→Inhibitory

### 📁 Saved Run Notebooks

These notebooks display the output of previously trained networks and visualize the learned connectivity and oscillatory behavior:

- `NN_STDP_saved_run_EE.ipynb` – Excitatory→Excitatory
- `NN_STDP_saved_run_EI.ipynb` – Excitatory→Inhibitory
- `NN_STDP_saved_run_IE.ipynb` – Inhibitory→Excitatory
- `NN_STDP_saved_run_II.ipynb` – Inhibitory→Inhibitory

### Analysis Module

- `functions_spectrum.py` - A collection of utility functions used for data analysis and spectral computations.

## Requirements

- Python 3.x
- Jupyter Notebook
- brian2
- numpy
- matplotlib

## Citation

If you use this code or build upon this work, please cite or acknowledge the project and authorship appropriately.