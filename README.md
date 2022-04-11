# Attack-and-Defense-Adversarial-Security-of-Data-driven-FDC-Systems
This repository contains the implementations of all adversarial attack and defense methods in the paper: 
> Attack and Defense: Adversarial Security of Data-driven FDC Systems.
> 
> Yue Zhuo and Zhiqiang Ge*

We also released the trained model checkpoints and calculated adversarial samples.

## Repository Structure
`attack/` contains the the implementations of adversarial attack methods

`defense/` contains the the implementations of adversarial defense methods

`data/` contains the original TEP data with the normal condition and first 15 faults

`models/` contains the the trained model checkpoints of defense methods and calculated adversarial samples of attack methods

`TEP/` for fault classification

`TEP_FD/` for fault detection

## Run
Run `main.py` can iteratively train the defensive models and calculate adversarial samples, to reproduce the main results of benchmark in the paper.

Configure variable `dataset_name` to adjust the task (faule classification or detection).

## Code Reference
We reproduce all the codes of adversarial attack and defense methods for FDC systems. Part of codes refer to the following repos:
### Adversarial attacks
*  **C&W** in [advtorch](https://github.com/duggalrahul/REST/tree/master/advertorch)
*  **SPSA, DeepFool** in [cleverhans](https://github.com/cleverhans-lab/cleverhans)
*  [**UAP**](https://github.com/sajabdoli/UAP)
*  **MILP** in [VeriGauge](https://github.com/AI-secure/VeriGauge)

### Adversarial defense
* [**Input Gradient Regularization**](https://github.com/cfinlay/tulip)
* [**IBP-training**](https://github.com/pawelmorawiecki/Interval_bound_propagation/tree/de525e3300750abf85f92833ee61f65f1ea6c3eb)
