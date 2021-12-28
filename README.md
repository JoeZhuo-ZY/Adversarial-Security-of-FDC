# Attack-and-Defense-Adversarial-Security-of-Data-driven-FDC-Systems
This repository contains the trained model checkpoints and crafted adversarial samples in the paper: 
> Attack and Defense: Adversarial Security of Data-driven FDC Systems.
> 
> Yue Zhuo and Zhiqiang Ge*
## Repository Structure
`data/` contains the original TEP data with the normal condition and first 15 faults

`models/` contains the the trained model checkpoints of defense methods and crafted adversarial samples of attack methods

`TEP/` for fault classification

`TEP_FD/` for fault detection

## Code Reference
We reproduce all the codes of adversarial attack and defense methods for FDC systems. Part of codes refer to the following repos:
### Adversarial attacks
*  **C&W** [advtorch](https://github.com/duggalrahul/REST/tree/master/advertorch)
*  **SPSA, DeepFool** [cleverhans](https://github.com/cleverhans-lab/cleverhans)
*  **UAP** https://github.com/sajabdoli/UAP
*  **MILP** [VeriGauge](https://github.com/AI-secure/VeriGauge)

### Adversarial defense
* **Input Gradient Regularization** https://github.com/cfinlay/tulip
* **IBP-training** https://github.com/pawelmorawiecki/Interval_bound_propagation/tree/de525e3300750abf85f92833ee61f65f1ea6c3eb
