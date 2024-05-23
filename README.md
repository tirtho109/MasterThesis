# MasterThesis

## Title
Model Learning using Machine Learning and Domain Decomposition.

## Tasks

### 1. Main Objective
 Implementing algorithms for learning the mechanisms of two ordinary differential equations systems, namely:

- **Saturated Growth Model:** 
  $$u' = u(C - u)$$ 
  with $u_0 >0$, and $C$ is a positive parameter.
  
- **Competition Model with coexistence and survival:** 
  $$u' = u(1 - a_1u - a_2v)$$ 
  $$v' = rv(1 - b_1u - b_2v)$$
  with $u_0>0 \quad \text{and}\quad v_0>0$, and $r, a_1, a_2, b_1, b_2$ are all positive parameters. 

using [SciANN](https://github.com/sciann/sciann), [FBPINNs](https://github.com/benmoseley/FBPINNs) and [PySINDy](https://github.com/dynamicslab/pysindy).


### 2. Additional Objectives

- Investigating how the selection of data from different time intervals influences the quality of learned models.
- Apply multi-level domain decomposition techniques for different time intervals, separating the dynamic and quasi-constant phases.
- Visualize the loss landscape for approaches with and without domain decomposition.

## Methods

- **SciANN** [TrySciANN](https://github.com/tirtho109/MasterThesis/tree/TrySciANN) repo. 
- **FBPINNs** [TryFBPINNs](https://github.com/tirtho109/MasterThesis/tree/TryFBPINNs) repo.
- **PySINDy** [TryPySINDy](https://github.com/tirtho109/MasterThesis/tree/TryPySINDy) repo.


