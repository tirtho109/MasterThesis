# PySINDy Implementation

The PySINDy package adopts a distinct approach to modeling dynamical systems compared to SciANN and FBPINNs. PySINDy directly identifies the underlying equations' coefficients from training data rather than learning specific model parameters such as $C$ in the Saturated Growth Model or $r, a_1, a_2, b_1, b_2$ in the Competition model. Instead, the learning process determines the coefficients for a predefined feature library used during model training.

### Saturated Growth Model 
As a result, the Saturated Growth Model is rewritten in the coefficient form as,
$$u' = u(C-u) = Cu- u^2 $$
$$u' = \alpha_1 \cdot u - \alpha_2 \cdot u^2 $$
where $\alpha_1 = C \quad \text{and} \quad \alpha_2 = 1$

For $C=1$, the true equation for Saturated Growth is:
$$u' = 1 \cdot u - 1 \cdot u^2$$

### Competition Model

Similarly, in Competition model, we rewrite the equations in coefficient form as:
$$ u' = u(1-a_1 u- a_2 v) $$
$$v' = rv(1- b_1 u - b_2 v) $$
$$u' = \alpha_1 \cdot u + \alpha_2 \cdot v + \alpha_3 \cdot u^2  + \alpha_4 \cdot uv + \alpha_5 \cdot v^2$$
$$v' =  \beta_1 \cdot u + \beta_2 \cdot v + \beta_3 \cdot u^2 + \beta_4 \cdot uv + \beta_5 \cdot v^2$$
$$u' = 1 \cdot u + 0 \cdot v + (-a_1) \cdot u^2 + (-a_2) \cdot uv + 0 \cdot v^2$$
$$u' = 1 \cdot u + 0 \cdot v + (-a_1) \cdot u^2 + (-a_2) \cdot uv + 0 \cdot v^2$$
So, we generalize the coefficients as:
$$\alpha_1 = 1; \quad \alpha_2 = 0 ;\quad \alpha_3 = -a_1; \quad \alpha_4 = -a_2; \quad \alpha_5 = 0$$
$$\beta_1 = 0;\quad \beta_2= r;\quad \beta_3 = 0; \quad \beta_5 = -rb_1 ;\quad \beta_4 = -rb_2$$
#### Coexistence
For Competition model with coexistence case, with $$r = 0.5;  a_1 = 0.7;  a_2 =0.3;  b_1 = 0.3;  b_2 = 0.6$$, the underlying true equations have the following form:
$$u' = 1 \cdot u + 0 \cdot v + (-0.7) \cdot u^2 + (-0.3) \cdot uv + 0 \cdot v^2$$
$$v' =  0 \cdot u + 0.5 \cdot v + 0 \cdot u^2 + (-0.15) \cdot uv + (-0.3) \cdot v^2$$

#### Survival
Similarly for the Competition model with Survival case, with $$r = 0.5;  a_1 = 0.3;  a_2 =0.6;  b_1 = 0.7;  b_2 = 0.3$$, the underlying true equation is:
$$ u' = 1 \cdot u + 0 \cdot v + (-0.3) \cdot u^2 + (-0.6) \cdot uv + 0 \cdot v^2$$
$$v' =  0 \cdot u + 0.5 \cdot v + 0 \cdot u^2 + (-0.35) \cdot uv + (-0.15) \cdot v^2$$

By implementing the PySINDy approach, we compare the coefficient of the learned model with the true coefficient in case of Saturated Growth and  Competition model with coexistence and survival. The implementation is similar for both models except for the internal constraints based on the individual model.


# Steps for PySINDy Implementation

## Step 1: Set Fixed Parameters
Define and initialize necessary parameters:
- **Domain**: $(0, 24)$
- **Number of training data**: 100
- **Sparsity of training data**: True
- **Number of test data**: 500
- **Initial conditions**: SG Model $0.01$, adn Comp Model $(2,1)$.
- **Time interval for training data selection**: $[0,10],[10,24],[0,24]$
- **Saturated Growth Model**: $C$
- **Competition Model**: $r, a_1, a_2, b_1, b_2$

## Step 2: Generate Training and Test Data
Create data sets using the fixed parameters:
- **Models**: `SaturatedGrowthModel`, `CompetitionModel`
- **Domain**: $(0, 24)$
- **Initial condition**: must include initial conditions for training

## Step 3: Set Feature Names and Feature Library
Define features and construct the feature library:
- **Saturated Growth Model**: $u$, polynomial degree 2, without bias.
  $$\texttt{Feature}_{SGModel} = [u, u^2]$$
- **Competition Model**: $u, v$, polynomial degree 2, without bias. 
$$\texttt{Feature}_{CompModel} = [u, v, u^2, uv, v^2]
$$

## Step 4: Set Constraints on the Coefficients for the Optimizer
Apply constraints during optimization using `ConstrainedSR3`:

### Saturated Growth Model:
- $\alpha_1 = C$
- $\alpha_2 = 1$ (equality constraint)

### Competition Model: 10 coefficients
- Known: $\alpha_1 = 1$, $\alpha_2 = 0$, $\alpha_5 = 0$, $\beta_1 = 0$, $\beta_3 = 0$ (equality constraints)
- Unknown: $\alpha_3 = -a_1$, $\alpha_4 = -a_2$, $\beta_2 = r$, $\beta_4 = -rb_1$, $\beta_5 = -rb_2$

## Step 5: Find the Best $\lambda$ by Fitting Several Models
Experiment with various $\lambda$ values using a `Pareto` curve:

### Optimizer Setup:
- `constraint_rhs` and `constraint_lhs` for known coefficients
- Regularization threshold $\lambda$
- Thresholder: $l_2$
- Trimming fraction for outliers

### Model Setup:
- Differentiation methods: Finite Difference, Smoothed Finite Difference
- Fit the model with training data
- Collect model coefficients for each $\lambda$

### Pareto Curve:
- Identify $\lambda$ with the lowest error

## Step 6: Rerun the Model with the Best $\lambda$
Set up the optimizer with the best $\lambda$ and finalize the model coefficients with the training data.

## Step 7: Postprocessing
Analyze and interpret the model output:

- Compare the learned model with the true model
- Export MSE error on the test data
- Export model coefficients for further comparison

<!-- This summary encapsulates the key steps and parameters for implementing the PySINDy approach, ensuring the essential equations and notations are clearly presented. -->
