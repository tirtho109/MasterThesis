"""
Defines some helper functions for plotting
"""

import jax.numpy as jnp

from fbpinns.analysis import load_model
from fbpinns.analysis import FBPINN_solution as FBPINN_solution_
from fbpinns.analysis import PINN_solution as PINN_solution_
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from problems import SaturatedGrowthModel, CompetitionModel
"""
i = epochs

"""

def load_FBPINN(tag, problem, network, l, w, h, p, n, rootdir="results/"):
    run = f"FBPINN_{tag}_{problem.__name__}_{network.__name__}_{l}-levels_{w}-overlap_{h}-layers_{p}-hidden_{n[0]}-n"
    c, model = load_model(run, rootdir=rootdir)
    i,t,l1n = model[-1][:,0], model[-1][:,3], model[-1][:,-1]
    return c, model, i, t, l1n

def load_PINN(tag, problem, network, h, p, n, rootdir="results/"):
    run = f"PINN_{tag}_{problem.__name__}_{network.__name__}_{h}-layers_{p}-hidden_{n[0]}-n"
    c, model = load_model(run, rootdir=rootdir)
    i,t,l1n = model[-1][:,0], model[-1][:,3], model[-1][:,-1]
    return c, model, i, t, l1n

def exact_solution(c, model):
    all_params, domain, problem = model[1], c.domain, c.problem
    x_batch = domain.sample_interior(all_params=all_params, key=None, sampler="grid", batch_shape=c.n_test)
    u_exact = problem.exact_solution(all_params, x_batch, batch_shape=c.n_test)
    return u_exact.reshape(c.n_test)

def FBPINN_solution(c, model):
    all_params, domain = model[1], c.domain
    x_batch = domain.sample_interior(all_params=all_params, key=None, sampler="grid", batch_shape=c.n_test)
    active = jnp.ones((all_params["static"]["decomposition"]["m"]))
    u_test = FBPINN_solution_(c, all_params, active, x_batch)
    return u_test.reshape(c.n_test)

def PINN_solution(c, model):
    all_params, domain = model[1], c.domain
    x_batch = domain.sample_interior(all_params=all_params, key=None, sampler="grid", batch_shape=c.n_test)
    u_test = PINN_solution_(c, all_params, x_batch)
    return u_test.reshape(c.n_test)

def plot_model_comparison(c, model, type=None, ax=None):
    if type not in ["FBPINN", "PINN"]:
        raise ValueError("Invalid type specified. Please use 'FBPINN' or 'PINN'.")
    u_exact, u_test, u_learned = get_us(c, model, type=type)
    x_batch = get_x_batch(c, model)
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(x_batch, u_exact[:, 0], 'r-', label='u true')
    ax.plot(x_batch, u_test[:, 0], 'r:', label='u pred')
    ax.plot(x_batch, u_learned[:, 0], 'r-.', label='u learned')
    if u_exact.shape[1]==2:
        ax.plot(x_batch, u_exact[:, 1], 'b-', label='v true')
        ax.plot(x_batch, u_test[:, 1], 'b:', label='v pred')
        ax.plot(x_batch, u_learned[:, 1], 'b-.', label='v learned')
    ax.legend()
    ax.set_xlabel("Time", fontsize=14)
    ax.set_ylabel("Population", fontsize=14)
    ax.set_title(f"{type} Models Comparison")
    if ax is None:
        plt.show()

def get_x_batch(c, model):
    """
    Extract x_batch
    """
    all_params, domain= model[1], c.domain
    x_batch = domain.sample_interior(all_params=all_params, key=None, sampler="grid", batch_shape=c.n_test)
    return x_batch

def get_us(c, model, type=None):
    """
    Extract u_exact, u_test, u_learned
    using model and constants
    """
    if type not in ["FBPINN", "PINN"]:
        raise ValueError("Invalid type specified. Please use 'FBPINN' or 'PINN'.")
    all_params, domain, problem, active = model[1], c.domain, c.problem, model[3]
    x_batch = domain.sample_interior(all_params=all_params, key=None, sampler="grid", batch_shape=c.n_test)
    u_exact = problem.exact_solution(all_params, x_batch, batch_shape=c.n_test)
    if type=="FBPINN":
        u_test = FBPINN_solution_(c, all_params, active, x_batch)
    elif type=="PINN":
        u_test = PINN_solution_(c, all_params, x_batch)
    else:
        raise ValueError("Invalid type specified. Please use 'FBPINN' or 'PINN'.")
    u_learned = problem.learned_solution(all_params, x_batch.reshape(-1))
    return u_exact, u_test, u_learned

def export_mse_mae(u_exact, u_test, u_learned, file_path=None):
    """
    Calculate MSE and MAE of the predicted model and learned model
    """
    mse_test = np.mean((u_exact - u_test)**2)
    mse_learned = np.mean((u_exact - u_learned)**2)
    mae_test = np.mean(np.abs(u_exact - u_test))
    mae_learned = np.mean(np.abs(u_exact - u_learned))
    metrics_df = pd.DataFrame({
        'Metric': ['MSE', 'MAE'],
        'Test': [mse_test, mae_test],
        'Learned': [mse_learned, mae_learned]
    })
    if file_path is not None:
        metrics_df.to_csv(file_path, index=False)
    else:
        metrics_df.to_csv('metrics.csv', index=False)

def export_parameters(c, model, file_path=None):
    """
    This function takes the uploaded constants and jax model.And, 
    export the learned and true params in a csv file.
    """
    all_params = model[1]  
    if c.problem==CompetitionModel:
        true_keys = ('r_true', 'a1_true', 'a2_true', 'b1_true', 'b2_true')
        learned_keys = ('r', 'a1', 'a2', 'b1', 'b2')
        true_params = [float(all_params['static']["problem"][key]) for key in true_keys]
        learned_params = [float(all_params['trainable']["problem"][key]) for key in learned_keys]
        data = {
            'Parameter': ['r', 'a1', 'a2', 'b1', 'b2'],
            'True': true_params,
            'Learned': learned_params
        }
    elif c.problem==SaturatedGrowthModel:
        true_params = [float(all_params['static']['problem']['C_true'])]
        learned_params = [float(all_params['trainable']['problem']['C'])]
        data = {
            'Parameter': ['C'],
            'True': true_params,
            'Learned': learned_params
        }
    else:
        raise ValueError("Unsupported problem type.")

    # Create DataFrame and save to CSV
    parameters_df = pd.DataFrame(data)
    if file_path is not None:
        parameters_df.to_csv(file_path, index=False)
    else:
        parameters_df.to_csv('parameters.csv', index=False)