import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from util import SaturatedGrowthModel, CompetitionModel

# Model Comparison
def plot_models(t_end, initial_condition, params, original_model, sindy_model, ax=None):
    t_span = (0, t_end)
    t_eval = np.linspace(t_span[0], t_span[1], 500)
    sol = solve_ivp(original_model, t_span, initial_condition, args=(params,), t_eval=t_eval)
    t = sol.t
    u_sol = sol.y[0].T 
    if original_model==CompetitionModel:
        v_sol = sol.y[1].T

    # syndy model output
    out = sindy_model.simulate(initial_condition, t_eval, integrator='odeint')
    u_predicted = out[:, 0]
    if original_model==CompetitionModel:
        v_predicted = out[:, 1]

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 3), dpi=300)
    ax.plot(t, u_sol, label='Original u', color='blue')
    ax.plot(t, u_predicted, label='SINDy u', linestyle='--', color='green')
    if original_model==CompetitionModel:
        ax.plot(t, v_sol, label='Original v', color='red')
        ax.plot(t, v_predicted, label='SINDy v', linestyle='--', color='black')
    ax.set_xlabel('Time (t)')
    ax.set_ylabel('Population')
    ax.set_title('Comparison of Original and SINDy Models')
    ax.set_xlim(-0.5, 25)
    ax.set_ylim(-0.2, max(u_sol)+0.5)
    ax.legend()
    if ax is None:
         ax.show()

def plot_pareto(coefs, opt, model, threshold_scan, x_test, t_test, ax1=None, ax2=None):
    dt = t_test[1] - t_test[0]
    mse = np.zeros(len(threshold_scan))
    mse_sim = np.zeros(len(threshold_scan))
    for i in range(len(threshold_scan)):
        opt.coef_ = coefs[i]
        mse[i] = model.score(x_test, dt, metric=mean_squared_error)
        x_test_sim = model.simulate(x_test[0,:], t_test, integrator="odeint")
        if np.any(x_test_sim > 1e4):
            x_test_sim = np.clip(x_test_sim, None, 1e4)
        mse_sim[i] = np.sum((x_test - x_test_sim) ** 2)

    if ax1 is None:
        fig, ax1 = plt.subplots(1, 1, figsize=(5, 3), dpi=300)
    ax1.semilogy(threshold_scan, mse, "bo")
    ax1.semilogy(threshold_scan, mse, "b")
    ax1.set_ylabel(r"$\dot{X}$ RMSE", fontsize=16)
    ax1.set_xlabel(r"$\lambda$", fontsize=16)
    ax1.tick_params(axis='x', labelsize=16)
    ax1.tick_params(axis='y', labelsize=16)
    ax1.grid(True)
    
    if ax2 is None:
        fig, ax2 = plt.subplots(1, 1, figsize=(5, 3), dpi=300)
    ax2.semilogy(threshold_scan, mse_sim, "bo")
    ax2.semilogy(threshold_scan, mse_sim, "b")
    ax2.set_ylabel(r"$\dot{X}$ RMSE", fontsize=16)
    ax2.set_xlabel(r"$\lambda$", fontsize=16)
    ax2.tick_params(axis='x', labelsize=16)
    ax2.tick_params(axis='y', labelsize=16)
    ax2.grid(True)

    # minimum MSE
    min_mse_index = np.argmin(mse)
    min_mse_sim_index = np.argmin(mse_sim)
    
    # lambda values for the minimum MSE
    lambda_min_mse = threshold_scan[min_mse_index]
    lambda_min_mse_sim = threshold_scan[min_mse_sim_index]

    return lambda_min_mse, lambda_min_mse_sim
