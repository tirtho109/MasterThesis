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
        fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    ax.plot(t, u_sol, label='Original u', color='blue')
    ax.plot(t, u_predicted, label='SINDy u', linestyle='--', color='green')
    if original_model==CompetitionModel:
        ax.plot(t, v_sol, label='Original v', color='red')
        ax.plot(t, v_predicted, label='SINDy v', linestyle='--', color='black')
    ax.set_xlabel('Time (t)')
    ax.set_ylabel('Population')
    ax.set_title('Comparison of Original and SINDy Models')
    ax.set_xlim(-0.01, 25)
    # ax.set_ylim(-0.0, max(u_predicted)+0.05)
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
            x_test_sim = np.clip(x_test_sim, 0, 10)
        mse_sim[i] = np.mean((x_test - x_test_sim) ** 2)

    if ax1 is None:
        fig, ax1 = plt.subplots(1, 1, figsize=(5, 3))
    ax1.semilogy(threshold_scan, mse, "bo")
    ax1.semilogy(threshold_scan, mse, "b")
    ax1.set_ylabel(r"$\dot{X}$ MSE", fontsize=16)
    ax1.set_xlabel(r"$\lambda$", fontsize=16)
    ax1.tick_params(axis='x', labelsize=16)
    ax1.tick_params(axis='y', labelsize=16)
    ax1.grid(True)
    
    if ax2 is None:
        fig, ax2 = plt.subplots(1, 1, figsize=(5, 3))
    ax2.semilogy(threshold_scan, mse_sim, "bo")
    ax2.semilogy(threshold_scan, mse_sim, "b")
    ax2.set_ylabel(r"X MSE", fontsize=16)
    ax2.set_xlabel(r"$\lambda$", fontsize=16)
    ax2.tick_params(axis='x', labelsize=16)
    ax2.tick_params(axis='y', labelsize=16)
    ax2.grid(True)

    if ax1 is None:
        ax1.show()
    if ax2 is None:
        ax2.show()

    # minimum MSE
    min_mse_index = np.argmin(mse)
    min_mse_sim_index = np.argmin(mse_sim)
    
    # lambda values for the minimum MSE
    lambda_min_mse = threshold_scan[min_mse_index]
    lambda_min_mse_sim = threshold_scan[min_mse_sim_index]

    return lambda_min_mse, lambda_min_mse_sim

def plot_data_and_derivative(x, dt, deriv, num_features = 1):
    if num_features == 1:
        feature_name = ["x",]
    elif num_features == 2:
        feature_name = ["x", "y",]
    elif num_features == 3:
        feature_name = ["x", "y", "z"]
    else:
        raise ValueError("Number of features must be 1, 2, or 3.")
    plt.figure(figsize=(20, 5))
    for i in range(num_features):
        plt.subplot(1, num_features, i + 1)
        plt.plot(x[:, i], label=feature_name[i])
        plt.grid(True)
        plt.xlabel("t", fontsize=24)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.legend(fontsize=24)
    x_dot = deriv(x, t=dt)
    plt.figure(figsize=(20, 5))
    for i in range(num_features):
        plt.subplot(1, num_features, i + 1)
        plt.plot(x_dot[:, i], label=r"$\dot{" + feature_name[i] + "}$")
        plt.grid(True)
        plt.xlabel("t", fontsize=24)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.legend(fontsize=24)

# Make an errorbar coefficient plot from the results of ensembling
def plot_ensemble_results(
    model, mean_ensemble, std_ensemble, mean_library_ensemble, std_library_ensemble
    ):
    
    # get n_features and n_equations
    n_features = len(model.get_feature_names())
    n_equations = len(model.feature_names)
    # Plot results
    xticknames = model.get_feature_names()
    for i in range(n_features): #  --> number of features
        xticknames[i] = "$" + xticknames[i] + "$"
    plt.figure(figsize=(18, 4))
    colors = ["b", "r", "k"]
    plt.subplot(1, 2, 1)
    plt.xlabel("Candidate terms", fontsize=22)
    plt.ylabel("Coefficient values", fontsize=22)
    # set title to  "Ensemble"
    plt.title("Ensemble", fontsize=22)
    for i in range(n_equations): # --> number of equations
        plt.errorbar(
            range(n_features), #  --> number of features
            mean_ensemble[i, :],
            yerr=std_ensemble[i, :],
            fmt="o",
            color=colors[i],
            label=r"Equation for $\dot{" + model.feature_names[i] + r"}$",
        )
    ax = plt.gca()
    plt.grid(True)
    ax.set_xticks(range(n_features)) # --> number of features
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax.set_xticklabels(xticknames, verticalalignment="top")
    plt.subplot(1, 2, 2)
    plt.xlabel("Candidate terms", fontsize=22)
    # set title to "Library Ensemble"
    plt.title("Library Ensemble", fontsize=22)
    for i in range(n_equations): # --> number of equations
        plt.errorbar(
            range(n_features), # --> number of features
            mean_library_ensemble[i, :],
            yerr=std_library_ensemble[i, :],
            fmt="o",
            color=colors[i],
            label=r"Equation for $\dot{" +model.feature_names[i] + r"}$",
        )
    ax = plt.gca()
    plt.grid(True)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=16, loc="upper right")
    ax.set_xticks(range(n_features)) # --> number of features
    ax.set_xticklabels(xticknames, verticalalignment="top")