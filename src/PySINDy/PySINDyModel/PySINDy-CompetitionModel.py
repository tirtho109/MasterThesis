import sys
import os
import numpy as np
import pysindy as ps
import inspect
from scipy.integrate import solve_ivp
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import copy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse

from pysindy.feature_library import GeneralizedLibrary, PolynomialLibrary, FourierLibrary, CustomLibrary, ConcatLibrary


# Input interface for python. 
parser = argparse.ArgumentParser(description='''
        PySINDy code for Separating longtime behavior and learning of mechanics  \n
        Saturated Growth Model'''
)

# arguments for training data generator
parser.add_argument('-ic', '--initial_conditions', help='Initial conditions(u0,v0) for the model (default [2,1])', type=float, default=[2,1])
parser.add_argument('--tend', help='End time for the model simulation (default 24)', type=int, default=24)
parser.add_argument('--model_type', help='Survival or co-existence model (default survival**)', type=str, nargs=1, default=['survival'])

# arguments for training data generator
parser.add_argument('-nx', '--numx', help='Num Node in X (default 100)', type=int, nargs=1, default=[100])
parser.add_argument('--sparse', help='Sparsity of training data (default True)', type=bool, nargs=1, default=[False])
parser.add_argument('-tl', '--time_limit', help='Time window for the training data (default [10, 30])', type=int, nargs=2, default=[10, 30])
parser.add_argument('-nl', '--noise_level', help='Level of noise in training data (default 0.005)', type=float, default=0.005)
parser.add_argument('-sf', '--show_figure', help='Show training data (default True)', type=bool, nargs=1, default=[True])
parser.add_argument('-nt','--num_threshold', help='Number of threshold to be scanned (default 100)', type=int, default=100)
parser.add_argument('-mt','--max_threshold', help='Maximum threshold to be scanned (default 1)', type=float, default=1)

parser.add_argument('-op', '--outputpath', help='Output path (default ./file_name)', type=str, nargs=1, default=['output'])
parser.add_argument('-of', '--outputprefix', help='Output path (default res**)', type=str, nargs=1, default=['res'])

# Optimizer
parser.add_argument('-fl', '--feature_library', help='Feature Library to choose for SINDy algorithm (default poly_library**)', type=str, nargs=1, default=['poly'])
parser.add_argument('-po', '--poly_order', help='Polynomial order for poly library (default 2)', type=int, default=2)
parser.add_argument('-opt', '--optimizer', help='optimizer for SINDy algorithm (default SR3**)', type=str, nargs=1, default=['SR3'])
parser.add_argument('-th', '--thresholder', help='thresholder for SINDy optimizer algorithm (default l1**)', type=str, nargs=1, default=['l1'])

parser.add_argument('--plot', help='Plot the model', nargs='?', default=False)

parser.add_argument('--gpu', help='Use GPU if available (default False)', type=bool, nargs=1, default=[False])
args = parser.parse_args()

if not args.gpu[0]:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Manage Files
if not os.path.isdir(args.outputpath[0]):
        os.mkdir(args.outputpath[0])

#TODO
folder_name = (
    f"CompModel_{args.model_type[0]}_"
    f"tl_{'-'.join(map(str, args.time_limit))}_"
    f"nl_{args.noise_level}_"
    f"numx_{args.numx[0]}_"
    f"sparse_{args.sparse[0]}_"
    f"fl_{args.feature_library[0]}_"
    f"opt_{args.optimizer[0]}_"
)
output_folder = os.path.join(args.outputpath[0], folder_name)

if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

output_file_name = os.path.join(output_folder, args.outputprefix[0])
fname = (
    f"{output_file_name}_"
    f"tl_{'-'.join(map(str, args.time_limit))}_"
    f"nl_{args.noise_level}_"
    f"numx_{args.numx[0]}_"
    f"sparse_{args.sparse[0]}_"
    f"fl_{args.feature_library[0]}_"
    f"opt_{args.optimizer[0]}_"
)
        
# Plot functions
def competitionModel(t, q, params):
        u, v = q
        # r, a1, a2, b1,b2
        r, a1, a2, b1,b2 = params
        dudt = u*(1-a1*u- a2*v)  
        dvdt = r*v*(1-b1*u-b2*v)
        return [dudt, dvdt]

def plot_pareto(coefs, opt, model, threshold_scan, x_test, t_test, ax1=None, ax2=None):
    dt = t_test[1] - t_test[0]
    mse = np.zeros(len(threshold_scan))
    mse_sim = np.zeros(len(threshold_scan))
    for i in range(len(threshold_scan)):
        opt.coef_ = coefs[i]
        mse[i] = model.score(x_test, t=dt, metric=mean_squared_error)
        x_test_sim = model.simulate(x_test[0,:], t_test, integrator="odeint")
        if np.any(x_test_sim > 1e4):
            x_test_sim = np.clip(x_test_sim, None, 1e4)
        mse_sim[i] = np.sum((x_test - x_test_sim) ** 2)

    if ax1 is None:
        plt.figure()
        plt.semilogy(threshold_scan, mse, "bo")
        plt.semilogy(threshold_scan, mse, "b")
        plt.ylabel(r"$\dot{X}$ RMSE", fontsize=16)
        plt.xlabel(r"$\lambda$", fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(True)
    else:
        ax1.semilogy(threshold_scan, mse, "bo")
        ax1.semilogy(threshold_scan, mse, "b")
        ax1.set_ylabel(r"$\dot{X}$ RMSE", fontsize=16)
        ax1.set_xlabel(r"$\lambda$", fontsize=16)
        ax1.grid(True)

    if ax2 is None:
        plt.figure()
        plt.semilogy(threshold_scan, mse_sim, "bo")
        plt.semilogy(threshold_scan, mse_sim, "b")
        plt.ylabel(r"$\dot{X}$ RMSE", fontsize=16)
        plt.xlabel(r"$\lambda$", fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(True)
    else:
        ax2.semilogy(threshold_scan, mse_sim, "bo")
        ax2.semilogy(threshold_scan, mse_sim, "b")
        ax2.set_ylabel(r"$\dot{X}$ RMSE", fontsize=16)
        ax2.set_xlabel(r"$\lambda$", fontsize=16)
        ax2.grid(True)

    # minimum MSE
    min_mse_index = np.argmin(mse)
    min_mse_sim_index = np.argmin(mse_sim)
    
    # lambda values for the minimum MSE
    lambda_min_mse = threshold_scan[min_mse_index]
    lambda_min_mse_sim = threshold_scan[min_mse_sim_index]

    return lambda_min_mse, lambda_min_mse_sim

# Model Comparison
def plot_models(t_end, initial_condition, params, original_model, sindy_model, ax=None):
    t_span = (0, t_end)
    t_eval = np.linspace(t_span[0], t_span[1], 500)
    sol = solve_ivp(original_model, t_span, initial_condition, args=(params,), t_eval=t_eval)
    t = sol.t
    u_sol = sol.y[0].T 
    v_sol = sol.y[1].T

    # syndy model output
    out = sindy_model.simulate(initial_condition, t_eval, integrator='odeint')
    u_predicted = out[:, 0]
    v_predicted = out[:, 1]

    if ax is None:
        plt.figure(figsize=(5, 3))
        plt.plot(t, u_sol, label='Original u', color='blue')
        plt.plot(t, u_predicted, label='SINDy u', linestyle='--', color='yellow')
        plt.plot(t, v_sol, label='Original v', color='red')
        plt.plot(t, v_predicted, label='SINDy v', linestyle='--', color='black')
        plt.xlabel('Time (t)')
        plt.ylabel('Population')
        plt.title('Comparison of Original and SINDy Models')
        plt.legend()
        plt.show()
    else:
        ax.plot(t, u_sol, label='Original u', color='blue')
        ax.plot(t, u_predicted, label='SINDy u', linestyle='--', color='yellow')
        ax.plot(t, v_sol, label='Original v', color='red')
        ax.plot(t, v_predicted, label='SINDy v', linestyle='--', color='black')
        ax.set_xlabel('Time (t)')
        ax.set_ylabel('Population')
        ax.set_title('Comparison of Original and SINDy Models')
        ax.legend()

# Create training and test datasets
def create_datasets(model, tend, numx, initial_conditions, params, training_time_limit, noise_level, show_figure=False, save_path=None, sparse=False):
    tend = args.tend
    t_span = (0, tend)
    if sparse:
         t_eval = np.sort(np.random.uniform(0, tend, numx))
         t_eval[0]  = 0.0
    else:
         t_eval = np.linspace(t_span[0], t_span[1], numx)

    # training datasets
    sol = solve_ivp(model, t_span, initial_conditions, args=(params,), t_eval=t_eval)
    t_train = sol.t
    x_train = sol.y.T

    mask = (t_train == 0) | ((t_train >= training_time_limit[0]) & (t_train <= training_time_limit[1]))
    t_filtered = t_train[mask]
    x_filtered = x_train[mask]

    noise = np.random.normal(0, noise_level, x_filtered.shape)
    x_filtered_noisy = x_filtered + noise
    u_train = x_filtered_noisy[:, 0]
    v_train = x_filtered_noisy[:, 1]

    # Test datasets
    sol_test = solve_ivp(model, t_span, initial_conditions, args=(params,), t_eval=t_eval)
    t_test = sol_test.t
    x_test = sol_test.y.T
    u_test = sol_test.y[0]
    v_test = sol_test.y[1]

    if show_figure:
        plt.figure(figsize=(5, 3))
        plt.scatter(t_filtered, u_train, label='u(t)')
        plt.scatter(t_filtered, v_train, label='v(t)')
        plt.xlabel('Time (t)')
        plt.ylabel('u')
        plt.title('Competition Model Data')
        plt.legend()
        if save_path:
            if os.path.basename(save_path) == '':
                raise ValueError("Please provide a file name with the save path.")
            plt.savefig(save_path)
        else:
            plt.show()

    return x_train, t_train, x_test, t_test
        
def run_comp_model():
    
    if args.model_type[0]=="survival":
        # [r, a1, a2, b1, b2, e0, f0, e1, f1, e2, f2, e3, f3, e4, f4]
        params = [0.5, 0.3, 0.6, 0.7, 0.3] # , 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    else:
        params = [0.5, 0.7, 0.3, 0.3, 0.6] # , 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    np.random.seed(0)
    plot_save_file = "training_data_plot.png"
    plot_save_path = os.path.join(output_folder, plot_save_file)
    x_train, t_train, x_test, t_test = create_datasets(model=competitionModel, 
                                                       tend=args.tend, 
                                                       numx=args.numx[0],
                                                       initial_conditions=args.initial_conditions, 
                                                       params=params, 
                                                       training_time_limit=args.time_limit, 
                                                       noise_level=args.noise_level, 
                                                       show_figure=args.show_figure,
                                                       save_path=plot_save_path if args.show_figure[0] else None,
                                                       sparse=args.sparse[0])
    
    threshold_scan = np.linspace(0, args.max_threshold, args.num_threshold)
    coefs = []
    feature_names = ["u", "v"]
    feature_library = None
    # feature library
    if args.feature_library[0]=="poly":
            feature_library = PolynomialLibrary(degree=args.poly_order)
    elif args.feature_library[0]=="fourier":
            feature_library = FourierLibrary()
    elif args.feature_library[0]=="combined":
            poly_library = PolynomialLibrary(degree=args.poly_order)
            fourier_library = FourierLibrary()
            feature_library = GeneralizedLibrary([poly_library, fourier_library])
    elif args.feature_library[0] == "custom":
         custom_functions = [lambda u,v:u, 
                    lambda u,v:v, 
                    lambda u,v:u**2, 
                    lambda u,v: v**2, 
                    lambda u,v:u*v]
         function_names = [lambda u, v: f"{u}",
                    lambda u, v: f"{v}",
                    lambda u, v: f"{u}^2",
                    lambda u, v: f"{v}^2",
                    lambda u, v: f"{u}{v}"]
         feature_library = CustomLibrary(library_functions=custom_functions, 
                                        function_names=function_names, 
                                        include_bias=False).fit(x_train[:,0], x_train[:,1])

    for i, threshold in enumerate(threshold_scan):
        if args.optimizer[0]=="SR3":
             optimizer = ps.SR3(threshold=threshold, thresholder=args.thresholder[0])
        elif args.optimizer[0]=="SINDyPI":
             optimizer = ps.SINDyPI(threshold=threshold, thresholder=args.thresholder[0])
        modelX = ps.SINDy(feature_names=feature_names,
                          optimizer=optimizer,
                          feature_library=feature_library)
        modelX.fit(x_train, t_train, quiet=True)
        coefs.append(modelX.coefficients())

    # loss plot
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(12,6), dpi=300)
    lambda_mse, lambda_mse_sim = plot_pareto(coefs, optimizer, modelX, threshold_scan, x_test, t_test, ax1=ax1, ax2=ax2)
    fig.suptitle("Loss plot")
    plt.savefig("{}_Loss.png".format(output_file_name))

    # model comparison
    if args.model_type[0] == "survival":
        r, a1, a2, b1, b2 = [0.5, 0.3, 0.6, 0.7, 0.3]
        original_eq = f"u' = u(1-{a1}*u, {a2}*v)\nv' = {r}*v(1-{b1}*u -{b2}*v)"
    else:  # Default to co-existence model
        r, a1, a2, b1, b2 = [0.5, 0.7, 0.3, 0.3, 0.6]
        original_eq = f"u' = u(1-{a1}*u - {a2}*v)\nv' = {r}*v(1-{b1}*u -{b2}*v)"

    if args.optimizer[0]=="SR3":
        opt_out = ps.SR3(threshold=lambda_mse, thresholder=args.thresholder[0])
    elif args.optimizer[0]=="SINDyPI":
        opt_out = ps.SINDyPI(threshold=lambda_mse, thresholder=args.thresholder[0])
    model_out = ps.SINDy(feature_names=feature_names,
                         optimizer=opt_out,
                         feature_library=feature_library)
    model_out.fit(x_train, t_train, quiet=True)

    fig, (ax1, ax2) = plt.subplots(2, 1,figsize=(6,6), dpi=300)
    plot_models(t_end=args.tend,
                initial_condition=args.initial_conditions,
                params=params,
                original_model=competitionModel,
                sindy_model=model_out, ax = ax1)
    
    sindy_eq = model_out.equations()

    ax2.axis('off') 

    lambda_mse_formatted = f"{lambda_mse:.3f}"
    equation_text = f"$\\lambda = {lambda_mse_formatted}$ \n"
    equation_text += f"\nSINDy Model Equations:\n{'' if len(sindy_eq) == 0 else sindy_eq[0]}\n{'' if len(sindy_eq) < 2 else sindy_eq[1]}"
    equation_text += f"\n\nOriginal Model Equation:\n{original_eq}"
    ax2.text(0.5, 0.5, equation_text, ha='center', va='center', fontsize=10)
    plt.savefig("{}_Model_Comparison.png".format(output_file_name))
    print("done")
    
if __name__=="__main__":
    run_comp_model()