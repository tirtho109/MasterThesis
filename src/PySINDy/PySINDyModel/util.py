import os
import shutil
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd

def SaturatedGrowthModel(t, u, C):
        return u*(C-u)

def CompetitionModel(t, q, params):
        u, v = q
        # r, a1, a2, b1,b2
        r, a1, a2, b1,b2 = params
        dudt = u*(1-a1*u- a2*v)  
        dvdt = r*v*(1-b1*u-b2*v)
        return [dudt, dvdt]

def clear_dir(directory):
    """
    Removes all files in the given directory.
    """
    # important! if None passed to os.listdir, current directory is wiped (!)
    if not os.path.isdir(directory): raise Exception(f"{directory} is not a directory")
    if type(directory) != str: raise Exception(f"string type required for directory: {directory}")
    if directory in ["..",".", "","/","./","../","*"]: raise Exception("trying to delete current directory, probably bad idea?!")

    for f in os.listdir(directory):
        path = os.path.join(directory, f)
        try:
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
        except Exception as e:
            print(e)

# Create training and test datasets
def create_datasets(model, tend, numx, nTest, initial_conditions, params, training_time_limit, noise_level, show_figure=False, save_path=None, sparse=False, seed=0):
    t_span = (0, tend)
    if training_time_limit[0] < 0:
        training_time_limit[0] = 0
    if training_time_limit[1] > tend:
        training_time_limit[1] = tend
    np.random.seed(seed)
    if sparse:
         t_eval = np.sort(np.random.uniform(training_time_limit[0], training_time_limit[1], numx))
         t_eval[0]  = 0.0
    else:
         t_eval = np.linspace(training_time_limit[0], training_time_limit[1], numx)
         t_eval[0]  = 0.0

    # training datasets
    sol = solve_ivp(model, t_span, initial_conditions, args=(params,), t_eval=t_eval)
    t_train = sol.t
    x_train = sol.y.T

    noise = np.random.normal(0, noise_level, x_train.shape)
    noise[0,:] = 0 # no noise in the initial conditions
    x_train_noisy = x_train + noise
    u_train = x_train_noisy[:, 0]
    if model==CompetitionModel:
        v_train = x_train_noisy[:, 1]

    # Test datasets
    t_eval = np.linspace(0, tend, nTest )
    sol_test = solve_ivp(model, t_span, initial_conditions, args=(params,), t_eval=t_eval)
    t_test = sol_test.t
    x_test = sol_test.y.T

    if show_figure:
        plt.figure(figsize=(5, 3))
        plt.scatter(t_train, u_train, label='u(t)')
        if model==CompetitionModel:
            plt.scatter(t_train, v_train, label='v(t)')
        plt.xlabel('Time (t)')
        plt.ylabel('Population')
        if model==CompetitionModel:
            plt.title(f'Comp Model Training Data, n-{noise_level}, t-{training_time_limit}')
        else:
            plt.title(f'SGModel Training Data, n-{noise_level}, t-{training_time_limit}')
        plt.legend()
        if save_path:
            if os.path.basename(save_path) == '':
                raise ValueError("Please provide a file name with the save path.")
            plt.savefig(save_path)
        else:
            plt.show()

    return x_train_noisy, t_train, x_test, t_test

def export_mse_mae(sindy_model, x_test, t_test, file_path=None):
    x_test_sim = sindy_model.simulate(x_test[0,:], t_test, integrator="odeint")
    if np.any(x_test_sim > 1e4):
        x_test_sim = np.clip(x_test_sim, 0, 10)
    # Calculate MSE
    squared_errors = (x_test - x_test_sim) ** 2
    mse = np.mean(squared_errors)

    # Calculate MAE
    absolute_errors = np.abs(x_test - x_test_sim)
    mae = np.mean(absolute_errors)

    metrics_df = pd.DataFrame({
        'Metric': ['MSE', 'MAE'],
        'Test': [mse, mae],
    })
    if file_path is not None:
        metrics_df.to_csv(file_path, index=False)
    else:
        metrics_df.to_csv('metrics.csv', index=False)

def export_coefficients(coefficients, file_path=None):
    u_coefs = coefficients[0]
    alpha_names = [f'alpha_{i+1}' for i in range(len(u_coefs))]
    
    df_data = {alpha: [coef] for alpha, coef in zip(alpha_names, u_coefs)}
    
    if len(coefficients) > 1:
        v_coefs = coefficients[1]
        beta_names = [f'beta_{i+1}' for i in range(len(v_coefs))]
        for beta, coef in zip(beta_names, v_coefs):
            df_data[beta] = [coef]

    coefficients_df = pd.DataFrame(df_data)
    
    # Save to CSV
    if file_path:
        coefficients_df.to_csv(file_path, index=False)
        print(f"Coefficients exported to {file_path}")
    else:
        print(coefficients_df)

def export_thereshold(optimizer, file_path=None):
    threshold = optimizer.threshold
    threshold_df = pd.DataFrame({'Threshold': [threshold]})
    
    if file_path:
        threshold_df.to_csv(file_path, index=False)
        print(f"Threshold exported to {file_path}")
    else:
        print(threshold_df)

