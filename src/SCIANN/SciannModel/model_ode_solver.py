import numpy as np
import sciann as sn 
import matplotlib.pyplot as plt
import csv

import os
import sys
import pandas as pd
from scipy.integrate import odeint
import plotly.graph_objects as go
import plotly.io as pio

from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def _model(self, state, t):
        """
        Defines the DE model.
        This abstract method must be implemented by child classes.
        """
        pass

    def solve_ode(self, initial_conditions, t_span):
        """
        Solves the ODE using scipy's ````odeint````.
        """
        solution = odeint(self._model, initial_conditions, t_span)
        return t_span, solution
    
    @abstractmethod
    def plot_solution(self, initial_conditions, t_end):
        """
        Plots the solution of the ODE.
        This abstract method must be implemented by child classes.
        """
        pass

class SaturatedGrowthModel(BaseModel):
    def __init__(self, C, initial_conditions, tend):
        super().__init__()
        self.C = C
        self.initial_conditions = initial_conditions
        self.tend = tend

    def _model(self, u, t):
        return u*(self.C -u)
    
    def plot_solution(self, ax=None, set_title=None):
        t_span = np.arange(0, self.tend, 0.1)
        _, solution = self.solve_ode(initial_conditions=self.initial_conditions, t_span=t_span)
        u = solution[:, 0]

        if ax is None:
            fig, ax = plt.subplots()
            ax.plot(t_span, u)
            if set_title is None:
                ax.set_title("SG Model")
            else:
                ax.set_title(set_title)
            ax.set_xlabel("Time")
            ax.set_ylabel("Population")
            plt.show() 
        else:
            ax.plot(t_span, u)
            if set_title is not None:
                ax.set_title(set_title)
            ax.set_xlabel("Time")
            ax.set_ylabel("Population")

class CompetitionModel(BaseModel):
    def __init__(self, params, initial_conditions, tend):
        super().__init__()
        self.params = params
        self.initial_conditions = initial_conditions
        self.tend = tend

    def _model(self, q, t):
        u, v = q
        r, a1, a2, b1,b2, e0, f0, e1, f1, e2, f2, e3, f3, e4,f4 = self.params
        dudt = u*(1-a1*u- a2*v)  +e0 + e1*u + e2*v + e3*u*u*v + e4*u*v*v
        dvdt = r*v*(1-b1*u-b2*v) +f0 + f1*u + f2*v + f3*u*u*v + f4*u*v*v
        return [dudt, dvdt]
    
    def plot_solution(self, ax=None, set_title=None):
        t_span = np.arange(0, self.tend, 0.1)
        _, solution = self.solve_ode(initial_conditions=self.initial_conditions, t_span=t_span)
        u, v = solution[:, 0], solution[:, 1]

        created_fig = False
        if ax is None:
            fig, ax = plt.subplots()
            created_fig = True

        ax.plot(t_span, u, label="Population 1")
        ax.plot(t_span, v, label="Population 2")
        if set_title is None:
            ax.set_title("Comp Model")
        else:
            ax.set_title(set_title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Population")
        ax.legend()

        if created_fig:
            return fig

def create_train_data(model, tend, 
                      initial_conditions, numpoints=500, 
                      sparse=False, time_limit=None, 
                      noise_level=0.0, show_figure=False):
    """
    Generates training data for a given model.
    
    Parameters:
    model : instance of a model class, must have a solve_ode method
    tend : end time for the data generation
    initial_cond : initial conditions for the ODE
    numpoints : number of data points to generate
    sparse : whether to generate data points sparsely
    time_limit : time window for data generation
    noise_level : level of noise to add to the data

    Returns:
    A tuple of (tTrain, training_data), where training_data can be a list of arrays for models with multiple outputs.
    """
    # check conditions
    if tend <= 0:
        raise ValueError("End time 'tend' must be greater than zero.")
    if numpoints <=1:
        raise ValueError("Numer of points 'numpoints' must be greater than one.")
    if time_limit is not None:
        if not isinstance(time_limit, (list, tuple)) or len(time_limit) != 2:
            raise ValueError("Time limit 'time_limit' must be a list or tuple with two elements.")
        if not 0 <= time_limit[0] < time_limit[1]:
            raise ValueError("Time limit 'time_limit' must contain increasing values greater than zero.")

    # sparsity 
    if sparse:
        tTrain = np.sort(np.random.uniform(0, tend, numpoints))
        tTrain[0] = 0.0
    else:
        tTrain = np.linspace(0, tend, numpoints)

    t_span, sol = model.solve_ode(initial_conditions, tTrain)

    if time_limit is None:
        time_limit = [0, tend]

    mask = (t_span == t_span[0]) | ((t_span >= time_limit[0]) & (t_span <= time_limit[1]))
    # mask = ((t_span >= time_limit[0]) & (t_span <= time_limit[1]))
    tTrain = t_span[mask]
    sol = sol[mask]

    noise = np.random.normal(0, noise_level, sol.shape)
    sol_noisy = sol.copy()
    for col in range(sol_noisy.shape[1]):
        sol_noisy[1:, col] += noise[1:, col]

    if show_figure:
        for i in range(sol_noisy.shape[1]):
            plt.scatter(tTrain, sol_noisy[:,i], label=f'Cell Population {i+1}')
            plt.legend()
        plt.title('Training Points')
        plt.xlabel('t')
        plt.ylabel('Population')
        plt.show()

    return tTrain, sol_noisy


def model_comparison(true_model, predicted_model):
    t_span = np.arange(0, true_model.tend, 0.01)
    _, true_sol = true_model.solve_ode(true_model.initial_conditions, t_span)
    _, predicted_sol = predicted_model.solve_ode(predicted_model.initial_conditions, t_span)

    fig, ax = plt.subplots()
    
    # Check if the solution includes both u and v
    if true_sol.shape[1] == 2:
        # Plot u and v for both true and predicted solutions
        ax.plot(t_span, true_sol[:, 0], label='True u', linestyle='-', color='blue', linewidth=2)
        ax.plot(t_span, true_sol[:, 1], label='True v', linestyle='-', color='green', linewidth=2)
        ax.plot(t_span, predicted_sol[:, 0], label='Predicted u', linestyle='--', color='blue',linewidth=2)
        ax.plot(t_span, predicted_sol[:, 1], label='Predicted v', linestyle='--', color='green', linewidth=2)
    else:
        # Plot only u for both true and predicted solutions
        ax.plot(t_span, true_sol[:, 0], label='True Solution', linestyle='-', linewidth=2)
        ax.plot(t_span, predicted_sol[:, 0], label='Predicted Solution', linestyle='--', linewidth=2)

    ax.set_title('Model Comparison')
    ax.set_xlabel('Time')
    ax.set_ylabel('Population')
    ax.legend()

    return fig