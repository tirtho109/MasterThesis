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
    def __init__(self, C):
        super().__init__()
        self.C = C

    def _model(self, u, t):
        return u*(self.C -u)
    
    def plot_solution(self, initial_conditions, t_end):
        t_span = np.arange(0,t_end, 0.1)
        _, solution = self.solve_ode(initial_conditions=initial_conditions, t_span=t_span)
        u = solution[:, 0]
        plt.plot(t_span, u)
        plt.plot(t_span, u)
        plt.title("Saturated Growth Model")
        plt.xlabel("Time")
        plt.ylabel("Population")
        plt.show()

class CompetitionModel(BaseModel):
    def __init__(self, params):
        super().__init__()
        self.params = params

    def _model(self, q, t):
        u, v = q
        r, a1, a2, b1,b2, e0, f0, e1, f1, e2, f2, e3, f3, e4,f4 = self.params
        dudt = u*(1-a1*u- a2*v)  +e0 + e1*u + e2*v + e3*u*u*v + e4*u*v*v
        dvdt = r*v*(1-b1*u-b2*v) +f0 + f1*u + f2*v + f3*u*u*v + f4*u*v*v
        return [dudt, dvdt]
    
    def plot_solution(self, initial_conditions, t_end):
        # initu, initv = initial_conditions
        t_span = np.arange(0, t_end, 0.1)
        _, solution = self.solve_ode(initial_conditions=initial_conditions, t_span=t_span)
        u, v = solution[:, 0], solution[:, 1]
        plt.plot(t_span, u, label="Population 1")
        plt.plot(t_span, v, label="Population 2")
        plt.title("Competition Model")
        plt.xlabel("Time")
        plt.ylabel("Population")
        plt.legend()
        plt.show()

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
        raise ValueError("End time 'ten' must be greater than zero.")
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
    else:
        tTrain = np.linspace(0, tend, numpoints)

    t_span, sol = model.solve_ode(initial_conditions, tTrain)

    if time_limit is None:
        time_limit = [0, tend]

    mask = (t_span == t_span[0]) | ((t_span >= time_limit[0]) & (t_span <= time_limit[1]))
    tTrain = t_span[mask]
    sol = sol[mask]

    noise = np.random.normal(0, noise_level, sol.shape)
    sol_noisy = sol + noise

    if show_figure:
        for i in range(sol_noisy.shape[1]):
            plt.scatter(tTrain, sol_noisy[:,i], label=f'Cell Population {i+1}')
            plt.legend()
        plt.title('Training Points')
        plt.xlabel('t')
        plt.ylabel('# of Cell')
        plt.show()

    return tTrain, sol_noisy