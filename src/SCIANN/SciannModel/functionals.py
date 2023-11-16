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


# Analytical sol'n
###################################################################################
# model 1: Saturated Growth Model
def saturated_growth(u, t, C):
    return u * (C - u)

def saturated_growth_ode_solver(t, initial_cond, params):
    initu = initial_cond
    C = params
    res = odeint(saturated_growth, initu, t, args=(C,))
    return res

def saturated_growth_solveDE(initu, C, tend):
    initial_cond = [initu]
    params = [C]
    tspan = np.arange(0, tend, 0.1)
    sol = saturated_growth_ode_solver(tspan, initial_cond, params)
    u = sol[:, 0]
    plt.plot(tspan, u)
    plt.show()

####################################################################################
# Model 2: Competition Model
def competition_model(q, t, r, a1, a2, b1, b2, e0,f0, e1,f1, e2,f2, e3,f3, e4,f4):

    u, v = q

    dudt = u*(1-a1*u- a2*v)  +e0 + e1*u + e2*v + e3*u*u*v + e4*u*v*v
    dvdt = r*v*(1-b1*u-b2*v) +f0 + f1*u + f2*v + f3*u*u*v + f4*u*v*v
    
    return [dudt, dvdt]

def competition_model_ode_solver(t, initial_cond, params):
    initu, initv = initial_cond
    #r, a1, a2, b1, b2 = params
    r, a1, a2, b1, b2, e0,f0, e1,f1, e2,f2, e3,f3, e4,f4 = params
    #res = odeint(ode_model, [initu, initv], t, args=(r, a1, a2, b1, b2))
    res = odeint(competition_model, [initu, initv], t, args=(r, a1, a2, b1, b2, e0,f0, e1,f1, e2,f2, e3,f3, e4,f4)) #odeint from scipy.integrate
    return res

def competition_model_solveDGL(initu, initv, r, a1, a2, b1, b2, e0,f0, e1,f1, e2,f2, e3,f3, e4,f4, tend):
    initial_conditions = [initu, initv]
    #params = [r, a1, a2, b1, b2]
    params = [r, a1, a2, b1, b2, e0,f0, e1,f1, e2,f2, e3,f3, e4,f4]
    tspan = np.arange(0, tend, 0.1)
    sol = competition_model_ode_solver(tspan, initial_conditions, params)
    u,v = sol[:, 0], sol[:, 1]
    
    plt.plot(tspan, u)
    plt.plot(tspan, v)
    plt.show()

###################################################################################
# Create Trining Data
'''
    create_train_data(args)------> create custom training data
    inputs:
    tend: end time
    ODE_solver: set the solver (i.e. saturated_growth_ode_solver, competition_model_ode_solver)
    initial_cond: [initu, initv] or [initu]
    model_name: "saturated_growth" or "competition_model"
    numpoints: set custom #of training data point. Default:500
    sparse: Set "True" for sparse data selection. Default:False
    time_limit: set time window (i.e. [a,b])
    noise_level: set noise as required. Default:0.0
    outputs:
    training_input: tTrain
    training_constraints: uTrain / [uTrain, vTrain]
'''
def create_train_data(tend, 
                      ODE_solver, 
                      params, 
                      initial_cond, 
                      model_name=None,  
                      numpoints = 500, 
                      sparse=False, 
                      time_limit=None, 
                      noise_level=0.0):
    
    if model_name is None:
        print("Please insert a model name.")
        return
    
    if sparse==False:
        tTrain = np.linspace(0, tend, numpoints)
    else:
        tTrain = np.sort(np.random.uniform(0, tend, numpoints))

    # solve the problem
    sol = ODE_solver(tTrain, initial_cond=initial_cond, params=params)

    if time_limit is None:
        time_limit = [0, tend]
    
    # select the time window including the IC
    mask = (tTrain==tTrain[0]) | ((tTrain >= time_limit[0]) & (tTrain <= time_limit[1]))
    tTrain = tTrain[mask]

    # add noise(optional)
    noise = np.random.normal(0, noise_level, tTrain.shape)


    # plot and return training data
    if model_name == "saturated_growth":
        uTrain = sol[:,0][mask] + noise
        plt.scatter(tTrain, uTrain, label='Cell Population')
        plt.legend()
        plt.title('Training Points')
        plt.xlabel('t')
        plt.ylabel('# of Cell')
        plt.show()
        return tTrain, uTrain
    
    elif model_name=="competition_model":
        uTrain = sol[:,0][mask] + noise
        vTrain = sol[:,1][mask] + noise
        plt.scatter(tTrain, uTrain, label='Cell Population 1')
        plt.scatter(tTrain, vTrain, label='Cell Population 2')
        plt.legend()
        plt.title('Training Points')
        plt.xlabel('t')
        plt.ylabel('# of Cell')
        plt.show()
        return tTrain, [uTrain, vTrain]
    else:
        print(f"Unknown model name: {model_name}. Please insert a valid model name.")
        return
    
################################################################################
# Create SciANN model
'''
    create_NN_model(args)----> creates NN model
    network_architechture: set your custom network architecture.
    activation: set your custom activatin function.
    model_name: set your choosen model name(i.e. saturated_growth, competition_model)
    higher_order: set higher_order polynomial for "competition model". Defaults:False
'''
def create_NN_model(network_architechture=3*[5], 
                    activation='tanh', 
                    model_name=None, 
                    higher_order=False):

    if model_name is None:
        print("Please insert a model name.")
        return
    
    if model_name=="saturated_growth":
        # Model Input
        t = sn.Variable("t", dtype='float64')
        # NN model architecture
        u = sn.Functional("u", t, network_architechture, activation)
        # initialize the target parameters
        C = sn.Parameter(0.5, inputs=t, name="C")
        learnable_parameter = [C]
        # set up PINN-model
        u_t = sn.diff(u,t)
        # Define data constraints
        d1 = sn.Data(u)
        # define model constraints
        c1 = sn.Tie(u_t, u*(C-u))
        # set the model
        model = sn.SciModel(t, [d1, c1])
        return model, learnable_parameter
    
    elif model_name=="competition_model":
        # Model Input
        t = sn.Variable("t", dtype='float64')
        # NN model architecture
        u = sn.Functional("u", t, 3*[5], 'tanh')
        v = sn.Functional("v", t, 3*[5], 'tanh')
        # set up derivatives
        u_t = sn.diff(u,t)
        v_t = sn.diff(v,t)
        # data constraints
        d1 = sn.Data(u)
        d2 = sn.Data(v)

        # initialize the target parameters
        a1 = sn.Parameter(0.5, inputs=t, name="a1") #a1 = 0.3 or 0.7
        a2 = sn.Parameter(0.5, inputs=t, name="a2") #a2 = 0.6 or 0.3
        b1 = sn.Parameter(0.5, inputs=t, name="b1" ) #b1 = 0.7 or 0.3
        b2 = sn.Parameter(0.5, inputs = t, name = "b2") #b2 = 0.3 or 0.6
        r = sn.Parameter(0.5, inputs=t, name="r" ) #r = 0.5

        learnable_parameter = [r, a1, a2, b1, b2]

        if higher_order==False:
            # model constraints
            c1 = sn.Tie(u_t, u*(1-a1*u-a2*v))
            c2 = sn.Tie(v_t, r*v*(1-b1*u-b2*v))
            # set the model
            model = sn.SciModel(t, [d1, d2, c1, c2])
            return model, learnable_parameter
        else:
            # add higher order target parameters
            e0 = sn.Parameter(0.5, inputs=t, name="e0")
            e1 = sn.Parameter(0.5, inputs=t, name="e1")
            e2 = sn.Parameter(0.5, inputs=t, name="e2")
            e3 = sn.Parameter(0.5, inputs=t, name="e3")
            e4 = sn.Parameter(0.5, inputs=t, name="e4")

            f0 = sn.Parameter(0.5, inputs=t, name="f0")
            f1 = sn.Parameter(0.5, inputs=t, name="f1")
            f2 = sn.Parameter(0.5, inputs=t, name="f2")
            f3 = sn.Parameter(0.5, inputs=t, name="f3")
            f4 = sn.Parameter(0.5, inputs=t, name="f4")

            learnable_parameter.extend([e0, e1, e2, e3, e4, f0, f1, f2, f3, f4])

            # model constraints
            c1 = sn.Tie(u_t, u*(1-a1*u-a2*v)+e0 +e1*u+e2*v+ e3*u*u*v +e4*u*v*v)
            c2 = sn.Tie(v_t, r*v*(1-b1*u-b2*v)+f0 +f1*u+f2*v + f3*u*u*v +f4*u*v*v)

            # set the model
            model = sn.SciModel(t, [d1, d2, c1, c2])
            return model, learnable_parameter
    else:
        print(f"Unknown model name: {model_name}. Please insert a valid model name.")
        return

################################################################################
# Train the model
def train_model(model, 
                model_input, 
                learnable_parameter,
                data_constraints, 
                model_name=None, 
                epochs=500,
                batch_size=25,
                shuffle=True,
                learning_rate=0.001,
                reduce_lr_after = 100,
                stop_loss_value=1e-8,
                verbose=0):
    
    if model_name is None:
        print("Please insert a model name.")
        return
    
    if model_name=="saturated_growth":
        input_data = [model_input]
        data_d1 = data_constraints 
        data_c1 = 'zeros'   # ode constraints
        target_data = [data_d1, data_c1]

        # Train
        history = model.train(x_true=input_data,
                            y_true=target_data,
                            epochs=epochs,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            learning_rate=learning_rate,
                            reduce_lr_after=reduce_lr_after,
                            stop_loss_value=stop_loss_value,
                            verbose=verbose)
        return history, learnable_parameter
        
    elif model_name=="competition_model":
        input_data=[model_input]
        data_d1, data_d2 = [data_constraints]
        data_c1 = 'zeros'
        data_c2 = 'zeros'
        target_data = [data_d1, data_d2, data_c1, data_c2]

        ## Train
        history = model.train(x_true=input_data,
                            y_true=target_data,
                            epochs=epochs,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            learning_rate=learning_rate,
                            reduce_lr_after=reduce_lr_after,
                            stop_loss_value=stop_loss_value,
                            verbose=verbose)
        return history, learnable_parameter
    else:
        print(f"Unknown model name: {model_name}. Please insert a valid model name.")
        return

'''
let's see some output:
step1:
def create_train_data(tend, 
                      ODE_solver, 
                      params, 
                      initial_cond, 
                      model_name=None,  
                      numpoints = 500, 
                      sparse=False, 
                      time_limit=None, 
                      noise_level=0.0)
out: model_input, data_constrains
step2:
def create_NN_model(network_architechture=3*[5], 
                    activation='tanh', 
                    model_name=None, 
                    higher_order=False):
out: model, learnable_parameter
step3:
def train_model(model, 
                model_input, 
                data_constraints, 
                learnable_parameter,
                model_name=None, 
                epochs=500,
                batch_size=25,
                shuffle=True,
                learning_rate=0.001,
                reduce_lr_after = 100,
                stop_loss_value=1e-8,
                verbose=0):
out: history, learnable_parameter
'''
