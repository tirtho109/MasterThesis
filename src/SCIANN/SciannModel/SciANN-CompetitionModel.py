import os, sys, time
import numpy as np
import random
from sciann.utils.math import diff
from sciann import SciModel, Functional, Parameter
from sciann import Data, Tie
from sciann import Variable, Field

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse
from model_ode_solver import *

# Input interface for python. 
parser = argparse.ArgumentParser(description='''
        SciANN code for Separating longtime behavior and learning of mechanics  \n
        Saturated Growth Model'''
)

parser.add_argument('-l', '--layers', help='Num layers and neurons (default 4 layers each 40 neurons [5, 5, 5])', type=int, nargs='+', default=[5]*3)
parser.add_argument('-af', '--actf', help='Activation function (default tanh)', type=str, nargs=1, default=['tanh'])
parser.add_argument('-nx', '--numx', help='Num Node in X (default 100)', type=int, nargs=1, default=[100])
parser.add_argument('-bs', '--batchsize', help='Batch size for Adam optimizer (default 25)', type=int, nargs=1, default=[25])
parser.add_argument('-e', '--epochs', help='Maximum number of epochs (default 2000)', type=int, nargs=1, default=[2000])
parser.add_argument('-lr', '--learningrate', help='Initial learning rate (default 0.001)', type=float, nargs=1, default=[0.001])
parser.add_argument('-rlr', '--reduce_learning_rate', help='Reduce learning rate (default 100)', type=int, nargs=1, default=[100])
parser.add_argument('-in', '--independent_networks', help='Use independent networks for each var (default True)', type=bool, nargs=1, default=[True])
parser.add_argument('-v', '--verbose', help='Show training progress (default 2) (check Keras.fit)', type=int, nargs=1, default=[2])

# model parameters
parser.add_argument('-ic', '--initial_conditions', help='Initial conditions(u0,v0) for the model (default [2,1])', type=float, default=[2,1])
parser.add_argument('--tend', help='End time for the model simulation (default 24)', type=int, default=24)
parser.add_argument('--model_type', help='Survival or co-existence model (default survival**)', type=str, nargs=1, default=['survival'])
# parser.add_argument('--higher_order', help='Higher order model parameter (default False)', type=bool, nargs=1, default=[False])

# arguments for training data generator
parser.add_argument('--sparse', help='Sparsity of training data (default True)', type=bool, nargs=1, default=[True])
parser.add_argument('-tl', '--time_limit', help='Time window for the training data (default [10, 30])', type=int, nargs=2, default=[10, 30])
parser.add_argument('-nl', '--noise_level', help='Level of noise in training data (default 0.005)', type=float, default=0.005)
parser.add_argument('-sf', '--show_figure', help='Show training data (default True)', type=bool, nargs=1, default=[True])

parser.add_argument('--shuffle', help='Shuffle data for training (default True)', type=bool, nargs=1, default=[True])
parser.add_argument('--stopafter', help='Patience argument from Keras (default 500)', type=int, nargs=1, default=[500])
parser.add_argument('--savefreq', help='Frequency to save weights (each n-epoch)', type=int, nargs=1, default=[100000])
parser.add_argument('--dtype', help='Data type for weights and biases (default float64)', type=str, nargs=1, default=['float64'])
parser.add_argument('--gpu', help='Use GPU if available (default False)', type=bool, nargs=1, default=[False])
parser.add_argument('-op', '--outputpath', help='Output path (default ./file_name)', type=str, nargs=1, default=['output'])
parser.add_argument('-of', '--outputprefix', help='Output path (default res**)', type=str, nargs=1, default=['res'])

parser.add_argument('--plot', help='Plot the model', nargs='?', default=False)

args = parser.parse_args()

if not args.gpu[0]:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Manage Files
if not os.path.isdir(args.outputpath[0]):
        os.mkdir(args.outputpath[0])

folder_name = (
    f"CompModel_{args.model_type[0]}_"
    f"actf_{args.actf[0]}_"
    f"layers_{'x'.join(map(str, args.layers))}_"
    f"numx_{args.numx[0]}_"
    f"bs_{args.batchsize[0]}_"
    f"epochs_{args.epochs[0]}_"
    f"lr_{args.learningrate[0]}_"
    f"tl_{'-'.join(map(str, args.time_limit))}_"
    f"nl_{args.noise_level}"
)

output_folder = os.path.join(args.outputpath[0], folder_name)

if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

output_file_name = os.path.join(output_folder, args.outputprefix[0])
fname = (
    f"{output_file_name}_"
    f"actf_{args.actf[0]}_"
    f"layers_{'x'.join(map(str, args.layers))}_"
    f"numx_{args.numx[0]}_"
    f"bs_{args.batchsize[0]}_"
    f"epochs_{args.epochs[0]}_"
    f"lr_{args.learningrate[0]}_"
    f"tl_{'-'.join(map(str, args.time_limit))}_"
    f"nl_{args.noise_level}"
)

#TODO
# Energy plots
def plot_energy(phi_values, u, v, ax=None, set_title=None):
    if ax is None:
        plt.figure(figsize=(4, 3))
        ax = plt.gca()
    
    contour = ax.contourf(u, v, phi_values, levels=50, cmap='jet')
    plt.colorbar(contour, ax=ax)
    
    if set_title is None:
        ax.set_title("Energy Landscape (Lyapunov Function)")
    else:
        ax.set_title(set_title)
    
    ax.set_xlabel("$x_1$ (Virus)")
    ax.set_ylabel("$x_2$ (Immune Cells)")

    if ax is None:
        plt.show()

# Calculate Lyapunov Function
"""
    I choose, A11 = b1*b2/u , A22 = a1*a2/v, A12 = 0, A21 = 0

"""
def phi_comp(u, v, params):
     r, a1, a2, b1, b2 = params
     #TODO Check later (sign)
     return (- b1 * b2 * u + 0.5 * b1 * b2 * a1 * u**2 + b1 * b2 * a2 * v * u -
            a1 * a2 * r * v + a1 * a2 * r * b1 * u * v + 0.5 * a1 * a2 * r * b2 * v**2)


network_description = (
     f"Layers: {'x'.join(map(str, args.layers))}\n"
     f"Activation Function: {args.actf[0]}\n"
     f"Num Node in X: {args.numx[0]}\n"
     f"Batch Size: {args.batchsize[0]}\n"
     f"Epochs: {args.epochs[0]}\n"
     f"Learning Rate: {args.learningrate[0]}\n"
     f"Time Window: {'-'.join(map(str, args.time_limit))}\n"
     f"Independent Networks: {args.independent_networks[0]}\n"
     f"Noise Level: {args.noise_level}\n"
     f"Sparse: {args.sparse[0]}\n"
     f"Model Type: {args.model_type[0]}"
)

def train_comp_model():

    # NN Setup
    t = Variable("t", dtype=args.dtype[0])              # input

    if args.independent_networks:  
        u = Functional("u", t, args.layers, args.actf[0])   # output
        v = Functional("v", t, args.layers, args.actf[0])
    else:
         u, v = Functional(
                        ["u", "v"], t, 
                        args.layers, 
                        args.actf[0]).split()
    
    r = Parameter(0.5, inputs=t, name="r" )                 # Learnable parameters
    a1 = Parameter(0.5, inputs=t, name="a1")
    a2 = Parameter(0.5, inputs=t, name="a2") 
    b1 = Parameter(0.5, inputs=t, name="b1" ) 
    b2 = Parameter(0.5, inputs = t, name = "b2") 
    

    # if args.higher_order:
    #       e0 = Parameter(0.5, inputs=t, name='e0')
    #       e1 = Parameter(0.5, inputs=t, name='e1')
    #       e2 = Parameter(0.5, inputs=t, name='e2')
    #       e3 = Parameter(0.5, inputs=t, name='e3')
    #       e4 = Parameter(0.5, inputs=t, name='e4')

    #       f0 = Parameter(0.5, inputs=t, name='f0')
    #       f1 = Parameter(0.5, inputs=t, name='f1')
    #       f2 = Parameter(0.5, inputs=t, name='f2')
    #       f3 = Parameter(0.5, inputs=t, name='f3')
    #       f4 = Parameter(0.5, inputs=t, name='f4')

    u_t = diff(u,t)
    v_t = diff(v,t)

    d1 = Data(u)
    d2 = Data(v)

    # if args.higher_order:
    #     c1 = Tie(u_t, u*(1-a1*u-a2*v)+e0 +e1*u+e2*v+ e3*u*u*v +e4*u*v*v)
    #     c2 = Tie(v_t, r*v*(1-b1*u-b2*v)+f0 +f1*u+f2*v + f3*u*u*v +f4*u*v*v)
    # else:
    #     c1 = Tie(u_t, u*(1-a1*u-a2*v))
    #     c2 = Tie(v_t, r*v*(1-b1*u-b2*v))

    c1 = Tie(u_t, u*(1-a1*u-a2*v))
    c2 = Tie(v_t, r*v*(1-b1*u-b2*v))

    model = SciModel(
        inputs=[t],
        targets=[d1, d2, c1, c2],
        loss_func="mse"
    )
    with open("{}_summary".format(fname), "w") as fobj:
        model.summary(print_fn=lambda x: fobj.write(x + '\n'))

    # prepeare training data
    #TODO: What about e's and f's has some value
    if args.model_type[0]=="survival":
         # [r, a1, a2, b1, b2, e0, f0, e1, f1, e2, f2, e3, f3, e4, f4]
         comp_params = [0.5, 0.3, 0.6, 0.7, 0.3] # , 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    else:
         comp_params = [0.5, 0.7, 0.3, 0.3, 0.6] # , 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    comp_model = CompetitionModel(params=comp_params,
                                  initial_conditions=args.initial_conditions,
                                  tend=args.tend)
    np.savetxt(fname+"_params", comp_params, delimiter=', ')
    np.random.seed(0)
    plot_save_file = "training_data_plot.png"
    plot_save_path = os.path.join(output_folder, plot_save_file)
    input_data, data_constraints = comp_model.generate_training_data(
                                                                numpoints=args.numx[0], 
                                                                sparse=args.sparse[0], 
                                                                time_limit=args.time_limit, 
                                                                noise_level=args.noise_level, 
                                                                show_figure=args.show_figure[0],
                                                                save_path=plot_save_path if args.show_figure[0] else None
                                                                )
    data_d1, data_d2 = data_constraints[:, 0], data_constraints[:, 1]
    data_c1 = 'zeros'
    data_c2 = 'zeros'
    target_data = [data_d1, data_d2, data_c1, data_c2]

    training_time = time.time()
    history = model.train(
        x_true=[input_data],
        y_true=target_data,
        epochs=args.epochs[0],
        batch_size=args.batchsize[0],
        shuffle=args.shuffle[0],
        learning_rate=args.learningrate[0],
        reduce_lr_after=args.reduce_learning_rate[0],
        stop_after=args.stopafter[0],
        verbose=args.verbose[0]
    )
    training_time = time.time() - training_time

    weights_file_name = fname + "_weights.hdf5"

    # Save the model weights as an .hdf5 file
    model.save_weights(weights_file_name)

    for loss in history.history:
        np.savetxt(fname+"_{}".format("_".join(loss.split("/"))), 
                    np.array(history.history[loss]).reshape(-1, 1))
    
    time_steps = np.linspace(0, training_time, len(history.history["loss"]))
    np.savetxt(fname+"_Time", time_steps.reshape(-1,1))

    # Post-processing
    tspan = np.linspace(0, args.tend, args.numx[0])
    # Learned Params
    r_pred = r.eval(model, tspan)[0]
    a1_pred = a1.eval(model, tspan)[0]
    a2_pred = a2.eval(model, tspan)[0]
    b1_pred = b1.eval(model, tspan)[0]
    b2_pred = b2.eval(model, tspan)[0]
    # if args.higher_order:
    #      e0_pred = e0.eval(model, tspan)[0]
    #      e1_pred = e1.eval(model, tspan)[0]
    #      e2_pred = e2.eval(model, tspan)[0]
    #      e3_pred = e3.eval(model, tspan)[0]
    #      e4_pred = e4.eval(model, tspan)[0]
    #      f0_pred = f0.eval(model, tspan)[0]
    #      f1_pred = f1.eval(model, tspan)[0]
    #      f2_pred = f2.eval(model, tspan)[0]
    #      f3_pred = f3.eval(model, tspan)[0]
    #      f4_pred = f4.eval(model, tspan)[0]
    # print(r_pred, a1_pred, a2_pred, b1_pred, b2_pred)
    # print("r: {}".format(r.value))
    # print("a1: {}".format(a1.value))
    # print("a2: {}".format(a2.value))
    # print("b1: {}".format(b1.value))
    # print("b2: {}".format(b2.value))
    # print(args.model_type[0])
    # print(comp_params)
    learned_comp_params = [r_pred, a1_pred, a2_pred, b1_pred, b2_pred]
    comp_learned_model = CompetitionModel(params=learned_comp_params, initial_conditions=args.initial_conditions, tend=args.tend)
    _, sol = comp_learned_model.solve_ode(initial_conditions=args.initial_conditions, t_span=tspan)
    u_learned = sol[:, 0]           # using learned params
    v_learned = sol[:, 1]

    u_pred = u.eval(model, tspan)   # NN pred
    v_pred = v.eval(model, tspan)

    np.savetxt(fname+"_t", tspan, delimiter=', ')
    np.savetxt(fname+"_learned_params", learned_comp_params, delimiter=', ')

    combined_learned_data = np.column_stack((tspan, u_learned, v_learned))
    np.savetxt(fname+"_t_u_v_learned.csv", combined_learned_data, delimiter=', ', header='t, u_learned, v_learned', comments='')

    combined_pred_data = np.column_stack((tspan, u_pred, v_pred))
    np.savetxt(fname+"_t_u_v_pred.csv", combined_pred_data, delimiter=', ', header='t, u_pred, v_pred', comments='')

def plot():
     # Loss plots
    total_loss = np.loadtxt(fname+"_loss")
    u_loss = np.loadtxt(fname+"_u_loss")
    ode1_loss = np.loadtxt(fname+"_sub_2_loss")
    v_loss = np.loadtxt(fname+"_v_loss")
    ode2_loss = np.loadtxt(fname+"_sub_4_loss")
    time = np.loadtxt(fname+"_time")

    fig, ax = plt.subplots(1, 3, figsize=(9, 4), dpi=300)
    
    cust_semilogx(ax[0], None, total_loss/total_loss[0], "epochs", "L/L0", label="Total_Loss")
    cust_semilogx(ax[0], None, u_loss/u_loss[0],xlabel=None, ylabel=None, label="U_Loss")
    cust_semilogx(ax[0], None, ode1_loss/ode1_loss[0],xlabel=None, ylabel=None, label="ODE1_Loss")
    cust_semilogx(ax[0], None, v_loss/v_loss[0],xlabel=None, ylabel=None, label="V_Loss")
    cust_semilogx(ax[0], None, ode2_loss/ode2_loss[0],xlabel=None, ylabel=None, label="ODE2_Loss")
    ax[0].legend(fontsize='small')

    cust_semilogx(ax[1], None, total_loss,  "epochs", "L", label="Total_Loss")
    cust_semilogx(ax[1], None, u_loss, xlabel=None, ylabel=None, label="U_Loss")
    cust_semilogx(ax[1], None, ode1_loss, xlabel=None, ylabel=None, label="ODE1_Loss")
    cust_semilogx(ax[1], None, v_loss, xlabel=None, ylabel=None, label="V_Loss")
    cust_semilogx(ax[1], None, ode2_loss, xlabel=None, ylabel=None, label="ODE2_Loss")
    ax[1].legend(fontsize='small')
    

    ax[2].axis('off')  # Turn off axis
    ax[2].text(0.5, 0.5, network_description, ha='center', va='center', fontsize='small')

    fig.suptitle("Loss")
    plt.subplots_adjust(wspace=0.35)
    plt.savefig("{}_loss.png".format(output_file_name))



    # Model plot
    param_true = np.loadtxt(fname+"_params")
    param_learned = np.loadtxt(fname+"_learned_params")
    true_comp_model = CompetitionModel(param_true, args.initial_conditions, args.tend)
    learned_comp_model= CompetitionModel(param_learned, args.initial_conditions, args.tend)

    fig, ax = plt.subplots(1, 2, figsize=(7, 4), dpi=300)
    tspan = np.linspace(0, args.tend, args.numx[0])
    nn_model = np.loadtxt(fname+"_t_u_v_pred.csv", delimiter=', ', skiprows=1)
    model_comparison(true_comp_model, learned_comp_model, nn_model, tspan, ax[0])

    ax[1].axis('off')  # Turn off axis
    ax[1].text(0.5, 0.5, network_description, ha='center', va='center', fontsize='small')
    plt.subplots_adjust(wspace=0.1)
    plt.savefig("{}_Sol.png".format(output_file_name))

    # Energy plot
    fig = plt.figure(figsize=(12, 6), dpi=300)
    gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1]) 

    ax1 = fig.add_subplot(gs[0, 0])  
    ax2 = fig.add_subplot(gs[0, 1]) 

    # Generate and plot energy landscapes
    u_range = np.linspace(0, 2, 500)
    v_range = np.linspace(0, 2, 500)
    u, v = np.meshgrid(u_range, v_range)

    phi_comp_values_learned = phi_comp(u, v, param_learned)
    plot_energy(phi_values=phi_comp_values_learned, u=u, v=v, ax=ax1, set_title=args.model_type[0]+" Learned")

    phi_comp_value_true = phi_comp(u, v, param_true)
    plot_energy(phi_values=phi_comp_value_true, u=u, v=v, ax=ax2, set_title=args.model_type[0]+" True")

    param_learned_formatted = ['{:.2f}'.format(param) for param in param_learned]

    # table
    columns = ['r', 'a1', 'a2', 'b1', 'b2']
    rows = ['Learned Params', 'True Params']
    cell_text = [param_learned_formatted, param_true]

    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis('tight')
    ax3.axis('off')
    table = ax3.table(cellText=cell_text, rowLabels=rows, colLabels=columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.2)

    plt.savefig("{}_Energy.png".format(output_file_name))

if __name__=="__main__":
    if args.plot==False:
         train_comp_model()
         plot()
    else:
         plot()
