import os, sys, time
import numpy as np
import random
from sciann.utils.math import diff
from sciann import SciModel, Functional, Parameter
from sciann import Data, Tie
from sciann import Variable, Field

import matplotlib.pyplot as plt
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

parser.add_argument('-in', '--independent_networks', help='Use independent networks for each var (default True)', type=bool, nargs=1, default=[True])
parser.add_argument('-v', '--verbose', help='Show training progress (default 2) (check Keras.fit)', type=int, nargs=1, default=[2])

# model parameters
parser.add_argument('-ic', '--initial_conditions', help='Initial conditions for the model (default 0.01)', type=float, default=0.01)
parser.add_argument('--tend', help='End time for the model simulation (default 24)', type=int, default=24)

# arguments for training data generator
parser.add_argument('--sparse', help='Sparsity of training data (default True)', type=bool, nargs=1, default=[True])
parser.add_argument('-tl', '--time_limit', help='Time window for the training data (default [10, 30])', type=int, nargs=2, default=[10, 30])
parser.add_argument('-nl', '--noise_level', help='Level of noise in training data (default 0.05)', type=float, default=0.005)
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
        "SGModel_"
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
    f"Model Type: Saturated Growth" 
)

def train_sg_model():

    # NN Setup
    t = Variable("t", dtype=args.dtype[0])              # input
    u = Functional("u", t, args.layers, args.actf[0])   # output
    C = Parameter(0.5, inputs=t, name="C")              # learnable param
    u_t = diff(u,t)
    d1 = Data(u)                                        # constraints
    c1 = Tie(u_t, u*(C-u))

    # Define the optimization model (set of inputs and constraints)
    model = SciModel(
        inputs=[t],
        targets=[d1, c1],
        loss_func="mse"
    )
    with open("{}_summary".format(fname), "w") as fobj:
        model.summary(print_fn=lambda x: fobj.write(x + '\n'))

    # prepare training data
    sg_model = SaturatedGrowthModel(C=1.0, initial_conditions=args.initial_conditions, tend=args.tend)
    np.savetxt(fname+"_C_true",[1.0], delimiter=', ')
    np.random.seed(0)
    plot_save_file = "training_data_plot.png"
    plot_save_path = os.path.join(output_folder, plot_save_file)
    input_data, data_constraints = sg_model.generate_training_data(
                                                                numpoints=args.numx[0], 
                                                                sparse=args.sparse[0], 
                                                                time_limit=args.time_limit, 
                                                                noise_level=args.noise_level, 
                                                                show_figure=args.show_figure[0],
                                                                save_path=plot_save_path if args.show_figure[0] else None
                                                                )
    
    data_d1 = data_constraints
    data_c1 = 'zeros'
    target_data = [data_d1, data_c1]

    training_time = time.time()
    history = model.train(
        x_true=[input_data],
        y_true=target_data,
        epochs=args.epochs[0],
        batch_size=args.batchsize[0],
        shuffle=args.shuffle[0],
        learning_rate=args.learningrate[0],
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

    C_pred = C.eval(model, tspan)      # Learned param
    print(C_pred)
    sg_learned_model = SaturatedGrowthModel(C=C_pred, initial_conditions=args.initial_conditions, tend=args.tend)
    _, u_learned = sg_learned_model.solve_ode(initial_conditions=args.initial_conditions, t_span=tspan)

    u_pred = u.eval(model, tspan)   	# NN pred

    np.savetxt(fname+"_t", tspan, delimiter=', ')
    np.savetxt(fname+"_C_learned", C_pred, delimiter=', ')
    np.savetxt(fname+"_u_NN_pred", u_pred, delimiter=', ')
    # np.savetxt(fname+"_u_learned_pred", u_learned, delimiter=', ')
    combined_data = np.column_stack((tspan, u_pred))
    np.savetxt(fname+"_t_u_pred.csv", combined_data, delimiter=', ', header='t, u_pred', comments='')


    # print(u_learned.shape)
    # print(u_pred.shape)

def plot():
    # Loss plots
    total_loss = np.loadtxt(fname+"_loss")
    u_loss = np.loadtxt(fname+"_u_loss")
    ode_loss = np.loadtxt(fname+"_sub_2_loss")
    time = np.loadtxt(fname+"_time")

    fig, ax = plt.subplots(1, 3, figsize=(9, 4), dpi=300)
    
    cust_semilogx(ax[0], None, total_loss/total_loss[0], "epochs", "L/L0", label="Total_Loss")
    cust_semilogx(ax[0], None, u_loss/u_loss[0],xlabel=None, ylabel=None, label="U_Loss")
    cust_semilogx(ax[0], None, ode_loss/ode_loss[0],xlabel=None, ylabel=None, label="ODE_Loss")
    ax[0].legend(fontsize='small')

    cust_semilogx(ax[1], None, total_loss,  "epochs", "L", label="Total_Loss")
    cust_semilogx(ax[1], None, u_loss, xlabel=None, ylabel=None, label="U_Loss")
    cust_semilogx(ax[1], None, ode_loss, xlabel=None, ylabel=None, label="ODE_Loss")
    ax[1].legend(fontsize='small')

    ax[2].axis('off')  # Turn off axis
    ax[2].text(0.5, 0.5, network_description, ha='center', va='center', fontsize='small')

    fig.suptitle("Loss")
    plt.subplots_adjust(wspace=0.35)
    plt.savefig("{}_loss.png".format(output_file_name))

    # Model plot
    C_true = np.loadtxt(fname+"_C_true")
    C_learned = np.loadtxt(fname+"_C_learned")
    true_sg_model = SaturatedGrowthModel(C_true, args.initial_conditions, args.tend)
    learned_sg_model= SaturatedGrowthModel(C_learned, args.initial_conditions, args.tend)

    fig, ax = plt.subplots(1, 2, figsize=(7, 4), dpi=300)
    tspan = np.linspace(0, args.tend, args.numx[0])
    nn_model = np.loadtxt(fname+"_t_u_pred.csv", delimiter=', ', skiprows=1)
    model_comparison(true_sg_model, learned_sg_model, nn_model, tspan, ax[0])

    ax[1].axis('off')  # Turn off axis
    ax[1].text(0.5, 0.5, network_description, ha='center', va='center', fontsize='small')
    plt.subplots_adjust(wspace=0.1)
    plt.savefig("{}_Sol.png".format(output_file_name))

if __name__=="__main__":
    if args.plot==False:
         train_sg_model()
         plot()
    else:
         plot()
    

