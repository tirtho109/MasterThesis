import sys
import os

module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)

import jax
import numpy as np
import argparse
import jax.numpy as jnp
from fbpinns.domains import RectangularDomainND
from problems import CompetitionModel
from fbpinns.decompositions import RectangularDecompositionND
from fbpinns.networks import FCN
from fbpinns.constants import Constants, get_subdomain_ws
from fbpinns.trainers import FBPINNTrainer, PINNTrainer
from fbpinns.analysis import load_model, FBPINN_solution, PINN_solution
import matplotlib.pyplot as plt
from plot import plot_model_comparison, get_us, export_mse_mae
import pandas as pd

# Input interface for python. 
parser = argparse.ArgumentParser(description='''
        FBPINNs code for Separating longtime behavior and learning of mechanics  \n
        Saturated Growth Model'''
)

parser.add_argument('-ic', '--initial_conditions', help='Initial conditions(u0,v0) for the model (default [2,1])', type=float, default=[2,1])
parser.add_argument('--model_type', help='Survival or co-existence model (default survival**)', type=str, nargs=1, default=['survival'])
parser.add_argument('--tend', help='End time for the model simulation (default 24)', type=int, default=24)
parser.add_argument('-nx','--numx', help='Number of data points (default 50)', type=int, default=50)
parser.add_argument('-tl', '--time_limit', help='Time window for the training data (default [0, 24])', type=int, nargs=2, default=[0, 24])

parser.add_argument('-nsub','--num_subdomain', help='Number of sub-domains (default 15)', type=int, default=15)
parser.add_argument('-wo','--window_overlap', help='Window overlap for each subdomain (default 1.9)', type=float, default=1.9)
parser.add_argument('--unnorm_mean', help='Mean for unnormalization (default 0.)', type=float, default=0.)
parser.add_argument('--unnorm_sd', help='Standard deviation for unnormalization (default 1.)', type=float, default=1.)

parser.add_argument('-l', '--layers', help='Num layers and neurons (default 2 layers [1, 32, 2])', type=int, nargs='+', default=[1, 32, 2])
parser.add_argument('-pl', '--pinns_layers', help='Num of pinns layers and neurons (default 3 layers [1, 5, 5, 5, 2])', type=int, nargs='+', default=[1, 5, 5, 5, 2])

parser.add_argument('-nc','--num_collocation', help='Number of collocation points (default 2000)', type=int, default=2000)
parser.add_argument('-nt','--num_test', help='Number of test points (default 200)', type=int, default=200)
parser.add_argument('-e','--epochs', help='Number of epochs (default 50000)', type=int, default=50000)

parser.add_argument('--rootdir', type=str, default='CompModels', help='Root directory for saving models and summaries (default: "CompModels")')
parser.add_argument('--tag', type=str, default='ablation', help='Tag for identifying the run (default: "ablation")')
parser.add_argument('--sparse', help='Sparsity of training data (default True)', type=bool, nargs=1, default=[False])
parser.add_argument('-pt','--pinn_trainer', help='Whether to train PINN trainer (default Faöse)', type=bool, nargs=1, default=[False])

args = parser.parse_args()

def train_comp_model():

    module_path = os.path.abspath(os.path.join('..'))
    if module_path not in sys.path:
        sys.path.append(module_path)
        
    # step 1: Define Domain
    domain = RectangularDomainND
    domain_init_kwargs = dict(
        xmin = np.array([0.,]),
        xmax = np.array([args.tend,], dtype=float)
    )

    # step 2: Define problem ode to solve
    problem = CompetitionModel
    if args.model_type[0] == "survival":
        params = [0.5, 0.3, 0.6, 0.7, 0.3]
    elif args.model_type[0] == "coexistence":
        params = [0.5, 0.7, 0.3, 0.3, 0.6]
    else:
        raise ValueError(f"Invalid model type: '{args.model_type}'. Expected 'survival' or 'coexistence'.")
    problem_init_kwargs = dict(
        params=params, 
        u0=args.initial_conditions[0], 
        v0 = args.initial_conditions[1], 
        sd=0.1, 
        time_limit=args.time_limit, 
        numx=args.numx,
    )

    # step 3: Define domain decomposition used by the FBPINNs
    decomposition = RectangularDecompositionND

    subdomain_xs = [np.linspace(0,args.tend,args.num_subdomain)]
    decomposition_init_kwargs = dict(
        subdomain_xs = subdomain_xs,
        subdomain_ws = [get_subdomain_ws(subdomain_xs, args.window_overlap)],
        unnorm = (args.unnorm_mean, args.unnorm_sd),
    )

    # step 4: define the NN placed in each subdomain
    network = FCN
    network_init_kwargs = dict(
        layer_size = args.layers,
    )

    # step 5: Create a constants object

    class ModifiedConstants(Constants):
        @property
        def summary_out_dir(self):
            return f"{args.rootdir}/summaries/{self.run}/"
        @property
        def model_out_dir(self):
            return f"{args.rootdir}/models/{self.run}/"
        
    tag = args.tag
    h = len(args.layers) - 2  # Number of hidden layers
    p = sum(args.layers[1:-1])  # Sum of neurons in hidden layers
    n = (args.num_collocation,) # number of training point(collocation)
    run = f"FBPINN_{tag}_{problem.__name__}_{args.model_type[0]}_{network.__name__}_{args.num_subdomain}-ns_{args.window_overlap}-overlap_{h}-layers_{p}-hidden_{n[0]}-nCol_"
    run += f"{args.epochs}-e_{args.numx}-nData_{args.time_limit}-timeLimit_{args.num_test}-nTest_"

    c = ModifiedConstants(
        run=run,
        domain=domain,
        domain_init_kwargs=domain_init_kwargs,
        problem=problem,
        problem_init_kwargs=problem_init_kwargs,
        decomposition=decomposition,
        decomposition_init_kwargs=decomposition_init_kwargs,
        network=network,
        network_init_kwargs=network_init_kwargs,
        ns=(n,),# use 200 collocation points for training
        n_test=(args.num_test,),# use 500 points for testing
        n_steps=args.epochs,# number of training steps
        clear_output=True,
        save_figures=True,
        show_figures=False,
    )

    # Train the FBPINNs usnig FBPINNTrainer
    FBPINNrun = FBPINNTrainer(c)
    FBPINNrun.train()

    # import model
    c_out, model = load_model(run, rootdir=args.rootdir+"/")

    # plots
    # 1. model comparisoin
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    plot_model_comparison(c_out, model, type="FBPINN", ax=ax)
    file_path = os.path.join(c.summary_out_dir, "model_comparison.png")
    plt.savefig(file_path)

    # Mse & Mae
    u_exact, u_test, u_learned = get_us(c_out, model, type="FBPINN")
    file_path = os.path.join(c.summary_out_dir, "metrices.csv")
    export_mse_mae(u_exact, u_test, u_learned, file_path)

    # plots
    # 2. N-l1 test loss vs training steps
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    i,t,l1n = model[-1][:,0], model[-1][:,3], model[-1][:,-1]
    ax.plot(i, l1n, label=f"FBPINN {h} l {p} hu {args.num_subdomain} ns")
    ax.set_yscale('log')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Normalized l1 test loss')
    ax.set_title('Loss vs Training Steps')
    ax.legend()
    file_path = os.path.join(c.summary_out_dir, "normalized_loss.png")
    plt.savefig(file_path)

    if args.pinn_trainer[0]:
        # PINN trainer
        h = len(args.pinns_layers) - 2  # Number of hidden layers
        p = sum(args.pinns_layers[1:-1])  # Sum of neurons in hidden layers
        run = f"PINN_{tag}_{problem.__name__}_{args.model_type[0]}_{network.__name__}_{h}-layers_{p}-hidden_{n[0]}-nCol_"
        run += f"{args.epochs}-e_{args.numx}-nData_{args.time_limit}-timeLimit_{args.num_test}-nTest_"
        c["network_init_kwargs"] = dict(layer_sizes=args.pinns_layers)# use a larger neural network
        c["run"] = run
        PINNrun = PINNTrainer(c)
        PINNrun.train()

        # import model
        c_out, model = load_model(run, rootdir=args.rootdir+"/")

        # plot1: Model Comparison
        fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
        plot_model_comparison(c_out, model, type="PINN", ax=ax)
        file_path = os.path.join(c.summary_out_dir, "model_comparison.png")
        plt.savefig(file_path)
        
        # Mse & Mae
        u_exact, u_test, u_learned = get_us(c_out, model, type="PINN")
        file_path = os.path.join(c.summary_out_dir, "metrices.csv")
        export_mse_mae(u_exact, u_test, u_learned, file_path)

        # plots
        # 2. N-l1 test loss vs training steps
        fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
        i,t,l1n = model[-1][:,0], model[-1][:,3], model[-1][:,-1]
        ax.plot(i, l1n, label=f"PINN {h} l {p} hu")
        ax.set_yscale('log')
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Normalized l1 test loss')
        ax.set_title('Loss vs Training Steps')
        ax.legend()
        file_path = os.path.join(c.summary_out_dir, "normalized_loss.png")
        plt.savefig(file_path)


if __name__=="__main__":
    train_comp_model()

        