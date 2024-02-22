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
from FBPINNs.FBPINNsModel.problems import SaturatedGrowthModel
from fbpinns.decompositions import RectangularDecompositionND
from fbpinns.networks import FCN
from fbpinns.constants import Constants

# Input interface for python. 
parser = argparse.ArgumentParser(description='''
        FBPINNs code for Separating longtime behavior and learning of mechanics  \n
        Saturated Growth Model'''
)

parser.add_argument('-ic', '--initial_conditions', help='Initial conditions for the model (default 0.01)', type=float, default=0.01)
parser.add_argument('--tend', help='End time for the model simulation (default 24)', type=int, default=24)
parser.add_argument('-nx','--numx', help='Number of data points (default 50)', type=int, default=50)
parser.add_argument('-tl', '--time_limit', help='Time window for the training data (default [0, 24])', type=int, nargs=2, default=[0, 24])

parser.add_argument('-nsub','--num_subdomain', help='Number of sub-domains (default 15)', type=int, default=15)
parser.add_argument('-ww','--window_width', help='Window width for each subdomain (default 1.9)', type=float, default=1.9)
parser.add_argument('--unnorm_mean', help='Mean for unnormalization (default 0.)', type=float, default=0.)
parser.add_argument('--unnorm_sd', help='Standard deviation for unnormalization (default 1.)', type=float, default=1.)

parser.add_argument('-l', '--layers', help='Num layers and neurons (default 2 layers [1, 32, 1])', type=int, nargs='+', default=[1, 32, 1])

parser.add_argument('-nc','--num_collocation', help='Number of collocation points (default 2000)', type=int, default=2000)
parser.add_argument('-nt','--num_test', help='Number of test points (default 200)', type=int, default=200)
parser.add_argument('-e','--epochs', help='Number of epochs (default 50000)', type=int, default=50000)

from fbpinns.trainers import FBPINNTrainer

args = parser.parse_args()

def train_fbpinns():

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
    problem = SaturatedGrowthModel
    problem_init_kwargs = dict(
        C=1, u0=args.initial_conditions, sd=0.1, time_limit=args.time_limit, numx=args.numx,
    )

    # step 3: Define domain decomposition used by the FBPINNs
    decomposition = RectangularDecompositionND

    decomposition_init_kwargs = dict(
        subdomain_xs = [np.linspace(0,args.tend,args.num_subdomain)],
        subdomain_ws = [(args.window_width)*np.ones((args.num_subdomain,))],
        unnorm = (args.unnorm_mean, args.unnorm_sd),
    )

    # step 4: define the NN placed in each subdomain
    network = FCN
    network_init_kwargs = dict(
        layer_size = args.layers,
    )

    # step 5: Create a constants object

    c = Constants(
        domain=domain,
        domain_init_kwargs=domain_init_kwargs,
        problem=problem,
        problem_init_kwargs=problem_init_kwargs,
        decomposition=decomposition,
        decomposition_init_kwargs=decomposition_init_kwargs,
        network=network,
        network_init_kwargs=network_init_kwargs,
        ns=((args.num_collocation,),),# use 200 collocation points for training
        n_test=(args.num_test,),# use 500 points for testing
        n_steps=args.epochs,# number of training steps
        clear_output=True,
        save_figures=True,
        show_figures=False,
    )

    # Train the FBPINNs usnig FBPINNTrainer
    run = FBPINNTrainer(c)
    all_params = run.train()

if __name__=="__main__":
    train_fbpinns()