import os
import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt
import argparse
from plot import plot_pareto, plot_models
from util import SaturatedGrowthModel, clear_dir, create_datasets, export_mse_mae
from pysindy.feature_library import GeneralizedLibrary, PolynomialLibrary, FourierLibrary, CustomLibrary

# Input interface for python. 
parser = argparse.ArgumentParser(description='''
        PySINDy code for Separating longtime behavior and learning of mechanics  \n
        Saturated Growth Model'''
)

# arguments for training data generator
parser.add_argument('-ic', '--initial_conditions', help='Initial conditions(u0) for the model (default [2,1])', type=float, default=0.01)
parser.add_argument('--tend', help='End time for the model simulation (default 24)', type=int, default=24)

# arguments for training data generator
parser.add_argument('-nx', '--numx', help='Num Node in X (default 100)', type=int, nargs=1, default=[100])
parser.add_argument('-nT', '--nTest', help='# of test point(default 200)', type=int, nargs=1, default=[200])
parser.add_argument('--sparse', help='Sparsity of training data (default False)', type=bool, nargs=1, default=[False])
parser.add_argument('-tl', '--time_limit', help='Time window for the training data (default [10, 24])', type=int, nargs=2, default=[10, 24])
parser.add_argument('-nl', '--noise_level', help='Level of noise in training data (default 0.005)', type=float, default=0.005)
parser.add_argument('-sf', '--show_figure', help='Show training data (default True)', type=bool, nargs=1, default=[True])
parser.add_argument('-nt','--num_threshold', help='Number of threshold(lambda) to be scanned (default 100)', type=int, default=100)
parser.add_argument('-mt','--max_threshold', help='Maximum threshold(lambda) to be scanned (default 1)', type=float, default=1)

parser.add_argument('-op', '--outputpath', help='Output path (default ./file_name)', type=str, nargs=1, default=['SGModels'])
parser.add_argument('-of', '--outputprefix', help='Output path (default res**)', type=str, nargs=1, default=['res'])

# Optimizer
parser.add_argument('-fl', '--feature_library', help='Feature Library to choose for SINDy algorithm (default custom_library**)', type=str, nargs=1, default=['custom'])
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

folder_name = (
    f"SGModel_"
    f"tl_{'-'.join(map(str, args.time_limit))}_"
    f"nl_{args.noise_level}_"
    f"nX_{args.numx[0]}_"
    f"nT_{args.nTest[0]}_"
    f"sp_{args.sparse[0]}_"
    f"fl_{args.feature_library[0]}_"
    f"opt_{args.optimizer[0]}_"
    f"mt_{args.max_threshold}_"
    f"nt_{args.num_threshold}_"
    f"th_{args.thresholder[0]}_"
    f"ic_{args.initial_conditions}_"
)
output_folder = os.path.join(args.outputpath[0], folder_name)

if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
if args.plot==False:    
    clear_dir(output_folder)

output_file_name = os.path.join(output_folder, args.outputprefix[0])


def run_sg_model():
    C = 1
    np.random.seed(0)
    plot_save_file = "training_data_plot.png"
    plot_save_path = os.path.join(output_folder, plot_save_file)
    x_train, t_train, x_test, t_test = create_datasets(model=SaturatedGrowthModel, 
                                                       tend=args.tend, 
                                                       numx=args.numx[0],
                                                       nTest=args.nTest[0],
                                                       initial_conditions=[args.initial_conditions], 
                                                       params=C, 
                                                       training_time_limit=args.time_limit, 
                                                       noise_level=args.noise_level, 
                                                       show_figure=args.show_figure,
                                                       save_path=plot_save_path if args.show_figure[0] else None,
                                                       sparse=args.sparse[0])

    threshold_scan = np.linspace(0, args.max_threshold, args.num_threshold)
    coefs = []
    feature_names = ["u"]
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
         custom_functions = [lambda u:u, 
                            lambda u:u**2]
         function_names = [
                            lambda u : f"{u}",
                            lambda u : f"{u}^2"
                         ]
         custom_library = CustomLibrary(library_functions=custom_functions, 
                                        function_names=function_names, 
                                        include_bias=False).fit(x_train)
         
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

    # model comparision

    original_eq = f"u' = u(u -{C}*u)"
    # extract model based on lambda_mse
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
                initial_condition=[args.initial_conditions],
                params=C,
                original_model=SaturatedGrowthModel,
                sindy_model=model_out, ax = ax1)
    
    sindy_eq = model_out.equations()

    ax2.axis('off') 

    lambda_mse_formatted = f"{lambda_mse:.3f}"
    equation_text = f"$\\lambda = {lambda_mse_formatted}$ \n"
    equation_text += f"\nSINDy Model Equation:\n{'' if len(sindy_eq) == 0 else sindy_eq[0]}\n{'' if len(sindy_eq) < 2 else sindy_eq[1]}"
    equation_text += f"\n\nOriginal Model Equation:\n{original_eq}"
    ax2.text(0.5, 0.5, equation_text, ha='center', va='center', fontsize=10)
    plt.savefig("{}_Model_Comparison.png".format(output_file_name))

    # export mse
    file_path = os.path.join(output_folder, "metrices.csv")
    export_mse_mae(model_out, x_test, t_test, file_path=file_path)
    print("done")

if __name__=="__main__":
    run_sg_model()