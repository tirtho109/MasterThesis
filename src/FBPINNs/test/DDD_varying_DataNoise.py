import sys
import os

module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import pandas as pd
import seaborn as sns
import argparse
from fbpinns.domains import RectangularDomainND
from FBPINNsModel.problems import CompetitionModel, SaturatedGrowthModel
from fbpinns.decompositions import RectangularDecompositionND
from fbpinns.networks import FCN
from fbpinns.constants import Constants
from fbpinns.trainers import FBPINNTrainer
from fbpinns.analysis import load_model
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from FBPINNsModel.plot import plot_model_comparison, get_us, export_mse_mae, export_parameters, get_x_batch
from FBPINNsModel.subdomain_helper import get_subdomain_xsws

parser = argparse.ArgumentParser(description='''
        FBPINN code for Separating longtime behavior and learning of mechanics  \n
        To find the best number of subdomain for different time limit cases'''
)

parser.add_argument('--train', help='To train the whole model (default False)', type=bool, nargs=1, default=[False])
parser.add_argument('--plot', help='Plot the model', nargs='?', default=False)

args = parser.parse_args()


def plot_DDD_varying_DataNoise():
    class ModifiedConstants(Constants):
        @property
        def summary_out_dir(self):
            return f"{rootdir}/summaries/{self.run}/"
        @property
        def model_out_dir(self):
            return f"{rootdir}/models/{self.run}/"

    numx = 100
    nCol = 200
    nTest = 500
    lambda_phy = 1e0
    lambda_data = 1e0
    sparse = True
    #noise_level = 0.005
    tbegin = 0
    tend = 24
    wo = 1.9
    nsub=2
    wi = 1.0005
    tag = "DDD"
    epochs = 50000
    sampler='grid'

    # Varying parameters
    # noise_levels = np.arange(0, 0.051, 0.005, dtype=float) 
    noise_levels = np.arange(0, 0.051, 0.01, dtype=float)
    # array([0.   , 0.005, 0.01 , 0.015, 0.02 , 0.025, 0.03 , 0.035, 0.04 ,
    #     0.045, 0.05 ])
    time_limits = [[0,10], [10,24], [0, 24]]

    # type of problem and other fix params
    layer1 = [1, 5, 5, 5, 1]
    layer2 = [1, 5, 5, 5, 2]
    layers = [layer1, layer2, layer2]
    problem_types = [SaturatedGrowthModel, CompetitionModel, CompetitionModel]
    params_type = [1,  [0.5, 0.7, 0.3, 0.3, 0.6], [0.5, 0.3, 0.6, 0.7, 0.3]] # sg, coexistence, survival
    cases = ["sg", "coexistence", "survival"]

    parentdir = "DDD_DataNoise"
    rootdirs = ["DDD_DataNoise_sg", "DDD_DataNoise_coexistence", "DDD_DataNoise_survival"]
    rootdirs_with_parent = [os.path.join(parentdir, rd) for rd in rootdirs]

    # collector dataframes
    collcted_df = pd.DataFrame(columns=['Time Limit', 'Noise Level', 'MSE Learned', 'MSE Test', 'Model Type', 'Learned Parameters'])

    for (layer, problem, params, rootdir, name) in zip(layers, problem_types, params_type, rootdirs_with_parent, cases):
    
        #step 1
        domain = RectangularDomainND
        domain_init_kwargs = dict(
            xmin = np.array([tbegin,]), 
            xmax = np.array([tend,])
            )
        #step 2
        problem_kwargs_set = []
        if problem.__name__=="CompetitionModel":
            for tl in time_limits:
                if not isinstance(tl, (list, tuple)) or len(tl) != 2:
                    raise ValueError(f"Invalid time_limit format: {tl}")
                for noise_level in noise_levels:
                    problem_init_kwargs = dict(
                        params=params, u0=2, v0=1, 
                        sd=0.1, time_limit=tl, 
                        numx=numx, lambda_phy=lambda_phy,
                        lambda_data=lambda_data,
                        sparse=[sparse], noise_level=noise_level,
                    )
                    problem_kwargs_set.append((problem_init_kwargs, tl))
        else:
            for tl in time_limits:
                if not isinstance(tl, (list, tuple)) or len(tl) != 2:
                    raise ValueError(f"Invalid time_limit format: {tl}")
                for noise_level in noise_levels:
                    problem_init_kwargs = dict(
                        C=params, u0=0.01, 
                        sd=0.1, time_limit=tl, 
                        numx=numx, lambda_phy=lambda_phy,
                        lambda_data=lambda_data,
                        sparse=[sparse], noise_level=noise_level,
                    )
                    problem_kwargs_set.append((problem_init_kwargs, tl))
        # step 3
        decomposition = RectangularDecompositionND
        decomposition_kwargs_set = []
        for tl in time_limits:
            subdomain_xs, subdomain_ws = get_subdomain_xsws(tl, tbegin, tend, nsub, wo, wi)
            decomposition_init_kwargs = dict(
                subdomain_xs = subdomain_xs,
                subdomain_ws = subdomain_ws,
                unnorm=(0.,1.),
            )
            decomposition_kwargs_set.append((decomposition_init_kwargs, tl))

        # step 4
        network = FCN# place a fully-connected network in each subdomain
        network_init_kwargs=dict(
            layer_sizes=layer,# with 2 hidden layers
        )

        h = len(layer) -2
        p = sum(layer[1:-1])
        n=(nCol,)

        runs = []
        for (problem_kwargs, tl) in problem_kwargs_set:
            for (decomposition_kwargs, tld) in decomposition_kwargs_set:
                if tl==tld:
                    run = f"FBPINN_{tag}_{name}_{nsub}_nsub_{problem_kwargs["noise_level"]}_noise_{tl}_tl"
                    runs.append(run)
                    c = ModifiedConstants(
                        run=run,
                        domain=domain,
                        domain_init_kwargs=domain_init_kwargs,
                        problem=problem,
                        problem_init_kwargs=problem_kwargs,
                        decomposition=decomposition,
                        decomposition_init_kwargs=decomposition_kwargs,
                        network=network,
                        network_init_kwargs=network_init_kwargs,
                        ns=((nCol,),),# use 200 collocation points for training
                        n_test=(nTest,),# use 500 points for testing
                        n_steps=epochs,# number of training steps
                        clear_output=True,
                        sampler=sampler,
                        show_figures=False,
                        save_figures=True,
                    )
                    if args.train[0]:
                        FBPINNrun = FBPINNTrainer(c)
                        FBPINNrun.train()
                        
                        # import model 
                        c_out, model = load_model(run, rootdir=rootdir+"/")

                        # plots
                        # 1. model comparisoin
                        fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
                        plot_model_comparison(c_out, model, type="FBPINN", ax=ax)
                        file_path = os.path.join(c.summary_out_dir, "model_comparison.png")
                        plt.savefig(file_path)

                        # Export params(true & learned)
                        file_path = os.path.join(c.summary_out_dir, "parameters.csv")
                        export_parameters(c, model, file_path)

                        # Mse & Mae
                        u_exact, u_test, u_learned = get_us(c_out, model, type="FBPINN")
                        file_path = os.path.join(c.summary_out_dir, "metrices.csv")
                        export_mse_mae(u_exact, u_test, u_learned, file_path)

                        # plots
                        # 2. N-l1 test loss vs training steps
                        fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
                        i,t,l1n = model[-1][:,0], model[-1][:,3], model[-1][:,-1]
                        ax.plot(i, l1n, label=f"FBPINN {h} l {p} hu {nsub} ns")
                        ax.set_yscale('log')
                        ax.set_xlabel('Training Steps')
                        ax.set_ylabel('Normalized l1 test loss')
                        ax.set_title('Loss vs Training Steps')
                        ax.legend()
                        file_path = os.path.join(c.summary_out_dir, "normalized_loss.png")
                        plt.savefig(file_path)

        # model comparison plot(whole)
        models_fig, models_ax = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
        for run in runs:
            c_out, model = load_model(run, rootdir=rootdir+"/")
            nl = run.split("_")[5]
            tl = c_out.problem_init_kwargs['time_limit']
            column_index = 0 if tl == [0,10] else 1 if tl == [10,24] else -1 
            if column_index != -1:
                u_exact, u_test, u_learned = get_us(c_out, model, type="FBPINN")
                x_batch = get_x_batch(c, model)
                labels  = ["u", "v"]    
                for i in range(0, u_exact.shape[1]):
                    models_ax[0, column_index].plot(x_batch, u_learned[:,i], '-.', label=f"{labels[i]}-{nl}")
                    models_ax[1, column_index].plot(x_batch, u_test[:,i], ':', label=f"{labels[i]}-{nl}") 

        for i in range(u_exact.shape[1]):  
            models_ax[0,0].plot(x_batch, u_exact[:, i], label=f"{labels[i]}-true") 
            models_ax[1,0].plot(x_batch, u_exact[:, i], label=f"{labels[i]}-true")  
            models_ax[0,1].plot(x_batch, u_exact[:, i], label=f"{labels[i]}-true") 
            models_ax[1,1].plot(x_batch, u_exact[:, i], label=f"{labels[i]}-true")   
        for i in range(2):  # For each row
            for j in range(2):  # For each column
                models_ax[i][j].set_title(f"{'Learned' if i == 0 else 'Test'}-tl[{'0-10' if j == 0 else '10-24'}]")
                models_ax[i][j].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=3 if problem.__name__=="SaturatedGrowthModel" else 4)
        models_fig.tight_layout()
        file_path = f"{parentdir}/Models_plot({name}).png"
        models_fig.savefig(file_path)

        # plot text, heatmaps and loss
        fig = plt.figure(figsize=(12,10), dpi=300)
        gs = gridspec.GridSpec(3, 1, height_ratios=[1, 5, 4])
        gs_lossplot = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[2], wspace=0.3) 

        ax1 = fig.add_subplot(gs_lossplot[0, 0])
        ax2 = fig.add_subplot(gs_lossplot[0, 1])
        ax3 = fig.add_subplot(gs_lossplot[0, 2])

        # Create Dataset for plotting
        df = pd.DataFrame(columns=['Time Limit', 'Noise Level', 'MSE Learned', 'MSE Test'])
        for run in runs:
            c_out, model = load_model(run, rootdir=rootdir+"/")

            parts = c_out.run.split("_")
            noise_index = 5
            noise_level = parts[noise_index]

            tl = c_out.problem_init_kwargs['time_limit']
            tl_key = f"{tl[0]}-{tl[1]}"

            # plot loss###########################
            i,t,l1n = model[-1][:,0], model[-1][:,3], model[-1][:,-1]

            axis_map = {'0-10': ax1, '0-24': ax2, '10-24': ax3}
            ax = axis_map.get(tl_key)
            if ax is None:
                raise ValueError(f"Invalid time_limit key: {tl_key}")
            
            # Now plot on the determined axis
            ax.plot(i, l1n, label=f"nl-{noise_level}")
            ax.set_yscale('log')
            ax.set_title(f'Time Limit: {tl_key}')
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Normalized l1 test loss')
            ax.legend(ncol=3, bbox_to_anchor=(0.5, -0.2), 
                      loc='upper center', fontsize='small')
            #################################
            
            u_exact, u_test, u_learned = get_us(c_out, model, type="FBPINN")
            mse_test = np.mean((u_exact - u_test)**2)
            mse_learned = np.mean((u_exact - u_learned)**2)
            
            new_row = pd.DataFrame({'Time Limit': [tl_key], 'Noise Level': [noise_level],
                                'MSE Learned': [mse_learned], 'MSE Test': [mse_test]})
            df = pd.concat([df, new_row], ignore_index=True)

            # extract parameters
            parameters_csv_pattern = os.path.join(rootdir+f"/summaries/{run}/", "parameters.csv")
            parameters = pd.read_csv(parameters_csv_pattern)
            
            true_params = parameters['True'].tolist()
            learned_params = parameters['Learned'].tolist()
            if len(true_params)==1:
                true_params = true_params[0]
                learned_params = learned_params[0]
            else:
                true_params = np.array(true_params)
                learned_params = np.array(learned_params)
            new_collcected_row = pd.DataFrame({'Time Limit': [tl_key], 'Noise Level': [noise_level],
                                'MSE Learned': [mse_learned], 'MSE Test': [mse_test],
                                'Model Type': [name], 'Learned Parameters': [learned_params]})
            collcted_df = pd.concat([collcted_df, new_collcected_row], ignore_index=True)
        
        df['MSE Learned'] = pd.to_numeric(df['MSE Learned'], errors='coerce')
        df['MSE Test'] = pd.to_numeric(df['MSE Test'], errors='coerce')

        pivot_learned = df.pivot(index="Noise Level", columns="Time Limit", values="MSE Learned")
        pivot_test = df.pivot(index="Noise Level", columns="Time Limit", values="MSE Test")

        pivot_learned_log = pivot_learned.map(lambda x: np.log10(x + 1e-10))  # Adding a small number to avoid log(0)
        pivot_test_log = pivot_test.map(lambda x: np.log10(x + 1e-10))

        ax0 = fig.add_subplot(gs[0, 0])
        lambda_phy_tex = r"$\lambda_{\mathrm{phy}}$"
        lambda_data_tex = r"$\lambda_{\mathrm{data}}$"

        params_text = (f"• nD: {numx} " +
                    f"• nC: {nCol} " +
                    f"• nT: {nTest} " +
                    f"• layer: {layer} " +
                    f"• epochs: {epochs} " + "\n"
                    f"• {lambda_phy_tex}: {lambda_phy} " +
                    f"• {lambda_data_tex}: {lambda_data} " + 
                    f"• Sparse: {sparse} " + 
                    f"• Opt: Adam " +
                    f"• lr: 0.001 " +
                    f"• DD: Nonuniform " + "\n"
                    f"• nsub: {nsub} " +
                    f"• wo: {wo} " +
                    f"• wi: {wi} " +
                    f"• Problem: {problem.__name__ if hasattr(problem, '__name__') else problem}"+
                    f"({name})")
        
        ax0.text(0.5, 0.5, params_text, ha='center', va='center', fontsize=12)
        ax0.set_frame_on(False)
        ax0.get_xaxis().set_visible(False)
        ax0.get_yaxis().set_visible(False)

        # heatmaps
        gs_heatmaps = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], wspace=0.3) 

        ax4 = fig.add_subplot(gs_heatmaps[0, 0])
        ax5 = fig.add_subplot(gs_heatmaps[0, 1])

        def highlight_min(data, ax, highlight_color='red'):
            for col in data.columns:
                min_row = data[col].idxmin()
                ax.add_patch(plt.Rectangle((data.columns.tolist().index(col), data.index.tolist().index(min_row)),
                                            1, 1, fill=True, facecolor=highlight_color, alpha=0.5, edgecolor=highlight_color, lw=2))

        # Heatmap for MSE 
        sns.heatmap(pivot_learned_log, annot=True, fmt=".2f", cmap='viridis', ax=ax4)
        highlight_min(pivot_learned_log, ax4)
        ax4.set_title('Log MSE Learned')
        ax4.invert_yaxis()

        sns.heatmap(pivot_test_log, annot=True, fmt=".2f", cmap='viridis', ax=ax5)
        highlight_min(pivot_test_log, ax5)
        ax5.set_title('Log MSE Test')
        ax5.invert_yaxis()

        plt.suptitle('MSE Value by Time Limit and Data Noise', fontsize=14, verticalalignment='top')#, y=0.95)
        plt.subplots_adjust(hspace=0.2, top=0.88)
        plt.tight_layout()
        file_path = f"{parentdir}/MSE_varying_DataNoise({name}).png"
        plt.savefig(file_path)

        # export collected data
        collcted_df.to_csv(f"{parentdir}/FBPINN_collected_info_varying_noise.csv", index=False)

        print("DONE")

if __name__=="__main__":
    plot_DDD_varying_DataNoise()