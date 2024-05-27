# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 17:06:27 2024

@author: kf120
"""
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glypruneabc.simulation as sfs
from glypruneabc.utils import unflatten_params
from pyabc.visualization.kde import kde_1d
from pyabc.visualization import plot_kde_1d, plot_kde_2d

from pyabc import ABCSMC, RV, Distribution
from pyabc.epsilon import MedianEpsilon


def max_a_posteriori_estimate(df, w):
    """
    Compute the maximum a posteriori (MAP) estimate given a DataFrame of parameter samples and their corresponding weights.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the parameter samples.
    w : numpy.ndarray
        Array of weights corresponding to the samples in df.

    Returns
    -------
    dict
        Dictionary with parameter names as keys and their MAP estimates as values.
    """
    map_estimates = {}
    for column in df.columns:
        xvals, pdf = kde_1d(df, w, column)
        map_estimate = xvals[np.argmax(pdf)]
        map_estimates[column] = map_estimate

    return map_estimates

def simulator_wrapper(df_from_pkl, input_params, fixed_est_params, var_est_params, xobs_IgG, xobs_HCP, abc_smc_mode=True, return_results=False):
    """
    Wrapper function for the simulator, merging fixed and variable parameters to be estimated and converting to nested format.

    Parameters
    ----------
    df_from_pkl : dict
        Dictionary containing various dataframes and numpy arrays structured for pathway analysis.
    input_params : dict
        Dictionary of input parameters required for the simulation.
    fixed_est_params : dict
        Dictionary of fixed estimated parameters.
    var_est_params : dict
        Dictionary of variable estimated parameters.
    xobs_IgG : pandas.DataFrame
        Observed data for IgG.
    xobs_HCP : pandas.DataFrame
        Observed data for HCP.
    abc_smc_mode : bool, optional
        Flag indicating whether to run the simulator in ABC SMC mode (default is True).
    return_results : bool, optional
        Flag indicating whether to return the results from the simulator (default is False).

    Returns
    -------
    depends on the simulator function
        The result from the simulator function, depending on the value of return_results.
    """
    # Merge variable parameters with fixed parameters
    full_params = {**fixed_est_params, **var_est_params}

    # Convert flat parameter dictionary to nested format
    nested_parameters = unflatten_params(full_params)
    
    # Call the actual simulator function with static and dynamic parameters
    return sfs.simulator(df_from_pkl, input_params, nested_parameters, xobs_IgG, xobs_HCP, abc_smc_mode=abc_smc_mode, return_results=return_results)

def get_reaction_rates_wrapper(N_f, df_from_pkl, input_params, fixed_est_params, var_est_params):
    """
    Wrapper function for getting reaction rates, merging fixed and variable parameters and converting to nested format.

    Parameters
    ----------
    N_f : numpy.ndarray
        Array of metabolite concentrations.
    df_from_pkl : dict
        Dictionary containing various dataframes and numpy arrays structured for pathway analysis.
    input_params : dict
        Dictionary of input parameters required for the simulation.
    fixed_est_params : dict
        Dictionary of fixed estimated parameters.
    var_est_params : dict
        Dictionary of variable estimated parameters.

    Returns
    -------
    depends on the get_reaction_rates function
        The result from the get_reaction_rates function.
    """
    # Merge variable parameters with fixed parameters
    full_params = {**fixed_est_params, **var_est_params}
    
    # Convert flat parameter dictionary to nested format
    nested_parameters = unflatten_params(full_params)
    
    # Call the actual function
    return sfs.get_reaction_rates(N_f, df_from_pkl, input_params, nested_parameters)


def distance_rmse(xpred, xobs):
    """
    Calculate the Root Mean Squared Error (RMSE) between predicted and observed values.

    Parameters
    ----------
    xpred : dict
        Dictionary of predicted values.
    xobs : dict
        Dictionary of observed values.

    Returns
    -------
    float
        The Root Mean Squared Error (RMSE) between the predicted and observed values.
    """
    xpred = np.array(list(xpred.values()))
    xobs = np.array(list(xobs.values()))

    # Residuals
    residuals = xpred - xobs

    # Root mean squared error (RMSE)
    rmse = np.sqrt(np.mean(residuals ** 2))

    return rmse


def visualize_abc_results(df_from_pkl, input_params, fixed_est_params, df_final_gen, w_final_gen, xobs_IgG, xobs_HCP, IgG_HCP_xobs, labels):
    """
    Visualize the results of an Approximate Bayesian Computation (ABC) simulation.

    Parameters
    ----------
    df_from_pkl : dict
        Dictionary containing various dataframes and numpy arrays structured for pathway analysis.
    input_params : dict
        Dictionary of input parameters required for the simulation.
    fixed_est_params : dict
        Dictionary of fixed estimated parameters.
    df_final_gen : pandas.DataFrame
        DataFrame containing the parameter samples of the final generation.
    w_final_gen : numpy.ndarray
        Array of weights corresponding to the samples in df_final_gen.
    xobs_IgG : pandas.DataFrame
        Observed data for IgG.
    xobs_HCP : pandas.DataFrame
        Observed data for HCP.
    IgG_HCP_xobs : dict
        Dictionary of observed values for both IgG and HCP.
    labels : list of str
        List of labels for the output files.

    Returns
    -------
    tuple
        Predicted values array and a tuple containing reaction rates for IgG and HCP.
    """
    # Compute the maximum a posteriori estimate
    var_est_p_flat_particles = max_a_posteriori_estimate(df_final_gen, w_final_gen)
    
    # Run the simulator wrapper with the estimated parameters
    Np, xpred_all_dict = simulator_wrapper(df_from_pkl, input_params, fixed_est_params, var_est_p_flat_particles, xobs_IgG, xobs_HCP, abc_smc_mode=True, return_results=True)
        
    # Get reaction rates for IgG and HCP
    rates_IgG_all, rates_HCP_all = get_reaction_rates_wrapper(Np, df_from_pkl, input_params, fixed_est_params, var_est_p_flat_particles)
    
    # Prepare the data for visualization
    xpred_all = np.array(list(xpred_all_dict.values()))
    xobs = np.array([IgG_HCP_xobs[k] for k in IgG_HCP_xobs.keys()])
    xobs_err = xobs * 0.05
    
    # Create a figure with two subplots
    fig, (ax_IgG, ax_HCP) = plt.subplots(1, 2, figsize=(12, 6))
    
    IgG_names = xobs_IgG['Tag']
    HCP_names = xobs_HCP['Tag']
    N_pos_IgG = np.arange(len(IgG_names))
    N_pos_HCP = np.arange(len(HCP_names))
    
    # Plot the first subplot with df_IgG
    ax_IgG.bar(N_pos_IgG-0.2, xobs[:len(IgG_names)], yerr=xobs_err[:len(IgG_names)], capsize=5, width=0.4, label='Experimental')
    ax_IgG.bar(N_pos_IgG+0.2, xpred_all[:len(IgG_names)], width=0.4, hatch='///', label='Prediction')
    ax_IgG.set(xlabel='Oligosaccharide species', ylabel='Relative abundance (%)')
    ax_IgG.set_xticks(N_pos_IgG, IgG_names, rotation=45)
    ax_IgG.set_ylim([0, 100])
    ax_IgG.set_yticks(np.arange(0, 110, 10))
    ax_IgG.set_title('IgG glycoprofile')
    ax_IgG.legend()
    
    # Plot the second subplot with df_HCP
    ax_HCP.bar(N_pos_HCP-0.2, xobs[len(IgG_names):], yerr=xobs_err[len(IgG_names):], capsize=5, width=0.4, label='Experimental')
    ax_HCP.bar(N_pos_HCP+0.2, xpred_all[len(IgG_names):], width=0.4, hatch='///', label='Prediction')
    ax_HCP.set(xlabel='Oligosaccharide species', ylabel='Relative abundance (%)')
    ax_HCP.set_xticks(N_pos_HCP, HCP_names, rotation=45)
    ax_HCP.set_ylim([0, 100])
    ax_HCP.set_yticks(np.arange(0, 110, 10))
    ax_HCP.set_title('HCP glycoprofile')
    ax_HCP.legend()
    
    # Add a common title for the whole figure
    fig.suptitle('Simulation solution with estimated parameters', fontsize=16)
    
    # Parity plot
    plt.tight_layout()
    plt.savefig(f"ABC_ParityPlot_{labels[0]}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Reaction Rates Plot
    fig = sfs.visualize_reaction_rates(df_from_pkl, rates_IgG_all, rates_HCP_all)
    fig.savefig(f"ABC_ReacPlot_{labels[0]}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return xpred_all, (rates_IgG_all, rates_HCP_all)
    

def visualize_1D_kde_plots(enzyme_list, protein_list, df_final_gen, w_final_gen):
    """
    Generate and save 1D Kernel Density Estimation (KDE) plots for enzymes and their kinetic parameters.

    Parameters
    ----------
    enzyme_list : list of str
        List of enzymes to visualize.
    protein_list : list of str
        List of proteins associated with the enzymes.
    df_final_gen : pandas.DataFrame
        DataFrame containing the parameter samples of the final generation.
    w_final_gen : numpy.ndarray
        Array of weights corresponding to the samples in df_final_gen.

    Returns
    -------
    None
    """
    # Loop to generate plots for each enzyme
    for enzyme in enzyme_list:
        fig, axes = plt.subplots(1, 3, figsize=(10 * 3, 8))  # Create a figure with 3 subplots
        plots_created = 0  # Tracker for created plots

        # Plot the cENZ_{enzyme} parameter
        cENZ_col = f'cENZ_{enzyme}'
        if cENZ_col in df_final_gen.columns:
            ax = axes[0]  # First subplot for cENZ
            plot_kde_1d(df_final_gen, w_final_gen, x=cENZ_col, ax=ax)
            ax.set_title(f'{enzyme} - cENZ')
            plots_created += 1

        # Loop to plot Km_{enzyme}_protein for each protein in protein_list
        for i, protein in enumerate(protein_list, start=1):
            Km_col = f'Km_{enzyme}_{protein}'
            if Km_col in df_final_gen.columns:
                ax = axes[i]  # Subsequent subplots for Km values
                plot_kde_1d(df_final_gen, w_final_gen, x=Km_col, ax=ax)
                ax.set_title(f'{enzyme} - Km_{protein}')
                plots_created += 1

        plt.savefig(f"1D_KDE_plot_{enzyme}.png", dpi=300, bbox_inches='tight')
        plt.close()  # Show the plot


def visualize_2D_kde_plots(enzyme_list, protein_list, df_final_gen, w_final_gen):
    """
    Generate and save 2D Kernel Density Estimation (KDE) plots for enzymes and their kinetic parameters.

    Parameters
    ----------
    enzyme_list : list of str
        List of enzymes to visualize.
    protein_list : list of str
        List of proteins associated with the enzymes.
    df_final_gen : pandas.DataFrame
        DataFrame containing the parameter samples of the final generation.
    w_final_gen : numpy.ndarray
        Array of weights corresponding to the samples in df_final_gen.

    Returns
    -------
    None
    """
    # Loop to generate plots for each enzyme
    for enzyme in enzyme_list:
        fig, axes = plt.subplots(1, len(protein_list), figsize=(10 * len(protein_list), 8))
        plots_created = 0
    
        for i, protein in enumerate(protein_list):
            cENZ_col = f'cENZ_{enzyme}'
            Km_col = f'Km_{enzyme}_{protein}'
    
            if Km_col in df_final_gen.columns:
                ax = axes if len(protein_list) == 1 else axes[i]
                plot_kde_2d(df_final_gen, w_final_gen, x=cENZ_col, y=Km_col, ax=ax)
                ax.set_title(f'{enzyme} - {protein}')
                plots_created += 1
    
        # If no plots were created for this enzyme (data not found), remove the empty figure
        if plots_created == 0:
            plt.close(fig)
        else:
            plt.savefig(f"2D_KDE_plot_{enzyme}", dpi=300, bbox_inches='tight')
            plt.close()
            
def run_abc_smc(suffix, stage_label, enzymes, igg_data, hcp_data, df_from_pkl, input_params, fixed_est_params, var_est_p_bounds_dict, sampler, pop_size, min_eps, max_nr_pop, proteins=['IgG', 'HCP'], show_kde=False):
    """
    Run the Approximate Bayesian Computation Sequential Monte Carlo (ABC-SMC) algorithm and visualize results.

    Parameters
    ----------
    suffix : str
        Suffix to append to the output filenames to uniquely identify them.
    stage_label : str
        Label for the current stage of the analysis.
    enzymes : list of str
        List of enzymes to be considered in the analysis.
    igg_data : dict
        Dictionary of observed IgG data with tags as keys and relative abundances as values.
    hcp_data : dict
        Dictionary of observed HCP data with tags as keys and relative abundances as values.
    df_from_pkl : dict
        Dictionary containing various dataframes and numpy arrays structured for pathway analysis.
    input_params : dict
        Dictionary of input parameters required for the simulation.
    fixed_est_params : dict
        Dictionary of fixed estimated parameters.
    var_est_p_bounds_dict : dict
        Dictionary of variable estimated parameter bounds with parameter names as keys and (min, max) tuples as values.
    sampler : object
        Sampler object for the ABC-SMC algorithm.
    pop_size : int
        Population size for the ABC-SMC algorithm.
    min_eps : float
        Minimum epsilon value for the ABC-SMC algorithm.
    max_nr_pop : int
        Maximum number of populations for the ABC-SMC algorithm.
    proteins : list of str, optional
        List of proteins associated with the enzymes (default is ['IgG', 'HCP']).
    show_kde : bool, optional
        Flag to indicate whether to show KDE plots (default is False).

    Returns
    -------
    dict
        Dictionary with parameter names as keys and their MAP estimates as values.
    numpy.ndarray
        Array of predicted values.

    """
    # Set acceptable format for experimental data
    df_xobs_IgG = pd.DataFrame(list(igg_data.items()), columns=['Tag', 'Relative Abundance (%)'])
    df_xobs_HCP = pd.DataFrame(list(hcp_data.items()), columns=['Tag', 'Relative Abundance (%)'])

    xobs_IgG_dict = {f"{index}_IgG": row["Relative Abundance (%)"] for index, row in df_xobs_IgG.set_index('Tag').iterrows()}
    xobs_HCP_dict = {f"{index}_HCP": row["Relative Abundance (%)"] for index, row in df_xobs_HCP.set_index('Tag').iterrows()}

    IgG_HCP_xobs_dict = {**xobs_IgG_dict, **xobs_HCP_dict}

    # Specify priors
    priors = Distribution(**{key: RV("uniform", a, b-a) for key, (a, b) in var_est_p_bounds_dict.items()})

    # Run ABC
    abc = ABCSMC(models=lambda var_params: simulator_wrapper(df_from_pkl, input_params, fixed_est_params, var_params, df_xobs_IgG, df_xobs_HCP, abc_smc_mode=True, return_results=False),
                 parameter_priors=priors,
                 distance_function=distance_rmse,
                 population_size=pop_size,
                 sampler=sampler,
                 eps=MedianEpsilon())

    db_path = os.path.join(os.getcwd(), f"ABC_DATABASE_{stage_label}_{suffix}.db")
    abc.new("sqlite:///" + db_path, {"xobs": np.array(list(IgG_HCP_xobs_dict.values()))})

    start_time = time.time()

    history = abc.run(minimum_epsilon=min_eps, max_nr_populations=max_nr_pop)

    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"SMC-ABC Total time elapsed: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")

    # Get results
    df_end_gen, w_end_gen = history.get_distribution()
    MAP = max_a_posteriori_estimate(df_end_gen, w_end_gen)

    df_end_gen.to_pickle(f"df_end_gen_{stage_label}_{suffix}.pkl")
    np.save(f"w_end_gen_{stage_label}_{suffix}.npy", w_end_gen)

    for param in var_est_p_bounds_dict.keys():
        if param in MAP:
            print(f"{param}: {MAP[param]:.2f}")
        else:
            print(f"{param} not found in MAP.")
            
    # Visualize results
    xpred_all, rates_data = visualize_abc_results(df_from_pkl, input_params, fixed_est_params, df_end_gen, w_end_gen, df_xobs_IgG, df_xobs_HCP, IgG_HCP_xobs_dict, [stage_label])
    if show_kde:
        visualize_1D_kde_plots(enzymes, proteins, df_end_gen, w_end_gen)
        visualize_2D_kde_plots(enzymes, proteins, df_end_gen, w_end_gen)
    
    return MAP, xpred_all