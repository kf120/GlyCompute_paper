# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 12:00:55 2024

@author: kf120
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import root

from glycompute.utils import RootNotConvergedException
from glycompute.model import GlycoSimModel, michaelis_menten_kinetics, sequential_bi_bi_kinetics


# Solve GlycoSimModel
def simulator(df_from_pkl, input_params, est_params, df_xobs_IgG, df_xobs_HCP, method='hybr', abc_smc_mode=False, return_results=False):
    """
    Run model simulation.

    Parameters
    ----------
    df_from_pkl : pandas.DataFrame
        DataFrame containing the initial data for the simulation.
    input_params : dict
        Dictionary of input parameters required for the simulation.
    est_params : dict
        Dictionary of estimated parameters for the model.
    df_xobs_IgG : pandas.DataFrame
        DataFrame containing observed data for IgG.
    df_xobs_HCP : pandas.DataFrame
        DataFrame containing observed data for HCP.
    method : str, optional
        Method used for solving the simulation (default is 'hybr').
    abc_smc_mode : bool, optional
        If True, use ABC-SMC mode for the simulation (default is False).
    return_results : bool, optional
        If True and `abc_smc_mode` is True, return detailed results (default is False).

    Returns
    -------
    tuple or dict
        If `abc_smc_mode` is False, returns a tuple (N, df_IgG, df_HCP) where:
            - N : int
                Number of species predicted by the simulation.
            - df_IgG : pandas.DataFrame
                DataFrame containing simulated results for IgG.
            - df_HCP : pandas.DataFrame
                DataFrame containing simulated results for HCP.
        If `abc_smc_mode` is True:
            - If `return_results` is True, returns a tuple (N, IgG_HCP_pred_dict) where:
                - N : int
                    Number of iterations or some count relevant to the simulation.
                - IgG_HCP_pred_dict : dict
                    Dictionary containing predicted results for IgG and HCP.
            - If `return_results` is False, returns IgG_HCP_pred_dict alone.
    """    
    Species = df_from_pkl['SPECIES']
    Compartments = df_from_pkl['COMPARTMENTS']
    Proteins = df_from_pkl['PROTEINS']
    
    composition_df = df_from_pkl['composition_df']
    
    IgG_names = df_xobs_IgG['Tag']
    HCP_names = df_xobs_HCP['Tag']
            
    N0 = np.zeros(Species*Compartments*Proteins)

    root_solution = root(GlycoSimModel, N0, args=(df_from_pkl, input_params, est_params), method=method)
    
    if not root_solution.success:
        raise RootNotConvergedException('Root finding did not converge')
    
    # Collect concentrations of species at steady state
    N = root_solution.x

    N1in = 100.
    Nnames = ['N' + str(n) for n in range(1, Species+1)]

    # Calculation of relative abundance for all species (%), 4th compartment
    xpred_IgG = 100 * N[Species*3:Species*4] / N1in
    df_xpred_IgG = pd.DataFrame(xpred_IgG, columns=['Relative Abundance (%)'])
    df_xpred_IgG.index = Nnames

    xpred_HCP = 100 * N[Species*7:Species*8] / N1in
    df_xpred_HCP = pd.DataFrame(xpred_HCP, columns=['Relative Abundance (%)'])
    df_xpred_HCP.index = Nnames
    
    # Add the 'Tag' column to the simulated dataframes
    df_xpred_IgG['Tag'] = composition_df['Tag']
    df_xpred_HCP['Tag'] = composition_df['Tag']
    
    # Group by 'Tag' and sum the concentrations for IgG and HCP
    df_xpred_tagged_IgG = df_xpred_IgG.groupby('Tag').sum()
    df_xpred_tagged_HCP = df_xpred_HCP.groupby('Tag').sum()
    
    # Map prediction data to experimental data
    df_IgG = pd.DataFrame(index=IgG_names, columns=['Experimental', 'Prediction'])

    # Map experimental data to predicted data for IgG
    for name in IgG_names:
        df_IgG.loc[name, 'Experimental'] = df_xobs_IgG[df_xobs_IgG['Tag'].str.strip() == name]['Relative Abundance (%)'].values[0] if name in df_xobs_IgG['Tag'].str.strip().values else 0
        df_IgG.loc[name, 'Prediction'] = df_xpred_tagged_IgG.loc[name.strip(), 'Relative Abundance (%)'] if name.strip() in df_xpred_tagged_IgG.index else 0

    # Create a dataframe to store mapped data for HCP
    df_HCP = pd.DataFrame(index=HCP_names, columns=['Experimental', 'Prediction'])

    # Map experimental data to predicted data for HCP
    for name in HCP_names:
        df_HCP.loc[name, 'Experimental'] = df_xobs_HCP[df_xobs_HCP['Tag'].str.strip() == name]['Relative Abundance (%)'].values[0] if name in df_xobs_HCP['Tag'].str.strip().values else 0
        df_HCP.loc[name, 'Prediction'] = df_xpred_tagged_HCP.loc[name.strip(), 'Relative Abundance (%)'] if name.strip() in df_xpred_tagged_HCP.index else 0
    
    # Return prediction data for IgG and HCP in a form compatible to pyabc
    df_IgG_dict = {f"{index}_IgG": row["Prediction"] for index, row in df_IgG.iterrows()}
    df_HCP_dict = {f"{index}_HCP": row["Prediction"] for index, row in df_HCP.iterrows()}
    
    IgG_HCP_pred_dict = {**df_IgG_dict, **df_HCP_dict} 
    
    if abc_smc_mode:
        if return_results:    
            return N, IgG_HCP_pred_dict
        else:
            return IgG_HCP_pred_dict
    else:
        return N, df_IgG, df_HCP
    
def visualize_goodness_of_fit(df_IgG, df_HCP, figsize=(10, 5), save_png=False, show_plot=False):
    """
    Visualizes the goodness of fit between experimental and predicted data for IgG and HCP glycoprofiles.

    Parameters
    ----------
    df_IgG : pandas.DataFrame
        DataFrame containing 'Experimental' and 'Prediction' columns for IgG glycoprofiles.
    df_HCP : pandas.DataFrame
        DataFrame containing 'Experimental' and 'Prediction' columns for HCP glycoprofiles.
    figsize : tuple of int, optional
        Figure size for the plot (default is (10, 5)).
    save_png : bool, optional
        Whether to save the plot as a PNG file (default is False).
    show_plot : bool, optional
        Whether to display the plot (default is False).

    Returns
    -------
    matplotlib.figure.Figure
        The created figure containing the subplots.

    Notes
    -----
    The function creates a side-by-side bar plot comparing experimental and predicted glycoprofiles for IgG and HCP.
    """
    IgG_names = list(df_IgG.index)
    HCP_names = list(df_HCP.index)
    
    # Create a figure with two subplots
    fig, (ax_IgG, ax_HCP) = plt.subplots(1, 2, figsize=figsize)

    # Set position indices for the bars
    N_pos_IgG = np.arange(len(IgG_names))
    N_pos_HCP = np.arange(len(HCP_names))

    # Plot IgG glycoprofile
    ax_IgG.bar(N_pos_IgG-0.2, df_IgG['Experimental'], width=0.4, label='Experimental')
    ax_IgG.bar(N_pos_IgG+0.2, df_IgG['Prediction'], width=0.4, label='Prediction', hatch='///')
    ax_IgG.set_xlabel('Oligosaccharide species')
    ax_IgG.set_ylabel('Relative abundance (%)')
    ax_IgG.set_xticks(N_pos_IgG)
    ax_IgG.set_xticklabels(IgG_names, rotation=45)
    ax_IgG.set_ylim([0, 100])
    ax_IgG.set_yticks(np.arange(0, 110, 10))
    ax_IgG.set_title('IgG glycoprofile')
    ax_IgG.legend()
    ax_IgG.grid(True)

    # Plot HCP glycoprofile
    ax_HCP.bar(N_pos_HCP-0.2, df_HCP['Experimental'], width=0.4, label='Experimental')
    ax_HCP.bar(N_pos_HCP+0.2, df_HCP['Prediction'], width=0.4, label='Prediction', hatch='///')
    ax_HCP.set_xlabel('Oligosaccharide species')
    ax_HCP.set_ylabel('Relative abundance (%)')
    ax_HCP.set_xticks(N_pos_HCP)
    ax_HCP.set_xticklabels(HCP_names, rotation=45)
    ax_HCP.set_ylim([0, 100])
    ax_HCP.set_yticks(np.arange(0, 110, 10))
    ax_HCP.set_title('HCP glycoprofile')
    ax_HCP.legend()
    ax_HCP.grid(True)

    # Add a common title for the whole figure
    fig.suptitle('Simulation solution with estimated parameters', fontsize=16)

    # Show the plot
    plt.tight_layout()
  
    if save_png:
        plt.savefig("ParityPlot.png", dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()

    return fig
        
def get_reaction_rates(N_f, df_from_pkl, input_params, est_params):
    """
    Calculate reaction rates for IgG and HCP.

    Parameters
    ----------
    N_f : int
        Some integer parameter related to the reaction rates calculation.
    df_from_pkl : pandas.DataFrame
        DataFrame containing initial data for the simulation.
    input_params : dict
        Dictionary of input parameters required for the simulation.
    est_params : dict
        Dictionary of estimated parameters for the model.

    Returns
    -------
    tuple
        A tuple containing two elements:
        - NR_IgG : list or array
            The calculated reaction rates for IgG.
        - NR_HCP : list or array
            The calculated reaction rates for HCP.
    """
    
    Species = df_from_pkl['SPECIES']
    Compartments = df_from_pkl['COMPARTMENTS']
    
    # Enzymatic reaction rates and the material balances across all Golgi compartments
    for c in range(Compartments):
        
        # Indices for proteins -> Species        
        indices_N_IgG = (Species * c, Species * (c + 1))
        indices_N_HCP = (Species * Compartments + Species * c, Species * Compartments + Species * (c + 1))
       
        # Michaelis-Menten Kinetics for ManI and ManII
        NR_ManI_IgG = michaelis_menten_kinetics(N_f, df_from_pkl, *indices_N_IgG, 'ManI', 'IgG', input_params, est_params)
        NR_ManI_HCP = michaelis_menten_kinetics(N_f, df_from_pkl, *indices_N_HCP, 'ManI', 'HCP', input_params, est_params)
        NR_ManII_IgG = michaelis_menten_kinetics(N_f, df_from_pkl, *indices_N_IgG, 'ManII', 'IgG', input_params, est_params)
        NR_ManII_HCP = michaelis_menten_kinetics(N_f, df_from_pkl, *indices_N_HCP, 'ManII', 'HCP', input_params, est_params)
        
        # Sequential Bi-Bi Kinetics for other enzymes
        NR_GnTI_IgG = sequential_bi_bi_kinetics(N_f, df_from_pkl, *indices_N_IgG, 'GnTI', 'IgG', input_params, est_params, 'UDPGlcNAc')
        NR_GnTI_HCP = sequential_bi_bi_kinetics(N_f, df_from_pkl, *indices_N_HCP, 'GnTI', 'HCP', input_params, est_params, 'UDPGlcNAc')
        NR_GnTII_IgG = sequential_bi_bi_kinetics(N_f, df_from_pkl, *indices_N_IgG, 'GnTII', 'IgG', input_params, est_params, 'UDPGlcNAc')
        NR_GnTII_HCP = sequential_bi_bi_kinetics(N_f, df_from_pkl, *indices_N_HCP, 'GnTII', 'HCP', input_params, est_params, 'UDPGlcNAc')
        NR_GnTIV_HCP = sequential_bi_bi_kinetics(N_f, df_from_pkl, *indices_N_HCP, 'GnTIV', 'HCP', input_params, est_params, 'UDPGlcNAc')
        NR_GnTV_HCP = sequential_bi_bi_kinetics(N_f, df_from_pkl, *indices_N_HCP, 'GnTV', 'HCP', input_params, est_params, 'UDPGlcNAc')
        NR_a6FucT_IgG = sequential_bi_bi_kinetics(N_f, df_from_pkl, *indices_N_IgG, 'a6FucT', 'IgG', input_params, est_params, 'GDPFuc')
        NR_a6FucT_HCP = sequential_bi_bi_kinetics(N_f, df_from_pkl, *indices_N_HCP, 'a6FucT', 'HCP', input_params, est_params, 'GDPFuc')
        NR_b4GalT_IgG = sequential_bi_bi_kinetics(N_f, df_from_pkl, *indices_N_IgG, 'b4GalT', 'IgG', input_params, est_params, 'UDPGal')
        NR_b4GalT_HCP = sequential_bi_bi_kinetics(N_f, df_from_pkl, *indices_N_HCP, 'b4GalT', 'HCP', input_params, est_params, 'UDPGal')
        NR_a3SiaT_HCP = sequential_bi_bi_kinetics(N_f, df_from_pkl, *indices_N_HCP, 'a3SiaT', 'HCP', input_params, est_params, 'CMPNeuAc')
        
        # Calculate total reaction rates
        NR_IgG = sum([NR_ManI_IgG, NR_ManII_IgG, NR_GnTI_IgG, NR_GnTII_IgG, NR_a6FucT_IgG, NR_b4GalT_IgG])
        NR_HCP = sum([NR_ManI_HCP, NR_ManII_HCP, NR_GnTI_HCP, NR_GnTII_HCP, NR_GnTIV_HCP, NR_GnTV_HCP, NR_a6FucT_HCP, NR_b4GalT_HCP, NR_a3SiaT_HCP]) # same indices as above
    
    return NR_IgG, NR_HCP

def visualize_reaction_rates(df_from_pkl, rates_IgG_all, rates_HCP_all, figsize=(18, 12), save_png=False, show_plot=False):
    """
    Creates visualizations of reaction rates for IgG and HCP across multiple Golgi compartments.

    Parameters
    ----------
    df_from_pkl : pandas.DataFrame
        DataFrame containing the 'REACTIONS' and 'COMPARTMENTS' information.
    rates_IgG_all : list or array
        A list or array containing the reaction rates for IgG for all compartments concatenated.
    rates_HCP_all : list or array
        A list or array containing the reaction rates for HCP for all compartments concatenated.
    figsize : tuple of int, optional
        Figure size for the plot (default is (18, 12)).
    save_png : bool, optional
        Whether to save the plot as a PNG file (default is False).
    show_plot : bool, optional
        Whether to display the plot (default is False).

    Returns
    -------
    matplotlib.figure.Figure
        The created figure containing the subplots.

    Notes
    -----
    Each compartment is assumed to have an equal number of reactions, determined by dividing the total length
    of rates by the number of compartments.
    """
    Reactions = df_from_pkl['REACTIONS']
    Compartments = df_from_pkl['COMPARTMENTS']
    
    # Create a figure with 4 subplots (2 rows, 2 columns)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize)
    axes = axes.flatten()  # Flatten the axes array for easy iteration

    for i in range(Compartments):  # Loop over four Golgi compartments
        golgi_comp = i
        indices_NR = (Reactions * golgi_comp, Reactions * (golgi_comp + 1))

        rates_IgG = rates_IgG_all[indices_NR[0]:indices_NR[1]]
        rates_HCP = rates_HCP_all[indices_NR[0]:indices_NR[1]]

        # Creating DataFrame for the current compartment
        data = pd.DataFrame({
            'Reaction Number': range(1, Reactions + 1),  # Assuming Reactions is the number of reactions per compartment
            'IgG Rates': rates_IgG,
            'HCP Rates': rates_HCP
        })

        # Plotting on the current subplot
        ax = axes[i]
        ax.plot(data['Reaction Number'], data['IgG Rates'], label='IgG', marker='o', color='red')
        ax.plot(data['Reaction Number'], data['HCP Rates'], label='HCP', marker='x', color='green')
        ax.set_title(f'Reaction Rates in Golgi Compartment {golgi_comp + 1}')
        ax.set_xlabel('Reaction Number')
        ax.set_ylabel('Rates')
        ax.set_xticks(data['Reaction Number'])
        ax.legend()
        ax.grid(True)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    if save_png:
        plt.savefig("ReacPlot.png", dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()
    return fig