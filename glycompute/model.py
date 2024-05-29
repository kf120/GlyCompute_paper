# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 12:03:49 2024

@author: kf120
"""

import numpy as np

def michaelis_menten_kinetics(N, df_from_pkl, start_N_idx, end_N_idx, enzyme, protein, input_params, est_params):
    """
    Calculate reaction rates using Michaelis-Menten kinetics.

    Parameters
    ----------
    N : numpy.ndarray
        Array of glycans species concentrations.
    df_from_pkl : dict
        Dictionary containing various data structures for pathway analysis.
    start_N_idx : int
        Starting index for glycan species in Golgi commpartment.
    end_N_idx : int
        Ending index for glycan species in Golgi commpartment.
    enzyme : str
        The enzyme involved in the reaction.
    protein : str
        The protein involved in the reaction.
    input_params : dict
        Dictionary of input parameters required for the simulation.
    est_params : dict
        Dictionary of estimated parameters for the model.

    Returns
    -------
    numpy.ndarray
        The calculated reaction rates.
    """
    S = df_from_pkl['rn_spec'][enzyme][start_N_idx:end_N_idx] @ N[start_N_idx:end_N_idx] / est_params['Km'][enzyme][protein]
    N_NR = N[start_N_idx:end_N_idx] @ np.abs(df_from_pkl['F_sub'][0:df_from_pkl['F_sub_comp'].shape[0], :])
    return (df_from_pkl['n'][enzyme] * est_params['cENZ'][enzyme] * df_from_pkl['localization'][enzyme] * input_params['kf'][enzyme] * N_NR) / (est_params['Km'][enzyme][protein] * (1 + S))

def sequential_bi_bi_kinetics(N, df_from_pkl, start_N_idx, end_N_idx, enzyme, protein, input_params, est_params, NSD_key):
    """
    Calculate reaction rates using sequential Bi-Bi kinetics.

    Parameters
    ----------
    N : numpy.ndarray
        Array of glycans species concentrations.
    df_from_pkl : dict
        Dictionary containing various data structures for pathway analysis.
    start_N_idx : int
        Starting index for glycan species in Golgi commpartment.
    end_N_idx : int
        Ending index for glycan species in Golgi commpartment.
    enzyme : str
        The enzyme involved in the reaction.
    protein : str
        The protein involved in the reaction.
    input_params : dict
        Dictionary of input parameters required for the simulation.
    est_params : dict
        Dictionary of estimated parameters for the model.
    NSD_key : str
        Key for the NSD (non-substrate dependent) parameter in the input parameters.

    Returns
    -------
    numpy.ndarray
        The calculated reaction rates.
    """
    S = df_from_pkl['rn_spec'][enzyme][start_N_idx:end_N_idx] @ N[start_N_idx:end_N_idx] / est_params['Km'][enzyme][protein]
    N_NR = N[start_N_idx:end_N_idx] @ np.abs(df_from_pkl['F_sub'][0:df_from_pkl['F_sub_comp'].shape[0], :])
    return (df_from_pkl['n'][enzyme] * est_params['cENZ'][enzyme] * df_from_pkl['localization'][enzyme] * input_params['kf'][enzyme] * input_params['cNSD'][NSD_key]['golg'] * N_NR) / (
        est_params['Km'][enzyme][protein] * input_params['Kmd'][enzyme][protein] * (1 + (input_params['cNSD'][NSD_key]['golg'] / input_params['Kmd'][enzyme][protein]) * (1 + S)))

# GlycoSimModel description (function)
def GlycoSimModel(N, df_from_pkl, input_params, est_params):   
    """
    Defines the GlycoSimModel to calculate the residuals for species concentrations across Golgi compartments.
    
    Parameters
    ----------
    N : numpy.ndarray
        Array of glycan concentrations.
    df_from_pkl : dict
        Dictionary containing various data structures for pathway analysis.
    input_params : dict
        Dictionary of input parameters required for the simulation.
    est_params : dict
        Dictionary of estimated parameters for the model.
    
    Returns
    -------
    numpy.ndarray
        Array of residuals for species concentrations.
    
    Notes
    -----
    This function models the enzymatic reaction rates and the material balances across all Golgi compartments.
    Michaelis-Menten kinetics is used for ManI and ManII enzymes, while sequential Bi-Bi kinetics is used for other enzymes.
    """
    
    Species = df_from_pkl['SPECIES']
    Reactions = df_from_pkl['REACTIONS']
    Compartments = df_from_pkl['COMPARTMENTS']
    Proteins = df_from_pkl['PROTEINS']
    
    F = df_from_pkl['F']
    
    # Initialize arrays for residuals
    dN = np.zeros((Species*Compartments*Proteins, ))

    # Linear velocity (L/min) according to del Val et al. (2016)
    # MW (g/mol = Da), q (mg/h)
    Vel_IgG = (input_params['q_IgG'] / input_params['N1in']) * (1 / input_params['MW_IgG']) * 1000 * 1/60 * (1 / input_params['V_golg']) * input_params['GS_IgG']
    Vel_HCP = (input_params['q_HCP'] / input_params['N1in']) * (1 / input_params['MW_HCP']) * 1000 * 1/60 * (1 / input_params['V_golg']) * input_params['GS_HCP']

    # Concentration (uM) of initial glycoforms (Man9 and Man8) entering the Golgi apparatus, after transferring from the ER
    Nin_IgG = np.zeros((Species, )) 
    Nin_IgG[0] = est_params['Man9propIgG'] * input_params['N1in']; Nin_IgG[1] = (1-est_params['Man9propIgG']) * input_params['N1in']
    
    Nin_HCP = np.zeros((Species, ))
    Nin_HCP[0] = est_params['Man9propHCP'] * input_params['N1in']; Nin_HCP[1] = (1-est_params['Man9propHCP']) * input_params['N1in']
    
    # Enzymatic reaction rates and the material balances across all Golgi compartments
    for c in range(Compartments):
        
        # Indices for proteins -> Species        
        indices_N_IgG = (Species * c, Species * (c + 1))
        indices_N_HCP = (Species * Compartments + Species * c, Species * Compartments + Species * (c + 1))
        
        # Indices for proteins -> Reactions  
        indices_NR = (Reactions * c, Reactions * (c + 1))
        
        # Michaelis-Menten Kinetics for ManI and ManII
        NR_ManI_IgG = michaelis_menten_kinetics(N, df_from_pkl, *indices_N_IgG, 'ManI', 'IgG', input_params, est_params)
        NR_ManI_HCP = michaelis_menten_kinetics(N, df_from_pkl, *indices_N_HCP, 'ManI', 'HCP', input_params, est_params)
        NR_ManII_IgG = michaelis_menten_kinetics(N, df_from_pkl, *indices_N_IgG, 'ManII', 'IgG', input_params, est_params)
        NR_ManII_HCP = michaelis_menten_kinetics(N, df_from_pkl, *indices_N_HCP, 'ManII', 'HCP', input_params, est_params)

        # Sequential Bi-Bi Kinetics for other enzymes
        NR_GnTI_IgG = sequential_bi_bi_kinetics(N, df_from_pkl, *indices_N_IgG, 'GnTI', 'IgG', input_params, est_params, 'UDPGlcNAc')
        NR_GnTI_HCP = sequential_bi_bi_kinetics(N, df_from_pkl, *indices_N_HCP, 'GnTI', 'HCP', input_params, est_params, 'UDPGlcNAc')
        NR_GnTII_IgG = sequential_bi_bi_kinetics(N, df_from_pkl, *indices_N_IgG, 'GnTII', 'IgG', input_params, est_params, 'UDPGlcNAc')
        NR_GnTII_HCP = sequential_bi_bi_kinetics(N, df_from_pkl, *indices_N_HCP, 'GnTII', 'HCP', input_params, est_params, 'UDPGlcNAc')
        NR_GnTIV_HCP = sequential_bi_bi_kinetics(N, df_from_pkl, *indices_N_HCP, 'GnTIV', 'HCP', input_params, est_params, 'UDPGlcNAc')
        NR_GnTV_HCP = sequential_bi_bi_kinetics(N, df_from_pkl, *indices_N_HCP, 'GnTV', 'HCP', input_params, est_params, 'UDPGlcNAc')
        NR_a6FucT_IgG = sequential_bi_bi_kinetics(N, df_from_pkl, *indices_N_IgG, 'a6FucT', 'IgG', input_params, est_params, 'GDPFuc')
        NR_a6FucT_HCP = sequential_bi_bi_kinetics(N, df_from_pkl, *indices_N_HCP, 'a6FucT', 'HCP', input_params, est_params, 'GDPFuc')
        NR_b4GalT_IgG = sequential_bi_bi_kinetics(N, df_from_pkl, *indices_N_IgG, 'b4GalT', 'IgG', input_params, est_params, 'UDPGal')
        NR_b4GalT_HCP = sequential_bi_bi_kinetics(N, df_from_pkl, *indices_N_HCP, 'b4GalT', 'HCP', input_params, est_params, 'UDPGal')
        NR_a3SiaT_HCP = sequential_bi_bi_kinetics(N, df_from_pkl, *indices_N_HCP, 'a3SiaT', 'HCP', input_params, est_params, 'CMPNeuAc')

        # Calculate total reaction rates
        NR_IgG = sum([NR_ManI_IgG, NR_ManII_IgG, NR_GnTI_IgG, NR_GnTII_IgG, NR_a6FucT_IgG, NR_b4GalT_IgG])
        NR_HCP = sum([NR_ManI_HCP, NR_ManII_HCP, NR_GnTI_HCP, NR_GnTII_HCP, NR_GnTIV_HCP, NR_GnTV_HCP, NR_a6FucT_HCP, NR_b4GalT_HCP, NR_a3SiaT_HCP])

        # Material balances for each Golgi compartment
        if c==0:
            dN[indices_N_IgG[0]:indices_N_IgG[1]] = Vel_IgG * (Nin_IgG - N[indices_N_IgG[0]:indices_N_IgG[1]]) + F[indices_N_IgG[0]:indices_N_IgG[1], indices_NR[0]:indices_NR[1]] @ NR_IgG[indices_NR[0]:indices_NR[1]]
            dN[indices_N_HCP[0]:indices_N_HCP[1]] = Vel_HCP * (Nin_HCP - N[indices_N_HCP[0]:indices_N_HCP[1]]) + F[indices_N_IgG[0]:indices_N_IgG[1], indices_NR[0]:indices_NR[1]] @ NR_HCP[indices_NR[0]:indices_NR[1]]
                        
        else:
            dN[indices_N_IgG[0]:indices_N_IgG[1]] = Vel_IgG * (N[(indices_N_IgG[0]-Species): (indices_N_IgG[1]-Species)] - N[indices_N_IgG[0]:indices_N_IgG[1]]) + F[indices_N_IgG[0]:indices_N_IgG[1], indices_NR[0]:indices_NR[1]] @ NR_IgG[indices_NR[0]:indices_NR[1]]
            dN[indices_N_HCP[0]:indices_N_HCP[1]] = Vel_HCP * (N[(indices_N_HCP[0]-Species): (indices_N_HCP[1]-Species)] - N[indices_N_HCP[0]:indices_N_HCP[1]]) + F[indices_N_IgG[0]:indices_N_IgG[1], indices_NR[0]:indices_NR[1]] @ NR_HCP[indices_NR[0]:indices_NR[1]]
        
    return dN