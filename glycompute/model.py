# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 12:03:49 2024

@author: kf120
"""

import numpy as np

def michaelis_menten_kinetics(N, df_from_pkl, start_N_idx_1, end_N_idx_1, start_N_idx_2, end_N_idx_2, enzyme, protein_1, protein_2, input_params, est_params):
    """
    Calculate reaction rates using Michaelis-Menten kinetics.

    Parameters
    ----------
    N : numpy.ndarray
        Array of glycans species concentrations.
    df_from_pkl : dict
        Dictionary containing various data structures for pathway analysis.
    start_N_idx_1 : int
        Starting index for glycan species for protein 1 in Golgi commpartment.
    end_N_idx_1 : int
        Ending index for glycan species for protein 1 in Golgi commpartment.
    start_N_idx_2 : int
        Starting index for glycan species for protein 2 in Golgi commpartment.
    end_N_idx_2 : int
        Ending index for glycan species for protein 2 in Golgi commpartment.    
    enzyme : str
        The enzyme involved in the reaction.
    protein_1 : str
        Protein 1 involved in the reaction.
    protein_2 : str
        Protein 2 involved in the reaction.
    input_params : dict
        Dictionary of input parameters required for the simulation.
    est_params : dict
        Dictionary of estimated parameters for the model.

    Returns
    -------
    numpy.ndarray
        The calculated reaction rates.
    """
    S1 = df_from_pkl['rn_spec'][enzyme][start_N_idx_1:end_N_idx_1] @ N[start_N_idx_1:end_N_idx_1] / est_params['Km'][enzyme][protein_1]
    S2 = df_from_pkl['rn_spec'][enzyme][start_N_idx_2:end_N_idx_2] @ N[start_N_idx_2:end_N_idx_2] / est_params['Km'][enzyme][protein_2]
    N_NR_1 = N[start_N_idx_1:end_N_idx_1] @ np.abs(df_from_pkl['F_sub'][0:df_from_pkl['F_sub_comp'].shape[0], :])
    N_NR_2 = N[start_N_idx_2:end_N_idx_2] @ np.abs(df_from_pkl['F_sub'][0:df_from_pkl['F_sub_comp'].shape[0], :])
    
    r1 = (df_from_pkl['n'][enzyme] * est_params['cENZ'][enzyme] * df_from_pkl['localization'][enzyme] * input_params['kf'][enzyme] * N_NR_1) / (est_params['Km'][enzyme][protein_1] * (1 + S1 + S2))
    r2 = (df_from_pkl['n'][enzyme] * est_params['cENZ'][enzyme] * df_from_pkl['localization'][enzyme] * input_params['kf'][enzyme] * N_NR_2) / (est_params['Km'][enzyme][protein_2] * (1 + S1 + S2))
    return r1, r2

def sequential_bi_bi_kinetics(N, df_from_pkl, start_N_idx_1, end_N_idx_1, start_N_idx_2, end_N_idx_2, enzyme, protein_1, protein_2, input_params, est_params, NSD_key):
    """
    Calculate reaction rates using sequential Bi-Bi kinetics.

    Parameters
    ----------
    N : numpy.ndarray
        Array of glycans species concentrations.
    df_from_pkl : dict
        Dictionary containing various data structures for pathway analysis.
    start_N_idx_1 : int
        Starting index for glycan species for protein 1 in Golgi commpartment.
    end_N_idx_1 : int
        Ending index for glycan species for protein 1 in Golgi commpartment.
    start_N_idx_2 : int
        Starting index for glycan species for protein 2 in Golgi commpartment.
    end_N_idx_2 : int
        Ending index for glycan species for protein 2 in Golgi commpartment.    
    enzyme : str
        The enzyme involved in the reaction.
    protein_1 : str
        Protein 1 involved in the reaction.
    protein_2 : str
        Protein 2 involved in the reaction.
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
    
    S1 = df_from_pkl['rn_spec'][enzyme][start_N_idx_1:end_N_idx_1] @ N[start_N_idx_1:end_N_idx_1] / est_params['Km'][enzyme][protein_1]
    S2 = df_from_pkl['rn_spec'][enzyme][start_N_idx_2:end_N_idx_2] @ N[start_N_idx_2:end_N_idx_2] / est_params['Km'][enzyme][protein_2]
    N_NR_1 = N[start_N_idx_1:end_N_idx_1] @ np.abs(df_from_pkl['F_sub'][0:df_from_pkl['F_sub_comp'].shape[0], :])
    N_NR_2 = N[start_N_idx_2:end_N_idx_2] @ np.abs(df_from_pkl['F_sub'][0:df_from_pkl['F_sub_comp'].shape[0], :])
    
    r1 = (df_from_pkl['n'][enzyme] * est_params['cENZ'][enzyme] * df_from_pkl['localization'][enzyme] * input_params['kf'][enzyme] * input_params['cNSD'][NSD_key]['golg'] * N_NR_1) / (est_params['Km'][enzyme][protein_1] * input_params['Kmd'][enzyme][protein_1] * (1 + (input_params['cNSD'][NSD_key]['golg'] / input_params['Kmd'][enzyme][protein_1]) * (1 + S1 + S2)))
    r2 = (df_from_pkl['n'][enzyme] * est_params['cENZ'][enzyme] * df_from_pkl['localization'][enzyme] * input_params['kf'][enzyme] * input_params['cNSD'][NSD_key]['golg'] * N_NR_2) / (est_params['Km'][enzyme][protein_2] * input_params['Kmd'][enzyme][protein_2] * (1 + (input_params['cNSD'][NSD_key]['golg'] / input_params['Kmd'][enzyme][protein_2]) * (1 + S1 + S2)))

    return r1, r2

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

    # Linear velocity (Golgi length/min) according to del Val et al. (2016)
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
        NR_ManI_IgG, NR_ManI_HCP = michaelis_menten_kinetics(N, df_from_pkl, *indices_N_IgG, *indices_N_HCP, 'ManI', 'IgG', 'HCP', input_params, est_params)
        NR_ManII_IgG, NR_ManII_HCP = michaelis_menten_kinetics(N, df_from_pkl, *indices_N_IgG, *indices_N_HCP, 'ManII', 'IgG', 'HCP', input_params, est_params)

        # Sequential Bi-Bi Kinetics for other enzymes
        NR_GnTI_IgG, NR_GnTI_HCP = sequential_bi_bi_kinetics(N, df_from_pkl, *indices_N_IgG, *indices_N_HCP, 'GnTI', 'IgG', 'HCP', input_params, est_params, 'UDPGlcNAc')
        NR_GnTII_IgG, NR_GnTII_HCP = sequential_bi_bi_kinetics(N, df_from_pkl, *indices_N_IgG, *indices_N_HCP, 'GnTII', 'IgG', 'HCP', input_params, est_params, 'UDPGlcNAc')
        _, NR_GnTIV_HCP = sequential_bi_bi_kinetics(N, df_from_pkl, *indices_N_IgG, *indices_N_HCP, 'GnTIV', 'IgG', 'HCP', input_params, est_params, 'UDPGlcNAc')
        _, NR_GnTV_HCP = sequential_bi_bi_kinetics(N, df_from_pkl, *indices_N_IgG, *indices_N_HCP, 'GnTV', 'IgG', 'HCP', input_params, est_params, 'UDPGlcNAc')
        NR_a6FucT_IgG, NR_a6FucT_HCP = sequential_bi_bi_kinetics(N, df_from_pkl, *indices_N_IgG, *indices_N_HCP, 'a6FucT', 'IgG', 'HCP', input_params, est_params, 'GDPFuc')
        NR_b4GalT_IgG, NR_b4GalT_HCP = sequential_bi_bi_kinetics(N, df_from_pkl, *indices_N_IgG, *indices_N_HCP, 'b4GalT', 'IgG', 'HCP', input_params, est_params, 'UDPGal')
        _, NR_a3SiaT_HCP = sequential_bi_bi_kinetics(N, df_from_pkl, *indices_N_IgG, *indices_N_HCP, 'a3SiaT', 'IgG', 'HCP', input_params, est_params, 'CMPNeuAc')

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
