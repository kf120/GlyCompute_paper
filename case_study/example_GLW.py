# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 13:21:18 2024

@author: kf120
"""
import sys
import os

# Determine the path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add the parent directory of the script to sys.path to ensure access to "glycompute" module
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

import random
import warnings
import numpy as np
import pickle

import glycompute.graph as gfs
import glycompute.pathway as pfs
import glycompute.strategy as sfs

from glycompute.utils import create_and_change_dir, save_dict_pickle
from pyabc.sampler import MulticoreEvalParallelSampler, SingleCoreSampler

warnings.filterwarnings('ignore') 


def main():
    print("Initializing environment and seeding for reproducibility...")
    # Seed everything for reproducibility
    SEED = 0
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    
    GRAPH_NAME = 'CHOGlycoNET'
    SUFFIX = 'GLW'
    MODE = 'ABC_Sequential'
    
    '''
    LOAD GRAPH
    '''
    print("Loading graph...")
    G = gfs.load_graph_from_json(f'{GRAPH_NAME}_{SUFFIX}.json')
    create_and_change_dir(f'{GRAPH_NAME}_{MODE}_{SUFFIX}')
    
    
    '''
    PATHWAY
    '''
    print("Extracting pathway information...")
    pfs.export_graph_to_excel_format(G, name=f'{GRAPH_NAME}_{SUFFIX}')
    pfs.extract_pathway_info_to_pkl(f'{GRAPH_NAME}_{SUFFIX}.xlsx', suffix=SUFFIX)
    df_from_pkl = pfs.load_pathway_info_from_pkl(suffix=SUFFIX)
    
    
    '''
    PARAMETER ESTIMATION STRATEGY
    '''
    print("Characterizing directed graph and extracting node features...")
    df_graph_features = gfs.characterize_directed_graph(G, source_node='N1', return_csv=False)
    df_with_node_features = df_graph_features[0]
    
    composition_df = df_from_pkl['composition_df']
    
    print("Reading experimental data for IgG and HCP...")
    # Define the names and experimental relative abundances for IgG and HCP
    igg_data = {
        'M5': 1.0,
        'A2': 10.0,
        'FA2': 54.0,
        'A2G1': 2.0,
        'FA2G1': 28.0,
        'A2G2': 1.0,
        'FA2G2': 4.0
    }
    
    hcp_data = {
        'M9': 9.6,
        'M8': 11.0,
        'M7': 11.5,
        'M6': 20.3,
        'M5': 15.9,
        'FA3': 3.0,
        'A2G2S1': 1.1,
        'FA2G2S2': 2.0,
        'FA2G2S1': 2.4,
        'FA2G2': 1.5,
        'FA2G1': 1.1,
        'FA1': 3.0,
        'FA2': 6.8
    }
    
    print("Identifying critical tags based on betweeness centrality and dominance frontier membership...")
    critical_tags = sfs.filter_critical_tags_by_centrality_and_membership(G, df_with_node_features, composition_df, betweenness_percentile=50, membership_percentile=50, betweenness_tolerance=0.05)
    critical_tags = critical_tags[:3] # Top 3
    # critical_tags = ['M5A1', 'FA2', 'FA2G2']
    
    acceptable_enzymes = ['ManI', 'ManII', 'GnTI', 'GnTII', 'a6FucT', 'GnTIV', 'GnTV', 'b4GalT', 'a3SiaT']
    
    print("Propose parameter estimation stages...")
    stage_data_igg = sfs.propose_stages_for_parameter_estimation(G, critical_tags, igg_data, composition_df, acceptable_enzymes)
    sfs.print_stages_data(stage_data_igg)
    
    stage_data_hcp = sfs.propose_stages_for_parameter_estimation(G, critical_tags, hcp_data, composition_df, acceptable_enzymes)
    sfs.print_stages_data(stage_data_hcp)
    
    print("Saving stage data dictionaries to pickle files...")
    # Save the dictionaries to pickle files for future use
    with open('stage_data_igg.pkl', 'wb') as f:
        pickle.dump(stage_data_igg, f)
    
    with open('stage_data_hcp.pkl', 'wb') as f:
        pickle.dump(stage_data_hcp, f)
        
    print("Reading input data for model parameters...")
    '''
    INPUT DATA
    '''
    # Enzyme kf (min-1) for all reactions (turnover ratio)
    kf = {"ManI": 888., "ManII": 1924., "GnTI": 1022., "GnTII": 1406., "GnTIV": 187., "GnTV": 1410.,
          "a6FucT": 291., "b4GalT": 872., "a3SiaT": 491.
          }
    
    
    # Enzyme Kmd (uM) for all reactions (dissociation constants of NSDs from enzymes)
    Kmd = {
            "GnTI": {"IgG": 170., "HCP": 170.},
            "GnTII": {"IgG": 960., "HCP": 960.},
            "GnTIV": {"HCP": 8300},
            "GnTV": {"HCP": 5390},
            "a6FucT": {"IgG": 46., "HCP": 46.},
            "b4GalT": {"IgG": 65., "HCP": 65.},
            "a3SiaT": {"HCP": 57.}
        }
    
    # Precursor molecule concentrations in the cytosol (mM) (taken from Pavlos' PhD, p. 169, Fig. 5.6)
    cNSD = {
        'UDPGlcNAc': {'cyt': 1.9},
        'UDPGal': {'cyt': 0.4},
        'GDPFuc': {'cyt': 0.14},
        'CMPNeuAc': {'cyt': 0.2},
    }
    
    # Calculate Golgi concentrations for NSDs assumption that intra-Golgi concentration (uM) of NSDs is 20 times greater than the cytosolic one (mM)
    for key in cNSD.keys():
            cNSD[key]['golg'] = 20 * 1000 * cNSD[key]['cyt']
                
    
    # Additional parameters specific to the process
    # q_IgG and q_HCP are in mg/(cell h)
    process_params_input = {
        'N1in': 100,
        'MW_IgG': 150000,
        'MW_HCP': 46167,
        'GS_IgG': 2,
        'GS_HCP': 0.0809,
        'q_IgG': 18e-9/60,
        'q_HCP': 18e-9/60,
        'V_golg': 6.25e-15,
    }
    
    input_p = {**process_params_input,
              'kf': kf,
              'Kmd': Kmd,
              'cNSD': cNSD
            }
    
    '''
    ABC-SMC
    '''
    pop_size = 200 #200
    min_eps = 2.3
    max_nr_pop = 25 #25
    
    # sampler = MulticoreEvalParallelSampler(n_procs=64) # use parallelization
    sampler = SingleCoreSampler() # default option
    print(f"Running ABC-SMC parameter estimation...")

    
    def get_var_est_p_bounds_dict(enzymes, stage_number):
        bounds_dict = {}
        if stage_number == 1:
            bounds_dict = {
                'Man9propIgG': [0.05, 0.95],
                'Man9propHCP': [0.05, 0.95]
            }
        for enzyme in enzymes:
            # if enzyme not in ['GnTV']:  # Only add if not in the list of specific enzymes
            bounds_dict[f'cENZ_{enzyme}'] = [0.001, 35] #[0.001, 35]
            bounds_dict[f'Km_{enzyme}_HCP'] = [10, 10000]
        return bounds_dict
    
    def get_fixed_est_p(MAP_results, stage_number):
        fixed_est_p = {
            'Km_ManI_IgG': 61,
            
            'cENZ_ManII': 0,
            'Km_ManII_IgG': 100,
            'Km_ManII_HCP': 10000,
            
            'cENZ_GnTI': 0,
            'Km_GnTI_IgG':  260,
            'Km_GnTI_HCP': 10000,
            
            'cENZ_GnTII': 0,
            'Km_GnTII_IgG': 190,
            'Km_GnTII_HCP': 10000,
            
            'cENZ_GnTIV': 0,
            'Km_GnTIV_HCP': 10000,
    
            'cENZ_GnTV': 0,
            'Km_GnTV_HCP': 10000,
            
            'cENZ_a6FucT': 0,
            'Km_a6FucT_IgG': 25,
            'Km_a6FucT_HCP': 10000,
    
            'cENZ_b4GalT': 0,
            'Km_b4GalT_IgG': 430,
            'Km_b4GalT_HCP': 10000,
            
            'cENZ_a3SiaT': 0,
            'Km_a3SiaT_HCP': 10000
        }
        
        if stage_number > 1:
            fixed_est_p['Man9propIgG'] = MAP_results['S1']['Man9propIgG']
            fixed_est_p['Man9propHCP'] = MAP_results['S1']['Man9propHCP']
    
        for stage_key, stage_value in MAP_results.items():
            for param, value in stage_value.items():
                fixed_est_p[param] = value
    
        return fixed_est_p
    
    # Run the parameter estimation
    MAP_results, xpred_results = sfs.run_parameter_estimation(SUFFIX, stage_data_igg, stage_data_hcp, df_from_pkl, input_p, get_var_est_p_bounds_dict, get_fixed_est_p, sampler, pop_size, min_eps, max_nr_pop)
    save_dict_pickle(xpred_results, suffix=SUFFIX)
    print(f"Parameter estimation complete and results stored in {GRAPH_NAME}_{MODE}_{SUFFIX} directory.")

    
if __name__ == "__main__":
    main()