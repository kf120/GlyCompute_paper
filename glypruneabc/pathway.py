# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 15:59:55 2024

@author: kf120
"""

import os
import pandas as pd
import numpy as np
from collections import defaultdict
from natsort import natsorted, natsort_keygen

# Set the option to handle downcasting explicitly in the future
pd.set_option('future.no_silent_downcasting', True)


def export_graph_to_excel_format(graph, name):
    """
    Exports graph information into an Excel format suitable for pathway extraction.

    Parameters
    ----------
    graph : networkx.Graph
        The graph from which data is extracted.
    name : str
        The base name for the output Excel file, which will include the graph information.

    Returns
    -------
    str
        The filename of the created Excel file.

    Notes
    -----
    This function creates two sheets in an Excel file:
    - One for node data with columns for node index and structure.
    - One for edge data with columns for edge index, child node, parent node, enzyme, and enzyme type.
    The output is saved as an Excel file named using the provided base name.
    """
    # Extract the edge list from the graph
    edge_list = list(graph.edges(data=True))

    # Get unique node names
    unique_nodes = set(graph.nodes)

    # Create the DataFrame for node data
    node_data = []
    for node in natsorted(unique_nodes):
        row = [node, graph.nodes[node].get('LinearCode')]
        node_data.append(row)

    node_df = pd.DataFrame(node_data, columns=['Index_N', 'LinearCode'])

    # Create the DataFrame for edge data
    edge_data = []
    for edge in natsorted(edge_list, key=lambda x: (x[0], x[1])):
        source, target, edge_attrs = edge
        enzyme = edge_attrs.get('Enzyme')
        gh_or_gt = 'GH' if enzyme in ('ManI', 'ManII') else 'GT'
        row = [edge_attrs.get('Reaction Index'), graph.nodes[target].get('LinearCode'), graph.nodes[source].get('LinearCode'), enzyme, gh_or_gt]
        edge_data.append(row)

    edge_df = pd.DataFrame(edge_data, columns=['Index_NR', 'Child', 'Parent', 'Enzyme', 'Enzyme type'])

    # Concatenate the node and edge DataFrames
    df = pd.concat([node_df, edge_df], axis=1)
    df.columns = ['Index_N', 'LinearCode', 'Index_NR', 'Child', 'Parent', 'Enzyme', 'Enzyme type']
    
    #TODO: not required, it creates confusion -> when commented out, weird behaviour
    # Rename the entries in Index_N (N1, N2, ...) and Index_NR (NR1, NR2, ...)
    df.loc[df['Index_N'].notna(), 'Index_N'] = ['N'+str(i+1) for i in range(df['Index_N'].count())]
    df.loc[df['Index_NR'].notna(), 'Index_NR'] = ['NR'+str(i+1) for i in range(df['Index_NR'].count())]

    # Save the DataFrame as an Excel file
    excel_filename = f"{name}.xlsx"
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Network', index=False, startrow=1)

    return excel_filename

def extract_pathway_info_to_pkl(excel_filename, suffix):
    """
    Extracts and processes pathway information from an Excel file and saves it as various pickle (.pkl) files.

    Parameters
    ----------
    excel_filename : str
        Filename of the Excel file containing the graph data.
    suffix : str
        Suffix to append to the output file names to uniquely identify them.

    Returns
    -------
    None

    Notes
    -----
    The function reads species and reactions data, processes them, and saves as separate .pkl files for easy access and manipulation in further analyses. It creates the following pickle files:
    - species_df_<suffix>.pkl
    - reactions_df_<suffix>.pkl
    - enzymes_df_<suffix>.pkl
    - enz_reac_df_<suffix>.pkl
    - enz_spec_df_<suffix>.pkl
    - F_df_<suffix>.pkl
    - F_sub_df_<suffix>.pkl
    - composition_df_<suffix>.pkl
    """
    
    def preprocess_df(df):
        """
        Preprocesses a DataFrame by stripping whitespace and converting all entries to strings.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame to preprocess.

        Returns
        -------
        pandas.DataFrame
            The preprocessed DataFrame.
        """
        df = df.dropna()
        df.columns = df.columns.str.strip()
        df = df.astype('string')
        df[df.columns] = df.apply(lambda x: x.str.strip())
        return df

    def preprocess_pivot_df(df, species_df):
        """
        Preprocesses a pivot DataFrame by ensuring all species are represented and sorting.

        Parameters
        ----------
        df : pandas.DataFrame
            The pivot DataFrame to preprocess.
        species_df : pandas.DataFrame
            The DataFrame containing species information.

        Returns
        -------
        pandas.DataFrame
            The preprocessed pivot DataFrame.
        """
        missing_rows = list(set(species_df['Index_N']) - set(df.index))
        missing_cols = list(set(species_df['Index_N']) - set(df.columns))
        new_rows = pd.DataFrame(index=missing_rows)
        new_cols = pd.DataFrame(columns=missing_cols)
        if any(elem in df.index for elem in species_df['Index_N']):
            df = pd.concat([df, new_rows], axis=0, sort=False)
        if any(elem in df.columns for elem in species_df['Index_N']):
            df = pd.concat([df, new_cols], axis=1, sort=False)
        df = df.fillna(0).infer_objects()
        df = df.sort_index(axis=0, key=natsort_keygen())
        df = df.sort_index(axis=1, key=natsort_keygen())
        return df

    # Build the path to the Excel file
    excel_path = os.path.join(os.getcwd(), excel_filename)

    # Process Species DataFrame
    species_df = pd.read_excel(excel_path, sheet_name='Network', usecols='A, B', skiprows=[0])
    species_df = preprocess_df(species_df)
    species_df.to_pickle(f'species_df_{suffix}.pkl')

    # Process Reactions DataFrame
    reactions_df = pd.read_excel(excel_path, sheet_name='Network', usecols='C, D, E, F, G', skiprows=[0])
    reactions_df = preprocess_df(reactions_df)
    species_dict = dict(zip(species_df['LinearCode'], species_df['Index_N']))
    reactions_df['Child'] = reactions_df['Child'].map(species_dict)
    reactions_df['Parent'] = reactions_df['Parent'].map(species_dict)
    reactions_df.to_pickle(f'reactions_df_{suffix}.pkl')

    # Process Enzymes DataFrame
    enzymes_df = reactions_df[['Enzyme']].drop_duplicates()
    enzymes_df = pd.DataFrame(enzymes_df)
    enzymes_df.to_pickle(f'enzymes_df_{suffix}.pkl')

    # Process Enzymes x Reactions Pivot Table
    enzym_reac_df = reactions_df.pivot_table(values='Enzyme type', index='Enzyme', columns='Index_NR', aggfunc='count')
    enzym_reac_df = preprocess_pivot_df(enzym_reac_df, species_df)
    enzym_reac_df.to_pickle(f'enz_reac_df_{suffix}.pkl')

    # Process Enzymes x Species Pivot Table
    enzym_spec_df = reactions_df.pivot_table(values='Enzyme type', index='Enzyme', columns='Parent', aggfunc=pd.Series.nunique)
    enzym_spec_df = preprocess_pivot_df(enzym_spec_df, species_df)
    enzym_spec_df.to_pickle(f'enz_spec_df_{suffix}.pkl')

    # Stoichiometric Matrix (Species (Parent+Child) x Reactions)
    stoich_df = pd.DataFrame(0, index=species_df['Index_N'], columns=reactions_df['Index_NR'])
    for idx, row in reactions_df.iterrows():
        parent = row['Parent']
        child = row['Child']
        reaction_idx = row['Index_NR']
        if parent in stoich_df.index:
            stoich_df.at[parent, reaction_idx] = -1
        if child in stoich_df.index:
            stoich_df.at[child, reaction_idx] = 1
    stoich_df.to_pickle(f'F_df_{suffix}.pkl')

    # Substrate Stoichiometric Matrix (Species (Parent) x Reactions)
    stoich_sub_df = reactions_df.pivot_table(values='Enzyme type', index='Parent', columns='Index_NR', aggfunc='count')
    stoich_sub_df = preprocess_pivot_df(stoich_sub_df, species_df)
    stoich_sub_df.to_pickle(f'F_sub_df_{suffix}.pkl')
    
    species_df = species_df.rename(columns={col: col.strip().replace(" ", "") for col in species_df.columns})  # Remove whitespace from column names
    species_df = species_df.map(lambda x: x.strip() if isinstance(x, str) else x)  # Remove whitespace from all string entries

    # Create composition dataframe
    composition_df = species_df.copy()

    # Add a new column 'M_count' with the count of 'M' in the 'LinearCode' column
    composition_df['Man_count'] = composition_df['LinearCode'].str.count('M')

    # Add a new column 'GN_count' with the (total) count of 'GN' in the 'LinearCode' column
    composition_df['GN_count'] = composition_df['LinearCode'].str.count('GN')

    # Add a new column 'F_count' with the count of 'F' in the 'LinearCode' column
    composition_df['CoreFuc_count'] = composition_df['LinearCode'].str.count('Fa6')

    # Add a new column 'A_count' with the count of 'Ab' in the 'LinearCode' column
    composition_df['Gal_count'] = composition_df['LinearCode'].str.count('Ab') 

    # Add a new column 'NN_count' with the count of 'NN' in the 'LinearCode' column
    composition_df['Sial_count'] = composition_df['LinearCode'].str.count('NN')
    
    # Add a new column 'LeX_count' with the count of 'Fa3' in the 'LinearCode' column
    composition_df['LeX_count'] = composition_df['LinearCode'].str.count('Fa3')
    
    # Add a new column 'PolyLacNAc_count' with the count of 'GNb3' in the 'LinearCode' column
    composition_df['PolyLacNAc_count'] = composition_df['LinearCode'].str.count('GNb3')

    # Tag species with glycosylation traits

    # High mannose glycans
    composition_df['HM'] = composition_df.apply(lambda x: 'Yes' if x['Man_count'] >= 5 and x['GN_count'] <= 2 else 'No', axis=1)

    # High branching glycans (consider the specific linkages that denote tri- and tetra-antennary N-glycans)
    composition_df['HB'] = composition_df.apply(
    lambda x: 'Yes' if (('GNb4)' in x['LinearCode'] or 'GNb6)' in x['LinearCode'])) and x['GN_count'] > 4 else 'No', axis=1)
    
    # Core fucosylated glycans
    composition_df['CoreFuc'] = composition_df['CoreFuc_count'].apply(lambda x: 'Yes' if x > 0 else 'No')
    
    # Galactosylated glycans
    composition_df['Gal'] = composition_df['Gal_count'].apply(lambda x: 'Yes' if x > 0 else 'No')

    # Sialylated glycans
    composition_df['Sial'] = composition_df['Sial_count'].apply(lambda x: 'Yes' if x > 0 else 'No')
    
    # Glycans with Lewis x antigen
    composition_df['LeX'] = composition_df['LeX_count'].apply(lambda x: 'Yes' if x > 0 else 'No')
    
    # Glycans with GlcNAc with b-1,3 linkage characteristic of PolyLacNac structures (may also be terminal)
    composition_df['PolyLacNAc'] = composition_df['PolyLacNAc_count'].apply(lambda x: 'Yes' if x > 0 else 'No')


    def generate_tag(row):
        """
        Generate a tag for each species based on its LinearCode.

        Parameters
        ----------
        row : pandas.Series
            A row of the DataFrame containing species information.

        Returns
        -------
        str
            The generated tag for the species.
        """
        # Initialize the fucose part based on the 'CoreFuc' column
        f_part = "F" if row['CoreFuc'] == 'Yes' else ""
        
        # High Mannose type or special case with GN_count=2 and HM='No'
        if row['HM'] == 'Yes' or (row['GN_count'] == 2 and row['HM'] == 'No'):
            return f"M{row['Man_count']}{f_part}"
        
        elif (row['GN_count'] == 3 and row['Man_count'] > 3):
            return f"M{row['Man_count']}{f_part}A{row['GN_count']-2}"
        
        # Default case for others
        else:
            # Determine the fucose part if F_count=1
            f_part = "F" if row['CoreFuc_count'] == 1 else ""
            
            # Determine the galactose part, if any
            gal_part = f"G{row['Gal_count']}" if row['Gal_count'] > 0 else ""
            
            # Determine the sialic acid part, if any
            s_part = f"S{row['Sial_count']}" if row['Sial_count'] > 0 else ""
            
            # Determine the single arm Ma6 part
            sa_part = "-Ma6" if ('(Ma6)' in row['LinearCode'] and row['GN_count'] >= 4) else ""
            
            # Determine the polylacnac part
            polylacnac_part = "-Poly" if row['PolyLacNAc_count'] > 0 else ""
            
            return f"{f_part}A{row['GN_count']-2}{gal_part}{s_part}{sa_part}{polylacnac_part}"


    # Generate 'Tag' column
    composition_df['Tag'] = composition_df.apply(lambda row: generate_tag(row), axis=1)

    # Reindex
    composition_df = composition_df.set_index('Index_N')
    
    # Add 'Trait' column
    columns_of_interest = ['HM', 'HB', 'CoreFuc', 'Gal', 'Sial', 'LeX', 'PolyLacNAc']
    
    # Get unique combinations of these columns
    subset_df = composition_df[columns_of_interest].drop_duplicates().reset_index(drop=True)
    
    def create_trait_name(row):
        """
        Create a trait name based on the presence of glycosylation traits.

        Parameters
        ----------
        row : pandas.Series
            A row of the DataFrame containing species information.

        Returns
        -------
        str
            The created trait name.
        """
        # Gather columns with 'Yes'
        active_columns = [col for col in columns_of_interest if row[col] == 'Yes']
        # Create a name by joining column names with underscores
        return '_'.join(active_columns) if active_columns else 'Hybrid/Complex'
    
    # Apply the function to each row in the DataFrame to create the 'Trait' column
    subset_df['Name'] = subset_df.apply(create_trait_name, axis=1)
    
    # Map each combination of values to a unique name
    combination_to_name = {tuple(row[columns_of_interest]): row['Name'] for index, row in subset_df.iterrows()}
    
    # Apply this mapping to the original DataFrame to create the 'Trait' column
    composition_df['Trait'] = composition_df.apply(lambda row: combination_to_name[tuple(row[col] for col in columns_of_interest)], axis=1)
    
    # Pickle
    composition_df.to_pickle(f'composition_df_{suffix}.pkl')
    
def load_pathway_info_from_pkl(suffix, species_name='species_df_', reactions_name='reactions_df_',
                               enzyme_name='enzymes_df_', enz_reac_name='enz_reac_df_', enz_spec_name='enz_spec_df_', 
                               stoich_name='F_df_', stoich_sub_name='F_sub_df_', composition_name='composition_df_'):
    """
    Loads various types of data related to a biological pathway from pickle files, using a common suffix.

    Parameters
    ----------
    suffix : str
        Suffix used to identify the specific pathway information.
    species_name : str, optional
        Base name for the species pickle file (default is 'species_df_').
    reactions_name : str, optional
        Base name for the reactions pickle file (default is 'reactions_df_').
    enzyme_name : str, optional
        Base name for the enzymes pickle file (default is 'enzymes_df_').
    enz_reac_name : str, optional
        Base name for the enzyme-reaction pickle file (default is 'enz_reac_df_').
    enz_spec_name : str, optional
        Base name for the enzyme-species pickle file (default is 'enz_spec_df_').
    stoich_name : str, optional
        Base name for the stoichiometric matrix pickle file (default is 'F_df_').
    stoich_sub_name : str, optional
        Base name for the substrate stoichiometric matrix pickle file (default is 'F_sub_df_').
    composition_name : str, optional
        Base name for the composition pickle file (default is 'composition_df_').

    Returns
    -------
    dict
        A dictionary containing all the loaded dataframes and numpy arrays structured for pathway analysis.

    Notes
    -----
    The function loads data from pickle files, processes it, and returns it in a structured format suitable for pathway analysis.
    The dictionary contains the following keys:
    - 'SPECIES': Number of species.
    - 'REACTIONS': Number of reactions.
    - 'PROTEINS': Number of proteins.
    - 'COMPARTMENTS': Number of Golgi compartments.
    - 'species_df': DataFrame of species information.
    - 'reactions_df': DataFrame of reactions information.
    - 'enzymes_df': DataFrame of enzymes information.
    - 'enz_reac_df': DataFrame of enzyme-reaction information.
    - 'enz_spec_df': DataFrame of enzyme-species information.
    - 'F_df': Stoichiometric matrix for all Golgi compartments.
    - 'F_sub_df': Substrate stoichiometric matrix for all Golgi compartments.
    - 'composition_df': DataFrame of composition information.
    - 'n_comp_spec': Auxiliary array for single Golgi compartment (enzyme x reactions by specificity).
    - 'n_comp': Auxiliary array for single Golgi compartment (enzyme x reactions).
    - 'n': Auxiliary array for all Golgi compartments (enzyme x reactions).
    - 'rn_comp_spec': Auxiliary array for single Golgi compartment (enzyme x species by specificity).
    - 'n_spec': Auxiliary array for all Golgi compartments (enzyme x reactions by specificity).
    - 'rn_spec': Auxiliary array for all Golgi compartments (enzyme x species by specificity).
    - 'localization': Enzyme localization across Golgi compartments.
    """
    # Build file paths based on suffix and base names
    species_df = pd.read_pickle(f'./{species_name}{suffix}.pkl')
    reactions_df = pd.read_pickle(f'./{reactions_name}{suffix}.pkl')
    enzymes_df = pd.read_pickle(f'./{enzyme_name}{suffix}.pkl')
    enz_reac_df = pd.read_pickle(f'./{enz_reac_name}{suffix}.pkl')
    enz_spec_df = pd.read_pickle(f'./{enz_spec_name}{suffix}.pkl')
    F_df = pd.read_pickle(f'./{stoich_name}{suffix}.pkl')
    F_sub_df = pd.read_pickle(f'./{stoich_sub_name}{suffix}.pkl')
    composition_df = pd.read_pickle(f'./{composition_name}{suffix}.pkl')

    ### Build auxiliary arrays for single Golgi compartment ###
    n_comp_spec = {}
    for enzyme in enzymes_df['Enzyme']:
        n_comp_spec[enzyme] = enz_reac_df.loc[enzyme].to_numpy()

    substrings = defaultdict(list)
    for enzyme, values in n_comp_spec.items():
        substr = enzyme.split('_')[0]
        substrings[substr].append(np.array(values))

    n_comp = {}
    for substr, arrays in substrings.items():
        n_comp[substr] = np.sum(arrays, axis=0)

    rn_comp_spec = {}
    for enzyme in enzymes_df['Enzyme']:
        rn_comp_spec[enzyme] = enz_spec_df.loc[enzyme].to_numpy()

    ### Build auxiliary arrays for all Golgi compartments ###
    SPECIES = species_df.index.size
    REACTIONS = reactions_df.index.size
    PROTEINS = 2
    COMPARTMENTS = 4

    n_spec = {key: np.tile(value, COMPARTMENTS) for key, value in n_comp_spec.items()}

    substrings = defaultdict(list)
    for enzyme, values in n_spec.items():
        substr = enzyme.split('_')[0]
        substrings[substr].append(np.array(values))

    n = {}
    for substr, arrays in substrings.items():
        n[substr] = np.sum(arrays, axis=0)

    rn_spec = {key: np.tile(value, COMPARTMENTS * PROTEINS) for key, value in rn_comp_spec.items()}

    F_comp = F_df.to_numpy()  # Single Golgi compartment (Species vs. Reactions)
    F = np.tile(F_comp, (COMPARTMENTS, COMPARTMENTS))  # All Golgi compartments (Species*Compartments vs. Reactions*Compartments)

    F_sub_comp = F_sub_df.to_numpy()  # Single Golgi compartment (Species x Reactions)
    F_sub = np.tile(F_sub_comp, (COMPARTMENTS, COMPARTMENTS))  # All Golgi compartments (Species*Compartments vs. Reactions*Compartments)

    dis = {
        'ManI': np.array([0.15, 0.40, 0.3, 0.15]),
        'ManII': np.array([0.15, 0.4, 0.3, 0.15]),
        'GnTI': np.array([0.20, 0.45, 0.2, 0.15]),
        'GnTII': np.array([0.20, 0.45, 0.2, 0.15]),
        'GnTIV': np.array([0.20, 0.45, 0.2, 0.15]),
        'GnTV': np.array([0.20, 0.45, 0.2, 0.15]),
        'a6FucT': np.array([0.20, 0.45, 0.2, 0.15]),            
        'b4GalT': np.array([0.0, 0.05, 0.2, 0.75]),
        'a3SiaT': np.array([0.0, 0.05, 0.2, 0.75]),
        'a3FucT': np.array([0.0, 0.05, 0.2, 0.75]),
        'iGnT': np.array([0.0, 0.05, 0.2, 0.75])
    }

    localization = {}
    for enzyme in n_comp.keys():
        localization[enzyme] = np.array([n_comp[enzyme] * dis[enzyme][k] for k in range(COMPARTMENTS)]).ravel()    

    df_from_pkl = {
        'SPECIES': SPECIES,
        'REACTIONS': REACTIONS,
        'PROTEINS': PROTEINS,
        'COMPARTMENTS': COMPARTMENTS,
        'species_df': species_df,
        'reactions_df': reactions_df,
        'enzymes_df': enzymes_df,
        'enz_reac_df': enz_reac_df,
        'enz_spec_df': enz_spec_df,
        'F_df': F_df,
        'F': F,
        'F_sub_df': F_sub_df,
        'F_sub': F_sub,
        'F_sub_comp': F_sub_comp,
        'composition_df': composition_df,
        'n_comp_spec': n_comp_spec,
        'n_comp': n_comp,
        'n': n,
        'rn_comp_spec': rn_comp_spec,
        'n_spec': n_spec,
        'rn_spec': rn_spec,
        'localization': localization
    }

    return df_from_pkl