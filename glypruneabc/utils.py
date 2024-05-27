# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 17:11:03 2024

@author: kf120
"""

import os
import pickle


def create_and_change_dir(directory_name):
    """
    Creates a new directory and changes the working directory to the newly created directory.

    Parameters
    ----------
    directory_name : str
        The name of the directory to create.

    Returns
    -------
    None

    Raises
    ------
    OSError
        If the directory cannot be created or if changing the working directory fails.
    """
    try:
        # Check if the directory already exists
        if not os.path.exists(directory_name):
            # Create a new directory
            os.makedirs(directory_name)
            print(f"Directory '{directory_name}' created successfully.")
        else:
            print(f"Directory '{directory_name}' already exists.")
        
        # Change the current working directory
        os.chdir(directory_name)
        print(f"Changed the current working directory to '{directory_name}'.")
        
    except OSError as error:
        print(f"Error: {error}")
        raise  # Re-raise the error to handle it further if needed
        
                    
def iupac_to_linearcode(iupac_condensed):
    """
    Converts a glycan from IUPAC-condensed format to LinearCode format.

    Parameters
    ----------
    iupac_condensed : str
        Glycan sequence in IUPAC-condensed format.

    Returns
    -------
    str
        Glycan in LinearCode format.

    Notes
    -----
    The function uses a predefined dictionary to replace IUPAC-condensed motifs with their corresponding LinearCode motifs.
    """
    replace_dic = {
        'Man(b1-4)': 'Mb4',
        'Man(a1-2)': 'Ma2',
        'Man(a1-3)': 'Ma3',
        'Man(a1-4)': 'Ma4',
        'Man(a1-6)': 'Ma6',
        'GlcNAc(b1-2)': 'GNb2',
        'GlcNAc(b1-3)': 'GNb3',
        'GlcNAc(b1-4)': 'GNb4',
        'GlcNAc(b1-6)': 'GNb6',
        'Gal(b1-4)': 'Ab4',
        'Fuc(a1-3)': 'Fa3',
        'Fuc(a1-6)': 'Fa6',
        'Neu5Ac(a2-3)': 'NNa3',
        'Neu5Ac(a2-6)': 'NNa6',
        'GlcNAc': 'GN;',
        '(': '(',
        ')': ')',
        '[': '(',
        ']': ')'
    }

    # Sort keys by length to avoid partial replacements
    for key in sorted(replace_dic.keys(), key=len, reverse=True):
        iupac_condensed = iupac_condensed.replace(key, replace_dic[key])
        linear_code = iupac_condensed
    
    return linear_code

def find_linearcode_difference(substrate, product):
    """
    Find the difference between the substrate and product LinearCode strings.

    Parameters
    ----------
    substrate : str
        LinearCode of the substrate node.
    product : str
        LinearCode of the product node.

    Returns
    -------
    str
        The difference between the product and the substrate.

    Notes
    -----
    The function identifies the common prefix and suffix between the substrate and product LinearCode strings and extracts the differing portion in the middle.
    """
    # Identify the common prefix
    i = 0
    while i < len(substrate) and i < len(product) and substrate[i] == product[i]:
        i += 1
    
    # Identify the common suffix
    j = 0
    while j < len(substrate) and j < len(product) and substrate[-(j+1)] == product[-(j+1)]:
        j += 1
    
    # Extract the difference
    difference = product[i:len(product)-j] if i + j <= len(product) else ''
    return difference


def infer_enzyme(substrate, product):
    """
    Infer the enzyme based on the difference between substrate and product LinearCode strings.

    Parameters
    ----------
    substrate : str
        LinearCode of the substrate node.
    product : str
        LinearCode of the product node.

    Returns
    -------
    str
        The inferred enzyme.

    Notes
    -----
    The function identifies the difference between the substrate and product LinearCode strings and attempts to infer the enzyme responsible for the transformation. It first checks if the difference length is within a specified maximum length for a direct match. If not, it attempts to identify the enzyme by progressively larger substrings of the difference.
    """
    difference = find_linearcode_difference(substrate, product)
    max_length_for_direct_match = 4  # maximum length for a direct match

    if len(difference) <= max_length_for_direct_match:
        # Enzyme identification based on the entire difference
        return identify_enzyme_by_difference(difference)
    else:
        # Enzyme identification based on substrings of the difference from left to right
        for i in range(1, len(difference) + 1):
            enzyme = identify_enzyme_by_difference(difference[:i])
            if enzyme != 'Unknown Enzyme':
                return enzyme
    return 'Unknown Enzyme'

def identify_enzyme_by_difference(difference):
    """
    Identify the enzyme based on the difference string.

    Parameters
    ----------
    difference : str
        The difference string between substrate and product.

    Returns
    -------
    str
        The inferred enzyme.

    Notes
    -----
    The function maps specific substrings in the difference string to corresponding enzymes.
    """
    # Enzyme identification based on difference
    if 'GNb2' in difference:
        return 'GnTII'
    if 'GNb4' in difference:
        return 'GnTIV'
    if 'GNb6' in difference:
        return 'GnTV'
    if 'GNb3' in difference:
        return 'iGnT'
    if 'Fa6' in difference:
        return 'a6FucT'
    if 'Fa3' in difference:
        return 'a3FucT'
    if 'Ab4' in difference:
        return 'b4GalT'   
    if 'NNa3' in difference:
        return 'a3SiaT'
    return 'Unknown Enzyme'


# Custom exceptions for specific cases
class OverflowException(Exception):
    pass

class RootNotConvergedException(Exception):
    pass

def flatten_params(params, parent_key='', sep='_'):
    """
    Recursively flattens a dictionary and concatenates keys from different levels using underscores.

    Parameters
    ----------
    params : dict
        The dictionary to flatten.
    parent_key : str, optional
        The base key to use for concatenation (default is an empty string).
    sep : str, optional
        The separator to use for concatenating keys (default is '_').

    Returns
    -------
    dict
        The flattened dictionary.

    Notes
    -----
    This function recursively traverses the dictionary, concatenating keys with the specified separator.
    """
    items = []
    for key, value in params.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            items.extend(flatten_params(value, new_key, sep=sep).items())
        else:
            items.append((new_key, value))
    return dict(items)

def unflatten_params(params, sep='_'):
    """
    Reconstructs a nested dictionary from a flat dictionary based on concatenated keys.

    Parameters
    ----------
    params : dict
        The flat dictionary to unflatten.
    sep : str, optional
        The separator used in the keys to denote different levels (default is '_').

    Returns
    -------
    dict
        The reconstructed nested dictionary.

    Notes
    -----
    This function splits the keys using the specified separator and constructs the nested dictionary structure.
    """
    result_dict = {}
    for key, value in params.items():
        parts = key.split(sep)
        d = result_dict
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return result_dict

def save_dict_pickle(dictionary, suffix):
    """
    Save a dictionary to a file using pickle.

    Parameters
    ----------
    dictionary : dict
        The dictionary to save.
    suffix : str
        The suffix to use for the filename.

    Returns
    -------
    None

    Notes
    -----
    The dictionary is saved with the filename format 'xpred_results_<suffix>.pkl'.
    """
    filename = f'xpred_results_{suffix}.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(dictionary, file)
    print(f'Dictionary saved to {filename}')
    
def linearcode_to_tag_mapping(composition_df):
    """
    Create mappings between LinearCode and Tag from the given DataFrame.

    Parameters
    ----------
    composition_df : pandas.DataFrame
        DataFrame containing 'LinearCode' and 'Tag' columns.

    Returns
    -------
    tuple of dict
        - linearcode_to_tag : dict
            Dictionary mapping LinearCode to Tag.
        - tag_to_linearcode : dict
            Dictionary mapping Tag to a list of LinearCodes.
    """
    # Map from LinearCode to Tag
    linearcode_to_tag = composition_df.set_index('LinearCode')['Tag'].to_dict()
    
    # Reverse the mapping from Tag to a list of LinearCodes
    tag_to_linearcode = {}
    for linearcode, tag in linearcode_to_tag.items():
        if tag not in tag_to_linearcode:
            tag_to_linearcode[tag] = [linearcode]
        else:
            tag_to_linearcode[tag].append(linearcode)
    
    return linearcode_to_tag, tag_to_linearcode