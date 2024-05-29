# -*- coding: utf-8 -*-
"""
Created on Sun May 26 12:31:18 2024

@author: kf120
"""
import numpy as np
import networkx as nx
import glycompute.utils as utf
import glycompute.graph as gfs
import glycompute.abc as afs

from collections import defaultdict


def identify_layers_and_traits(graph, composition_df):
    """
    Identify layers and critical traits of nodes in a directed graph based on topological sorting.

    Parameters
    ----------
    graph : networkx.DiGraph
        The directed graph to analyze.
    composition_df : pandas.DataFrame
        DataFrame containing node composition information with 'Trait' and 'Tag' columns.

    Returns
    -------
    tuple
        - layers : dict
            Dictionary where keys are layer numbers and values are lists of nodes in each layer.
        - unique_critical_tags : list
            List of unique critical tags identifying transitions between different traits.
    """
    sorted_nodes = list(nx.topological_sort(graph))
    node_to_trait = composition_df['Trait'].to_dict()
    node_to_tag = composition_df['Tag'].to_dict()
    
    node_layers = {}
    current_trait = None
    critical_tags = []

    for node in sorted_nodes:
        predecessors = list(graph.predecessors(node))
        if predecessors:
            node_layers[node] = max(node_layers[pred] for pred in predecessors) + 1
        else:
            node_layers[node] = 0

        trait = node_to_trait.get(node)
        if trait != current_trait:
            tag = node_to_tag.get(node)
            if tag:
                critical_tags.append(tag)
            current_trait = trait

    layers = defaultdict(list)
    for node, layer in node_layers.items():
        layers[layer].append(node)

    unique_critical_tags = list(dict.fromkeys(critical_tags))

    return layers, unique_critical_tags

def get_betweenness_centrality(df, composition_df, critical_tags):
    """
    Get betweenness centrality values for critical tags.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing nodes and their betweenness centrality values.
    composition_df : pandas.DataFrame
        DataFrame containing node composition information with 'Tag' columns.
    critical_tags : list of str
        List of critical tags to filter by their betweenness centrality values.

    Returns
    -------
    dict
        Dictionary with critical tags as keys and their betweenness centrality values as values.
    """
    node_to_tag = composition_df['Tag'].to_dict()
    
    critical_tags_betweenness = {}
    for _, row in df.iterrows():
        node = row['Node']
        betweenness = row['Betweenness Centrality']
        tag = node_to_tag.get(node)
        if tag in critical_tags:
            critical_tags_betweenness[tag] = betweenness
    
    return critical_tags_betweenness

def calculate_membership_in_dominance_frontiers(df, critical_tags, composition_df):
    """
    Calculate the membership of nodes in dominance frontiers for given critical tags.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing nodes and their dominance frontiers.
    critical_tags : list of str
        List of critical tags to filter dominance frontier memberships.
    composition_df : pandas.DataFrame
        DataFrame containing node composition information with 'Tag' columns.

    Returns
    -------
    dict
        Dictionary with critical tags as keys and lists of nodes that belong to their dominance frontiers as values.
    """
    node_to_tag = composition_df['Tag'].to_dict()
    membership_data = {tag: [] for tag in critical_tags}

    for _, row in df.iterrows():
        node = row['Node']
        frontier = row['Dominance Frontier']
        frontier_tags = {node_to_tag[n] for n in frontier if n in node_to_tag}
        for tag in critical_tags:
            if tag in frontier_tags:
                membership_data[tag].append(node)

    return membership_data

def filter_critical_tags_by_centrality_and_membership(graph, df_with_node_features, composition_df, betweenness_percentile=50, membership_percentile=50, betweenness_tolerance=0.05):
    """
    Filter and sort critical tags based on betweenness centrality and membership in dominance frontiers.

    Parameters
    ----------
    graph : networkx.DiGraph
        The directed graph to analyze.
    df_with_node_features : pandas.DataFrame
        DataFrame containing nodes and their features, including betweenness centrality and dominance frontiers.
    composition_df : pandas.DataFrame
        DataFrame containing node composition information with 'Tag' columns.
    betweenness_percentile : int, optional
        Percentile threshold for filtering by betweenness centrality (default is 50).
    membership_percentile : int, optional
        Percentile threshold for filtering by membership size in dominance frontiers (default is 50).
    betweenness_tolerance : float, optional
        Tolerance for considering betweenness values as close when sorting (default is 0.05).

    Returns
    -------
    list
        List of filtered and sorted critical tags based on the specified criteria.
    """
    # Identify layers and critical tags
    layers, critical_tags = identify_layers_and_traits(graph, composition_df)

    # Get critical tags with their betweenness centrality
    critical_tags_betweenness = get_betweenness_centrality(df_with_node_features, composition_df, critical_tags)

    # Sort critical tags by betweenness centrality
    sorted_critical_tags = sorted(critical_tags_betweenness.items(), key=lambda x: x[1], reverse=True)

    # Extract just the tags for calculating dominance frontier
    sorted_critical_tags_only = [tag for tag, _ in sorted_critical_tags]

    # Calculate membership
    membership_data = calculate_membership_in_dominance_frontiers(df_with_node_features, sorted_critical_tags_only, composition_df)

    # Calculate percentile thresholds
    betweenness_values = list(critical_tags_betweenness.values())
    betweenness_threshold = np.percentile(betweenness_values, betweenness_percentile) if betweenness_values else 0
    membership_sizes = [len(members) for members in membership_data.values()]
    membership_threshold = np.percentile(membership_sizes, membership_percentile) if membership_sizes else 0

    # Filter critical tags based on the betweenness threshold
    filtered_critical_tags = [
        (tag, betweenness, len(membership_data.get(tag, []))) for tag, betweenness in sorted_critical_tags
        if betweenness > betweenness_threshold and len(membership_data.get(tag, [])) >= membership_threshold
    ]

    # Sort primarily by betweenness, secondarily by membership size if betweenness is close
    final_sorted_tags = []
    for tag, betweenness, membership_size in filtered_critical_tags:
        inserted = False
        for i, (f_tag, f_betweenness, f_membership_size) in enumerate(final_sorted_tags):
            if abs(betweenness - f_betweenness) <= betweenness_tolerance and membership_size > f_membership_size:
                final_sorted_tags.insert(i, (tag, betweenness, membership_size))
                inserted = True
                break
        if not inserted:
            final_sorted_tags.append((tag, betweenness, membership_size))

    # Filter final critical tags based on membership threshold
    final_critical_tags = [
        (tag, betweenness, membership_size) for tag, betweenness, membership_size in final_sorted_tags
        if membership_size >= membership_threshold
    ]

    # Return the names of the tags in their sorted order
    return [tag for tag, _, _ in final_critical_tags]


def find_preceding_observed_nodes(graph, critical_nodes, observed_nodes):
    """
    Find all observed nodes that precede the given critical nodes in a topologically sorted graph.

    Parameters
    ----------
    graph : networkx.DiGraph
        The directed graph to search within.
    critical_nodes : list or str
        A list of critical nodes or a single critical node.
    observed_nodes : list of list
        A list of lists, where each sublist contains observed nodes.

    Returns
    -------
    list
        A list of nodes that are observed and precede any of the critical nodes in the topological order.
    """
    # Ensure critical_nodes is a list even if a single node is provided
    if not isinstance(critical_nodes, list):
        critical_nodes = [critical_nodes]

    # Ensure observed_nodes is a list of lists
    observed_nodes_sets = [set(lst) for lst in observed_nodes]

    # Perform a topological sort on the graph
    sorted_nodes = list(nx.topological_sort(graph))

    # Find the indices of all critical nodes in the sorted list
    critical_indices = [sorted_nodes.index(node) for node in critical_nodes]

    # Calculate the minimum index to consider all critical nodes
    min_critical_index = min(critical_indices)

    # Collect all preceding nodes that are also observed nodes
    preceding_observed = []
    for observed_set in observed_nodes_sets:
        if any(node in sorted_nodes[:min_critical_index] for node in observed_set):
            preceding_observed.extend(observed_set)

    return preceding_observed


def propose_stages_for_parameter_estimation(graph, critical_tags, experimental_profile, composition_df, acceptable_enzymes=None):
    """
    Construct stages for parameter estimation based on critical tags and experimental profiles.

    Parameters
    ----------
    graph : networkx.DiGraph
        The directed graph containing enzyme reaction pathways.
    critical_tags : list of str
        List of critical tags for identifying stages.
    experimental_profile : dict
        Dictionary containing experimental profile data with tags as keys and their abundances as values.
    composition_df : pandas.DataFrame
        DataFrame containing node composition information with 'LinearCode' and 'Tag' columns.
    acceptable_enzymes : list of str, optional
        List of acceptable enzymes to be considered in the estimation (default is None).

    Returns
    -------
    dict
        Dictionary containing stages data with estimated enzymes and updated tagged profiles.
    """
    linearcode_to_tag, tag_to_linearcode = utf.linearcode_to_tag_mapping(composition_df)
    stages_data = {}  # Dictionary to store data by stages
    estimated_enzymes_so_far = set()
    tagged_profile = experimental_profile.copy()  # Initialize tagged_profile here
    total_abundance = sum(experimental_profile.values())
    source_nodes = gfs.find_nodes_by_attribute(graph, 'LinearCode', tag_to_linearcode['M9'])
    observed_tags = set(experimental_profile.keys())
    observed_nodes = [gfs.find_nodes_by_attribute(graph, 'LinearCode', tag_to_linearcode[tag]) for tag in observed_tags]
    stage_number = 1  # Initialize stage number counter

    for critical_tag in critical_tags:
        critical_nodes = gfs.find_nodes_by_attribute(graph, 'LinearCode', tag_to_linearcode[critical_tag])
        print(f"\nProcessing {critical_tag} from {source_nodes} to {critical_nodes}")
        all_enzymes = set()
        unique_paths = set()
        stage_abundance = 0
        accounted_nodes = set()
        added_tags = set()

        for critical_node in critical_nodes:
            preceding_observed_nodes = find_preceding_observed_nodes(graph, critical_node, observed_nodes)
            print(f"  Preceding observed nodes for critical node {critical_node}: {preceding_observed_nodes}")

            for source_node in source_nodes:
                all_paths = list(nx.all_simple_paths(graph, source=source_node, target=critical_node))
                if not all_paths:
                    print(f"  Warning: No paths found from {source_node} to {critical_node}. This stage will not be processed.")
                    continue

                for path in all_paths:
                    path_tuple = tuple(path)
                    if path_tuple not in unique_paths:
                        unique_paths.add(path_tuple)
                        for i in range(len(path) - 1):
                            edge_data = graph.get_edge_data(path[i], path[i + 1])
                            if 'Enzyme' in edge_data and edge_data['Enzyme'] not in estimated_enzymes_so_far:
                                all_enzymes.add(edge_data['Enzyme'])

            for node in preceding_observed_nodes:
                node_tag = linearcode_to_tag.get(graph.nodes[node]['LinearCode'])
                if node_tag and node_tag in experimental_profile and node_tag not in added_tags:
                    stage_abundance += experimental_profile[node_tag]
                    added_tags.add(node_tag)
                    accounted_nodes.add(node)  # Mark this node as accounted for
                    print(f"    Adding {node_tag} abundance: {experimental_profile[node_tag]}; stage_abundance: {stage_abundance}")

        new_tagged_profile = tagged_profile.copy()
        for tag in tagged_profile.keys():
            tag_nodes = gfs.find_nodes_by_attribute(graph, 'LinearCode', tag_to_linearcode[tag])
            if not any(node in accounted_nodes for node in tag_nodes):
                new_tagged_profile[tag] = 0
            else:
                new_tagged_profile[tag] = experimental_profile.get(tag, 0)
        tagged_profile = new_tagged_profile

        if all_enzymes:
            remaining_abundance = total_abundance - stage_abundance
            print(f"Stage {stage_number} - total_abundance: {total_abundance}, stage_abundance: {stage_abundance}, remaining_abundance: {remaining_abundance}")

            for critical_node in critical_nodes:
                critical_node_tag = linearcode_to_tag.get(graph.nodes[critical_node]['LinearCode'])
                if critical_node_tag:
                    if critical_node_tag not in tagged_profile:
                        tagged_profile[critical_node_tag] = 0  # Initialize to zero if not present
                    tagged_profile[critical_node_tag] += remaining_abundance
                    break  # Assign remaining abundance to the first found critical node
                else:
                    print(f"  Warning: {critical_node_tag} not found in tagged_profile. Proceeding with remaining abundance assignment.")
                    
            print(f"  Tagged profile after assigning remaining abundance to critical node: {tagged_profile}")

            stages_data[f"Stage {stage_number}"] = {
                "Enzymes": list(all_enzymes),
                "Tagged Profile": tagged_profile.copy()
            }
            stage_number += 1
            estimated_enzymes_so_far.update(all_enzymes)

        source_nodes = critical_nodes

    # Final stage for enzymes not selected in previous stages
    remaining_enzymes = set(nx.get_edge_attributes(graph, 'Enzyme').values()) - estimated_enzymes_so_far
    if acceptable_enzymes is not None:
        remaining_enzymes &= set(acceptable_enzymes)
    if remaining_enzymes:
        stages_data[f"Stage {stage_number}"] = {
            "Enzymes": list(remaining_enzymes),
            "Tagged Profile": experimental_profile.copy()  # Use the original experimental profile
        }

    return stages_data

def print_stages_data(stages_data):
    """
    Print the enzymes and tagged profiles for each stage in the stages data.

    Parameters
    ----------
    stages_data : dict
        Dictionary containing stages data, where keys are stage names and values are dictionaries with 'Enzymes' and 'Tagged Profile'.

    Returns
    -------
    None
    """
    for stage, data in stages_data.items():
        print(f"{stage}:")
        print("  Enzymes:")
        enzymes = data.get("Enzymes", [])
        if enzymes:
            for enzyme in enzymes:
                print(f"    - {enzyme}")
        else:
            print("    - No enzymes listed.")

        print("  Tagged Profile:")
        tagged_profile = data.get("Tagged Profile", {})
        for tag, abundance in tagged_profile.items():
            print(f"    {tag}: {abundance}")

        print("\n")  # Add a newline for better separation between stages


def run_parameter_estimation(suffix, stages_data_igg, stages_data_hcp, df_from_pkl, input_p, get_var_est_params_bounds_dict, get_fixed_est_params, sampler, pop_size, min_eps, max_nr_pop):
    """
    Run parameter estimation for multiple stages and store the results.

    Parameters
    ----------
    suffix : str
        Suffix to append to the output filenames to uniquely identify them.
    stages_data_igg : dict
        Dictionary containing IgG data for different stages.
    stages_data_hcp : dict
        Dictionary containing HCP data for different stages.
    df_from_pkl : dict
        Dictionary containing various dataframes and numpy arrays structured for pathway analysis.
    input_p : dict
        Dictionary of input parameters required for the simulation.
    get_var_est_params_bounds_dict : function
        Function to get the variable estimated parameter bounds for a given stage.
    get_fixed_est_params : function
        Function to get the fixed estimated parameters for a given stage.
    sampler : object
        Sampler object for the ABC-SMC algorithm.
    pop_size : int
        Population size for the ABC-SMC algorithm.
    min_eps : float
        Minimum epsilon value for the ABC-SMC algorithm.
    max_nr_pop : int
        Maximum number of populations for the ABC-SMC algorithm.

    Returns
    -------
    dict
        Dictionary with stage numbers as keys and MAP estimates as values.
    dict
        Dictionary with stage numbers as keys and predicted values as values.
    """
    # Initialize dictionaries for storing results
    MAP_results = {}
    xpred_all_results = {}
    
    stage_number = 1  # Initialize stage number counter
    for stage in stages_data_igg.keys():
        enzymes = stages_data_igg[stage]['Enzymes']
        igg_data = stages_data_igg[stage]['Tagged Profile']
        hcp_data = stages_data_hcp[stage]['Tagged Profile']

        var_est_p_bounds_dict = get_var_est_params_bounds_dict(enzymes, stage_number)
        fixed_est_p = get_fixed_est_params(MAP_results, stage_number)
        
        print(f"\nRunning parameter estimation for stage {stage_number} with enzymes: {enzymes}")
        
        MAP_results[f'S{stage_number}'], xpred_all_results[f'S{stage_number}'] = afs.run_abc_smc(suffix, f'S{stage_number}', enzymes, igg_data, hcp_data, df_from_pkl, input_p, fixed_est_p, var_est_p_bounds_dict, sampler, pop_size, min_eps, max_nr_pop)
        
        stage_number += 1

    return MAP_results, xpred_all_results