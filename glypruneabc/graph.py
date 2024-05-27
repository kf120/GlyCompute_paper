# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 12:03:21 2024

@author: kf120
"""
import json
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pyvis.network import Network
from glycowork.motif.processing import canonicalize_iupac
from glycowork.network.biosynthesis import construct_network

import glypruneabc.utils as utf


def save_graph_to_json(graph, json_filename):
    """
    Save a NetworkX graph to a JSON file using the node-link format.

    Parameters
    ----------
    graph : networkx.Graph
        The graph to be saved.
    json_filename : str
        The filename to save the graph in JSON format.

    Returns
    -------
    None

    Notes
    -----
    This function converts the NetworkX graph to node-link format and saves it as a JSON file.
    """
    data = nx.node_link_data(graph)
    with open(json_filename, 'w') as f:
        json.dump(data, f, indent=4)
        
def load_graph_from_json(json_filename):
    """
    Load a graph from a JSON file using NetworkX.

    Parameters
    ----------
    json_filename : str
        Path to the JSON file containing the graph data.

    Returns
    -------
    networkx.Graph
        The graph loaded from the JSON file.
    """
    with open(json_filename, 'r') as f:
        data = json.load(f)
        graph = nx.readwrite.json_graph.node_link_graph(data)
    return graph

def get_node_edge_count(graph, filename):
    """
    Print the number of nodes and edges in a given NetworkX graph.

    Parameters
    ----------
    graph : networkx.Graph
        The graph to be analyzed.
    filename : str
        The filename or identifier to include in the print statements.

    Returns
    -------
    None

    """
    print(f'Number of nodes {filename}:', graph.number_of_nodes())
    print(f'Number of edges {filename}:', graph.number_of_edges())


def add_nodes_and_edges(graph, extra_nodes, extra_edges):
    """
    Add extra nodes and edges to the graph.

    Parameters
    ----------
    graph : networkx.DiGraph
        The directed graph.
    extra_nodes : list of tuples
        List of extra nodes to be added, each as a tuple (node, attributes).
    extra_edges : list of tuples
        List of extra edges to be added, each as a tuple (node1, node2, attributes).

    Returns
    -------
    None
    """
    for node, attrs in extra_nodes:
        graph.add_node(node, **attrs)
    for edge in extra_edges:
        u, v, attrs = edge
        graph.add_edge(u, v, **attrs)

def remove_node_attributes(graph, attribute_names):
    """
    Remove multiple node attributes from all nodes in the graph.

    Parameters
    ----------
    graph : networkx.Graph
        The graph from which the node attributes will be removed.
    attribute_names : list of str
        The names of the attributes to remove.

    Raises
    ------
    TypeError
        If attribute_names is not a list.
    """
    if not isinstance(attribute_names, list):
        raise TypeError("attribute_names must be a list")

    for node in graph.nodes():
        for attribute_name in attribute_names:
            if attribute_name in graph.nodes[node]:
                del graph.nodes[node][attribute_name]

def remove_edge_attributes(graph, attribute_names):
    """
    Remove multiple edge attributes from all edges in the graph.

    Parameters
    ----------
    graph : networkx.Graph
        The graph from which the edge attributes will be removed.
    attribute_names : list of str
        The names of the attributes to remove.

    Raises
    ------
    TypeError
        If attribute_names is not a list.
    """
    if not isinstance(attribute_names, list):
        raise TypeError("attribute_names must be a list")

    for u, v in graph.edges():
        for attribute_name in attribute_names:
            if attribute_name in graph[u][v]:
                del graph[u][v][attribute_name]
                
def add_linearcode_and_update_labels(graph):
    """
    Add LinearCode attribute to each node in the graph based on its IUPAC-condensed name,
    and update the node labels to use the LinearCode attribute.

    Parameters
    ----------
    graph : networkx.DiGraph
        The directed graph with nodes named using IUPAC-condensed notation.

    Returns
    -------
    None
    """
    mapping = {}
    for node in graph.nodes():
        try:
            linear_code = utf.iupac_to_linearcode(node)
            graph.nodes[node]['LinearCode'] = linear_code
            mapping[node] = linear_code
        except ValueError as e:
            print(f"Error converting node {node}: {e}")
    
    # Relabel the nodes using the mapping
    nx.relabel_nodes(graph, mapping, copy=False)
    
def add_enzyme_edge_attributes(graph):
    """
    Add enzyme attribute to each edge in the graph based on the difference between LinearCode of connected nodes.

    Parameters
    ----------
    graph : networkx.DiGraph
        The directed graph with nodes having LinearCode attribute.

    Returns
    -------
    None
    """
    for u, v in graph.edges():
        substrate = graph.nodes[u].get('LinearCode')
        product = graph.nodes[v].get('LinearCode')
        if substrate and product:
            enzyme = utf.infer_enzyme(substrate, product)
            graph[u][v]['Enzyme'] = enzyme


def add_reaction_index_edge_attributes(graph, start_index=1):
    """
    Add 'Reaction Index' attribute to all edges in the graph.

    Parameters
    ----------
    graph : networkx.DiGraph
        The directed graph.
    start_index : int, optional
        The starting index for the 'Reaction Index' attribute (default is 1).

    Returns
    -------
    None
    """
    # Find the highest existing Reaction Index
    max_index = start_index - 1
    for _, _, attrs in graph.edges(data=True):
        if 'Reaction Index' in attrs:
            try:
                index = int(attrs['Reaction Index'][2:])
                if index > max_index:
                    max_index = index
            except ValueError:
                continue

    # Assign Reaction Index to all edges without it
    reaction_index = max_index + 1
    for u, v, attrs in graph.edges(data=True):
        if 'Reaction Index' not in attrs:
            attrs['Reaction Index'] = f'NR{reaction_index}'
            reaction_index += 1
            
def rename_nodes_topologically(graph):
    """
    Rename the nodes of the graph graph based on topological sorting.

    Parameters
    ----------
    graph : networkx.DiGraph
        The directed graph.

    Returns
    -------
    networkx.DiGraph
        The graph with renamed nodes.
    dict
        A dictionary mapping old node names to new node names.
    """
    # Perform topological sorting
    topo_sort = list(nx.topological_sort(graph))
    
    # Create a mapping from old node names to new node names
    mapping = {old_name: f'N{i+1}' for i, old_name in enumerate(topo_sort)}
    
    # Relabel nodes in the graph
    graph = nx.relabel_nodes(graph, mapping)
    
    return graph, mapping


def characterize_directed_graph(graph, source_node, return_csv=True):
    """
    Calculate and save various metrics for a directed graph.

    Parameters
    ----------
    graph : networkx.DiGraph
        The directed graph to characterize.
    source_node : node
        The source node for dominance frontier calculations.
    return_csv : bool, optional
        Whether to save the metrics to CSV files (default is True).

    Returns
    -------
    tuple
        A tuple containing three pandas.DataFrames:
        - df_eval_node: DataFrame with node-specific metrics.
        - df_eval_edge: DataFrame with edge-specific metrics.
        - df_eval_graph: DataFrame with whole graph metrics.
    """
    # Node-specific metrics
    in_degrees = dict(graph.in_degree())
    out_degrees = dict(graph.out_degree())
    closeness_centrality = nx.closeness_centrality(graph)
    betweenness_centrality = nx.betweenness_centrality(graph)
    pagerank = nx.pagerank(graph)
    hubs, authorities = nx.hits(graph, max_iter=500)
    clustering_coeffs = nx.clustering(graph)
    dominance_frontiers = nx.dominance_frontiers(graph, source_node)
    dominance_frontier_list = {node: list(frontier) for node, frontier in dominance_frontiers.items()}
    dominance_frontier_sizes = {node: len(frontier) for node, frontier in dominance_frontiers.items()}

    # Edge betweenness centrality
    edge_betweenness_centrality = nx.edge_betweenness_centrality(graph)

    # Whole graph metrics
    density = nx.density(graph)
    degree_assortativity = nx.degree_assortativity_coefficient(graph)
    diameter = nx.diameter(graph.to_undirected()) if nx.is_weakly_connected(graph) else float('inf')
    strong_components = list(nx.strongly_connected_components(graph))
    weak_components = list(nx.weakly_connected_components(graph))
    avg_clustering = nx.average_clustering(graph)

    # DataFrames creation
    df_eval_node = pd.DataFrame({
        "Node": list(graph.nodes()),
        **{attr: [data.get(attr, None) for node, data in graph.nodes(data=True)] for attr in set().union(*(d.keys() for _, d in graph.nodes(data=True)))},
        "In-Degree": [in_degrees[node] for node in graph.nodes()],
        "Out-Degree": [out_degrees[node] for node in graph.nodes()],
        "Closeness Centrality": [closeness_centrality[node] for node in graph.nodes()],
        "Betweenness Centrality": [betweenness_centrality[node] for node in graph.nodes()],
        "PageRank": [pagerank[node] for node in graph.nodes()],
        "Hub Score": [hubs[node] for node in graph.nodes()],
        "Authority Score": [authorities[node] for node in graph.nodes()],
        "Clustering Coefficient": [clustering_coeffs[node] for node in graph.nodes()],
        "Dominance Frontier": [dominance_frontier_list[node] for node in graph.nodes()],
        "Dominance Frontier Size": [dominance_frontier_sizes[node] for node in graph.nodes()]
    })

    df_eval_edge = pd.DataFrame({
        "Edge": list(edge_betweenness_centrality.keys()),
        **{attr: [data.get(attr, None) for _, _, data in graph.edges(data=True)] for attr in set().union(*(d.keys() for _, _, d in graph.edges(data=True)))},
        "Edge Betweenness": list(edge_betweenness_centrality.values())
    })

    df_eval_graph = pd.DataFrame({
        "Metric": ["Density", "Assortativity Coefficient", "Diameter", "Number of Strong Components", "Number of Weak Components", "Average Clustering Coefficient"],
        "Value": [density, degree_assortativity, diameter, len(strong_components), len(weak_components), avg_clustering]
    })
    
    if return_csv:
        
        # Save DataFrames to CSV files for further analysis
        save_node_csv="directed_node_metrics.csv"
        save_edge_csv="directed_edge_metrics.csv"
        save_graph_csv="directed_graph_metrics.csv"
        
        df_eval_node.to_csv(save_node_csv, index=False)
        df_eval_edge.to_csv(save_edge_csv, index=False)
        df_eval_graph.to_csv(save_graph_csv, index=False)

    return df_eval_node, df_eval_edge, df_eval_graph

def visualize_graph(graph, name):
    """
    Visualizes a given NetworkX graph using Pyvis with hierarchical layout and customized node and edge tooltips.

    Parameters
    ----------
    graph : networkx.Graph
        The graph to visualize.
    name : str
        The base name for the output HTML file where the visualization will be saved.

    Returns
    -------
    None
    """
    # Create a Pyvis network object
    nt = Network(height='800px', width='100%', directed=True)

    # Add nodes with tooltips based on node attributes
    for node, attrs in graph.nodes.data():
        code = attrs.get('LinearCode', '')
        trait = attrs.get('Trait', '')
        title = f"Node: {node}\nLinearCode: {code}\nTrait: {trait}"
        nt.add_node(node, title=title)

    # Add edges with tooltips based on edge attributes
    for u, v, attrs in graph.edges.data():
        idx = attrs.get('Reaction Index', '')
        enzyme = attrs.get('Enzyme', '')
        genes = attrs.get('Genes', '')
        title = f'Reaction Index: {idx}\nEnzyme: {enzyme}\nGenes: {genes}'
        nt.add_edge(u, v, title=title)

    # Configure the hierarchical layout
    options = {
        'layout': {
            'hierarchical': {
                'enabled': True,
                'levelSeparation': 150,
                'nodeSpacing': 100,
                'treeSpacing': 200,
                'direction': 'UD',
                'sortMethod': 'directed'
            }
        }
    }

    # Apply the options
    options_json = json.dumps(options)
    nt.set_options(options_json)

    # Save and show the visualization
    nt.show(f'{name}.html', notebook=False)

def visualize_graph_with_legend(graph, name):
    """
    Visualize a directed graph with hierarchical layout and enzyme-based edge coloring, and creates a legend for enzyme reaction counts.

    Parameters
    ----------
    graph : networkx.DiGraph
        The graph to be visualized.
    name : str
        Base name for saving the visualization (HTML) and the legend (PNG).

    Returns
    -------
    None
    """
    # Create a pyvis network object
    nt = Network(width="1000px", height="750px", directed=True)

    # Load the graph
    nt.from_nx(graph)
    
    # Hide node labels
    for node in nt.nodes:
        node['label'] = ''
    
    options = {
        'layout': {
            'hierarchical': {
                'enabled': True,
                'levelSeparation': 150,  # vertical distance between levels
                'nodeSpacing': 100,     # horizontal distance between nodes
                'treeSpacing': 200,     # distance between different trees in the forest
                'direction': 'UD',      # direction of layout
                'sortMethod': 'directed' # method for positioning nodes at different levels
            }
        }
    }

    # Convert the options dictionary to a JSON-formatted string
    options_str = json.dumps(options)
    nt.set_options(options_str)

    # Set the edge colors based on enzyme attribute
    edge_colors = {
        "GnTI": "blue",
        "GnTII": "green",
        "GnTIV": "red",
        "GnTV": "cyan",
        "ManI": "#DAA520",
        "ManII": "orange",
        "a3FucT": "purple",
        "a3SiaT": "grey",
        "a6FucT": "pink",
        "b4GalT": "brown",
        "iGnT": "lightblue"
    }

    for edge in nt.edges:
        enzyme = edge['Enzyme']
        if enzyme in edge_colors:
            edge['color'] = edge_colors[enzyme]

    # Show the graph
    nt.show(f"viz_{name}.html", notebook=False)

    # Create legend
    enzymes = list(edge_colors.keys())
    colors = list(edge_colors.values())

    # Count the number of reactions for each enzyme
    reaction_counts = {enzyme: sum(1 for _, _, data in graph.edges(data=True) if data.get('Enzyme') == enzyme) for enzyme in enzymes}

    # Generate the legend patches, excluding enzymes with reaction_counts of 0
    legend_patches = [mpatches.Patch(color=color, label=f"{enzyme} ({reaction_counts[enzyme]} reactions)") for enzyme, color in zip(enzymes, colors) if reaction_counts[enzyme] > 0]

    # Plot the legend
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.axis('off')
    ax.legend(handles=legend_patches, title="Enzyme (#Reactions)", loc='center', fontsize='medium', title_fontsize='large')
    plt.tight_layout()

    # Save the legend as an image
    plt.savefig(f"legend_{name}.png", dpi=600, bbox_inches='tight')
    plt.show()


def construct_graph_with_shortest_path_finding(G, node_attr, source_value, sink_values, json_filename):
    """
    Generate a graph to include only the nodes that are part of the shortest paths from
    the source node to any of the sink nodes. Saves the resulting graph as a JSON file.

    Parameters
    ----------
    G : networkx.Graph
        The original graph.
    node_attr : str
        Attribute name to identify the nodes.
    source_value : str
        Attribute value for the source node.
    sink_values : list of str
        List of attribute values for sink nodes.
    json_filename : str
        The filename to save the pruned graph in JSON format.

    Returns
    -------
    networkx.Graph
        The pruned subgraph.
    """
    # Find source and sink nodes based on attributes
    source_nodes = [n for n, attr in G.nodes(data=True) if attr.get(node_attr) == source_value]
    if not source_nodes:
        raise ValueError("No source node found; check the source_value.")
    if len(source_nodes) > 1:
        raise ValueError("Multiple source nodes found; ensure unique source_value.")
    source_node = source_nodes[0]

    # Ensure sink values is a list with unique entries
    if isinstance(sink_values, str):
        sink_values = [sink_values]
    sink_nodes = [n for n, attr in G.nodes(data=True) if attr.get(node_attr) in sink_values]

    # Find all nodes reachable via the shortest path from source_node to any sink_node
    reachable_nodes = set()
    for sink_node in sink_nodes:
        try:
            paths = nx.all_shortest_paths(G, source=source_node, target=sink_node)
            for path in paths:
                reachable_nodes.update(path)
        except nx.NetworkXNoPath:
            print(f"No path between {source_nodes} and {sink_node}.")
            continue

    # Create the subgraph
    mG = G.subgraph(reachable_nodes).copy()
    
    # Return the node and edge counts of the subgraph
    get_node_edge_count(mG, json_filename)
    
    # Rename nodes based on topological sorting
    mG, node_mapping = rename_nodes_topologically(mG)
    
    # Save the graph to a JSON file
    save_graph_to_json(mG, json_filename)
    
    return mG

def construct_graph_with_minimum_flow_reachability(G, node_attr, source_value, sink_values, json_filename):
    """
    Construct a graph based on minimum flow reachability from a source node to multiple sink nodes.
    Saves the resulting graph as a JSON file.

    Parameters
    ----------
    G : networkx.Graph
        The original graph.
    node_attr : str
        Attribute name to identify the nodes.
    source_value : str
        Attribute value for the source node.
    sink_values : list of str
        List of attribute values for sink nodes.
    json_filename : str
        The filename to save the pruned graph in JSON format.

    Returns
    -------
    networkx.Graph
        The pruned subgraph, if successful.
    """
    # Find source and sink nodes based on attributes
    source_nodes = [n for n, attr in G.nodes(data=True) if attr.get(node_attr) == source_value]
    if not source_nodes:
        raise ValueError("No source node found; check the source_value.")
    if len(source_nodes) > 1:
        raise ValueError("Multiple source nodes found; ensure unique source_value.")
    source_node = source_nodes[0]
    
    # Ensure sink values is a list with unique entries
    if isinstance(sink_values, str):
        sink_values = [sink_values]
    sink_nodes = [n for n, attr in G.nodes(data=True) if attr.get(node_attr) in sink_values]

    # Create a graph for the flow problem
    flow_graph = G.copy()

    # Add a super source and super sink
    super_source = 'SuperSource'
    super_sink = 'SuperSink'
    flow_graph.add_node(super_source)
    flow_graph.add_node(super_sink)

    # Connect the super source to the actual source
    flow_graph.add_edge(super_source, source_node, capacity=float('inf'))

    # Connect each sink to the super sink with specific capacities indicating the minimum required flow
    for sink in sink_nodes:
        flow_graph.add_edge(sink, super_sink, capacity=1)  # Each sink must receive exactly one unit of flow

    # Compute the maximum flow from the super source to the super sink
    flow_value, flow_dict = nx.maximum_flow(flow_graph, super_source, super_sink)

    # Check if all sinks receive exactly one unit of flow
    if all(flow_dict[sink][super_sink] == 1 for sink in sink_nodes) and flow_value == len(sink_nodes):
        # Extract the subgraph that includes all nodes and edges involved in any flow-carrying path
        flow_edges = [(u, v) for u in flow_dict for v in flow_dict[u] if flow_dict[u][v] > 0]
        mG = flow_graph.edge_subgraph(flow_edges).copy()
        mG.remove_nodes_from([super_source, super_sink])
        
        # Check that the subgraph contains all sink nodes
        subgraph_sink_values = [attr[node_attr] for n, attr in mG.nodes(data=True) if attr.get(node_attr) in sink_values]
        if set(sink_values) == set(subgraph_sink_values):
            # Remove terminal nodes that are only one step removed from the source node
            nodes_to_remove = []
            for node in list(mG.nodes()):
                if mG.degree[node] == 1 and source_node in mG.neighbors(node):
                    nodes_to_remove.append(node)
            mG.remove_nodes_from(nodes_to_remove)

            # Check again if the subgraph contains all sink nodes after removal
            subgraph_sink_values = [attr[node_attr] for n, attr in mG.nodes(data=True) if attr.get(node_attr) in sink_values]
            if set(sink_values) == set(subgraph_sink_values):
                # Return the node and edge counts of the subgraph
                get_node_edge_count(mG, json_filename)
                
                # Rename nodes based on topological sorting
                mG, node_mapping = rename_nodes_topologically(mG)

                # Save the graph to a JSON file
                save_graph_to_json(mG, json_filename)
                
                return mG
            else:
                raise Exception("The final subgraph does not contain all sink nodes after removing terminal nodes.")
        else:
            raise Exception("The final subgraph does not contain all sink nodes.")
    else:
        raise Exception("Flow conditions not met; not all sinks are reachable with the required flow.")

def construct_graph_with_glycowork(source_value, sink_values, json_filename):
    """
    Generate graph using Glycowork and append the initial pathway designated by 
    ManI, GnTI, ManII. Saves the resulting graph as a JSON file.

    Parameters
    ----------
    source_value : str
        The IUPAC-condensed notation of the source node.
    sink_values : list of str
        List of IUPAC-condensed notations for sink nodes.
    json_filename : str
        The filename to save the pruned graph in JSON format.

    Returns
    -------
    networkx.Graph
        The pruned and updated graph with additional nodes, edges, and attributes.

    Notes
    -----
    Glycowork can construct a network of glycosyltransferases only, i.e., it can only add monosaccharides,
    but it cannot remove monosaccharides (glycosidases). Preprocessing was done to construct a graph compatible
    with the rest of the codebase.
    """
    obs_glycans = sink_values
    obs_glycans_iupac = []
    for i in range(len(obs_glycans)):
        obs_glycan_iupac = canonicalize_iupac(obs_glycans[i])
        obs_glycans_iupac.append(obs_glycan_iupac)
    
    mG_init = construct_network(obs_glycans_iupac, edge_type='enzyme', permitted_roots=canonicalize_iupac(source_value))
    
    # Preprocess generated graph
    add_linearcode_and_update_labels(mG_init)
    add_enzyme_edge_attributes(mG_init)
    remove_node_attributes(mG_init, ['IUPAC'])
    remove_edge_attributes(mG_init, ['weight', 'diffs'])
    
    # Manually add pathway designated by ManI, GnTI, ManII
    extra_nodes = [
        ('Ma2Ma2Ma3(Ma2Ma3(Ma2Ma6)Ma6)Mb4GNb4GN;', {'LinearCode': 'Ma2Ma2Ma3(Ma2Ma3(Ma2Ma6)Ma6)Mb4GNb4GN;'}),
        ('Ma2Ma3(Ma2Ma3(Ma2Ma6)Ma6)Mb4GNb4GN;', {'LinearCode': 'Ma2Ma3(Ma2Ma3(Ma2Ma6)Ma6)Mb4GNb4GN;'}),
        ('Ma2Ma2Ma3(Ma2Ma3(Ma6)Ma6)Mb4GNb4GN;', {'LinearCode': 'Ma2Ma2Ma3(Ma2Ma3(Ma6)Ma6)Mb4GNb4GN;'}),
        ('Ma3(Ma2Ma3(Ma2Ma6)Ma6)Mb4GNb4GN;', {'LinearCode': 'Ma3(Ma2Ma3(Ma2Ma6)Ma6)Mb4GNb4GN;'}),
        ('Ma2Ma3(Ma2Ma3(Ma6)Ma6)Mb4GNb4GN;', {'LinearCode': 'Ma2Ma3(Ma2Ma3(Ma6)Ma6)Mb4GNb4GN;'}),
        ('Ma3(Ma2Ma3(Ma6)Ma6)Mb4GNb4GN;', {'LinearCode': 'Ma3(Ma2Ma3(Ma6)Ma6)Mb4GNb4GN;'}),
        ('Ma3(Ma3(Ma6)Ma6)Mb4GNb4GN;', {'LinearCode': 'Ma3(Ma3(Ma6)Ma6)Mb4GNb4GN;'}),
        ('GNb2Ma3(Ma3(Ma6)Ma6)Mb4GNb4GN;', {'LinearCode': 'GNb2Ma3(Ma3(Ma6)Ma6)Mb4GNb4GN;'}),
        ('GNb2Ma3(Ma6Ma6)Mb4GNb4GN;', {'LinearCode': 'GNb2Ma3(Ma6Ma6)Mb4GNb4GN;'}),
        ('GNb2Ma3(Ma6)Mb4GNb4GN;', {'LinearCode': 'GNb2Ma3(Ma6)Mb4GNb4GN;'})
    ]

    extra_edges = [
        ('Ma2Ma2Ma3(Ma2Ma3(Ma2Ma6)Ma6)Mb4GNb4GN;', 'Ma2Ma3(Ma2Ma3(Ma2Ma6)Ma6)Mb4GNb4GN;', {'Reaction Index': 'NR1', 'Enzyme': 'ManI'}), 
        ('Ma2Ma2Ma3(Ma2Ma3(Ma2Ma6)Ma6)Mb4GNb4GN;', 'Ma2Ma2Ma3(Ma2Ma3(Ma6)Ma6)Mb4GNb4GN;', {'Reaction Index': 'NR2', 'Enzyme': 'ManI'}),
        ('Ma2Ma3(Ma2Ma3(Ma2Ma6)Ma6)Mb4GNb4GN;', 'Ma3(Ma2Ma3(Ma2Ma6)Ma6)Mb4GNb4GN;', {'Reaction Index': 'NR3', 'Enzyme': 'ManI'}),
        ('Ma2Ma3(Ma2Ma3(Ma2Ma6)Ma6)Mb4GNb4GN;', 'Ma2Ma3(Ma2Ma3(Ma6)Ma6)Mb4GNb4GN;', {'Reaction Index': 'NR4', 'Enzyme': 'ManI'}),
        ('Ma2Ma2Ma3(Ma2Ma3(Ma6)Ma6)Mb4GNb4GN;', 'Ma2Ma3(Ma2Ma3(Ma6)Ma6)Mb4GNb4GN;', {'Reaction Index': 'NR5', 'Enzyme': 'ManI'}),
        ('Ma3(Ma2Ma3(Ma2Ma6)Ma6)Mb4GNb4GN;', 'Ma3(Ma2Ma3(Ma6)Ma6)Mb4GNb4GN;', {'Reaction Index': 'NR6', 'Enzyme': 'ManI'}),
        ('Ma2Ma3(Ma2Ma3(Ma6)Ma6)Mb4GNb4GN;', 'Ma3(Ma2Ma3(Ma6)Ma6)Mb4GNb4GN;', {'Reaction Index': 'NR7', 'Enzyme': 'ManI'}),
        ('Ma3(Ma2Ma3(Ma6)Ma6)Mb4GNb4GN;', 'Ma3(Ma3(Ma6)Ma6)Mb4GNb4GN;', {'Reaction Index': 'NR8', 'Enzyme': 'ManI'}),
        ('Ma3(Ma3(Ma6)Ma6)Mb4GNb4GN;', 'GNb2Ma3(Ma3(Ma6)Ma6)Mb4GNb4GN;', {'Reaction Index': 'NR9', 'Enzyme': 'GnTI'}),
        ('GNb2Ma3(Ma3(Ma6)Ma6)Mb4GNb4GN;', 'GNb2Ma3(Ma6Ma6)Mb4GNb4GN;', {'Reaction Index': 'NR10', 'Enzyme': 'ManII'}),
        ('GNb2Ma3(Ma6Ma6)Mb4GNb4GN;', 'GNb2Ma3(Ma6)Mb4GNb4GN;', {'Reaction Index': 'NR11', 'Enzyme': 'ManII'}),
        ('GNb2Ma3(Ma6)Mb4GNb4GN;', 'GNb2Ma3(GNb2Ma6)Mb4GNb4GN;', {'Reaction Index': 'NR12', 'Enzyme': 'GnTII'}),
        ('GNb2Ma3(Ma6)Mb4GNb4GN;', 'GNb2Ma3(Ma6)Mb4GNb4(Fa6)GN;', {'Reaction Index': 'NR13', 'Enzyme': 'a6FucT'}),
    ]
    
    # Add extra nodes and edges
    add_nodes_and_edges(mG_init, extra_nodes, extra_edges)

    # Rename nodes based on topological sorting
    mG, node_mapping = rename_nodes_topologically(mG_init)

    # Add Reaction Index to all edges
    add_reaction_index_edge_attributes(mG)
    
    # Return the node and edge counts of the graph
    get_node_edge_count(mG, json_filename)

    # Save the graph to a JSON file
    save_graph_to_json(mG, json_filename)
    
    return mG

        
def compare_graphs(graphs, graph_names):
    """
    Compare a list of NetworkX graphs on various similarity metrics and return a DataFrame with the results.

    Parameters
    ----------
    graphs : list of networkx.Graph
        The list of graph objects to compare.
    graph_names : list of str
        The list of graph names corresponding to the graphs.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the comparison results.
    """
    # Initialize an empty list to store the comparison results
    comparison_results = []

    # Iterate over all pairs of graphs
    for i, graph1 in enumerate(graphs):
        for j, graph2 in enumerate(graphs):
            if i < j:
                # Calculate similarities and differences
                node_overlap = len(set(graph1.nodes()).intersection(set(graph2.nodes())))
                edge_overlap = len(set(graph1.edges()).intersection(set(graph2.edges())))
                is_isomorphic = nx.is_isomorphic(graph1, graph2)
                
                try:
                    graph_edit_distance = next(nx.optimize_graph_edit_distance(graph1, graph2))
                except StopIteration:
                    graph_edit_distance = np.nan

                # Normalize GED by the average number of edges
                avg_edges = (graph1.number_of_edges() + graph2.number_of_edges()) / 2
                normalized_ged_by_edges = graph_edit_distance / avg_edges if avg_edges != 0 else np.nan

                # Check for subgraph isomorphism
                is_subgraph1_in_2 = nx.algorithms.isomorphism.GraphMatcher(graph2, graph1).subgraph_is_isomorphic()
                is_subgraph2_in_1 = nx.algorithms.isomorphism.GraphMatcher(graph1, graph2).subgraph_is_isomorphic()

                # Append the results for this pair of graphs to the comparison_results list
                comparison_results.append({
                    'Graph Pair': f'{graph_names[i]} - {graph_names[j]}',
                    'Node Overlap': node_overlap,
                    'Edge Overlap': edge_overlap,
                    'Isomorphic': is_isomorphic,
                    'Graph Edit Distance': graph_edit_distance,
                    'Normalized GED by Edges': normalized_ged_by_edges,
                    f'{graph_names[i]} Subgraph of {graph_names[j]}': is_subgraph1_in_2,
                    f'{graph_names[j]} Subgraph of {graph_names[i]}': is_subgraph2_in_1
                })

    # Create a pandas DataFrame from the results
    df = pd.DataFrame(comparison_results)
    return df

def find_nodes_by_attribute(graph, attribute_name, attribute_values):
    """
    Find all nodes in the graph where the node's attribute `attribute_name` matches any of the `attribute_values`.

    Parameters
    ----------
    graph : networkx.Graph
        The graph to search within.
    attribute_name : str
        The name of the attribute to match.
    attribute_values : list or str
        A list of values or a single value of the attribute to match.

    Returns
    -------
    list
        A list of nodes for which the attribute matches any of the given values.
    """
    matched_nodes = []
    for node, attrs in graph.nodes(data=True):
        if attrs.get(attribute_name) in attribute_values:
            matched_nodes.append(node)
    
    return matched_nodes