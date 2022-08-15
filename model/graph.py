import collections
from typing import List, Optional, Tuple

import networkx as nx
from networkx import Graph

# Definition for a Node type used in NodeBlock and GraphBlock modules
Node = collections.namedtuple('Node', ['id', 'node_type', 'inputs'])


def get_graph_info(graph: Graph) -> Tuple[List[Node], List[int], List[int]]:
    """Return a list of Node objects given a connected Watts-Strogatz graph.

    Args:
        graph (Graph): The Watts-Strogatz graph to retrieve the nodes from.

    Returns:
        List[Node]: List of nodes in the graph.
        List[int]: List of input nodes.
        List[int]]: List of output nodes.
    """
    inputs, outputs, nodes = [], [], []

    for i in range(graph.number_of_nodes()):
        neighbours = list(graph.neighbors(i))
        node_type = 'intermediate'

        if i < min(neighbours):
            inputs.append(i)
            node_type = 'input'
        elif i > max(neighbours):
            outputs.append(i)
            node_type = 'output'

        node = Node(i, node_type, [n for n in neighbours if n < i])
        nodes.append(node)

    return nodes, inputs, outputs


def build_graph(nodes: int, k: int, p: float,
                seed: Optional[int] = None) -> Graph:
    """Build a Watts-Strogatz graph."""
    return nx.connected_watts_strogatz_graph(nodes, k, p, seed=seed)


def save_graph(graph: Graph, path: str) -> None:
    """Save graph as a `gpickle` file."""
    nx.write_gpickle(graph, path)


def load_graph(path: str) -> Graph:
    """Load a graph from a `gpickle` file."""
    return nx.read_gpickle(path)
