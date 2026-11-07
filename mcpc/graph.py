"""Graphs use vertices 0, ..., n-1 and triples (endpoint, center, endpoint)."""

from dataclasses import dataclass, field
from itertools import combinations

import networkx as nx

Triple = tuple[int, int, int]


def triple(i, k, j):
    return (min(i, j), k, max(i, j))


def triples(skeleton):
    return sorted(triple(i, k, j) for k in skeleton
                  for i, j in combinations(sorted(skeleton[k]), 2)
                  if not skeleton.has_edge(i, j))


@dataclass
class Pattern:
    n: int
    directed: set[tuple[int, int]] = field(default_factory=set)
    undirected: set[tuple[int, int]] = field(default_factory=set)
    ambiguous: set[Triple] = field(default_factory=set)

    def skeleton(self):
        graph = nx.Graph()
        graph.add_nodes_from(range(self.n))
        graph.add_edges_from(self.directed | self.undirected)
        return graph

    def to_dict(self):
        return {"n": int(self.n), **{name: [list(map(int, item)) for item in sorted(getattr(self, name))]
                               for name in ("directed", "undirected", "ambiguous")}}

    @classmethod
    def from_dict(cls, record):
        return cls(int(record["n"]), **{
            name: {tuple(map(int, item)) for item in record.get(name, [])}
            for name in ("directed", "undirected", "ambiguous")})


def cpdag(dag):
    """Return the Markov equivalence class of a DAG."""
    from causallearn.graph.Dag import Dag
    from causallearn.graph.GraphNode import GraphNode
    from causallearn.utils.DAG2CPDAG import dag2cpdag

    n = len(dag)
    if set(dag) != set(range(n)) or not nx.is_directed_acyclic_graph(dag):
        raise ValueError("expected a DAG on vertices 0, ..., n-1")
    nodes = [GraphNode(str(i)) for i in range(n)]
    graph = Dag(nodes)
    for i, j in sorted(dag.edges()):
        graph.add_directed_edge(nodes[i], nodes[j])
    matrix = dag2cpdag(graph).graph
    result = Pattern(n)
    for i, j in combinations(range(n), 2):
        a, b = matrix[i, j], matrix[j, i]
        if a == b == -1:
            result.undirected.add((i, j))
        elif a == -1 and b == 1:
            result.directed.add((i, j))
        elif a == 1 and b == -1:
            result.directed.add((j, i))
    return result
