#!/usr/bin/env python

import numpy as np
import pandas as pd
from collections import defaultdict
import statsmodels.api as sm
from statsmodels.formula.api import ols
import random


class Graph():
    """
    A class for creating graph objects. A graph will be
    defined by (i) a set of vertices and (ii) a dictionary
    mapping each vertex Vi to its set of parents pa_G(Vi).
    """

    def __init__(self, vertices, edges=set()):
        """
        Constructor for the Graph class. A Graph is created
        by accepting a set of vertices and edges as inputs

        Inputs:
        vertices: a set/list of vertex names
        edges: a set/list of tuples for example [(Vi, Vj), (Vj, Vk)] where the tuple (Vi, Vj) indicates Vi->Vj exists in G
        """
        self.vertices = set(vertices)
        self.parents = defaultdict(set)
        for parent, child in edges:
            self.parents[child].add(parent)

    def add_edge(self, parent, child):
        """
        Function to add an edge to the graph from parent -> child
        """

        self.parents[child].add(parent)

    def delete_edge(self, parent, child):
        """
        Function to delete an edge to the graph from parent -> child
        """

        if parent in self.parents[child]:
            self.parents[child].remove(parent)

    def edges(self):
        """
        Returns a list of tuples [(Vi, Vj), (Vx, Vy),...] corresponding to edges
        present in the graph
        """

        edges = []
        for v in self.vertices:
            edges.extend([(p, v) for p in self.parents[v]])

        return edges

    def produce_visualization_code(self, filename):
        """
        Function that outputs a text file with the necessary graphviz
        code that can be pasted into https://dreampuf.github.io/GraphvizOnline/
        to visualize the graph.
        """

        # set up a Digraph object in graphviz
        gviz_file = open(filename, "w")
        gviz_file.write("Digraph G { \n")

        # iterate over all vertices and add them to the graph
        for v in self.vertices:
            gviz_file.write('  {} [shape="plaintext"];\n'.format(v))

        # add edges between the vertices
        for v in self.vertices:
            for p in self.parents[v]:
                gviz_file.write('  {} -> {} [color="blue"];\n'.format(p, v))

        # close the object definition and close the file
        gviz_file.write("}\n")
        gviz_file.close()

def acyclic(G):
    """
    A function that uses depth first traversal to determine whether the
    graph G is acyclic.
    """

    return True

def bic_score(G, data):
    """
    Compute the BIC score for a given graph G and a dataset as a pandas data frame.

    Inputs:
    G: a Graph object as defined by the Graph class above
    data: a pandas data frame
    """

    return 1

def causal_discovery(data, num_steps=50):
    """
    Take in data and perform causal discovery according to a set of moves
    described in the write up for a given number of steps.
    """

    # initalize an empty graph as the optimal one and gets its BIC score
    G_star = Graph(vertices=data.columns)
    bic_star = bic_score(G_star, data)

    # forward phase of causal discovery:
    for i in range(num_steps):

        # attempt a random edge addition that does not create a cycle
        # if it improves the BIC score, update G_star and bic_star
        pass

    # backward phase of causal discovery
    for i in range(num_steps):

        # attempt a random edge deletion/reversal
        # pick the move that improves the BIC score (if any)
        pass

    return G_star



################################################
# Tests for your acyclic function
################################################

# G = X->Y<-Z, Z->X
G1 = Graph(vertices=["X", "Y", "Z"], edges=[("X", "Y"), ("Z", "Y"), ("Z", "X")])

# X->Y->Z, Z->X
G2 = Graph(vertices=["X", "Y", "Z"], edges=[("X", "Y"), ("Y", "Z"), ("Z", "X")])

# X->Y->Z, Y->Y
G3 = Graph(vertices=["X", "Y", "Z"], edges=[("X", "Y"), ("Y", "Z"), ("Y", "Y")])

print(acyclic(G1))
print(acyclic(G2))
print(acyclic(G3))


################################################
# Tests for your bic_score function
################################################
data = pd.read_csv("bic_test_data.txt")

# fit model for G1: A->B->C->D, B->D and get BIC
G1 = Graph(vertices=["A", "B", "C", "D"], edges=[("A", "B"), ("B", "C"), ("C", "D"), ("B", "D")])
print(bic_score(G1, data), acyclic(G1))
G1.produce_visualization_code("G1_viz.txt")

# fit model for G2: A<-B->C->D, B->D and get BIC
G2 = Graph(vertices=["A", "B", "C", "D"], edges=[("B", "A"), ("B", "C"), ("C", "D"), ("B", "D")])
print(bic_score(G2, data), acyclic(G2))

# fit model for G3: A->B<-C->D, B->D and get BIC
G3 = Graph(vertices=["A", "B", "C", "D"], edges=[("A", "B"), ("C", "B"), ("C", "D"), ("B", "D")])
print(bic_score(G3, data), acyclic(G3))

# fit model for G4: A<-B->C<-D, B->D and get BIC
G4 = Graph(vertices=["A", "B", "C", "D"], edges=[("B", "A"), ("B", "C"), ("D", "C"), ("B", "D")])
print(bic_score(G4, data), acyclic(G4))



################################################
# Tests for your causal_discovery function
################################################
np.random.seed(1000)
random.seed(100)
data = pd.read_csv("data.txt")
G_opt = causal_discovery(data)
# you can paste the code in protein_viz.txt into the online interface of Graphviz
G_opt.produce_visualization_code("protein_viz.txt")
