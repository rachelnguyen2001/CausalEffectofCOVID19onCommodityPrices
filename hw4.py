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

def depth_first_search(G, v, visited, current_path):
    visited.add(v)
    current_path.add(v)
    children = set()
    edges = G.edges()
    
    for u in G.vertices:
        if (v, u) in edges:
            if u in current_path:
                return True
            if u not in visited:
                children.add(u)

    for c in children:
        if depth_first_search(G, c, visited, current_path):
            return True
            
    current_path.remove(v)
    return False

def acyclic(G):
    """
    A function that uses depth first traversal to determine whether the
    graph G is acyclic.
    """
    visited = set()
    current_path = set()

    for v in G.vertices:
            if v not in visited and depth_first_search(G, v, visited, current_path):
                return False

    return True

def bic_score_helper(G, data, v):
    formula = v + ' ~ 1'
    parents = G.parents[v]

    for p in parents:
        formula += ' + '
        formula += p

    model = ols(formula=formula, data=data).fit()
    loglikelihood = model.llf
    bic = -2*loglikelihood + len(model.params) * np.log(data.shape[0])
    return bic

def is_pair_valid(G, edges, V):
    if (V[0], V[1]) in edges:
        return False
    else:
        G.add_edge(V[0], V[1])

        if acyclic(G):
            return True
        else:
            G.delete_edge(V[0], V[1])
            return False

def bic_score(G, data):
    """
    Compute the BIC score for a given graph G and a dataset as a pandas data frame.

    Inputs:
    G: a Graph object as defined by the Graph class above
    data: a pandas data frame
    """
    bic = 0.0

    for v in G.vertices:
        bic += bic_score_helper(G, data, v)

    return bic

def causal_discovery(data, num_steps=50):
    """
    Take in data and perform causal discovery according to a set of moves
    described in the write up for a given number of steps.
    """

    # initalize an empty graph as the optimal one and gets its BIC score
    G_star = Graph(vertices=data.columns)
    bic_star = bic_score(G_star, data)
    vertices = list(G_star.vertices)
    
    # forward phase of causal discovery:
    for i in range(num_steps):
        edges = G_star.edges()
        V = random.choices(vertices, k=2)
        # attempt a random edge addition that does not create a cycle
        # if it improves the BIC score, update G_star and bic_star
        
        while not is_pair_valid(G_star, edges, V):
            V = random.choices(vertices, k=2)

        bic_add = bic_score(G_star, data)
        
        if bic_add < bic_star:
            bic_star = bic_add
        else:
            G_star.delete_edge(V[0], V[1])
        

    # backward phase of causal discovery
    edges = list(G_star.edges())
    for i in range(num_steps):

        # attempt a random edge deletion/reversal
        # pick the move that improves the BIC score (if any)
        (v_i, v_j) = random.choice(edges)
        G_star.delete_edge(v_i, v_j)
        bic_del = bic_score(G_star, data)
        G_star.add_edge(v_j, v_i)
        bic_rev = bic_score(G_star, data)

        if bic_del < bic_rev and bic_del < bic_star:
            bic_star = bic_del
            G_star.delete_edge(v_j, v_i)
        elif bic_star < bic_rev:
            G_star.delete_edge(v_j, v_i)
            G_star.add_edge(v_i, v_j)

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

# print(acyclic(G1))
# print(acyclic(G2))
# print(acyclic(G3))

# print(acyclic(G1) == True)
# print(acyclic(G2) == False)
# print(acyclic(G3) == False)


################################################
# Tests for your bic_score function
################################################
data = pd.read_csv("bic_test_data.txt")

# fit model for G1: A->B->C->D, B->D and get BIC
G1 = Graph(vertices=["A", "B", "C", "D"], edges=[("A", "B"), ("B", "C"), ("C", "D"), ("B", "D")])
# print(bic_score(G1, data))
# print(acyclic(G1))
# print(bic_score(G1, data), acyclic(G1))
# G1.produce_visualization_code("G1_viz.txt")

# fit model for G2: A<-B->C->D, B->D and get BIC
G2 = Graph(vertices=["A", "B", "C", "D"], edges=[("B", "A"), ("B", "C"), ("C", "D"), ("B", "D")])
# print(acyclic(G2))
# print(bic_score(G2, data))
# print(bic_score(G2, data), acyclic(G2))

# fit model for G3: A->B<-C->D, B->D and get BIC
G3 = Graph(vertices=["A", "B", "C", "D"], edges=[("A", "B"), ("C", "B"), ("C", "D"), ("B", "D")])
# print(acyclic(G3))
# print(bic_score(G3, data))
# print(bic_score(G3, data), acyclic(G3))

# fit model for G4: A<-B->C<-D, B->D and get BIC
G4 = Graph(vertices=["A", "B", "C", "D"], edges=[("B", "A"), ("B", "C"), ("D", "C"), ("B", "D")])
# print(acyclic(G4))
# print(bic_score(G4, data))
# print(bic_score(G4, data), acyclic(G4))



################################################
# Tests for your causal_discovery function
################################################
np.random.seed(1000)
random.seed(100)
data = pd.read_csv("data.txt")
G_opt = causal_discovery(data)
# you can paste the code in protein_viz.txt into the online interface of Graphviz
G_opt.produce_visualization_code("protein_viz.txt")
