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
    """                                                                                                             
    A function that uses depth first traversal to determine whether a vertex v
    is part of a cycle in the graph G.
    """
    # Add v to the set of visited vertices and the set of vertices in the current path
    visited.add(v)
    current_path.add(v)

    # Children of v
    children = set()

    # Edges in G
    edges = G.edges()
    
    for u in G.vertices:

        if (v, u) in edges:
            
            if u in current_path:
                # A back edge which indicates a cycle
                return True
            if u not in visited:
                # Only visit unvisited vertices
                children.add(u)

    # Continue depth first traversal to the next level
    for c in children:
        if depth_first_search(G, c, visited, current_path):
            return True

    # Remove v from current_path when we return from our depth first traversal
    current_path.remove(v)
    
    return False

def acyclic(G):
    """
    A function that uses depth first traversal to determine whether the
    graph G is acyclic.
    """
    # Set of visited vertices
    visited = set()

    # Set of vertices in the current path
    current_path = set()

    for v in G.vertices:
        
        # Determine whether an unvisited vertex is part of a cycle in G
        if v not in visited and depth_first_search(G, v, visited, current_path):
            return False

    return True

def bic_score_helper(G, data, v):
     """                                                                                                            
     Compute the BIC score for a model fit for a variable v in a graph G given its parents
     and a dataset as a pandas data frame.                                  
                                                                                                                    
     Inputs:                                                                                                          
     G: a Graph object as defined by the Graph class above
     v: the variable we want to fit a model for
     data: a pandas data frame                                                                                        
    """
     # Formula for the model
     formula = v + ' ~ 1'

     # Parents of v
     parents = G.parents[v]

     for p in parents:
         formula += ' + '
         formula += p

     model = ols(formula=formula, data=data).fit()
     loglikelihood = model.llf
     bic = -2*loglikelihood + len(model.params) * np.log(data.shape[0])
    
     return bic

def bic_score(G, data):
    """
    Compute the BIC score for a given graph G and a dataset as a pandas data frame.

    Inputs:
    G: a Graph object as defined by the Graph class above
    data: a pandas data frame
    """
    bic = 0.0

    # Sum over the BIC scores for individual models fit for each variable given its parents
    for v in G.vertices:
        bic += bic_score_helper(G, data, v)

    return bic

def is_pair_valid(G, edges, V):
    """                                                                         
    Check if an edge is valid (i.e. does not exist in G and adding it to G does not make G cyclic)                                                 
    """
    
    if (V[0], V[1]) in edges:
        # The edge is already in G
        return False
    else:
        G.add_edge(V[0], V[1])

        if acyclic(G):
            return True
        else:
            # Adding the edge to G make it cyclic
            G.delete_edge(V[0], V[1])
            return False
        
def causal_discovery(data, num_steps=50):
    """
    Take in data and perform causal discovery according to a set of moves
    described in the write up for a given number of steps.
    """

    # initalize an empty graph as the optimal one and gets its BIC score
    G_star = Graph(vertices=data.columns)
    bic_star = bic_score(G_star, data)

    # Get a list of all the vertices in G
    vertices = list(G_star.vertices)
    
    # forward phase of causal discovery:
    for i in range(num_steps):
        edges = G_star.edges()

        # Pick two random vertices in the set of all vertices until we find a valid pair
        V = random.choices(vertices, k=2)
        
        while not is_pair_valid(G_star, edges, V):
            V = random.choices(vertices, k=2)

        # The BIC score after adding the new edge
        bic_add = bic_score(G_star, data)

        # Only keeping the added edge if it improves the BIC score
        if bic_add < bic_star:
            bic_star = bic_add
        else:
            G_star.delete_edge(V[0], V[1])

    # backward phase of causal discovery
    for i in range(num_steps):
        edges = G_star.edges()

        # Pick a random edge that exists in G_star
        (v_i, v_j) = random.choice(edges)

        # Attempt edge deletion
        G_star.delete_edge(v_i, v_j)
        bic_del = bic_score(G_star, data)

        # Attempt edge reversal
        G_star.add_edge(v_j, v_i)

        if acyclic(G_star):
            bic_rev = bic_score(G_star, data)
        else:
            # If reversing an edge makes the graph cyclic, make the BIC score for the edge reversal attempt big
            # so that it cannot be chosen
            bic_rev = bic_star + 1

        
        if bic_del < bic_rev and bic_del < bic_star:
            # When the BIC score for edge deletion is the best, delete the edge and update bic_star
            bic_star = bic_del
            G_star.delete_edge(v_j, v_i)
        elif bic_star < bic_rev:
            # When deleting and reversing an edge do not improve the BIC score, keep the bic_star and G_star unchanged
            G_star.delete_edge(v_j, v_i)
            G_star.add_edge(v_i, v_j)
        else:
            # When the BIC score for edge reversal is the best, reverse the edge and update bic_star
            bic_star = bic_rev

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
data = pd.read_csv("vaccine_no_null.csv2.txt")
G_opt = causal_discovery(data)
# you can paste the code in protein_viz.txt into the online interface of Graphviz
G_opt.produce_visualization_code("protein_viz.txt")
