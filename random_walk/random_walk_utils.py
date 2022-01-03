import igraph
import numpy as np
import pandas as pd
import random


class RandomWalkGraph(object):
    def __init__(self, walk):
        # walk is the number of times that the random walker will move in the graph
        # the number of nodes has to be the same as the number of objects in the image
        # the first graph will be fully connected
        self.walk = walk

    def graph_with_random_walk(self, graph, method, weights=None):
        # this method returns the optimized graph with random walk using a method of cut
        # methods -> central bias, classic random walk, random edge weight
        # all of them will use a type of random walk to cut the edges
        pass

    def graph_with_random_cut(self, graph, number_of_nodes_cutted):
        # this method returns the graph using a random cut method, the set of edges will be created randomly
        pass