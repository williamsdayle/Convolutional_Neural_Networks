import pickle as pk
import time
import os
from igraph.clustering import Dendrogram
import numpy as np
import pandas as pd
import igraph
from scipy.sparse import csr_matrix
from collections import defaultdict
from tqdm import tqdm
import random
import scipy.sparse as sp
import traceback
from scipy.spatial import distance
import sys
import argparse

class FullyConnectedGraph(object):
    def __init__(self, number_of_nodes):
        self.number_of_nodes = number_of_nodes

    def create_fc_graph(self):
        # This method creates a fully connected graph
        pass

class RandomWalkGraph(object):
    def __init__(self, walk):
        # walk is the number of times that the random walker will move in the graph
        # the number of nodes has to be the same as the number of objects in the image
        # the first graph will be fully connected
        self.walk = walk

    def central_bias_random_walk(self, graph, radius):
        # This method uses the radius to maintain the edges, using a full connected as start
        pass

    def classic_random_walk(self, graph):
        # This method uses the classic random walk from the igraph library
        pass

    def weighted_random_walk(self, graph, threshold):
        # This method uses the weighted edges and a threshold to maintain the edges insied the TH value
        pass

    def graph_with_random_cut(self, graph, number_of_nodes_cutted):
        # This method returns the graph using a random cut method, the set of edges will be created randomly
        # This method uses the optimized graph to create another one with random edges created
        pass

    def random_edge_creation(self, graph, seed):
        # This method we create a graph with random edges created, the seed value is the number of edges that will be created
        # The seed value has to be equal or smaller than the set of nodes
        pass

class DataCreation(object):
    def __init__(self, dataset, extractor, pooling, train_set_percentage, test_set_percentage, walk, k_fold_split=True, number_of_sets=5):
        self.dataset = dataset  # [VRD, MIT67, UNREL]
        self.extractor = extractor # [Pre trained methods]
        self.pooling = pooling
        self.train_set_percentage = train_set_percentage
        self.test_set_percentage = test_set_percentage
        self.k_fold_split = k_fold_split
        self.number_of_sets = number_of_sets
        self.random_walk_step = walk

class BoundingBox(object):
    '''
        This class will store all the bounding box data, data as: bb label, bb name, image label, image id, image name and etc..
    '''

    def __init__(self, image_id, image_label, image_name, bounding_box_id, bounding_box_label):
        self.image_id = image_id
        self.image_label = image_label
        self.image_name = image_name
        self.bounding_box_id = bounding_box_id
        self.bounding_box_label = bounding_box_label

    def get_bb_information(self):
        return 'Bounding Box ID => ' + self.bounding_box_id + \
               'Bounding Box Label => ' + self.bounding_box_label + \
               'Image that this Bounding Box belong => ' + self.image_id + \
               'Image name is => ' + self.image_name

class Image(object):
    '''
    This class will store all the image data, data as: label, name, number of bounding boxes and etc..
    '''

    def __init__(self, image_id, image_name, image_label, image_number_of_bounding_boxes):
        self.image_id = image_id
        self.image_name = image_name
        self.image_label = image_label
        self.image_number_of_bounding_boxes = image_number_of_bounding_boxes
        self.list_of_bounding_boxes = []

    def get_bb_information(self):
        return 'Image ID => ' + self.image_id + \
               'Image Label => ' + self.image_label + \
               'Image Name => ' + self.image_name + \
               'Number of Bounding Boxes => ' + self.image_number_of_bounding_boxes + \
               'Lista de Bounding Boxes => ' + self.list_of_bounding_boxes
