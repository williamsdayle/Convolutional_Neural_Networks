import enum
import numpy as np
import pandas as pd
import igraph
import random
from scipy.spatial import distance

class RandomWalkGraph(object):
    def __init__(self, walk):
        # walk is the number of times that the random walker will move in the graph
        # the number of nodes has to be the same as the number of objects in the image
        # the first graph will be fully connected
        self.walk = walk
        self.number_of_bboxes = 0

    def weighted_random_walk(self, graph, cutted_edges):
        """
        This method takes a graph G` and walk X random walks inside this graph
        The edges that are equal of bigger than the threshold will be cutted

        :param weights
        :param threshold

        return weighted_random_walk_graph
        """
        if cutted_edges > 0:
            mean_dists = []
            for i in range(len(graph.vs) - 1):
                source_connection = graph.vs[i]["features"]
                target_connection = graph.vs[i + 1]["features"]       
                euc_dist = distance.euclidean(source_connection, target_connection)
                if euc_dist != 0:
                    mean_dists.append(euc_dist)

            mean_dist = np.mean(mean_dists)
            temp_edges = []
            edges_to_build = len(graph.get_edgelist()) - cutted_edges
            for i in range(edges_to_build):
                seed_source = random.randint(0, len(graph.vs) - 1)
                seed_target = random.randint(0, len(graph.vs) - 1)
                for connection in graph.get_edgelist():
                    source_connection = connection[0]
                    target_connection = connection[1]
                    if target_connection == seed_target and source_connection == seed_source:
                        feature_source = graph.vs[source_connection]["features"]
                        feature_target = graph.vs[target_connection]["features"]                
                        euc_dist = distance.euclidean(feature_source, feature_target)
                        if euc_dist < mean_dist:
                            temp_edges.append((source_connection, target_connection))
            return temp_edges
        else:
            return graph.get_edgelist()

    def classic_random_walk(self, graph):
        '''
        This method builds the Random Walk and Random Cut graphs, using the euclidean distance introduced as weights param.
        We used the Computing Communities in Large Networks Using Random Walks approach, with the igraph lib.

        :param g:
        :param step:
        :return:
        '''
        self.number_of_bboxes = len(graph.vs)
        dendrogram = graph.community_walktrap(steps=self.walk)
        edges = []
        if self.number_of_bboxes > 3:
            for i in range(self.number_of_bboxes):
                edges.append((i, i))
            clusters = dendrogram.as_clustering(n=dendrogram.optimal_count)  # or clusters = dendrogram.as_clustering(n=len(dendrogram.merges))        
            for cluster, index_clust in zip(clusters, range(len(clusters))):
                for node in cluster:
                    edge = (index_clust, node)
                    if edge not in edges:
                        edges.append(edge)
        else:
            edges = graph.get_edgelist()
        rdg = igraph.Graph(directed=False)
        for i in range(self.number_of_bboxes):
            rdg.add_vertices(1)
        rdg.add_edges(edges)
        full_connected_edges = graph.get_edgelist()
        random_walk_edges_size = len(rdg.get_edgelist())
        cutted_edges = len(full_connected_edges) - random_walk_edges_size
        return rdg.get_edgelist(), cutted_edges

    def graph_with_random_cut(self, graph, cutted_edges):
        '''
        This method is used side-by-side with the classic random walk
        Using this method we want to prove that the random walk method uses some contextual information to work

        This method takes a graph G` and cuts the edges in random way
        The number of edges cutted has to be the same as the classic random walk

        :param graph:
        :param cutted_edges

        return random_cut_graph
        '''
        
        full_connected_edges = graph.get_edgelist()
        rdc = igraph.Graph(directed=False)
        for i in range(self.number_of_bboxes):
            rdc.add_vertices(1)
        full_connected_updated = len(full_connected_edges)
        edges_to_cut = cutted_edges
        while edges_to_cut > 0:
            position = random.randint(0, full_connected_updated - 1)
            source, target = full_connected_edges[position]
            if source != target:
                edges_to_cut = edges_to_cut - 1
                del full_connected_edges[position]
            else:
                pass
            full_connected_updated = len(full_connected_edges)
        rdc.add_edges(full_connected_edges)
        return rdc.get_edgelist()

    def random_edge_creation(self, graph, cutted_edges):
        '''
        This method creates a graph with random edges

        :param graph
        :param cutted_edges
        
        :return random_edge_creation
        '''
        self.number_of_bboxes = len(graph.vs)
        random_edge_graph = igraph.Graph(directed=False)
        for i in range(self.number_of_bboxes):
            random_edge_graph.add_vertices(1)
        edges = []
        if cutted_edges == 0:            
            return graph.get_edgelist()
        else:
            number_of_random_edges_to_create = len(graph.get_edgelist()) - cutted_edges
            while number_of_random_edges_to_create > 0:
                x, y = random.randint(0, self.number_of_bboxes - 1), random.randint(0, self.number_of_bboxes - 1)
                edges.append((x, y))
                number_of_random_edges_to_create-=1
            random_edge_graph.add_edges(edges)
            return random_edge_graph.get_edgelist()


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
