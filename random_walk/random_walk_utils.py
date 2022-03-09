import numpy as np
import pandas as pd
import igraph
import random

class RandomWalkGraph(object):
    def __init__(self, walk):
        # walk is the number of times that the random walker will move in the graph
        # the number of nodes has to be the same as the number of objects in the image
        # the first graph will be fully connected
        self.walk = walk
        self.cutted_edges = 0
        self.number_of_bboxes = 0

    def central_bias_random_walk(self, graph, radius):
        # This method uses the radius to maintain the edges, using a full connected as start
        pass

    def weighted_random_walk(self, graph, threshold):
        # This method uses the weighted edges and a threshold to maintain the edges insied the TH value
        pass

    def classic_random_walk(self, graph, weights):        
        '''
        This method builds the Random Walk and Random Cut graphs, using the euclidean distance introduced as weights param.
        We used the Computing Communities in Large Networks Using Random Walks approach, with the igraph lib.

        :param g:
        :param weights:
        :param step:
        :return:
        '''
        self.number_of_bboxes = len(graph.vs)
        dendrogram = graph.community_walktrap(weights=weights, steps=self.step)
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
        self.cutted_edges = len(full_connected_edges) - random_walk_edges_size
        return rdg.get_edgelist()

    def graph_with_random_cut(self, graph):
        # This method returns the graph using a random cut method, the set of edges will be created randomly
        # This method uses the optimized graph to create another one with random edges created
        
        full_connected_edges = graph.get_edgelist()
        rdc = igraph.Graph(directed=False)
        for i in range(self.number_of_bboxes):
            rdc.add_vertices(1)
        full_connected_updated = len(full_connected_edges)
        while self.cutted_edges > 0:
            position = random.randint(0, full_connected_updated - 1)
            source, target = full_connected_edges[position]
            if source != target:
                self.cutted_edges = self.cutted_edges - 1
                del full_connected_edges[position]
            else:
                pass
            full_connected_updated = len(full_connected_edges)
        rdc.add_edges(full_connected_edges)
        return rdc.get_edgelist()

    def random_edge_creation(self, graph, percentage):
        # This method we create a graph with random edges created, the percentage value will be the edge creation value maintain
        self.number_of_bboxes = len(graph.vs)
        random_edge_graph = igraph.Graph(directed=False)
        for i in range(self.number_of_bboxes):
            random_edge_graph.add_vertices(1)
        total_edges = len(graph.get_edgelist())
        number_of_random_edges_to_create = int((total_edges * percentage) / 100)
        edges = []
        while number_of_random_edges_to_create > 0:
            x, y = random.randint(0, self.number_of_bboxes), random.randint(0, self.number_of_bboxes)
            if (x, y) not in edges:
                edges.append((x, y))
                number_of_random_edges_to_create = number_of_random_edges_to_create - 1
            else:
                pass
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
