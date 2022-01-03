class RandomWalk(object):
    def __init__(self, walk):
        # walk is the number of times that the random walker will move in the graph
        # the number of nodes has to be the same as the number of objects in the image
        # the first graph will be fully connected
        self.walk = walk
    
    def create_graph_fully_connected(self, number_of_objects):
        # this method returns the number of objects in the image and create a fully connected graph
        # number of nodes -> {0, 1, 2, 3}
        # number of edges -> {0:[0, 1, 2, 3], 1:[0, 1, 2, 3, 4], 2:.....}
        pass
