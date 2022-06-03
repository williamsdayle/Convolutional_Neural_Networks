import pickle as pk
import time
import os
import numpy as np
import pandas as pd
import igraph
from scipy.sparse import csr_matrix
from collections import defaultdict
from tqdm import tqdm
import random
import traceback
from ..random_walk_utils import RandomWalkGraph
from ..random_walk_utils import BoundingBox
from ..random_walk_utils import Image

class Utils(object):
    def __init__(self):
        pass

    def array_from_feature_file(self, file_path):
        '''
        In this method we get all the values extract from the models, we saved in the feature_files path
        :param file_path:
        :return feature array:
        '''
        feature_vector = []

        try:
            with open(file_path, "r") as file:

                features = file.read()

            lst_features = features.split(",")

            dim = len(lst_features)

            count = 0

            for feature in lst_features:
                count += 1
                if (feature != "" or feature != " " or feature != None):
                    try:

                        feature_vector.append(float(feature))
                    except:

                        feature_vector.append(float(0))

                if (count == dim):
                    feature_vector = np.asarray(feature_vector)
                    break
        except:
            raise Exception("NO FEATURE FOUND")

        return feature_vector

    def build_image_and_bounding_box_data(self, DATASET):
        '''
            This method gets the dataset and build a list of all images with their respective bounding boxes and informations
            This method needs of some data out of this code:

            The data is the path config_arqs, in this method we used the daa from the {DATASET}_sub_class_bboxes.txt files

            :param DATASET:
            :return images : all images and their respect bounding boxes information:
        '''

        metadata_file = "/home/william/Mestrado/Projeto/Convolutional_Neural_Networks/src/config_arqs/{}_subclass_bboxes.txt".format(DATASET.lower())

        with open(metadata_file) as file:

            image_metadata = file.readlines()  # each lilne belongs to one image for the dataset

        images = []

        for line, index in zip(image_metadata, tqdm(range(len(image_metadata)))):

            line = line.rstrip()

            data = line.split(' ')

            image_data = data[:4]

            metadata = data[4:]

            image_name = image_data[0]
            image_label = image_data[2]
            image_id = int(image_data[1])
            bounding_box_number = int(image_data[3])

            image = Image(image_id=image_id, image_name=image_name,
                        image_label=image_label, image_number_of_bounding_boxes=bounding_box_number)

            bounding_box_labels = []

            for i in range(0, len(metadata), 5):

                bb_label = metadata[i]

                if bb_label.count('*') > 0:
                    bb_label = bb_label.replace('*', '_')

                bounding_box_labels.append(bb_label)

            bounding_box_list = []

            bb_id = 0

            for bounding_box_label in bounding_box_labels:
                bounding_box = BoundingBox(image_id=image_id, image_label=image_label, image_name=image_name,
                                        bounding_box_id=bb_id, bounding_box_label=bounding_box_label)

                bb_id = bb_id + 1

                bounding_box_list.append(bounding_box)

            random.shuffle(bounding_box_list)
            image.list_of_bounding_boxes = bounding_box_list

            images.append(image)

        return images

    def create_and_save_gcn_data(self, DATASET, EXTRACTOR, folds, FC, RW, RC, RWEC, REC, RANDOM_WALK_STEP):

        for kfold_size in tqdm(range(len(folds))):

            fold = folds[kfold_size]

            x = csr_matrix(fold['x'])
            allx = csr_matrix(fold['allx'])
            tx = csr_matrix(fold['tx'])

            y = fold['y']
            ally = fold['ally']
            ty = fold['ty']

            test_indexes = fold['test_index']

            gcn_graph_fc = self.create_pickle_file(FC)
            gcn_graph_rw = self.create_pickle_file(RW)
            gcn_graph_rc = self.create_pickle_file(RC)
            gcn_graph_rwec = self.create_pickle_file(RWEC)
            gcn_graph_rec = self.create_pickle_file(REC)
            

            DST_PATH = '/home/william/Mestrado/Projeto/Convolutional_Neural_Networks/gcn/data/{}'.format(DATASET)
            self.save(EXTRACTOR, DST_PATH, kfold_size, x, tx, allx, y, ty, ally, test_indexes, gcn_graph_fc)
            self.save(EXTRACTOR, DST_PATH, kfold_size, x, tx, allx, y, ty, ally, test_indexes, gcn_graph_rw, random_walk=True, step=RANDOM_WALK_STEP)
            self.save(EXTRACTOR, DST_PATH, kfold_size, x, tx, allx, y, ty, ally, test_indexes, gcn_graph_rc, random_cut=True, step=RANDOM_WALK_STEP)
            self.save(EXTRACTOR, DST_PATH, kfold_size, x, tx, allx, y, ty, ally, test_indexes, gcn_graph_rwec, random_weighted=True, step=RANDOM_WALK_STEP)
            self.save(EXTRACTOR, DST_PATH, kfold_size, x, tx, allx, y, ty, ally, test_indexes, gcn_graph_rec, random_edge=True, step=RANDOM_WALK_STEP)
               
    def create_data(self, DATASET, EXTRACTOR, POOLING, kfold, train, kfold_size=0):
        '''
        Here we create all the data for GCN, we based our model and intro files in the (KIPF; WELLING, 2016).

        X will be all our trainable data
        Y will be all the representative data, with their labels

        allx: Train data
        x: values with the true label, we are using supervised learning, so allx and x are equal
        tx: Test data

        allx: representative data for allx
        y; representative data for x
        ty: representative data for tx


        :param DATASET:
        :param EXTRACTOR:
        :param POOLING:
        :param FC:
        :param RW:
        :param RC:
        :param kfold:
        :param kfold_size:
        :param train:
        :param test:
        :return gcn data:
        '''

        folds = []  # X and Y will be added here

        if kfold == True:

            for fold_number in range(kfold_size):

                if os.path.isfile('/home/william/Mestrado/Projeto/Convolutional_Neural_Networks/information/benchmarks/{}_{}_TEST_FILES.csv'.format(fold_number, DATASET)) and os.path.isfile(
                        '/home/william/Mestrado/Projeto/Convolutional_Neural_Networks/information/benchmarks/{}_{}_TRAIN_FILES.csv'.format(fold_number, DATASET)):

                    fold = {'fold_number': 0, 'x': [], 'allx': [], 'tx': [], 'y': [], 'ally': [], 'ty': [],
                            'test_index': []}

                    x, allx, tx = [], [], []

                    df_train_loaded_files = pd.read_csv('/home/william/Mestrado/Projeto/Convolutional_Neural_Networks/information/benchmarks/{}_{}_TRAIN_FILES.csv'.format(fold_number, DATASET))
                    df_test_loaded_files = pd.read_csv('/home/william/Mestrado/Projeto/Convolutional_Neural_Networks/information/benchmarks/{}_{}_TEST_FILES.csv'.format(fold_number, DATASET))

                    df_train_loaded_labels = pd.read_csv('/home/william/Mestrado/Projeto/Convolutional_Neural_Networks/information/benchmarks/{}_{}_TRAIN_LABELS.csv'.format(fold_number, DATASET))
                    df_test_loaded_labels = pd.read_csv('/home/william/Mestrado/Projeto/Convolutional_Neural_Networks/information/benchmarks/{}_{}_TEST_LABELS.csv'.format(fold_number, DATASET))

                    train_data_loaded = np.array(df_train_loaded_files['Files'])
                    train_labels_loaded = np.array(df_train_loaded_labels['Labels'])

                    labels_aux = []

                    for label in train_labels_loaded:

                        if label not in labels_aux:
                            labels_aux.append(label)

                    labels_aux = sorted(labels_aux)

                    for file_loaded, label_loaded in zip(train_data_loaded, train_labels_loaded):

                        metadata_loaded = file_loaded.split('/')

                        loaded_extractor = metadata_loaded[7].split("_")[-1]

                        file = metadata_loaded[-1]

                        if EXTRACTOR != loaded_extractor:

                            file_loaded = "/{}/{}/{}/{}/{}".format("/".join(metadata_loaded[1:8]), DATASET, EXTRACTOR, label_loaded, file)

                        features = self.array_from_feature_file(file_loaded)

                        x.append(features)
                        allx.append(features)


                    test_data_loaded = np.array(df_test_loaded_files['Files'])
                    test_labels_loaded = np.array(df_test_loaded_labels['Labels'])

                    for file_loaded, label_loaded in zip(test_data_loaded, test_labels_loaded):

                        metadata_loaded = file_loaded.split('/')

                        loaded_extractor = metadata_loaded[7].split("_")[-1]

                        file = metadata_loaded[-1]

                        if EXTRACTOR != loaded_extractor:

                            file_loaded = "/{}/{}/{}/{}/{}".format("/".join(metadata_loaded[1:8]), DATASET, EXTRACTOR, label_loaded, file)

                        features = self.array_from_feature_file(file_loaded)

                        tx.append(features)

                    y = np.zeros((len(x), len(labels_aux)))
                    ally = np.zeros((len(allx), len(labels_aux)))
                    ty = np.zeros((len(tx), len(labels_aux)))

                    for file_loaded, label_loaded, index in zip(train_data_loaded, train_labels_loaded,
                                                                range(len(train_data_loaded))):
                        label_index_loaded = self.get_label_index(label=label_loaded, labels=labels_aux)

                        y[index][label_index_loaded] = 1
                        ally[index][label_index_loaded] = 1

                    for file_loaded, label_loaded, index in zip(test_data_loaded, test_labels_loaded,
                                                                range(len(test_data_loaded))):
                        label_index_loaded = self.get_label_index(label=label_loaded, labels=labels_aux)


                        ty[index][label_index_loaded] = 1

                    x = np.array(x)
                    allx = np.array(allx)
                    tx = np.array(tx)

                    print('X', x.shape)
                    print('ALLX', allx.shape)
                    print('TX', tx.shape)

                    print('Y', y.shape)
                    print('ALLY', ally.shape)
                    print('TY', ty.shape)

                    test_index = [a for a in range(x.shape[0], x.shape[0] + tx.shape[0])]

                    print('Test Index', len(test_index))

                    fold['x'] = x
                    fold['allx'] = allx
                    fold['tx'] = tx

                    fold['y'] = y
                    fold['ally'] = ally
                    fold['ty'] = ty

                    fold['test_index'] = test_index

                    folds.append(fold)

                else:

                    fold = {'fold_number': 0, 'x': [], 'allx': [], 'tx': [], 'y': [], 'ally': [], 'ty': [],
                            'test_index': []}

                    all_train_files = []
                    all_test_files = []

                    all_train_labels = []
                    all_test_labels = []

                    path = '/home/william/Mestrado/Projeto/Convolutional_Neural_Networks/src/feature_files_{}/{}/{}'.format(POOLING, DATASET, EXTRACTOR)

                    labels = sorted(os.listdir(path=path))

                    x, allx, tx = [], [], []
                    
                    used_labels = []

                    for label in labels:

                        label_path = os.path.join(path, label)

                        files = os.listdir(label_path)
                        
                        if len(files) > 10:
                            
                            used_labels.append(label)

                            number_of_files_in_the_current_label = len(files)

                            train_size = int(round((number_of_files_in_the_current_label * train), 1))

                            test_size = int(round((number_of_files_in_the_current_label - train_size), 1))
                            '''
                            Adding train values
                            '''
                            for file in files[:train_size]:

                                all_train_labels.append(label)

                                file_path = os.path.join(label_path, file)

                                all_train_files.append(file_path)

                                features = self.array_from_feature_file(file_path)

                                x.append(features)
                                allx.append(features)                            
                            '''
                            Adding test values
                            '''
                            for file in files[train_size: ]:

                                all_test_labels.append(label)

                                file_path = os.path.join(label_path, file)

                                all_test_files.append(file_path)

                                features = self.array_from_feature_file(file_path)

                                tx.append(features)

                    y = np.zeros((len(x), len(used_labels)))
                    ally = np.zeros((len(allx), len(used_labels)))
                    ty = np.zeros((len(tx), len(used_labels)))

                    label_index = 0
                    file_index_train = 0
                    file_index_test = 0

                    for label in used_labels:

                        label_path = os.path.join(path, label)

                        files = os.listdir(label_path)

                        number_of_files_in_the_current_label = len(files)

                        if number_of_files_in_the_current_label > 10:
                            train_size = int(round((number_of_files_in_the_current_label * train), 1))

                            test_size = int(round((number_of_files_in_the_current_label - train_size), 1))

                            train_files = random.sample(files, train_size)
                            test_files = [data for data in files if data not in train_files]

                            for i in range(train_size):

                                y[file_index_train][label_index] = 1

                                ally[file_index_train][label_index] = 1

                                file_index_train = file_index_train + 1


                            for j in range(test_size):

                                ty[file_index_test][label_index] = 1

                                file_index_test = file_index_test + 1

                            label_index = label_index + 1
                    self.create_gcn_labels_file(DATASET=DATASET, labels=used_labels)
                    df_train_files = pd.DataFrame(all_train_files, columns=['Files'])
                    df_test_files = pd.DataFrame(all_test_files, columns=['Files'])

                    df_train_labels = pd.DataFrame(all_train_labels, columns=['Labels'])
                    df_test_labels = pd.DataFrame(all_test_labels, columns=['Labels'])

                    df_test_files.to_csv('/home/william/Mestrado/Projeto/Convolutional_Neural_Networks/information/benchmarks/{}_{}_TEST_FILES.csv'.format(fold_number, DATASET))
                    df_train_files.to_csv('/home/william/Mestrado/Projeto/Convolutional_Neural_Networks/information/benchmarks/{}_{}_TRAIN_FILES.csv'.format(fold_number, DATASET))

                    df_test_labels.to_csv('/home/william/Mestrado/Projeto/Convolutional_Neural_Networks/information/benchmarks/{}_{}_TEST_LABELS.csv'.format(fold_number, DATASET))
                    df_train_labels.to_csv('/home/william/Mestrado/Projeto/Convolutional_Neural_Networks/information/benchmarks/{}_{}_TRAIN_LABELS.csv'.format(fold_number, DATASET))

                    x = np.array(x)
                    allx = np.array(allx)
                    tx = np.array(tx)

                    print('X', x.shape)
                    print('ALLX', allx.shape)
                    print('TX', tx.shape)

                    print('Y', y.shape)
                    print('ALLY', ally.shape)
                    print('TY', ty.shape)

                    test_index = [a for a in range(x.shape[0], x.shape[0] + tx.shape[0])]

                    print('Test Index', len(test_index))

                    fold['x'] = x
                    fold['allx'] = allx
                    fold['tx'] = tx

                    fold['y'] = y
                    fold['ally'] = ally
                    fold['ty'] = ty

                    fold['test_index'] = test_index

                    folds.append(fold)

        else:
            traceback.print_exc('ERRO')


        return folds

    def create_gcn_labels_file(self, DATASET, labels):

        '''
        This method creates a txt file with all the dataset labels, this file will be used in the label evaluation in
        classification report method in the train file.

        :param labels:
        :return:
        '''
        print("LABELS used", len(labels))
        f = open('/home/william/Mestrado/Projeto/Convolutional_Neural_Networks/gcn/{}_labels.txt'.format(DATASET), 'w')
        f.write(str(labels))
        f.close()

    def create_graph_data(self, DATASET, EXTRACTOR, POOLING, images, RANDOM_WALK_STEP):
        '''
        In this method we created all our graph structure...

        First we created a Full Connected edges based in (BUGATTI; SAITO; DAVIS, 2019)
        Then, we created a random walk edges based in Walktrap method from Igraph
        Then, we created a random cut edges to see how good our method with random walk was, compared to a random cut


        :param DATASET:
        :param EXTRACTOR:
        ;param POOLING:
        :param images:
        :param RANDOM_WALK_STEP:
        :param OPTMIZATION_METHOD
        :param radius = None
        :param threshold = None
        :param seed = None
        :return Full Connected, Random Walk, Random Cut GRAPHS:
        '''

        FC_GRAPH = igraph.Graph(directed=False)
        RANDOM_WALK_GRAPH = igraph.Graph(directed=False)
        RANDOM_CUT_GRAPH = igraph.Graph(directed=False)
        RANDOM_EDGE_ADD_GRAPH = igraph.Graph(directed=False)
        RANDOM_WEIGHTED_CUT_GRAPH = igraph.Graph(directed=False)
        global_graph = igraph.Graph(directed=False)        

        random_walk_object = RandomWalkGraph(RANDOM_WALK_STEP)

        lista_de_graph_obj_FC = []
        lista_de_graph_obj_RW = []
        lista_de_graph_obj_RC = []
        lista_de_graph_obj_REC = []
        lista_de_graph_obj_RWC = []
        
        graph_FC_size = 0
        graph_RW_size = 0
        graph_RC_size = 0
        graph_REC_size = 0
        graph_RWC_size = 0
        


        global_count = 0

        for image, index in zip(images, tqdm(range(len(images)))):

            graph = igraph.Graph(directed=False)
            random_walk_graph_current = igraph.Graph(directed=False)
            random_cut_graph_current = igraph.Graph(directed=False)
            weighted_random_walk_graph_current = igraph.Graph(directed=False)
            random_edge_creation_graph_current = igraph.Graph(directed=False)

            columns, indexes = [], []

            for i in range(image.image_number_of_bounding_boxes):

                graph.add_vertices(1)  # full connected graph
                random_walk_graph_current.add_vertices(1)  # random walk graph
                random_cut_graph_current.add_vertices(1)  # random cut graph
                global_graph.add_vertices(1)  # global graph for creation of other files
                weighted_random_walk_graph_current.add_vertices(1) # to cut edges with a threshold
                random_edge_creation_graph_current.add_vertices(1) # this graph will contain edges created random

                graph.vs[i]['image_name'] = image.image_name
                global_graph.vs[global_count]['image_name'] = image.image_name
                random_walk_graph_current.vs[i]['image_name'] = image.image_name
                random_cut_graph_current.vs[i]['image_name'] = image.image_name
                weighted_random_walk_graph_current.vs[i]['image_name'] = image.image_name
                random_edge_creation_graph_current.vs[i]['image_name'] = image.image_name

                graph.vs[i]['image_id'] = image.image_id
                global_graph.vs[global_count]['image_id'] = image.image_id
                random_walk_graph_current.vs[i]['image_id'] = image.image_id
                random_cut_graph_current.vs[i]['image_id'] = image.image_id
                weighted_random_walk_graph_current.vs[i]['image_id'] = image.image_id
                random_edge_creation_graph_current.vs[i]['image_id'] = image.image_id

                graph.vs[i]['image_label'] = image.image_label
                global_graph.vs[global_count]['image_label'] = image.image_label
                random_walk_graph_current.vs[i]['image_label'] = image.image_label
                random_cut_graph_current.vs[i]['image_label'] = image.image_label
                weighted_random_walk_graph_current.vs[i]['image_label'] = image.image_id
                random_edge_creation_graph_current.vs[i]['image_label'] = image.image_id

                graph.vs[i]['image_bounding_box_number'] = image.image_number_of_bounding_boxes
                global_graph.vs[global_count]['image_bounding_box_number'] = image.image_number_of_bounding_boxes
                random_walk_graph_current.vs[i]['image_bounding_box_number'] = image.image_number_of_bounding_boxes
                random_cut_graph_current.vs[i]['image_bounding_box_number'] = image.image_number_of_bounding_boxes
                weighted_random_walk_graph_current.vs[i]['image_bounding_box_number'] = image.image_number_of_bounding_boxes
                random_edge_creation_graph_current.vs[i]['image_bounding_box_number'] = image.image_number_of_bounding_boxes

                bounding_box_current = image.list_of_bounding_boxes[i]

                graph.vs[i]['bounding_box_id'] = bounding_box_current.bounding_box_id
                global_graph.vs[global_count]['bounding_box_id'] = bounding_box_current.bounding_box_id
                random_walk_graph_current.vs[i]['bounding_box_id'] = bounding_box_current.bounding_box_id
                random_cut_graph_current.vs[i]['bounding_box_id'] = bounding_box_current.bounding_box_id
                weighted_random_walk_graph_current.vs[i]['bounding_box_id'] = bounding_box_current.bounding_box_id
                random_edge_creation_graph_current.vs[i]['bounding_box_id'] = bounding_box_current.bounding_box_id


                graph.vs[i]['bounding_box_label'] = bounding_box_current.bounding_box_label
                global_graph.vs[global_count]['bounding_box_label'] = bounding_box_current.bounding_box_label
                random_walk_graph_current.vs[i]['bounding_box_label'] = bounding_box_current.bounding_box_label
                random_cut_graph_current.vs[i]['bounding_box_label'] = bounding_box_current.bounding_box_label
                weighted_random_walk_graph_current.vs[i]['bounding_box_label'] = bounding_box_current.bounding_box_label
                random_edge_creation_graph_current.vs[i]['bounding_box_label'] = bounding_box_current.bounding_box_label

                column = str(image.image_id) + \
                        '_' + bounding_box_current.bounding_box_label + \
                        '_' + str(bounding_box_current.bounding_box_id)
                columns.append(column)
                indexes.append(column)

                bounding_box_file = column + '.txt'
                if POOLING == 'max':
                    src_path = '/home/william/Mestrado/Projeto/Convolutional_Neural_Networks/src/feature_files_max/{}/{}/{}'.format(DATASET, EXTRACTOR, image.image_label)
                if POOLING == 'avg':
                    src_path = '/home/william/Mestrado/Projeto/Convolutional_Neural_Networks/src/feature_files_avg/{}/{}/{}'.format(DATASET, EXTRACTOR, image.image_label)
                path = os.path.join(src_path, bounding_box_file)
                global_graph.vs[global_count]['bb_path'] = path
                features = self.array_from_feature_file(path)

                graph.vs[i]['features'] = features
                global_graph.vs[global_count]['features'] = features
                random_walk_graph_current.vs[i]['features'] = features
                random_cut_graph_current.vs[i]['features'] = features
                weighted_random_walk_graph_current.vs[i]['features'] = features
                random_edge_creation_graph_current.vs[i]['features'] = features


            PRIMARY_GRAPH = igraph.Graph.Full(directed=False, n=image.image_number_of_bounding_boxes, loops=True)

            FC_EDGES = PRIMARY_GRAPH.get_edgelist()

            graph.add_edges(FC_EDGES)

            '''
            Here we calculate the euclidean distance between the objects from the image
            '''

            RANDOM_WALK_EDGES, CUTTED_EDGES = random_walk_object.classic_random_walk(graph)
            RANDOM_CUT_EDGES = random_walk_object.graph_with_random_cut(graph, CUTTED_EDGES)
            RANDOM_CREATION_EDGES = random_walk_object.random_edge_creation(graph, CUTTED_EDGES)
            RANDOM_WEIGHTED_EDGES = random_walk_object.weighted_random_walk(graph, CUTTED_EDGES)

            graph_FC_size += len(FC_EDGES)
            graph_RW_size += len(RANDOM_WALK_EDGES)
            graph_RC_size += len(RANDOM_CUT_EDGES)
            graph_REC_size += len(RANDOM_CREATION_EDGES)
            graph_RWC_size += len(RANDOM_WEIGHTED_EDGES)
     
            
            random_walk_graph_current.add_edges(
                RANDOM_WALK_EDGES)  # inserting all edges in the graph, this is the random walk edges
            
            random_cut_graph_current.add_edges(
                RANDOM_CUT_EDGES)  # inserting all edges in the graph, this is the random cut edges
            
            weighted_random_walk_graph_current.add_edges(
                RANDOM_WEIGHTED_EDGES)  # inserting all edges in the graph, this is the random walk weighted edges
            
            random_edge_creation_graph_current.add_edges(
                RANDOM_CREATION_EDGES)  # inserting all edges in the graph, this is the random creation edges
            
            image_adjacency_matrix_fc = graph.get_adjacency()
            image_data_frame_fc = pd.DataFrame(image_adjacency_matrix_fc, columns=columns, index=indexes)

            image_data_frame_fc.to_csv(
                '/home/william/Mestrado/Projeto/Convolutional_Neural_Networks/information/csv/{}/FULL CONNECTED/{}.csv'.format(DATASET, image.image_name.replace('.jpg', '')))

            image_adjacency_matrix_rw = random_walk_graph_current.get_adjacency()
            image_data_frame_rw = pd.DataFrame(image_adjacency_matrix_rw, columns=columns, index=indexes)

            image_data_frame_rw.to_csv('/home/william/Mestrado/Projeto/Convolutional_Neural_Networks/information/csv/{}/RANDOM WALK/{}_{}_{}.csv'.format(DATASET,
                                                                                    image.image_name.replace('.jpg',
                                                                                                                ''),
                                                                                    RANDOM_WALK_STEP,
                                                                                    EXTRACTOR))

            image_adjacency_matrix_rc = random_cut_graph_current.get_adjacency()
            image_data_frame_rc = pd.DataFrame(image_adjacency_matrix_rc, columns=columns, index=indexes)

            image_data_frame_rc.to_csv('/home/william/Mestrado/Projeto/Convolutional_Neural_Networks/information/csv/{}/RANDOM CUT/{}_{}_{}.csv'.format(DATASET,
                                                                                    image.image_name.replace('.jpg',
                                                                                                            ''),
                                                                                    RANDOM_WALK_STEP,
                                                                                    EXTRACTOR))
            
            
            image_adjacency_matrix_rcw = weighted_random_walk_graph_current.get_adjacency()
            image_data_frame_rcw = pd.DataFrame(image_adjacency_matrix_rcw, columns=columns, index=indexes)

            image_data_frame_rcw.to_csv('/home/william/Mestrado/Projeto/Convolutional_Neural_Networks/information/csv/{}/RANDOM WEIGHT/{}_{}_{}.csv'.format(DATASET,
                                                                                    image.image_name.replace('.jpg',
                                                                                                            ''),
                                                                                    RANDOM_WALK_STEP,
                                                                                    EXTRACTOR))
            
            
            image_adjacency_matrix_rec = random_edge_creation_graph_current.get_adjacency()
            image_data_frame_rec = pd.DataFrame(image_adjacency_matrix_rec, columns=columns, index=indexes)

            image_data_frame_rec.to_csv('/home/william/Mestrado/Projeto/Convolutional_Neural_Networks/information/csv/{}/RANDOM CREATION/{}_{}_{}.csv'.format(DATASET,
                                                                                    image.image_name.replace('.jpg',
                                                                                                            ''),
                                                                                    RANDOM_WALK_STEP,
                                                                                    EXTRACTOR))
            
            lista_de_graph_obj_FC.append(graph)
            lista_de_graph_obj_RC.append(random_cut_graph_current)
            lista_de_graph_obj_RW.append(random_walk_graph_current)
            lista_de_graph_obj_REC.append(random_edge_creation_graph_current)
            lista_de_graph_obj_RWC.append(weighted_random_walk_graph_current)

        fc_start = time.time()
        for g, index in zip(lista_de_graph_obj_FC, tqdm(range(len(lista_de_graph_obj_FC)))):
            FC_GRAPH = igraph.Graph.disjoint_union(FC_GRAPH, g)
        fc_stop = time.time()

        fc_build_time = fc_stop - fc_start

        rw_start = time.time()
        for g, index in zip(lista_de_graph_obj_RW, tqdm(range(len(lista_de_graph_obj_RW)))):
            RANDOM_WALK_GRAPH = igraph.Graph.disjoint_union(RANDOM_WALK_GRAPH, g)
        rw_stop = time.time()

        rw_build_time = rw_stop - rw_start

        rc_start = time.time()
        for g, index in zip(lista_de_graph_obj_RC, tqdm(range(len(lista_de_graph_obj_RC)))):
            RANDOM_CUT_GRAPH = igraph.Graph.disjoint_union(RANDOM_CUT_GRAPH, g)
        rc_stop = time.time()

        rc_build_time = rc_stop - rc_start
        
        rweighted_start = time.time()
        for g, index in zip(lista_de_graph_obj_RWC, tqdm(range(len(lista_de_graph_obj_RWC)))):
            RANDOM_WEIGHTED_CUT_GRAPH = igraph.Graph.disjoint_union(RANDOM_WEIGHTED_CUT_GRAPH, g)
        rweighted_stop = time.time()

        rwg_build_time = rweighted_stop - rweighted_start
        
        rb_start = time.time()
        for g, index in zip(lista_de_graph_obj_REC, tqdm(range(len(lista_de_graph_obj_REC)))):
            RANDOM_EDGE_ADD_GRAPH = igraph.Graph.disjoint_union(RANDOM_EDGE_ADD_GRAPH, g)
        rb_stop = time.time()

        rb_build_time = rb_stop - rb_start

        print('Full Connected Graph', FC_GRAPH.summary(), 'process time:', fc_build_time)
        print()
        print('Random Walk {} Graph'.format(RANDOM_WALK_STEP), RANDOM_WALK_GRAPH.summary(), 'process time:', rw_build_time)
        print()
        print('Random Cut {} Graph'.format(RANDOM_WALK_STEP), RANDOM_CUT_GRAPH.summary(), 'process time:', rc_build_time)
        print()
        print('Random Edge Cut Weighted Graph {}'.format(RANDOM_WALK_STEP), RANDOM_WEIGHTED_CUT_GRAPH.summary(), 'process time:', rwg_build_time)
        print()
        print('Random Edge Choiced {} Graph'.format(RANDOM_WALK_STEP), RANDOM_EDGE_ADD_GRAPH.summary(), 'process time:', rb_build_time)
        
        
        return FC_GRAPH, RANDOM_WALK_GRAPH, RANDOM_CUT_GRAPH, RANDOM_WEIGHTED_CUT_GRAPH, RANDOM_EDGE_ADD_GRAPH, fc_build_time, rw_build_time, rc_build_time, rwg_build_time, rb_build_time

    def create_pickle_file(self, graph):

        '''
        This method builds the graph structure to be used in the pickle file
        :param graph:
        :return the dict list with nodes and conenctions, in format of default dict:
        '''

        d_dictionary = defaultdict(list)

        node = 0

        for connection in graph.get_adjlist():

            d_dictionary[node] = connection

            node = node + 1

        d_dictionary = self.remove_duplicated_values(d_dictionary)

        print("Graph size: %s " % str(sum([len(d) for (k, d) in d_dictionary.items()])))

        return d_dictionary

    def get_label_index(self, label, labels):

        '''
        Method used in the data build process
        
        :param label:
        :return label index:
        '''

        for label_id, l in enumerate(labels):
            if l == label:
                return label_id

    def remove_repetidos(self, lista):
        l = []
        for i in lista:
            if i not in l:
                l.append(i)
        l.sort()
        return l

    def remove_duplicated_values(self, default_dict_list):

        dict_graph = defaultdict(list)

        for keys, values in default_dict_list.items():

            new_list = self.remove_repetidos(values)

            dict_graph[keys] = sorted(new_list)

        return dict_graph

    def save(self, model_name, dst_path, fold, x, tx, allx, y, ty, ally, test_indexes, graph, random_walk=False, random_edge=False, random_weighted=False, random_cut=False, step=None):

        '''
        Here we save all data in pickle format.
        This files will be loaded bt the train file.

        :param model_name:
        :param dst_path:
        :param fold:
        :param x:
        :param tx:
        :param allx:
        :param y:
        :param ty:
        :param ally:
        :param test_indexes:
        :param graph:
        :param random_walk:
        :param random_cut:
        :param step:
        :return:
        '''
        try:            
            if random_edge:
                pk.dump(x,
                        open(os.path.join(dst_path, "ind.mine_{}_conj_{}_step_{}.x".format(model_name, fold, step+30)), "wb"), protocol=2)
                pk.dump(tx,
                        open(os.path.join(dst_path, "ind.mine_{}_conj_{}_step_{}.tx".format(model_name, fold, step+30)), "wb"), protocol=2)
                pk.dump(allx,
                        open(os.path.join(dst_path, "ind.mine_{}_conj_{}_step_{}.allx".format(model_name, fold, step+30)),
                            "wb"), protocol=2)
                pk.dump(y,
                        open(os.path.join(dst_path, "ind.mine_{}_conj_{}_step_{}.y".format(model_name, fold, step+30)), "wb"), protocol=2)
                pk.dump(ty,
                        open(os.path.join(dst_path, "ind.mine_{}_conj_{}_step_{}.ty".format(model_name, fold, step+30)), "wb"), protocol=2)
                pk.dump(ally,
                        open(os.path.join(dst_path, "ind.mine_{}_conj_{}_step_{}.ally".format(model_name, fold, step+30)),
                            "wb"), protocol=2)
                pk.dump(graph,
                        open(os.path.join(dst_path, "ind.mine_{}_conj_{}_step_{}.graph".format(model_name, fold, step+30)),
                            "wb"), protocol=2)

                with open(os.path.join(dst_path, "ind.mine_{}_conj_{}_step_{}.test.index".format(model_name, fold, step+30)),
                        "w") as indexes:
                    indexes.write(str(test_indexes).replace(",", "\n").replace("[", "").replace("]", "").replace(" ", ""))
            
            if random_weighted:
                pk.dump(x,
                        open(os.path.join(dst_path, "ind.mine_{}_conj_{}_step_{}.x".format(model_name, fold, step+20)), "wb"), protocol=2)
                pk.dump(tx,
                        open(os.path.join(dst_path, "ind.mine_{}_conj_{}_step_{}.tx".format(model_name, fold, step+20)), "wb"), protocol=2)
                pk.dump(allx,
                        open(os.path.join(dst_path, "ind.mine_{}_conj_{}_step_{}.allx".format(model_name, fold, step+20)),
                            "wb"), protocol=2)
                pk.dump(y,
                        open(os.path.join(dst_path, "ind.mine_{}_conj_{}_step_{}.y".format(model_name, fold, step+20)), "wb"), protocol=2)
                pk.dump(ty,
                        open(os.path.join(dst_path, "ind.mine_{}_conj_{}_step_{}.ty".format(model_name, fold, step+20)), "wb"), protocol=2)
                pk.dump(ally,
                        open(os.path.join(dst_path, "ind.mine_{}_conj_{}_step_{}.ally".format(model_name, fold, step+20)),
                            "wb"), protocol=2)
                pk.dump(graph,
                        open(os.path.join(dst_path, "ind.mine_{}_conj_{}_step_{}.graph".format(model_name, fold, step+20)),
                            "wb"), protocol=2)

                with open(os.path.join(dst_path, "ind.mine_{}_conj_{}_step_{}.test.index".format(model_name, fold, step+20)),
                        "w") as indexes:
                    indexes.write(str(test_indexes).replace(",", "\n").replace("[", "").replace("]", "").replace(" ", ""))
            
            if random_walk:
                pk.dump(x,
                        open(os.path.join(dst_path, "ind.mine_{}_conj_{}_step_{}.x".format(model_name, fold, step)), "wb"), protocol=2)
                pk.dump(tx,
                        open(os.path.join(dst_path, "ind.mine_{}_conj_{}_step_{}.tx".format(model_name, fold, step)), "wb"), protocol=2)
                pk.dump(allx,
                        open(os.path.join(dst_path, "ind.mine_{}_conj_{}_step_{}.allx".format(model_name, fold, step)),
                            "wb"), protocol=2)
                pk.dump(y,
                        open(os.path.join(dst_path, "ind.mine_{}_conj_{}_step_{}.y".format(model_name, fold, step)), "wb"), protocol=2)
                pk.dump(ty,
                        open(os.path.join(dst_path, "ind.mine_{}_conj_{}_step_{}.ty".format(model_name, fold, step)), "wb"), protocol=2)
                pk.dump(ally,
                        open(os.path.join(dst_path, "ind.mine_{}_conj_{}_step_{}.ally".format(model_name, fold, step)),
                            "wb"), protocol=2)
                pk.dump(graph,
                        open(os.path.join(dst_path, "ind.mine_{}_conj_{}_step_{}.graph".format(model_name, fold, step)),
                            "wb"), protocol=2)

                with open(os.path.join(dst_path, "ind.mine_{}_conj_{}_step_{}.test.index".format(model_name, fold, step)),
                        "w") as indexes:
                    indexes.write(str(test_indexes).replace(",", "\n").replace("[", "").replace("]", "").replace(" ", ""))

            if random_cut:

                pk.dump(x, open(
                    os.path.join(dst_path, "ind.mine_{}_conj_{}_step_{}.x".format(model_name, fold, step + 10)), "wb"), protocol=2)
                pk.dump(tx, open(
                    os.path.join(dst_path, "ind.mine_{}_conj_{}_step_{}.tx".format(model_name, fold, step + 10)),
                    "wb"), protocol=2)
                pk.dump(allx, open(
                    os.path.join(dst_path, "ind.mine_{}_conj_{}_step_{}.allx".format(model_name, fold, step + 10)),
                    "wb"), protocol=2)
                pk.dump(y, open(
                    os.path.join(dst_path, "ind.mine_{}_conj_{}_step_{}.y".format(model_name, fold, step + 10)), "wb"), protocol=2)
                pk.dump(ty, open(
                    os.path.join(dst_path, "ind.mine_{}_conj_{}_step_{}.ty".format(model_name, fold, step + 10)),
                    "wb"), protocol=2)
                pk.dump(ally, open(
                    os.path.join(dst_path, "ind.mine_{}_conj_{}_step_{}.ally".format(model_name, fold, step + 10)),
                    "wb"), protocol=2)
                pk.dump(graph, open(
                    os.path.join(dst_path, "ind.mine_{}_conj_{}_step_{}.graph".format(model_name, fold, step + 10)),
                    "wb"), protocol=2)

                with open(os.path.join(dst_path,
                                    "ind.mine_{}_conj_{}_step_{}.test.index".format(model_name, fold, step + 10)),
                        "w") as indexes:
                    indexes.write(str(test_indexes).replace(",", "\n").replace("[", "").replace("]", "").replace(" ", ""))
            
            if random_weighted is False and random_cut is False and random_edge is False and random_walk is False:

                pk.dump(x, open(os.path.join(dst_path, "ind.mine_{}_conj_{}_step_0.x".format(model_name, fold)), "wb"), protocol=2)
                pk.dump(tx, open(os.path.join(dst_path, "ind.mine_{}_conj_{}_step_0.tx".format(model_name, fold)), "wb"), protocol=2)
                pk.dump(allx,
                        open(os.path.join(dst_path, "ind.mine_{}_conj_{}_step_0.allx".format(model_name, fold)), "wb"), protocol=2)
                pk.dump(y, open(os.path.join(dst_path, "ind.mine_{}_conj_{}_step_0.y".format(model_name, fold)), "wb"), protocol=2)
                pk.dump(ty, open(os.path.join(dst_path, "ind.mine_{}_conj_{}_step_0.ty".format(model_name, fold)), "wb"), protocol=2)
                pk.dump(ally,
                        open(os.path.join(dst_path, "ind.mine_{}_conj_{}_step_0.ally".format(model_name, fold)), "wb"), protocol=2)
                pk.dump(graph,
                        open(os.path.join(dst_path, "ind.mine_{}_conj_{}_step_0.graph".format(model_name, fold)), "wb"), protocol=2)

                with open(os.path.join(dst_path, "ind.mine_{}_conj_{}_step_0.test.index".format(model_name, fold)),
                        "w") as indexes:
                    indexes.write(str(test_indexes).replace(",", "\n").replace("[", "").replace("]", "").replace(" ", ""))

        except:
            traceback.print_exc()

    def save_connections(self, graph, DATASET, EXTRACTOR, type, walk=None):

        summary = graph.summary()

        if walk == None:

            path = '/home/william/Mestrado/Projeto/Convolutional_Neural_Networks/information/connections/{}/Full_Connected.txt'.format(DATASET)

            file = open(path, 'w')

            file.write(str(summary))

            file.close()

        else:

            path = '/home/william/Mestrado/Projeto/Convolutional_Neural_Networks/information/connections/{}/{}_{}_{}.txt'.format(DATASET, type, EXTRACTOR, walk)

            file = open(path, 'w')

            file.write(str(summary))

            file.close()

    def save_process_time(self, DATASET, FC_TIME, RW_TIME, RC_TIME, R_EDGE_TIME, R_WEIGHTED_TIME, WALK, MODEL):

        ##########################
        #
        # Saving full connected time process
        #
        ##########################

        path = '/home/william/Mestrado/Projeto/Convolutional_Neural_Networks/information/process_time/{}/Full Connected Time {}.txt'.format(DATASET, MODEL)

        file = open(path, 'w')

        file.write('Process Time: ' + str(FC_TIME) + '\n')

        file.close()

        ##########################
        #
        # Saving Random Walk time process
        #
        ##########################

        path = '/home/william/Mestrado/Projeto/Convolutional_Neural_Networks/information/process_time/{}/Random Walk {} Time {}.txt'.format(DATASET, WALK, MODEL)

        file = open(path, 'w')

        file.write('Process Time: ' + str(RW_TIME) + '\n')

        file.close()

        ##########################
        #
        # Saving Random Cut time process
        #
        ##########################

        path = '/home/william/Mestrado/Projeto/Convolutional_Neural_Networks/information/process_time/{}/Random Cut {} Time {}.txt'.format(DATASET, WALK, MODEL)

        file = open(path, 'w')

        file.write('Process Time: ' + str(RC_TIME) + '\n')

        file.close()
        
        ##########################
        #
        # Saving Random Weighted time process
        #
        ##########################

        path = '/home/william/Mestrado/Projeto/Convolutional_Neural_Networks/information/process_time/{}/Random Weighted {} Time {}.txt'.format(DATASET, WALK, MODEL)

        file = open(path, 'w')

        file.write('Process Time: ' + str(R_WEIGHTED_TIME) + '\n')

        file.close()
        
        ##########################
        #
        # Saving Random Edge time process
        #
        ##########################

        path = '/home/william/Mestrado/Projeto/Convolutional_Neural_Networks/information/process_time/{}/Random Edge {} Time {}.txt'.format(DATASET, WALK, MODEL)

        file = open(path, 'w')

        file.write('Process Time: ' + str(R_EDGE_TIME) + '\n')

        file.close()

    def show_information(self, DATASET, EXTRACTOR, TRAIN, TEST):
        '''
        This method just print informations in shell
        :param DATASET:
        :param EXTRACTOR:
        :return None:
        '''

        print('-----------------------DATA INFORMATION------------------------')
        print('This is a master dregree project, we desire to introduce random walk in GCNs')
        print('In this launch we used', DATASET, 'to create all of our data')
        print('To extract features we used the ', EXTRACTOR, 'convolutional neural network as extractor')
        print('We splited the data in ', TEST * 100, '% of test files and', TRAIN * 100, '% of train files')
        print('----------------------------------------------------------------')
