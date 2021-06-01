import igraph
import numpy
import pandas



def get_matrix_from_csv(dataset, model, step, random):

    if step == 0:

        file_name = 'FilterDataFrame/{}/{}_Full_Connected_ADJ_Matrix_CONJ_0_ImageName_266.jpg.csv'.format(dataset, model)
        df = pandas.read_csv(file_name)

    if step > 0:

        if random == True:
            file_name = 'FilterDataFrame/{}/{}_RandomCut_ADJ_step_{}_CONJ_0_ImageName_266.jpg.csv'.format(dataset, model, step)
            df = pandas.read_csv(file_name)
        else:
            file_name = 'FilterDataFrame/{}/{}_RandomWalk_ADJ_step_{}_CONJ_0_ImageName_266.jpg.csv'.format(dataset, model, step)
            df = pandas.read_csv(file_name)

    return df


def df_to_matrix(df):

    matrix = df.to_numpy()

    new_matrix = []

    labels = []

    for line in matrix:

        new_matrix.append(line[1:])
        name = line[0]
        splts = name.split('_')
        bb_label = str(' '.join(splts[1:]))
        labels.append(bb_label)

    return numpy.asarray(new_matrix), labels

datasets = ['UNREL']

models = ['VGG16', 'ResNet50', 'EfficientNet', 'InceptionV3', 'InceptionResNetV2', 'Xception']

steps = [i for i in range(11)]

for dataset in datasets:

    for model in models:

        for step in steps:

            if step == 0:


                file_name = 'graphs/UNREL Full Connected {} IMAGE 266 Graph Summary.txt'.format(model)
                full_connected = get_matrix_from_csv(dataset, model, step, False)
                matrix, labels = df_to_matrix(full_connected)
                graph = igraph.Graph.Adjacency(matrix.tolist(), mode=1)
                for i in range(len(graph.vs)):
                    graph.vs[i]['name'] = labels[i]
                visual_style = {}
                visual_style["vertex_label"] = graph.vs["name"]
                layout = graph.layout("kamada_kawai")
                summary = graph.summary()
                file = open(file_name, 'w')
                file.write(summary)
                file.close()
                out = igraph.plot(graph, layout=layout, **visual_style)
                out.save('graphs/UNREL Full Connected {} IMAGE 266.png'.format(model))

            else:
                file_name = 'graphs/UNREL Random Walk {} Step {} IMAGE 266 Graph Summary.txt'.format(model, step)
                random_walk = get_matrix_from_csv(dataset, model, step, False)
                matrix, labels = df_to_matrix(random_walk)
                graph = igraph.Graph.Adjacency(matrix.tolist(), mode=1)
                for i in range(len(graph.vs)):
                    graph.vs[i]['name'] = labels[i]
                visual_style = {}
                visual_style["vertex_label"] = graph.vs["name"]
                layout = graph.layout("kamada_kawai")
                out = igraph.plot(graph, layout=layout, **visual_style)
                out.save('graphs/UNREL Random Walk {} Step {} IMAGE 266.png'.format(model, step))
                summary = graph.summary()
                file = open(file_name, 'w')
                file.write(summary)
                file.close()
                random_cut = get_matrix_from_csv(dataset, model, step, True)
                matrix, labels = df_to_matrix(random_cut)
                graph = igraph.Graph.Adjacency(matrix.tolist(), mode=1)
                for i in range(len(graph.vs)):
                    graph.vs[i]['name'] = labels[i]
                visual_style = {}
                visual_style["vertex_label"] = graph.vs["name"]
                layout = graph.layout("kamada_kawai")
                out = igraph.plot(graph, layout=layout, **visual_style)
                out.save('graphs/UNREL Random Cut {} Step {} IMAGE 266.png'.format(model, step))
                file_name = 'graphs/UNREL Random Cut {} Step {} IMAGE 266 Graph Summary.txt'.format(model, step)
                summary = graph.summary()
                file = open(file_name, 'w')
                file.write(summary)
                file.close()