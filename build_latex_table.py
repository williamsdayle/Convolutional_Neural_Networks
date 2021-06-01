import pandas as pd
import numpy as np

def build_table(random_walk_path, full_connected_path, metric):

    rw_count = 0
    fc_count = 0
    tie_count = 0
    class_rw_won = []
    class_fc_won = []

    random_walk_data_frame = pd.read_csv(random_walk_path)
    full_connected_data_frame = pd.read_csv(full_connected_path)

    classes = random_walk_data_frame[random_walk_data_frame.columns[1]]
    supports = random_walk_data_frame[random_walk_data_frame.columns[-1]]

    if metric == 'Accuracy':
        results_rw = random_walk_data_frame[random_walk_data_frame.columns[2]]
        results_std_rw = random_walk_data_frame[random_walk_data_frame.columns[3]]

        results_fc = full_connected_data_frame[full_connected_data_frame.columns[2]]
        results_std_fc = full_connected_data_frame[full_connected_data_frame.columns[3]]

    elif metric == 'Precision':
        results_rw = random_walk_data_frame[random_walk_data_frame.columns[4]]
        results_std_rw = random_walk_data_frame[random_walk_data_frame.columns[5]]

        results_fc = full_connected_data_frame[full_connected_data_frame.columns[4]]
        results_std_fc = full_connected_data_frame[full_connected_data_frame.columns[5]]

    elif metric == 'Recall':
        results_rw = random_walk_data_frame[random_walk_data_frame.columns[6]]
        results_std_rw = random_walk_data_frame[random_walk_data_frame.columns[7]]

        results_fc = full_connected_data_frame[full_connected_data_frame.columns[6]]
        results_std_fc = full_connected_data_frame[full_connected_data_frame.columns[7]]

    elif metric == 'F1-Score':
        results_rw = random_walk_data_frame[random_walk_data_frame.columns[8]]
        results_std_rw = random_walk_data_frame[random_walk_data_frame.columns[9]]

        results_fc = full_connected_data_frame[full_connected_data_frame.columns[8]]
        results_std_fc = full_connected_data_frame[full_connected_data_frame.columns[9]]
    else:
        results_rw = []
        results_std_rw = []
        results_fc = []
        results_std_fc = []
        supports = []

    table_string = '\hline \n Classe & Caminhada Aleatória & Totalmente Conectado & Suporte \\\ \hline \n'

    for classe, result_rw, std_rw, result_fc, std_fc, support in zip(classes,
                                                                     results_rw, results_std_rw,
                                                                     results_fc, results_std_fc,
                                                                     supports):

        if classe.count('_') > 0:

            classe = classe.replace('_', '-')

        if (result_fc + std_fc) < (result_rw - std_rw):
            rw_count+=1
            class_rw_won.append(classe)

        elif (result_rw + std_rw) < (result_fc - std_fc):
            fc_count+=1
            class_fc_won.append(classe)

        else:
            tie_count+=1



        result_rw = str(float(round(result_rw * 100, 5))).replace('.', ',')
        std_rw = str(float(round(std_rw * 100, 5))).replace('.', ',')

        result_fc = str(float(round(result_fc * 100, 5))).replace('.', ',')
        std_fc = str(float(round(std_fc * 100, 5))).replace('.', ',')

        support = str(int(support))

        line_table = classe + ' & ' + str(result_rw) + ' $\\pm{' + str(std_rw) + '}$' + \
                              ' & ' + str(result_fc) + ' $\\pm{' + str(std_fc) + '}$ & ' + str(support) + ' \\\ \hline \n'
        table_string = table_string + line_table

    print('{} fc won {} time and rw won {} times, with {} ties'.format(metric, fc_count, rw_count, tie_count))
    print('Random walk won in {}'.format(class_rw_won))
    print('Full Connected won in {}'.format(class_fc_won))
    print('Ties {}'.format([class_ for class_ in classes if class_ not in class_fc_won and class_ not in class_rw_won]))
    return table_string


def main():

    #########################
    #
    # Random Walk best results
    #
    ########################

    dataset = 'UNREL'
    model = 'ResNet50'

    lr = 0.005
    drop = 0.3
    neuron = 256
    step = 3

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

    path_rw = 'results/per_configuration/{}/LR_{}_DROP_{}_NEURON_{}_STEP_{}_{}/Class Evaluation.csv'.format(
        dataset,
        lr,
        drop,
        neuron,
        step,
        model
    )
    #########################
    #
    # Full Connected best results
    #
    ########################

    dataset = 'UNREL'
    model = 'InceptionResNetV2'

    lr = 0.01
    drop = 0.9
    neuron = 256
    step = 0

    path_fc = 'results/per_configuration/{}/LR_{}_DROP_{}_NEURON_{}_STEP_{}_{}/Class Evaluation.csv'.format(
        dataset,
        lr,
        drop,
        neuron,
        step,
        model
    )

    for metric in metrics:

        table = build_table(path_rw, path_fc, metric)
        print('METRIC {}'.format(metric))
        print(table)

if __name__ == '__main__':
    main()