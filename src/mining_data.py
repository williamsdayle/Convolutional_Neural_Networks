import pandas as pd
import numpy as np
import os
from tqdm import tqdm


def build_data_and_analise(dataset, model):

    path  = 'results/{}-dataset/global_evaluation/{}'.format(dataset.lower(), model)

    all_results = []

    results = os.listdir(path)

    for result, index in zip(results, tqdm(range(len(results)))):

        configuration = {'learning-rate': 0, 'dropout': 0, 'step': 0, 'neuron': 0, 'result': 0.0, 'std':0.0}

        metadata_config_result = result.split('_')

        if metadata_config_result[1] == 'RandomWalk':

            step = int(metadata_config_result[2])

            config_metadata = metadata_config_result[3:]

            learning_rate = float(config_metadata[0])
            dropout = float(config_metadata[1])

            neuron = int(config_metadata[-1].replace('.txt', ''))

            if neuron == 35:
                neuron = 32

            configuration['learning-rate'] = learning_rate
            configuration['dropout'] = dropout
            configuration['step'] = step
            configuration['neuron'] = neuron


            file = open(os.path.join(path, result)).read()

            metadata = file.split('\n')

            std_line = metadata[0]

            std_line_metadatada = std_line.split(' ')

            std = float(std_line_metadatada[5])

            mean_line = metadata[1]

            mean_line_metadata = mean_line.split(' ')

            mean = float(mean_line_metadata[5])

            configuration['result'] = mean
            configuration['std'] = std

            all_results.append(configuration)

        elif metadata_config_result[1] == 'RandomCut':

            step = int(metadata_config_result[2])

            config_metadata = metadata_config_result[3:]

            learning_rate = float(config_metadata[0])
            dropout = float(config_metadata[1])

            neuron = int(config_metadata[-1].replace('.txt', ''))

            if neuron == 35:
                neuron = 32

            configuration['learning-rate'] = learning_rate
            configuration['dropout'] = dropout
            configuration['step'] = step
            configuration['neuron'] = neuron

            file = open(os.path.join(path, result)).read()

            metadata = file.split('\n')

            std_line = metadata[0]

            std_line_metadatada = std_line.split(' ')

            std = float(std_line_metadatada[5])

            mean_line = metadata[1]

            mean_line_metadata = mean_line.split(' ')

            mean = float(mean_line_metadata[5])

            configuration['result'] = mean
            configuration['std'] = std

            all_results.append(configuration)

        else:
            config_metadata = metadata_config_result[3:]

            learning_rate = float(config_metadata[0])
            dropout = float(config_metadata[1])
            step = 0

            neuron = int(config_metadata[-1].replace('.txt', ''))

            if neuron == 35:

                neuron = 32

            configuration['learning-rate'] = learning_rate
            configuration['dropout'] = dropout
            configuration['step'] = step
            configuration['neuron'] = neuron

            file = open(os.path.join(path, result)).read()

            metadata = file.split('\n')

            std_line = metadata[0]

            std_line_metadatada = std_line.split(' ')

            std = float(std_line_metadatada[5])

            mean_line = metadata[1]

            mean_line_metadata = mean_line.split(' ')

            mean = float(mean_line_metadata[5])

            configuration['result'] = mean
            configuration['std'] = std

            all_results.append(configuration)
    print('\n Data sorted')
    return sorted(all_results, key=lambda k: k['result'], reverse=True)

def analise_with_full_connected(dataset, model, sorted_data):

    for sorted_result, index in zip(sorted_data, tqdm(range(len(sorted_data)))):

        step = sorted_result['step']

        if step != 0:

            learning_rate = sorted_result['learning-rate']
            dropout = sorted_result['dropout']
            neuron = sorted_result['neuron']

            if model == 'VGG16' and neuron == 32 and dataset == 'UNREL':
                neuron = 35

            path = 'results/{}-dataset/global_evaluation/{}/{}_Full_Connected_{}_{}_{}_{}.txt'.format(dataset.lower(),
                                                                                                      model,
                                                                                                      dataset,
                                                                                                      learning_rate,
                                                                                                      dropout,
                                                                                                      model,
                                                                                                      neuron)
            file = open(path).read()

            metadata = file.split('\n')

            std_line = metadata[0]

            mean_line = metadata[1]

            std_line_metadata = std_line.split(' ')

            mean_line_metadata = mean_line.split(' ')

            mean = float(mean_line_metadata[5])

            std = float(std_line_metadata[5])

            sorted_result['fc_result'] = mean

            sorted_result['fc_std'] = std

        else:
            pass
    print('\n Analise with full connected made')
    return sorted_data

def compare_with_full_connected(sorted_data):

    for sorted_result, index in zip(sorted_data, tqdm(range(len(sorted_data)))):

        try:

            random_walk_result = sorted_result['result']
            full_connected_result = sorted_result['fc_result']

            random_walk_std = sorted_result['std']
            full_connected_std = sorted_result['fc_std']

            bigger_fc = full_connected_result + full_connected_std
            smaller_fc = full_connected_result - full_connected_std

            bigger_random_walk = random_walk_result + random_walk_std
            smaller_random_walk = random_walk_result - random_walk_std

            if random_walk_result > full_connected_result:

                if smaller_random_walk <= bigger_fc:
                    sorted_result['compare'] = 'tie'

                elif smaller_random_walk > bigger_fc:
                    sorted_result['compare'] = 'win'

            elif random_walk_result <= full_connected_result:

                if bigger_random_walk >= smaller_fc:

                    sorted_result['compare'] = 'tie'

                elif bigger_random_walk < smaller_fc:

                    sorted_result['compare'] = 'lose'
            else:
                print('Error')

        except:
            pass

    print('\n Comparison made')
    return sorted_data

def write_comparisons(dataset, model, comparisons):

    win_file_path = 'results/comparisons/{}/{}/{}_{}_win_comparisons.csv'.format(dataset, model, dataset, model)
    tie_file_path = 'results/comparisons/{}/{}/{}_{}_tie_comparisons.csv'.format(dataset, model, dataset, model)
    lose_file_path = 'results/comparisons/{}/{}/{}_{}_lose_comparisons.csv'.format(dataset, model, dataset, model)

    win = open(win_file_path, 'w')
    tie = open(tie_file_path, 'w')
    lose = open(lose_file_path, 'w')

    tie_value = 0
    win_value = 0
    lose_value = 0

    cabecalho = 'Método; Learning-rate; Dropout; Neuronios; Resultado RW; Desvio RW; Resultado FC; Desvio FC; \n'
    win.write(cabecalho)
    tie.write(cabecalho)
    lose.write(cabecalho)

    for comparison, index in zip(comparisons, tqdm(range(len(comparisons)))):

        try:
            compare = comparison['compare']

            if compare == 'win':

                string_save = '{}; {}; {}; {}; {}; {}; {}; {}; WIN \n'.format(
                    comparison['step'],
                    comparison['learning-rate'],
                    comparison['dropout'],
                    comparison['neuron'],
                    comparison['result'],
                    comparison['std'],
                    comparison['fc_result'],
                    comparison['fc_std']
                )

                win.write(string_save)
                win_value = win_value + 1

            if compare == 'tie':
                string_save = '{}; {}; {}; {}; {}; {}; {}; {}; TIE \n'.format(
                    comparison['step'],
                    comparison['learning-rate'],
                    comparison['dropout'],
                    comparison['neuron'],
                    comparison['result'],
                    comparison['std'],
                    comparison['fc_result'],
                    comparison['fc_std']
                )

                tie.write(string_save)
                tie_value = tie_value + 1
            if compare == 'lose':
                string_save = '{}; {}; {}; {}; {}; {}; {}; {}; LOSE \n'.format(
                    comparison['step'],
                    comparison['learning-rate'],
                    comparison['dropout'],
                    comparison['neuron'],
                    comparison['result'],
                    comparison['std'],
                    comparison['fc_result'],
                    comparison['fc_std']
                )

                lose.write(string_save)
                lose_value = lose_value + 1
        except:
            pass

    win.close()
    tie.close()
    lose.close()

    print('\n WIN RESULTS', win_value)
    print('\n TIE RESULTS', tie_value)
    print('\n LOSE RESULTS', lose_value)


model_to_analise = 'VGG16'
dataset_to_analise = 'MIT67'

best_results_path = 'results/{}-dataset/BEST_RESULTS.csv'.format(dataset_to_analise.lower())
best_result_file = open(best_results_path, 'w')

cabecalho_best = 'Modelo; Learning-rate; Dropout; Neuronio; Metodo ; Resultado RW;  \n'

best_result_file.write(cabecalho_best)

sorted_data = build_data_and_analise(dataset_to_analise, model_to_analise)

sorted_data = analise_with_full_connected(dataset_to_analise, model_to_analise, sorted_data)

comparisons = compare_with_full_connected(sorted_data)

write_comparisons(dataset_to_analise, model_to_analise, comparisons)


string_save_best = '{}; {}; {}; {}; {}; {} \n'.format(
    model_to_analise,
    sorted_data[0]['learning-rate'],
    sorted_data[0]['dropout'],
    sorted_data[0]['neuron'],
    sorted_data[0]['step'],
    sorted_data[0]['result']
)
best_result_file.write(string_save_best)

model_to_analise = 'ResNet50'

best_result_file.write(cabecalho_best)

sorted_data = build_data_and_analise(dataset_to_analise, model_to_analise)

sorted_data = analise_with_full_connected(dataset_to_analise, model_to_analise, sorted_data)

comparisons = compare_with_full_connected(sorted_data)

write_comparisons(dataset_to_analise, model_to_analise, comparisons)


string_save_best = '{}; {}; {}; {}; {}; {} \n'.format(
    model_to_analise,
    sorted_data[0]['learning-rate'],
    sorted_data[0]['dropout'],
    sorted_data[0]['neuron'],
    sorted_data[0]['step'],
    sorted_data[0]['result']
)
best_result_file.write(string_save_best)

model_to_analise = 'Xception'

best_result_file.write(cabecalho_best)

sorted_data = build_data_and_analise(dataset_to_analise, model_to_analise)

sorted_data = analise_with_full_connected(dataset_to_analise, model_to_analise, sorted_data)

comparisons = compare_with_full_connected(sorted_data)

write_comparisons(dataset_to_analise, model_to_analise, comparisons)


string_save_best = '{}; {}; {}; {}; {}; {} \n'.format(
    model_to_analise,
    sorted_data[0]['learning-rate'],
    sorted_data[0]['dropout'],
    sorted_data[0]['neuron'],
    sorted_data[0]['step'],
    sorted_data[0]['result']
)
best_result_file.write(string_save_best)

# model_to_analise = 'InceptionV3'
#
# best_result_file.write(cabecalho_best)
#
# sorted_data = build_data_and_analise(dataset_to_analise, model_to_analise)
#
# sorted_data = analise_with_full_connected(dataset_to_analise, model_to_analise, sorted_data)
#
# comparisons = compare_with_full_connected(sorted_data)
#
# write_comparisons(dataset_to_analise, model_to_analise, comparisons)
#
#
# string_save_best = '{}; {}; {}; {}; {}; {} \n'.format(
#     model_to_analise,
#     sorted_data[0]['learning-rate'],
#     sorted_data[0]['dropout'],
#     sorted_data[0]['neuron'],
#     sorted_data[0]['step'],
#     sorted_data[0]['result']
# )
# best_result_file.write(string_save_best)
#
model_to_analise = 'InceptionResNetV2'

best_result_file.write(cabecalho_best)

sorted_data = build_data_and_analise(dataset_to_analise, model_to_analise)

sorted_data = analise_with_full_connected(dataset_to_analise, model_to_analise, sorted_data)

comparisons = compare_with_full_connected(sorted_data)

write_comparisons(dataset_to_analise, model_to_analise, comparisons)


string_save_best = '{}; {}; {}; {}; {}; {} \n'.format(
    model_to_analise,
    sorted_data[0]['learning-rate'],
    sorted_data[0]['dropout'],
    sorted_data[0]['neuron'],
    sorted_data[0]['step'],
    sorted_data[0]['result']
)
best_result_file.write(string_save_best)
#
# model_to_analise = 'EfficientNet'
#
# best_result_file.write(cabecalho_best)
#
# sorted_data = build_data_and_analise(dataset_to_analise, model_to_analise)
#
# sorted_data = analise_with_full_connected(dataset_to_analise, model_to_analise, sorted_data)
#
# comparisons = compare_with_full_connected(sorted_data)
#
# write_comparisons(dataset_to_analise, model_to_analise, comparisons)
#
#
# string_save_best = '{}; {}; {}; {}; {}; {} \n'.format(
#     model_to_analise,
#     sorted_data[0]['learning-rate'],
#     sorted_data[0]['dropout'],
#     sorted_data[0]['neuron'],
#     sorted_data[0]['step'],
#     sorted_data[0]['result']
# )
# best_result_file.write(string_save_best)

best_result_file.close()


