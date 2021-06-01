import pandas as pd
import numpy as np
import os

def get_best_metrics_results(dataset, model, step, metric):


    path = 'results/per_configuration/{}'.format(dataset)

    paths = os.listdir(path)

    files = [p for p in paths if p.split('_')[7] == str(step) and p.split('_')[8] == model]

    temp_path = files[0]

    new_path = os.path.join(path, temp_path)

    final_path = os.path.join(new_path, 'Class Evaluation.csv')

    df = pd.read_csv(final_path)

    labels = df['Label']

    labels = list(labels.to_numpy())

    results = df[metric]

    results = list(results.to_numpy())

    bests = sorted(results, reverse=True)[:5]

    mins = sorted(results)[:5]

    labels_min = []

    for min_ in mins:

        for result, index in zip(results, range(len(results))):

            if result == min_ and labels[index] not in labels_min:

                labels_min.append(labels[index])

    labels_best = []

    for best in bests:

        for result, index in zip(results, range(len(results))):

            if result == best and labels[index] not in labels_best:

                labels_best.append(labels[index])

    str_best = ''
    str_min = ''

    for label, result in zip(labels_best, bests):

        str_best = str_best + label.replace(' ', '') + ' ' + str(result) + '\n'



    for label, result in zip(labels_min, mins):

        str_min = str_min + label.replace(' ', '') + ' ' + str(result) + '\n'


    temp_path = 'Top 5 ' + metric + '.txt'

    save_path = os.path.join(new_path, temp_path)

    file = open(save_path, 'w')

    file.write(str_best)

    file.close()

    temp_path = 'Down 5 ' + metric + '.txt'

    save_path = os.path.join(new_path, temp_path)

    file = open(save_path, 'w')

    file.write(str_min)

    file.close()

    print('Ranking for {} and model {} created..'.format(metric, model))


def main():

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

    dataset = 'UNREL'

    for metric in metrics:

        get_best_metrics_results(dataset, 'VGG16', 7, metric)
        get_best_metrics_results(dataset, 'VGG16', 0, metric)

        get_best_metrics_results(dataset, 'ResNet50', 3, metric)
        get_best_metrics_results(dataset, 'ResNet50', 0, metric)

        get_best_metrics_results(dataset, 'Xception', 7, metric)
        get_best_metrics_results(dataset, 'Xception', 0, metric)

        get_best_metrics_results(dataset, 'InceptionV3', 6, metric)
        get_best_metrics_results(dataset, 'InceptionV3', 0, metric)

        get_best_metrics_results(dataset, 'InceptionResNetV2', 1, metric)
        get_best_metrics_results(dataset, 'InceptionResNetV2', 0, metric)

        get_best_metrics_results(dataset, 'EfficientNet', 2, metric)
        get_best_metrics_results(dataset, 'EfficientNet', 0, metric)

if __name__ == '__main__':
    main()