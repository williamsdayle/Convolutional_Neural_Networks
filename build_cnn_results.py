import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

def get_class_accuracy(dataset, model, lr, drop):

    data_frames = []

    for i in range(5):

        path_for_files = 'results/{}-dataset/cnn/logs/LOG_FOLD_{}_LR_{}_DROP_{}_MODEL_{}.csv'.format(dataset.lower(),
                                                                                            i,
                                                                                            lr,
                                                                                            drop,
                                                                                            model)

        current_df = pd.read_csv(path_for_files)

        data_frames.append(current_df)

        columns = [column for column in current_df]

        columns.pop(0)

    all_data = []

    metadatas = []

    for column in columns:

        metadata = {'label': column, 'accuracy': 0, 'std': 0}

        metadatas.append(metadata)

    diagonals = []

    for i in range(5):

        data_frame = data_frames[i]

        data_frame = data_frame.to_numpy()

        matrix = []

        for j in range(data_frame.shape[0]):

            line = data_frame[j]

            line = line[1:]

            matrix.append(line)

        matrix = np.array(matrix)

        cm = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

        diagonal = cm.diagonal()

        diagonals.append(diagonal)

    for i in range(len(diagonals[0])):

        acc_1 = diagonals[0][i]
        acc_2 = diagonals[1][i]
        acc_3 = diagonals[2][i]
        acc_4 = diagonals[3][i]
        acc_5 = diagonals[4][i]

        mean = np.mean([acc_1, acc_2, acc_3, acc_4, acc_5])

        std = np.std([acc_1, acc_2, acc_3, acc_4, acc_5])

        metadata = metadatas[i]

        metadata['accuracy'] = float(round(mean, 7))
        metadata['std'] = float(round(std, 7))


    return metadatas


def create_confusion_matrix(dataset, model, lr, drop):

    data_frames = []

    for i in range(5):

        path = 'results/{}-dataset/cnn/logs/LOG_FOLD_{}_LR_{}_DROP_{}_MODEL_{}.csv'.format(dataset.lower(),
                                                                                            i,
                                                                                            lr,
                                                                                            drop,
                                                                                            model)


        df = pd.read_csv(path)

        data_frames.append(df)

        columns = [column for column in df.columns]

        columns.pop(0)

    all_data = []

    for column in columns:

        data_frame_1 = data_frames[0]
        data_frame_2 = data_frames[1]
        data_frame_3 = data_frames[2]
        data_frame_4 = data_frames[3]
        data_frame_5 = data_frames[4]

        current_column_in_df_1 = data_frame_1[column]
        current_column_in_df_2 = data_frame_2[column]
        current_column_in_df_3 = data_frame_3[column]
        current_column_in_df_4 = data_frame_4[column]
        current_column_in_df_5 = data_frame_5[column]

        means = []

        for column_data_df_1, column_data_df_2, column_data_df_3, column_data_df_4, column_data_df_5 in zip(
                current_column_in_df_1,
                current_column_in_df_2,
                current_column_in_df_3,
                current_column_in_df_4,
                current_column_in_df_5
        ):
            means.append(int(
                np.mean([column_data_df_1, column_data_df_2, column_data_df_3, column_data_df_4, column_data_df_5])))

        all_data.append(means)

    all_data = np.array(all_data)

    final_df = pd.DataFrame(all_data, columns=sorted(columns), index=sorted(columns))

    csv_path = 'results/{}-dataset/cnn/csv/{}/{}_{}_confusion_matrix.csv'.format(dataset.lower(),
                                                                                    model,
                                                                                    lr,
                                                                                    drop)

    final_df.to_csv(csv_path)

    confusion_matrix_path = 'results/{}-dataset/cnn/confusion_matrix/{}/{}_{}_confusion_matrix.png'.format(dataset.lower(),
                                                                                                            model,
                                                                                                            lr,
                                                                                                            drop)

    plt.figure(figsize=(20, 21))
    if dataset == 'UNREL':
        ax = sns.heatmap(final_df, annot=True, cmap='OrRd', fmt='2.0f', vmin=0, vmax=56)
    if dataset == 'MIT67':
        ax = sns.heatmap(final_df, annot=True, cmap='OrRd', fmt='2.0f', vmin=0, vmax=56)
    if dataset == 'VRD':
        ax = sns.heatmap(final_df, annot=True, cmap='OrRd', fmt='2.0f', vmin=0, vmax=56)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)

    plt.title('CONFUSION MATRIX')
    plt.ylabel('P R E V I S T O')
    plt.xlabel('R E A L')
    plt.savefig(confusion_matrix_path)
    plt.close()

    print('Heat Map created for', (lr, drop, model, dataset))


def create_train_plot_and_class_evaluation(dataset, model, lr, drop):

    train_results = []
    loss_results = []
    results_global = []

    for i in range(5):

        path = 'results/{}-dataset/cnn/logs/LOG_FOLD_{}_LR_{}_DROP_{}_MODEL_{}.txt'.format(dataset.lower(),
                                                                                           i,
                                                                                           lr,
                                                                                           drop,
                                                                                           model)

        file = open(path).read()

        file_metadata = file.split('\n')

        train_metadata = file_metadata[:2000]

        class_results = file_metadata[2004:2061]

        train = []
        loss = []

        for result in train_metadata:

            result_metadata = result.split('|')

            loss_brut = result_metadata[1]

            loss_metadata = loss_brut.split(' ')

            loss_result = loss_metadata[-2]

            loss.append(float(loss_result))

            acc_brut = result_metadata[-1]

            acc_metadata = acc_brut.split(' ')

            acc = acc_metadata[-1]

            train.append(float(acc))

        train_results.append(train)
        loss_results.append(loss)

        labels = []
        precisions = []
        recalls = []
        f_scores = []
        suports = []

        for class_result in class_results:

            report_metadata = class_result.split(' ')

            filter_data = [data for data in report_metadata if data != '']

            label = filter_data[0]
            precision = filter_data[1]
            recall = filter_data[2]
            f_score = filter_data[3]
            support = filter_data[4]

            labels.append(label)
            precisions.append(float(precision))
            recalls.append(float(recall))
            f_scores.append(float(f_score))
            suports.append(int(support))

        labels_sorted = sorted(labels)

        results_global.append([precisions, recalls, f_scores, suports])

    all_data = []

    metadata = get_class_accuracy(dataset, model, lr, drop)

    columns = ['Accuracy', 'Std Acuracy',
               'Precision', 'Std Precision',
               'Recall', 'Std Recall',
               'F_Score', 'Std F_Score',
               'Suport']

    for i in range(len(results_global[0][0])):

        precision_1 = results_global[0][0][i]
        precision_2 = results_global[1][0][i]
        precision_3 = results_global[2][0][i]
        precision_4 = results_global[3][0][i]
        precision_5 = results_global[4][0][i]

        recall_1 = results_global[0][1][i]
        recall_2 = results_global[1][1][i]
        recall_3 = results_global[2][1][i]
        recall_4 = results_global[3][1][i]
        recall_5 = results_global[4][1][i]

        f_score_1 = results_global[0][2][i]
        f_score_2 = results_global[1][2][i]
        f_score_3 = results_global[2][2][i]
        f_score_4 = results_global[3][2][i]
        f_score_5 = results_global[4][2][i]

        suport_1 = results_global[0][3][i]
        suport_2 = results_global[1][3][i]
        suport_3 = results_global[2][3][i]
        suport_4 = results_global[3][3][i]
        suport_5 = results_global[4][3][i]

        precision_mean = round(float(np.mean([precision_1, precision_2, precision_3, precision_4, precision_5])), 5)
        precision_std = round(float(np.std([precision_1, precision_2, precision_3, precision_4, precision_5])), 6)

        recall_mean = round(float(np.mean([recall_1, recall_2, recall_3, recall_4, recall_5])), 5)
        recall_std = round(float(np.std([recall_1, recall_2, recall_3, recall_4, recall_5])), 6)

        f_score_mean = round(float(np.mean([f_score_1, f_score_2, f_score_3, f_score_4, f_score_5])), 5)
        f_score_std = round(float(np.std([f_score_1, f_score_2, f_score_3, f_score_4, f_score_5])), 6)

        suport_mean = int(np.mean([suport_1, suport_2, suport_3, suport_4, suport_5]))

        label = labels_sorted[i]

        metadata_class = [metadata_c for metadata_c in metadata if metadata_c['label'] == label]

        mean_accuracy = metadata_class[0]['accuracy']

        std_accuracy = metadata_class[0]['std']

        all_data.append([mean_accuracy, std_accuracy,
                         precision_mean, precision_std,
                         recall_mean, recall_std,
                         f_score_mean, f_score_std,
                         suport_mean])

    final_df = pd.DataFrame(all_data, columns=columns, index=labels_sorted)

    path_to_save = 'results/{}-dataset/cnn/class_evaluation/{}/Class Evaluation LR {} Drop {}.csv'.format(dataset.lower(),
                                                                                                          model,
                                                                                                          lr,
                                                                                                          drop)

    final_df.to_csv(path_to_save)

    mean_train_accs = []

    for i in range(len(train_results[0])):

        train_1 = train_results[0][i]
        train_2 = train_results[1][i]
        train_3 = train_results[2][i]
        train_4 = train_results[3][i]
        train_5 = train_results[4][i]

        mean = np.mean([train_1, train_2, train_3, train_4, train_5])

        mean_train_accs.append(mean)

    mean_train_loss = []

    for i in range(len(loss_results[0])):
        train_1 = loss_results[0][i]
        train_2 = loss_results[1][i]
        train_3 = loss_results[2][i]
        train_4 = loss_results[3][i]
        train_5 = loss_results[4][i]

        mean = np.mean([train_1, train_2, train_3, train_4, train_5])

        mean_train_loss.append(mean)


    loss_path = 'results/{}-dataset/cnn/plots/{}/train/LOSS_PLOT_LR_{}_DROP_{}.png'.format(dataset.lower(),
                                                                                              model,
                                                                                              lr,
                                                                                              drop)

    acc_path = 'results/{}-dataset/cnn/plots/{}/train/ACCURACY_PLOT_LR_{}_DROP_{}.png'.format(dataset.lower(),
                                                                                              model,
                                                                                              lr,
                                                                                              drop)


    plt.title('Accuracy')
    plt.plot([i for i in range(2000)], mean_train_accs, color='g')
    # plt.show()
    plt.savefig(acc_path)
    plt.close()

    plt.title('Loss')
    plt.plot([i for i in range(2000)], mean_train_loss, color='g')
    # plt.show()
    plt.savefig(loss_path)
    plt.close()

    print('Class Evaluation and plots made for learning rate {} dropout {} and model  {}'.format(lr, drop, model))


def main():

    models = ['ResNet50', 'Xception', 'InceptionV3', 'InceptionResNetV2', 'EfficientNet', 'VGG16']
    lrs = [0.01, 0.05, 0.001, 0.005]
    drops = [0.3, 0.5, 0.8, 0.9]
    datasets = ['UNREL']

    for dataset in datasets:

        for model in models:

            for lr in lrs:

                for drop in drops:

                    create_confusion_matrix(dataset, model, lr, drop)
                    create_train_plot_and_class_evaluation(dataset, model, lr, drop)


if __name__ == '__main__':
    main()
