import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


global_file = open('results/{}-dataset/evaluation/InceptionResNetV2/GLOBAL_EVALUATION_IN_{}_WITH_{}.csv'.format('MIT67'.lower(), 'MIT67', 'InceptionResNetV2'), 'w')


def get_class_accuracy(step, lr, drop, neuron, model, dataset):
    data_frames = []

    for i in range(5):

        if step == 0:

            path_for_files = 'results/{}-dataset/logs/{}/FullConnected_KFOLD_{}_STEP_0_LR_{}_DROP_{}_ARC_{}_NEURON_{}.csv'.format(
                dataset.lower(),
                model,
                i,
                lr,
                drop,
                model,
                neuron)

        elif step > 0 and step <= 10:

            path_for_files = 'results/{}-dataset/logs/{}/Random_Walk_KFOLD_{}_STEP_{}_LR_{}_DROP_{}_ARC_{}_NEURON_{}.csv'.format(
                dataset.lower(),
                model,
                i,
                step,
                lr,
                drop,
                model,
                neuron)

        else:

            path_for_files = 'results/{}-dataset/logs/{}/Random_Cut_KFOLD_{}_STEP_{}_LR_{}_DROP_{}_ARC_{}_NEURON_{}.csv'.format(
                dataset.lower(),
                model,
                i,
                step,
                lr,
                drop,
                model,
                neuron)

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

def put_sub_plot_in_axes(axes, lst_results, method):

    if method == 'mean':
        index = 0

        std = lst_results[index]['Standard Deviation']
        mean = lst_results[index]['Mean-Accuracy']
        axes.text(x=index, y=index + (mean + std), s=str(mean + std), rotation=70)
        axes.text(x=index, y=index + mean, s=str(mean), rotation=70)
        axes.text(x=index, y=index + (mean - std), s=str(mean - std), rotation=70)

        index = 1

        std = lst_results[index]['Standard Deviation']
        mean = lst_results[index]['Mean-Accuracy']
        axes.text(x=index, y=index + (mean + std), s=str(mean + std), rotation=70)
        axes.text(x=index, y=index + mean, s=str(mean), rotation=70)
        axes.text(x=index, y=index + (mean - std), s=str(mean - std), rotation=70)

        index = 2

        std = lst_results[index]['Standard Deviation']
        mean = lst_results[index]['Mean-Accuracy']
        axes.text(x=index, y=index + (mean + std), s=str(mean + std), rotation=70)
        axes.text(x=index, y=index + mean, s=str(mean), rotation=70)
        axes.text(x=index, y=index + (mean - std), s=str(mean - std), rotation=70)

        index = 3

        std = lst_results[index]['Standard Deviation']
        mean = lst_results[index]['Mean-Accuracy']
        axes.text(x=index, y=index + (mean + std), s=str(mean + std), rotation=70)
        axes.text(x=index, y=index + mean, s=str(mean), rotation=70)
        axes.text(x=index, y=index + (mean - std), s=str(mean - std), rotation=70)

        index = 4

        std = lst_results[index]['Standard Deviation']
        mean = lst_results[index]['Mean-Accuracy']
        axes.text(x=index, y=index + (mean + std), s=str(mean + std), rotation=70)
        axes.text(x=index, y=index + mean, s=str(mean), rotation=70)
        axes.text(x=index, y=index + (mean - std), s=str(mean - std), rotation=70)

        index = 5

        std = lst_results[index]['Standard Deviation']
        mean = lst_results[index]['Mean-Accuracy']
        axes.text(x=index, y=index + (mean + std), s=str(mean + std), rotation=70)
        axes.text(x=index, y=index + mean, s=str(mean), rotation=70)
        axes.text(x=index, y=index + (mean - std), s=str(mean - std), rotation=70)

        index = 6

        std = lst_results[index]['Standard Deviation']
        mean = lst_results[index]['Mean-Accuracy']
        axes.text(x=index, y=index + (mean + std), s=str(mean + std), rotation=70)
        axes.text(x=index, y=index + mean, s=str(mean), rotation=70)
        axes.text(x=index, y=index + (mean - std), s=str(mean - std), rotation=70)

        index = 7

        std = lst_results[index]['Standard Deviation']
        mean = lst_results[index]['Mean-Accuracy']
        axes.text(x=index, y=index + (mean + std), s=str(mean + std), rotation=70)
        axes.text(x=index, y=index + mean, s=str(mean), rotation=70)
        axes.text(x=index, y=index + (mean - std), s=str(mean - std), rotation=70)

        index = 8

        std = lst_results[index]['Standard Deviation']
        mean = lst_results[index]['Mean-Accuracy']
        axes.text(x=index, y=index + (mean + std), s=str(mean + std), rotation=70)
        axes.text(x=index, y=index + mean, s=str(mean), rotation=70)
        axes.text(x=index, y=index + (mean - std), s=str(mean - std), rotation=70)

        index = 9

        std = lst_results[index]['Standard Deviation']
        mean = lst_results[index]['Mean-Accuracy']
        axes.text(x=index, y=index + (mean + std), s=str(mean + std), rotation=70)
        axes.text(x=index, y=index + mean, s=str(mean), rotation=70)
        axes.text(x=index, y=index + (mean - std), s=str(mean - std), rotation=70)

        index = 10
        std = lst_results[index]['Standard Deviation']
        mean = lst_results[index]['Mean-Accuracy']
        axes.text(x=index, y=index + (mean + std), s=str(mean + std), rotation=70)
        axes.text(x=index, y=index + mean, s=str(mean), rotation=70)
        axes.text(x=index, y=index + (mean - std), s=str(mean - std), rotation=70)

    else:

        index = 0

        std = lst_results[index]['Standard Deviation']
        mean = lst_results[index]['Median-Accuracy']
        axes.text(x=index, y=index + (mean + std), s=str(mean + std), rotation=70)
        axes.text(x=index, y=index + mean, s=str(mean), rotation=70)
        axes.text(x=index, y=index + (mean - std), s=str(mean - std), rotation=70)

        index = 1

        std = lst_results[index]['Standard Deviation']
        mean = lst_results[index]['Median-Accuracy']
        axes.text(x=index, y=index + (mean + std), s=str(mean + std), rotation=70)
        axes.text(x=index, y=index + mean, s=str(mean), rotation=70)
        axes.text(x=index, y=index + (mean - std), s=str(mean - std), rotation=70)

        index = 2

        std = lst_results[index]['Standard Deviation']
        mean = lst_results[index]['Median-Accuracy']
        axes.text(x=index, y=index + (mean + std), s=str(mean + std), rotation=70)
        axes.text(x=index, y=index + mean, s=str(mean), rotation=70)
        axes.text(x=index, y=index + (mean - std), s=str(mean - std), rotation=70)

        index = 3

        std = lst_results[index]['Standard Deviation']
        mean = lst_results[index]['Median-Accuracy']
        axes.text(x=index, y=index + (mean + std), s=str(mean + std), rotation=70)
        axes.text(x=index, y=index + mean, s=str(mean), rotation=70)
        axes.text(x=index, y=index + (mean - std), s=str(mean - std), rotation=70)

        index = 4

        std = lst_results[index]['Standard Deviation']
        mean = lst_results[index]['Median-Accuracy']
        axes.text(x=index, y=index + (mean + std), s=str(mean + std), rotation=70)
        axes.text(x=index, y=index + mean, s=str(mean), rotation=70)
        axes.text(x=index, y=index + (mean - std), s=str(mean - std), rotation=70)

        index = 5

        std = lst_results[index]['Standard Deviation']
        mean = lst_results[index]['Median-Accuracy']
        axes.text(x=index, y=index + (mean + std), s=str(mean + std), rotation=70)
        axes.text(x=index, y=index + mean, s=str(mean), rotation=70)
        axes.text(x=index, y=index + (mean - std), s=str(mean - std), rotation=70)

        index = 6

        std = lst_results[index]['Standard Deviation']
        mean = lst_results[index]['Median-Accuracy']
        axes.text(x=index, y=index + (mean + std), s=str(mean + std), rotation=70)
        axes.text(x=index, y=index + mean, s=str(mean), rotation=70)
        axes.text(x=index, y=index + (mean - std), s=str(mean - std), rotation=70)

        index = 7

        std = lst_results[index]['Standard Deviation']
        mean = lst_results[index]['Median-Accuracy']
        axes.text(x=index, y=index + (mean + std), s=str(mean + std), rotation=70)
        axes.text(x=index, y=index + mean, s=str(mean), rotation=70)
        axes.text(x=index, y=index + (mean - std), s=str(mean - std), rotation=70)

        index = 8

        std = lst_results[index]['Standard Deviation']
        mean = lst_results[index]['Median-Accuracy']
        axes.text(x=index, y=index + (mean + std), s=str(mean + std), rotation=70)
        axes.text(x=index, y=index + mean, s=str(mean), rotation=70)
        axes.text(x=index, y=index + (mean - std), s=str(mean - std), rotation=70)

        index = 9

        std = lst_results[index]['Standard Deviation']
        mean = lst_results[index]['Median-Accuracy']
        axes.text(x=index, y=index + (mean + std), s=str(mean + std), rotation=70)
        axes.text(x=index, y=index + mean, s=str(mean), rotation=70)
        axes.text(x=index, y=index + (mean - std), s=str(mean - std), rotation=70)

        index = 10
        std = lst_results[index]['Standard Deviation']
        mean = lst_results[index]['Median-Accuracy']
        axes.text(x=index, y=index + (mean + std), s=str(mean + std), rotation=70)
        axes.text(x=index, y=index + mean, s=str(mean), rotation=70)
        axes.text(x=index, y=index + (mean - std), s=str(mean - std), rotation=70)

def build_plots_from_configuration(step, lr, drop, neuron, model, dataset):

    train_results_lst = []

    loss_results_lst = []

    for i in range(5):

        if step == 0:

            path_for_files = 'results/{}-dataset/logs/{}/FullConnected_KFOLD_{}_STEP_0_LR_{}_DROP_{}_ARC_{}_NEURON_{}.txt'.format(
                dataset.lower(),
                model,
                i,
                lr,
                drop,
                model,
                neuron)

        elif step > 0 and step <= 10:

            path_for_files = 'results/{}-dataset/logs/{}/Random_Walk_KFOLD_{}_STEP_{}_LR_{}_DROP_{}_ARC_{}_NEURON_{}.txt'.format(
                dataset.lower(),
                model,
                i,
                step,
                lr,
                drop,
                model,
                neuron)

        else:

            path_for_files = 'results/{}-dataset/logs/{}/Random_Cut_KFOLD_{}_STEP_{}_LR_{}_DROP_{}_ARC_{}_NEURON_{}.txt'.format(
                dataset.lower(),
                model,
                i,
                step,
                lr,
                drop,
                model,
                neuron)

        file = open(path_for_files).read()

        all_lines = file.split('\n')

        train_results = all_lines[2:2002]

        current_file_train_result = []

        current_file_loss_result = []

        for train_result in train_results:

            metadata = train_result.split('|')

            loss_metadata = metadata[1]

            acc_metadata = metadata[2]

            loss_metadata = loss_metadata.split(' ')

            acc_metadata = acc_metadata.split(' ')

            acc = float(acc_metadata[2])
            loss = float(loss_metadata[2])

            current_file_loss_result.append(loss)
            current_file_train_result.append(acc)

        train_results_lst.append(current_file_train_result)
        loss_results_lst.append(current_file_loss_result)


    """
    Getting the loss from every file
    """


    loss = []

    for i in range(len(loss_results_lst[0])):

        moment_loss = []

        loss_1 = loss_results_lst[0][i]
        loss_2 = loss_results_lst[0][i]
        loss_3 = loss_results_lst[0][i]
        loss_4 = loss_results_lst[0][i]
        loss_5 = loss_results_lst[0][i]

        moment_loss.append(loss_1)
        moment_loss.append(loss_2)
        moment_loss.append(loss_3)
        moment_loss.append(loss_4)
        moment_loss.append(loss_5)

        loss_mean = np.mean(moment_loss)

        loss.append(loss_mean)

    '''
    Getting the accuracy from every file
    '''

    accs = []

    for i in range(len(train_results_lst[0])):

        moment_acc = []

        acc_1 = train_results_lst[0][i]
        acc_2 = train_results_lst[0][i]
        acc_3 = train_results_lst[0][i]
        acc_4 = train_results_lst[0][i]
        acc_5 = train_results_lst[0][i]

        moment_acc.append(acc_1)
        moment_acc.append(acc_2)
        moment_acc.append(acc_3)
        moment_acc.append(acc_4)
        moment_acc.append(acc_5)

        acc_mean = np.mean(moment_acc)

        accs.append(acc_mean)

    path_loss_fig = 'results/plots/{}-dataset/Train/{}/LOSS_PLOT_STEP_{}_LR_{}_DROP_{}_NEURON_{}_MODEL_{}.png'.format(
                    dataset.lower(),
                    model,
                    step,
                    lr,
                    drop,
                    neuron,
                    model)


    path_acc_fig = 'results/plots/{}-dataset/Train/{}/ACCURACY_PLOT_STEP_{}_LR_{}_DROP_{}_NEURON_{}_MODEL_{}.png'.format(
                    dataset.lower(),
                    model,
                    step,
                    lr,
                    drop,
                    neuron,
                    model)

    plt.title('Loss')
    plt.plot([i for i in range(2000)], loss, color='g')
    #plt.show()
    plt.savefig(path_loss_fig)
    plt.close()

    plt.title('Acc')
    plt.plot([i for i in range(2000)], accs, color='b')
    #plt.show()
    plt.savefig(path_acc_fig)
    plt.close()

    print('Plots saved for', (step, lr, drop, neuron, model, dataset))

def build_heat_map(step, lr, drop, neuron, model, dataset):

    data_frames = []

    for i in range(5):

        if step == 0:

            path_for_files = 'results/{}-dataset/logs/{}/FullConnected_KFOLD_{}_STEP_0_LR_{}_DROP_{}_ARC_{}_NEURON_{}.csv'.format(
                dataset.lower(),
                model,
                i,
                lr,
                drop,
                model,
                neuron)

        elif step > 0 and step <= 10:

            path_for_files = 'results/{}-dataset/logs/{}/Random_Walk_KFOLD_{}_STEP_{}_LR_{}_DROP_{}_ARC_{}_NEURON_{}.csv'.format(
                dataset.lower(),
                model,
                i,
                step,
                lr,
                drop,
                model,
                neuron)

        else:

            path_for_files = 'results/{}-dataset/logs/{}/Random_Cut_KFOLD_{}_STEP_{}_LR_{}_DROP_{}_ARC_{}_NEURON_{}.csv'.format(
                dataset.lower(),
                model,
                i,
                step,
                lr,
                drop,
                model,
                neuron)

        current_df = pd.read_csv(path_for_files)

        data_frames.append(current_df)

        columns = [column for column in current_df]

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

            means.append(int(np.mean([column_data_df_1, column_data_df_2, column_data_df_3, column_data_df_4, column_data_df_5])))

        all_data.append(means)

    all_data = np.array(all_data)

    final_df = pd.DataFrame(all_data, columns=columns, index=columns)

    csv_file = 'results/csv/{}-dataset/simple_results/{}/Confusion_Matrix_STEP_{}_LR_{}_DROP_{}_NEURON_{}_MODEL_{}.csv'.format(
        dataset.lower(),
        model,
        step,
        lr,
        drop,
        neuron,
        model)

    final_df.to_csv(csv_file)

    confusion_matrix_path = 'results/csv/{}-dataset/confusion_matrix/{}/Confusion_Matrix_STEP_{}_LR_{}_DROP_{}_NEURON_{}_MODEL_{}.png'.format(
        dataset.lower(),
        model,
        step,
        lr,
        drop,
        neuron,
        model)




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

    print('Heat Map created for', (step, lr, drop, neuron, model, dataset))

def get_acc_from_each_label(step, lr, drop, neuron, model, dataset):

    class_evaluation = []

    columns = []

    if dataset == 'UNREL' and model == 'VGG16' and neuron == 32:

        neuron = 35


    for i in range(5):

        if step == 0:

            path_for_files = 'results/{}-dataset/logs/{}/FullConnected_KFOLD_{}_STEP_0_LR_{}_DROP_{}_ARC_{}_NEURON_{}.txt'.format(
                dataset.lower(),
                model,
                i,
                lr,
                drop,
                model,
                neuron)

        elif step > 0 and step <= 10:

            path_for_files = 'results/{}-dataset/logs/{}/Random_Walk_KFOLD_{}_STEP_{}_LR_{}_DROP_{}_ARC_{}_NEURON_{}.txt'.format(
                dataset.lower(),
                model,
                i,
                step,
                lr,
                drop,
                model,
                neuron)

        else:

            path_for_files = 'results/{}-dataset/logs/{}/Random_Cut_KFOLD_{}_STEP_{}_LR_{}_DROP_{}_ARC_{}_NEURON_{}.txt'.format(
                dataset.lower(),
                model,
                i,
                step,
                lr,
                drop,
                model,
                neuron)

        file = open(path_for_files).read()

        fold = []

        if dataset == 'UNREL':

            all_lines = file.split('\n')

            class_acc = all_lines[2006:2063]

            columns = []

            for accs in class_acc:

                acc_metadata = accs.split('\n')

                simplex_data = []

                for metadata in acc_metadata:

                    line_filtered = metadata.split(' ')

                    for line in line_filtered:

                        if line != '':

                            simplex_data.append(line)

                if simplex_data[0] not in columns:
                    columns.append(simplex_data[0])
                simplex_data.pop(0)
                fold.append(simplex_data)

        if dataset == 'MIT67':

            all_lines = file.split('\n')

            class_acc = all_lines[2006:2059]

            columns = []

            for accs in class_acc:

                acc_metadata = accs.split('\n')

                simplex_data = []

                for metadata in acc_metadata:

                    line_filtered = metadata.split(' ')

                    for line in line_filtered:

                        if line != '':

                            simplex_data.append(line)

                if simplex_data[0] not in columns:
                    columns.append(simplex_data[0])
                simplex_data.pop(0)
                fold.append(simplex_data)
        class_evaluation.append(fold)

    data = []

    metadatas = get_class_accuracy(step, lr, drop, neuron, model, dataset)

    for i in range(len(class_evaluation[0])):

        pre_results = []

        precision_1 = float(class_evaluation[0][i][0])
        precision_2 = float(class_evaluation[1][i][0])
        precision_3 = float(class_evaluation[2][i][0])
        precision_4 = float(class_evaluation[3][i][0])
        precision_5 = float(class_evaluation[4][i][0])

        mean_precision = round(float(np.mean([precision_1, precision_2, precision_3, precision_4, precision_5])), 7)
        std_precision = round(float(np.std([precision_1, precision_2, precision_3, precision_4, precision_5])), 7)

        recall_1 = float(class_evaluation[0][i][1])
        recall_2 = float(class_evaluation[1][i][1])
        recall_3 = float(class_evaluation[2][i][1])
        recall_4 = float(class_evaluation[3][i][1])
        recall_5 = float(class_evaluation[4][i][1])

        mean_recall = round(float(np.mean([recall_1, recall_2, recall_3, recall_4, recall_5])), 7)
        std_recall = round(float(np.std([recall_1, recall_2, recall_3, recall_4, recall_5])), 7)

        fscore_1 = float(class_evaluation[0][i][2])
        fscore_2 = float(class_evaluation[1][i][2])
        fscore_3 = float(class_evaluation[2][i][2])
        fscore_4 = float(class_evaluation[3][i][2])
        fscore_5 = float(class_evaluation[4][i][2])

        mean_fscore = round(float(np.mean([fscore_1, fscore_2, fscore_3, fscore_4, fscore_5])), 7)
        std_fscore = round(float(np.std([fscore_1, fscore_2, fscore_3, fscore_4, fscore_5])), 7)

        metadata = metadatas[i]

        pre_results.append(metadata['accuracy'])
        pre_results.append(metadata['std'])
        pre_results.append(mean_precision)
        pre_results.append(std_precision)
        pre_results.append(mean_recall)
        pre_results.append(std_recall)
        pre_results.append(mean_fscore)
        pre_results.append(std_fscore)
        pre_results.append(class_evaluation[4][i][3])
        data.append(pre_results)

    data = np.array(data)

    df = pd.DataFrame(data, columns=['accuracy', 'accuracy std', 'precision', 'precision std', 'recall', 'recall std', 'f1-score', 'f1-score std', 'support'], index=columns)

    path = 'results/csv/{}-dataset/simple_results/{}/ClassEvaluation_STEP_{}_LR_{}_DROP_{}_NEURON_{}_MODEL_{}.csv'.format(dataset.lower(),
                                                                                                                       model,
                                                                                                                       step,
                                                                                                                       lr,
                                                                                                                       drop,
                                                                                                                       neuron,
                                                                                                                       model)

    df.to_csv(path)

    print('Class Evaluation Done for', (step, lr, drop, neuron, model, dataset))

def global_evaluation(step, lr, drop, neuron, model, dataset):

    if step == 0:

        path_to_the_file = 'results/{}-dataset/global_evaluation/{}/{}_Full_Connected_{}_{}_{}_{}.txt'.format(dataset.lower(),
                                                                                                              model,
                                                                                                           dataset,
                                                                                                           lr,
                                                                                                           drop,
                                                                                                           model,
                                                                                                           neuron)
        time_path = 'process_time/{}/Full Connected Time {}.txt'.format(dataset, model)
        size_path = 'process_size/{}/Full Connected graph size {}.txt'.format(dataset, model)
    elif step > 0 and step <=10:

        path_to_the_file = 'results/{}-dataset/global_evaluation/{}/{}_RandomWalk_{}_{}_{}_{}_{}.txt'.format(
            dataset.lower(),
            model,
            dataset,
            step,
            lr,
            drop,
            model,
            neuron)

        time_path = 'process_time/{}/Random Walk {} Time {}.txt'.format(dataset, step, model)
        size_path = 'process_size/{}/Random Walk graph size {} {}.txt'.format(dataset, model, step)
    else:
        path_to_the_file = 'results/{}-dataset/global_evaluation/{}/{}_RandomCut_{}_{}_{}_{}_{}.txt'.format(
            dataset.lower(),
            model,
            dataset,
            step,
            lr,
            drop,
            model,
            neuron)

        time_path = 'process_time/{}/Random Cut {} Time {}.txt'.format(dataset, step - 10, model)
        size_path = 'process_size/{}/Random Cut graph size {} {}.txt'.format(dataset, model, step - 10)

    file = open(path_to_the_file).read()
    metadata = file.split('\n')

    meta_std = metadata[0]
    meta_std = meta_std.split(' ')

    meta_mean = metadata[1]
    meta_mean = meta_mean.split(' ')

    meta_time_mean = metadata[2]
    meta_time_mean = meta_time_mean.split(' ')

    meta_nodes_mean = metadata[3]
    meta_nodes_mean = meta_nodes_mean.split(' ')

    meta_edges_mean = metadata[4]
    meta_edges_mean = meta_edges_mean.split(' ')

    meta_folds = metadata[5:len(metadata) - 1]

    folds = []

    for meta in meta_folds:

        meta_fold = meta.split(' ')

        folds.append(float(meta_fold[3]))

    median = np.median(folds)
    std = meta_std[5].replace('.', ',')
    mean = float(meta_mean[5])
    time = meta_time_mean[5].replace('.', ',')
    nodes = int(float(meta_nodes_mean[5]))
    edges = int(float(meta_edges_mean[4]))

    fl_build_time = open(time_path).read()
    lines = fl_build_time.split('\n')
    fsrt_line = lines[0]
    fsrt_line_metadata = fsrt_line.split(' ')
    time = fsrt_line_metadata[2].replace('.', ',')


    fl_size_process = open(size_path).read()
    lines = fl_size_process.split('\n')
    fsrt_line = lines[0]
    fsrt_line_metadata = fsrt_line.split(' ')
    bt_size = fsrt_line_metadata[4]


    scnd_line = lines[1]
    scnd_line_metadata = scnd_line.split(' ')
    mb_size = scnd_line_metadata[4].replace('.', ',')




    line_to_write = '{};{};{};{};{};{};{};{} \n'.format(mean, median, std, time, nodes, edges, bt_size, mb_size)

    global_file.write(line_to_write)

    print('Global evaluation done for', (step, lr, drop, neuron, model, dataset))

def build_plots_from_test_configuration(lr, drop, neuron, model, dataset):

    results = []


    for step in range(11):

        dict_result = {'Method':'',
                       'Step':0,
                       'Mean-Accuracy':0,
                       'Median-Accuracy':0,
                       'Standard Deviation':0,
                       'Time':0, 'Edges':0,
                       'Nodes':0}

        if step == 0:

            path_for_files = 'results/{}-dataset/global_evaluation/{}/{}_Full_Connected_{}_{}_{}_{}.txt'.format(
                dataset.lower(),
                model,
                dataset,
                lr,
                drop,
                model,
                neuron)

            file = open(path_for_files).read()

            lines = file.split('\n')

            fst_line_std_metadata = lines[0].split(' ')
            scd_line_mean_metadata = lines[1].split(' ')
            trd_line_time_metadata = lines[2].split(' ')
            frt_line_nodes_metadata = lines[3].split(' ')
            fth_line_edges_metadata = lines[4].split(' ')

            still_median_metadata = lines[5:len(lines) - 1]

            median = []

            for line in still_median_metadata:

                line_metadata = line.split(' ')

                med = float(line_metadata[3])
                median.append(med)

            std = float(fst_line_std_metadata[5])
            mean = float(scd_line_mean_metadata[5])
            time = float(trd_line_time_metadata[5])
            nodes = float(frt_line_nodes_metadata[5])
            edges = float(fth_line_edges_metadata[4])

            dict_result['Standard Deviation'] = std
            dict_result['Median-Accuracy'] = float(np.median(median))
            dict_result['Mean-Accuracy'] = mean
            dict_result['Method'] = 'Full Connected'
            dict_result['Edges'] = edges
            dict_result['Nodes'] = nodes
            dict_result['Time'] = time
            dict_result['Step'] = 0
            results.append(dict_result)

        elif step > 0 and step <= 10:

            path_for_files = 'results/{}-dataset/global_evaluation/{}/{}_RandomWalk_{}_{}_{}_{}_{}.txt'.format(
                dataset.lower(),
                model,
                dataset,
                step,
                lr,
                drop,
                model,
                neuron)

            file = open(path_for_files).read()

            lines = file.split('\n')

            fst_line_std_metadata = lines[0].split(' ')
            scd_line_mean_metadata = lines[1].split(' ')
            trd_line_time_metadata = lines[2].split(' ')
            frt_line_nodes_metadata = lines[3].split(' ')
            fth_line_edges_metadata = lines[4].split(' ')

            still_median_metadata = lines[5:len(lines) - 1]

            median = []

            for line in still_median_metadata:
                line_metadata = line.split(' ')

                med = float(line_metadata[3])
                median.append(med)

            std = float(fst_line_std_metadata[5])
            mean = float(scd_line_mean_metadata[5])
            time = float(trd_line_time_metadata[5])
            nodes = float(frt_line_nodes_metadata[5])
            edges = float(fth_line_edges_metadata[4])

            dict_result['Standard Deviation'] = std
            dict_result['Median-Accuracy'] = float(np.median(median))
            dict_result['Mean-Accuracy'] = mean
            dict_result['Method'] = 'Random Walk'
            dict_result['Edges'] = edges
            dict_result['Nodes'] = nodes
            dict_result['Time'] = time
            dict_result['Step'] = step
            results.append(dict_result)

        else:

            path_for_files = 'results/{}-dataset/global_evaluation/{}/{}_RandomCut_{}_{}_{}_{}_{}.txt'.format(
                dataset.lower(),
                model,
                dataset,
                step,
                lr,
                drop,
                model,
                neuron)

            file = open(path_for_files).read()

            lines = file.split('\n')

            fst_line_std_metadata = lines[0].split(' ')
            scd_line_mean_metadata = lines[1].split(' ')
            trd_line_time_metadata = lines[2].split(' ')
            frt_line_nodes_metadata = lines[3].split(' ')
            fth_line_edges_metadata = lines[4].split(' ')

            still_median_metadata = lines[5:len(lines) - 1]

            median = []

            for line in still_median_metadata:
                line_metadata = line.split(' ')

                med = float(line_metadata[3])
                median.append(med)

            std = float(fst_line_std_metadata[5])
            mean = float(scd_line_mean_metadata[5])
            time = float(trd_line_time_metadata[5])
            nodes = float(frt_line_nodes_metadata[5])
            edges = float(fth_line_edges_metadata[4])

            dict_result['Standard Deviation'] = std
            dict_result['Median-Accuracy'] = float(np.median(median))
            dict_result['Mean-Accuracy'] = mean
            dict_result['Method'] = 'Random Cut'
            dict_result['Edges'] = edges
            dict_result['Nodes'] = nodes
            dict_result['Time'] = time
            dict_result['Step'] = step
            results.append(dict_result)


    path = 'results/plots/{}-dataset/Test/{}/{}_Global Evaluation {} {} {} {}.png'.format(dataset.lower(),
                                                                                model,
                                                                                dataset,
                                                                                lr,
                                                                                drop,
                                                                                model,
                                                                                neuron)
    fig, axs = plt.subplots(2)
    fig.set_figheight(15)
    fig.set_figwidth(10)
    fig.suptitle('Global Evaluation')
    axs[0].set_title('Mean Values')
    axs[0].errorbar([result['Method'] + str(result['Step']) for result in results],
                 [result['Mean-Accuracy'] for result in results],
                 [result['Standard Deviation'] for result in results],
                 ecolor='b', color='r', fmt='-o')
    axs[0].set_xticklabels(labels=[result['Method'] + str(result['Step']) for result in results], rotation=45)


    axs[1].set_title('Median Values')
    axs[1].errorbar([result['Method'] + str(result['Step']) for result in results],
                    [result['Median-Accuracy'] for result in results],
                    [result['Standard Deviation'] for result in results],
                    ecolor='b', color='g', fmt='-o')
    axs[1].set_xticklabels(labels=[result['Method'] + str(result['Step']) for result in results], rotation=45)


    plt.savefig(path)
    plt.close()
    print('Global Evaluation done for', (lr, drop, neuron, model, dataset))

def main():

    steps = [i for i in range(11)]
    models = ['InceptionResNetV2']
    dropouts = [0.3, 0.5, 0.8, 0.9]
    neurons = [16, 32, 64, 128, 256]
    learning_rates = [0.01, 0.05, 0.001, 0.005]
    datasets = ['MIT67']

    for dataset in datasets:

        for model in models:

            for lr in learning_rates:

                for dropout in dropouts:

                    for neuron in neurons:

                        global_file.write('Lr_{}_drop_{}_neuron_{}'.format(lr, dropout, neuron) + '\n')
                        global_file.write('MÉTODO;MÉDIA;MEDIANA;DESVIO PADRÃO;BUILD TIME; NÓS; ARESTAS; SIZE B; SIZE MB;' + '\n')

                        for step in steps:

                            global_file.write('Step={};'.format(step))
                            build_plots_from_configuration(step=step, lr=lr, drop=dropout, neuron=neuron, model=model, dataset=dataset)
                            build_heat_map(step=step, lr=lr, drop=dropout, neuron=neuron, model=model, dataset=dataset)
                            get_acc_from_each_label(step=step, lr=lr, drop=dropout, neuron=neuron, model=model, dataset=dataset)
                            global_evaluation(step=step, lr=lr, drop=dropout, neuron=neuron, model=model, dataset=dataset)

                        build_plots_from_test_configuration(lr=lr, drop=dropout, neuron=neuron, model=model, dataset=dataset)

                        global_file.write('\n')

if __name__ == '__main__':
    main()

