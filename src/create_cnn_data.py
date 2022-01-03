import pandas as pd
import numpy as np
import os
import shutil as sh


def move_data_for_train_and_test(dataset, fold):

    image_dir = 'bounding_boxes/{}-dataset/images'.format(dataset.lower())

    fold_dir = 'data/{}/fold_{}'.format(dataset, fold)

    if os.path.isdir(fold_dir):
        pass
    else:
        os.mkdir(fold_dir)


    train_new_image_dir = 'data/{}/fold_{}/TRAIN'.format(dataset, fold)
    test_new_image_dir = 'data/{}/fold_{}/TEST'.format(dataset, fold)

    if os.path.isdir(train_new_image_dir):
        pass
    else:
        os.mkdir(train_new_image_dir)

    if os.path.isdir(test_new_image_dir):
        pass
    else:
        os.mkdir(test_new_image_dir)

    df_train_files = pd.read_csv('benchmarks/{}-dataset/{}_{}_TRAIN_FILES.csv'.format(dataset.lower(), fold, dataset))
    df_test_files = pd.read_csv('benchmarks/{}-dataset/{}_{}_TEST_FILES.csv'.format(dataset.lower(), fold, dataset))

    df_train_labels = pd.read_csv('benchmarks/{}-dataset/{}_{}_TRAIN_LABELS.csv'.format(dataset.lower(), fold, dataset))
    df_test_labels = pd.read_csv('benchmarks/{}-dataset/{}_{}_TEST_LABELS.csv'.format(dataset.lower(), fold, dataset))

    train_data_loaded = np.array(df_train_files['Files'])
    train_labels_loaded = np.array(df_train_labels['Labels'])

    test_data_loaded = np.array(df_test_files['Files'])
    test_labels_loaded = np.array(df_test_labels['Labels'])

    for train_file, train_label in zip(train_data_loaded, train_labels_loaded):

        train_file_metadata = train_file.split('/')

        image_name = train_file_metadata[4].replace('.txt', '.jpg')

        old_path = os.path.join(image_dir, os.path.join(train_label, image_name))
        new_path = os.path.join(train_new_image_dir, train_label)

        if os.path.isdir(new_path):
            pass
        else:
            os.mkdir(new_path)

        src_path = old_path
        dst_path = os.path.join(new_path, image_name)

        sh.copy2(src_path, dst_path)
        print(image_name, 'MOVED')


    for test_file, test_label in zip(test_data_loaded, test_labels_loaded):

        test_file_metadata = test_file.split('/')

        image_name = test_file_metadata[4].replace('.txt', '.jpg')

        old_path = os.path.join(image_dir, os.path.join(test_label, image_name))
        new_path = os.path.join(test_new_image_dir, test_label)

        if os.path.isdir(new_path):
            pass
        else:
            os.mkdir(new_path)

        src_path = old_path
        dst_path = os.path.join(new_path, image_name)

        sh.copy2(src_path, dst_path)
        print(image_name, 'MOVED')








dataset = 'UNREL'

folds = [i for i in range(5)]

for fold in folds:

    move_data_for_train_and_test(dataset, fold)


