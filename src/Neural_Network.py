import keras
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
import numpy as np
import pandas as pd
from keras.regularizers import l1
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, log_loss
import matplotlib.pyplot as plt
import seaborn as sn
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
import pickle
import os
import time
from glob import glob
import matplotlib.pyplot as plt
import matplotlib


tf.random.set_seed(1234)


def cm2df(cm, labels):
    df = pd.DataFrame()
    # rows
    for i, row_label in enumerate(labels):
        rowdata = {}
        # columns
        for j, col_label in enumerate(labels):
            rowdata[col_label] = cm[i, j]
        df = df.append(pd.DataFrame.from_dict({row_label: rowdata}, orient='index'))
    return df[labels]

def get_model(model):

    dim = (224, 224, 3)

    if model == 'ResNet50':
        return tf.keras.applications.resnet50.ResNet50(weights='imagenet', include_top=False,  input_shape=dim, pooling='avg')
    elif model == 'Xception':
        return tf.keras.applications.xception.Xception(weights='imagenet', include_top=False, input_shape=dim, pooling='avg')
    elif model == 'InceptionV3':
        return tf.keras.applications.inception_v3.InceptionV3(weights='imagenet', include_top=False, input_shape=dim, pooling='avg')
    elif model == 'VGG16':
        return tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=dim, pooling='avg')
    elif model == 'InceptionResNetV2':
        return tf.keras.applications.inception_resnet_v2.InceptionResNetV2(weights='imagenet', include_top=False,  input_shape=dim, pooling='avg')
    else:
        return efn.EfficientNetB7(weights="imagenet", include_top=False, input_shape=dim, pooling='avg')

def get_pre_process(model):

    if model == 'ResNet50':
        return tf.keras.applications.resnet50.preprocess_input
    elif model == 'Xception':
        return tf.keras.applications.xception.preprocess_input
    elif model == 'InceptionV3':
        return tf.keras.applications.inception_v3.preprocess_input
    elif model == 'VGG16':
        return tf.keras.applications.vgg16.preprocess_input
    elif model == 'InceptionResNetV2':
        return tf.keras.applications.inception_resnet_v2.preprocess_input
    else:
        return efn.preprocess_input

EPOCHS=2000
DATASET = 'UNREL'
batch_size = 32
models = ['ResNet50']
lrs = [0.01, 0.001, 0.05, 0.005]
dropouts = [0.3, 0.5, 0.8, 0.9]
folds = [i for i in range(5)]

for model_name in models:

    for lr in lrs:

        for dropout in dropouts:

            results = []

            for fold in folds:

                LOGS_FOLD = 'cnn_logs/{}/STD/LOGS_KFOLD/LOG_FOLD_{}_LR_{}_DROP_{}_MODEL_{}.txt'.format(DATASET, fold, lr, dropout, model_name)
                logs_file = open(LOGS_FOLD, 'w')

                train_dir = 'data/{}/fold_{}/TRAIN'.format(DATASET, fold)
                test_dir = 'data/{}/fold_{}/TEST'.format(DATASET, fold)

                labels = os.listdir(train_dir)
                classes = len(labels)

                try:
                    model_path = 'CNN_MODELS/{}_CNN_MODEL_{}.h5'.format(DATASET, model_name)

                    final_model = tf.keras.models.load_model(model_path)

                    print('Modelo carregado com sucesso!!')

                except:

                    base_model = get_model(model_name)

                    for layer in base_model.layers:
                        layer.trainable = False

                    x = base_model.output
                    x = tf.keras.layers.Dropout(dropout)(x)
                    predictions = tf.keras.layers.Dense(classes, activation='softmax')(x)
                    final_model = tf.keras.models.Model(base_model.input, predictions)
                    print('Modelo criado com sucesso!!')

                    final_model.layers[-1].trainable = True


                for l in final_model.layers:

                    print(l.name, l.trainable)

                # inicializando o objeto que ira recuperar as amostras de treino com a funcao de pre-processamento da ResNet50
                train_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                                                 shear_range=10,
                                                                                 zoom_range=0.2,
                                                                                 horizontal_flip=True,
                                                    preprocessing_function=get_pre_process(model_name))  # included in our dependencies

                # inicializando o objeto que ira recuperar as amostras de teste com a funcao de pre-processamento da InceptionV3
                test_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                   preprocessing_function=get_pre_process(model_name))
                train_it = train_data_gen.flow_from_directory(train_dir,
                                                             class_mode='categorical',
                                                             batch_size=batch_size,
                                                             target_size=(224, 224),
                                                             shuffle=True)
                # load and iterate test dataset
                test_it = test_data_gen.flow_from_directory(test_dir,
                                                             class_mode='categorical',
                                                             batch_size=32,
                                                             target_size=(224, 224),
                                                             shuffle=False)

                adam = tf.keras.optimizers.Adam(learning_rate=lr)

                # fucao de custo e metrica
                final_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])


                start = time.time()
                H = final_model.fit(train_it,
                                    steps_per_epoch=10,
                                    epochs=EPOCHS)

                final = time.time()

                process_time = final - start
                print('Evaluating..')
                loss = final_model.evaluate_generator(train_it, steps=32, verbose=1)
                predictions = final_model.predict_generator(test_it, steps=15)
                y_pred = np.argmax(predictions, axis=1)
                test_acc = accuracy_score(test_it.classes, y_pred)
                print('Train Loss:', loss[0])
                print('Train Accuracy:', loss[1])
                print('Test Accuracy:', test_acc)
                #print('Accuracy:', test_acc)

                ACC_LOG = 'cnn_logs/{}/ACC/fold_{}_ACC.txt'.format(DATASET, fold)

                acc_file = open(ACC_LOG, 'w')
                acc_file.write('Accuracy: ' + str(test_acc) + '\n')
                acc_file.close()

                classification = classification_report(test_it.classes, y_pred, target_names=labels)

                results.append(test_acc)

                accs = H.history['accuracy']
                losses = H.history['loss']
                epochs = [i for i in range(EPOCHS)]

                count = 0

                for loss, acc in zip(losses, accs):
                    logs_file.write('Epoch ' + str(count) + ' | Loss: ' + str(loss) + ' | Train_Accuracy: ' + str(acc) + '\n')
                    count = count + 1

                logs_file.write('Process Time: ' + str(process_time) + '\n')
                logs_file.write('Test Accuracy: ' + str(test_acc) + '\n')

                logs_file.write(str(classification))
                logs_file.close()

                cm = confusion_matrix(test_it.classes, y_pred)
                print('Confusion matrix done..')
                cm = cm2df(cm, labels)
                cm.to_csv(LOGS_FOLD.replace('.txt', '.csv'))
                final_model.save('CNN_MODELS/{}_CNN_MODEL_{}.h5'.format(DATASET, model_name))
                print('MODELO SALVO COM SUCESSO')

            STD_LOG = 'cnn_logs/{}/STD/{}_{}_{}.txt'.format(DATASET, model_name, lr, dropout)

            between = open(STD_LOG, 'w')
            print('Standard deviation of ' + model_name + ' = ',
                  np.std(results))
            print('Mean between the experiments is', np.mean(results))
            between.write('Standard deviation of ' + model_name + ' ' + str(
                np.std(results)))
            between.write('\n')
            between.write(
                'Mean between the experiments is ' + str(np.mean(results)) + '\n')
            sent = 0
            for result in results:
                between.write('Conjunto ' + str(sent) + '=' + str(result) + '\n')
                sent = sent + 1
            between.close()



















