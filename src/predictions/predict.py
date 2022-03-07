import torch as th
import cv2 as cv
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
from torchvision import transforms
from torch.autograd import Variable
from tensorflow.keras.preprocessing import image
import efficientnet.tfkeras as efn
from efficientnet.tfkeras import EfficientNetB7
import tensorflow as tf
import sys
from math import exp

def get_labels(base):

    f = open('{}_labels.txt'.format(base)).read()

    labels = []

    labels_in_file = f.split(',')

    for split in labels_in_file:

        if split.count('[') > 0:

            split = split.replace('[', '')

        if split.count('\'') > 0:

            split = split.replace('\'', '')

        if split.count(']') > 0:

            split = split.replace(']', '')

        labels.append(split)

    return labels

def pre_process_and_model(model):

    input_shape = (244, 244, 3)

    if model == 'VGG16':

        preprocessed_image = tf.keras.applications.vgg16.preprocess_input

        extractor_model = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, pooling="avg", input_shape=input_shape)

        return preprocessed_image, extractor_model

    if model == 'EfficientNet':

        from efficientnet.tfkeras import preprocess_input as efficient_preprocess

        preprocessed_image = efficient_preprocess

        extractor_model = efn.EfficientNetB7(weights="imagenet", include_top=False, pooling="avg", input_shape=input_shape)

        return preprocessed_image, extractor_model

    if model == 'ResNet50':

        preprocessed_image = tf.keras.applications.resnet50.preprocess_input

        extractor_model = tf.keras.applications.resnet50.ResNet50(weights='imagenet', include_top=False, pooling="avg", input_shape=input_shape)

        return preprocessed_image, extractor_model

    if model == 'Xception':

        preprocessed_image = tf.keras.applications.xception.preprocess_input

        extractor_model = tf.keras.applications.xception.Xception(weights='imagenet', include_top=False, pooling="avg", input_shape=input_shape)

        return preprocessed_image, extractor_model

    if model == 'InceptionResNetV2':

        preprocessed_image = tf.keras.applications.inception_resnet_v2.preprocess_input

        extractor_model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(weights='imagenet', include_top=False, pooling="avg", input_shape=input_shape)

        return preprocessed_image, extractor_model

    if model == 'InceptionV3':

        preprocessed_image = tf.keras.applications.inception_v3.preprocess_input

        extractor_model = tf.keras.applications.inception_v3.InceptionV3(weights='imagenet', include_top=False, pooling="avg", input_shape=input_shape)

        return preprocessed_image, extractor_model

def predict(base, image_name, model, true_label, walk, flatten_array):
	
    labels = get_labels(base)

    random_walk_path = 'models/{}/random_walk_model_{}_{}_{}.pth'.format(model, base, model, walk)

    fully_connected_path = 'models/{}/full_connected_model_{}_{}_0.pth'.format(model, base, model)

    random_walk_model = th.load(random_walk_path, map_location='cpu')
    fully_connected_model = th.load(fully_connected_path, map_location='cpu')

    print('Model loaded')

    test_transforms = th.FloatTensor(flatten_array)

    image_tensor = test_transforms.unsqueeze_(0)
    input = Variable(image_tensor)

    random_walk_model.eval()

    output_rdn = random_walk_model(input)

    _, predicted_rdn = th.max(output_rdn, 1)

    classification_rdn = predicted_rdn.data.cpu().numpy()

    index_rn = classification_rdn[0]

    output_numpy = output_rdn.data.cpu().numpy()

    probs_preprocess = []

    for i in output_numpy[0]:

        probs_preprocess.append(exp(i / 10000))

    rw_prob = probs_preprocess[index_rn] / sum(probs_preprocess)

    print('Predicted in this image with Random Walk model', labels[index_rn], rw_prob)

    fully_connected_model.eval()

    output_fc = fully_connected_model(input)

    _, predicted_fc = th.max(output_fc, 1)

    classification_fc = predicted_fc.data.cpu().numpy()

    index_fc = classification_fc[0]

    output_numpy = output_fc.data.cpu().numpy()

    probs_preprocess = []

    for i in output_numpy[0]:

        probs_preprocess.append(exp(i / 10000))

    fc_prob = probs_preprocess[index_fc] / sum(probs_preprocess)

    print('Predicted in this image with Fully Connected model', labels[index_fc], fc_prob)
    print('The true label was ' + true_label)

    cv_img = mpimg.imread('examples/{}/{}'.format(true_label, image_name))

    red_patch = mpatches.Patch(color='red', label='Random Walk ' + str(walk) + ' ' + labels[index_rn] + ' ' + str(round(rw_prob * 100, 3)) + '%')
    black_patch = mpatches.Patch(color='black', label='Full Connected ' + ' ' + labels[index_fc] + ' ' + str(round(fc_prob * 100, 3)) + '%')
    green_patch = mpatches.Patch(color='green', label='True Label ' + true_label)

    plt.legend(handles=[red_patch, black_patch, green_patch])
    plt.imshow(cv_img)
    plt.savefig('results/{}/{}/{}_prediction_walk_{}.jpg'.format(base, model, image_name.replace('.jpg', ''), walk), dpi=250)

def main():

    base = 'MIT67'
    models = ['ResNet50', 'VGG16', 'InceptionResNetV2', 'Xception']
    walks = [i for i in range(1, 11)]

    if base == 'MIT67':

        images = [

            {'image_name': '5.jpg',
             'true_label': 'airport_inside'},

            {'image_name': '6.jpg',
             'true_label': 'airport_inside'},

            {'image_name': '7.jpg',
             'true_label': 'airport_inside'},

            {'image_name': '2.jpg',
             'true_label': 'bedroom'},

            {'image_name': '4.jpg',
             'true_label': 'bedroom'},

            {'image_name': '1.jpg',
             'true_label': 'bedroom'},

            {'image_name': '26.jpg',
             'true_label': 'church_inside'},

            {'image_name': '27.jpg',
             'true_label': 'church_inside'},

            {'image_name': '11.jpg',
             'true_label': 'tv_studio'},

            {'image_name': '12.jpg',
             'true_label': 'tv_studio'},

            {'image_name': '8.jpg',
             'true_label': 'computerroom'},

            {'image_name': '9.jpg',
             'true_label': 'computerroom'},

        ]

    if base == 'UNREL':
        pass
    if base == 'VRD':
        pass

    for model in models:

        preprocess_input, pre_trained_model = pre_process_and_model(model)

        for image_ in images:

            print('Pre processing the image...')

            image_name = image_['image_name']

            true_label = image_['true_label']

            img = image.load_img('examples/{}/{}'.format(true_label, image_name), target_size=(244, 244))

            image_array = image.img_to_array(img)

            image_expanded = np.expand_dims(image_array, axis=0)

            preprocessed_image = preprocess_input(image_expanded)

            feature_array = pre_trained_model.predict(preprocessed_image)

            flatten_array = feature_array.flatten()

            print('Image processed')

            for walk in walks:

                print('Evaluating Random Walk step {}...'.format(walk), 'with {} in image {}'.format(model, image_name))

                predict(base=base, image_name=image_name, model=model, true_label=true_label, walk=walk,
                        flatten_array=flatten_array)

if __name__ == '__main__':
    main()















