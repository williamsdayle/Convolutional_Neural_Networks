import numpy as np
import os
import pickle
import scipy #from scipy.sparse import csr_matrix
import tensorflow as tf
import traceback

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

from tensorflow.keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications.nasnet import NASNetLarge, NASNetMobile

import tensorflow.keras.applications as app

#Importing EfficientNet
import efficientnet.tfkeras as efn
from efficientnet.tfkeras import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def named_model(name):

    if name == 'InceptionV3':
        return InceptionV3(weights='imagenet', include_top=False, pooling='max', input_shape=(224, 224, 3))

    if name == 'ResNet50':
        return ResNet50(weights='imagenet', include_top=False, pooling='max', input_shape=(224, 224, 3))

    if name == 'VGG16':
        return VGG16(weights='imagenet', include_top=False, pooling='max', input_shape=(224, 224, 3))

    if name == 'Xception':
        return Xception(weights='imagenet', include_top=False, pooling='max', input_shape=(224, 224, 3))

    if name == 'InceptionResNetV2':
        return InceptionResNetV2(weights='imagenet', include_top=False, pooling='max', input_shape=(224, 224, 3))

    if name == 'EfficientNet':
        return efn.EfficientNetB7(weights='imagenet', include_top=False, pooling='max', input_shape=(224, 224, 3))



def get_preprocessed_image(name, image_path):

    img = image.load_img(image_path, target_size=(224, 224, 3))

    # convert image to numpy array
    x = image.img_to_array(img)
    # the image is now in an array of shape (3, 224, 224)
    # but we need to expand it to (1, 2, 224, 224) as Keras is expecting a list of images
    x = np.expand_dims(x, axis=0)

    if name == 'InceptionV3':
        from tensorflow.keras.applications.inception_v3 import preprocess_input
        return preprocess_input(x)

    if name == 'ResNet50':
        from tensorflow.keras.applications.resnet50 import preprocess_input
        return preprocess_input(x)

    if name == 'VGG16':
        from tensorflow.keras.applications.vgg16 import preprocess_input
        return preprocess_input(x)

    if name == 'Xception':
        from tensorflow.keras.applications.xception import preprocess_input
        return preprocess_input(x)

    if name == 'InceptionResNetV2':
        from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
        return preprocess_input(x)

    if name == 'EfficientNet':
        from efficientnet.tfkeras import preprocess_input
        return preprocess_input(x)

def get_features(x, model):

    features = model.predict(x)[0]

    features_arr = np.char.mod('%f', features)

    return features_arr


def array_to_str(array):


    text = ''
    for value in array:

        text = text + '%.8f' % float(value) + ','

    final_text = text[:len(text) - 1]

    return final_text


def save_file(model, model_name, dataset):

    examples_path = os.path.join(BASE_DIR, 'Examples')

    features_example_path = os.path.join(BASE_DIR, 'Feature_Examples')

    model_path = os.path.join(os.path.join(features_example_path, dataset), model_name)

    if os.path.isdir(model_path):
        print('Path for the model already exists')
    else:
        os.mkdir(model_path)

    examples_path_dataset = os.path.join(examples_path, dataset)

    example_classes = os.listdir(examples_path_dataset)

    for classes in example_classes:

        example_class_path = os.path.join(examples_path_dataset, classes)

        features_example_class_path = os.path.join(model_path, classes)

        if os.path.isdir(features_example_class_path):

            print('Path para essa classe já foi criado')
        else:
            os.mkdir(features_example_class_path)

        class_images = os.listdir(example_class_path)

        size = len(class_images)

        for class_image in class_images:

            example_image_path = os.path.join(example_class_path, class_image)

            x = get_preprocessed_image(model_name, example_image_path)

            features = get_features(x, model)

            str_features = array_to_str(features)

            txt_file = class_image.replace('.jpg', '.txt')

            save_path = os.path.join(features_example_class_path, txt_file)

            file = open(save_path, 'w')

            file.write(str_features)

            file.close()

            print('Image', class_image, 'extracted and saved!')

            size = size - 1

            print('Still', size)


datasets = ['UNREL', 'MIT67', 'VRD']

models = ['EfficientNet', 'VGG16', 'ResNet50', 'InceptionResNetV2', 'InceptionV3', 'Xception']

for dataset in datasets:

    for model_str in models:

        model = named_model(model_str)

        save_file(model=model, model_name=model_str, dataset=dataset)










