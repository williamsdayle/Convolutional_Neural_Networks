from email.mime import base
import natsort
import glob
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

import matplotlib
import os
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


print("------------PACKAGES - VERSIONS------------")
print("numpy version: %s" % np.__version__)
print("tensorflow version: %s" % tf.__version__)
print("pickle version: %s" % pickle.format_version)
print("keras version: %s" % tf.keras.__version__)
print("matplotlib version: %s" % matplotlib.__version__)
print("BASE DIR {}".format(BASE_DIR))
print("-------------------------------------------")


def get_model(model_name, input_shape, pooling):
    base_model = None
    preprocessing_function = None
    # Prepare the model
    if model_name == "VGG16":
        from tensorflow.keras.applications.vgg16 import preprocess_input
        preprocessing_function = preprocess_input
        base_model = VGG16(weights='imagenet', include_top=False, pooling=pooling, input_shape=input_shape)

    elif model_name == "VGG19":
        from tensorflow.keras.applications.vgg19 import preprocess_input
        preprocessing_function = preprocess_input

        base_model = VGG19(weights='imagenet', include_top=False, pooling=pooling, input_shape=input_shape)

    elif model_name == "ResNet50":
        from tensorflow.keras.applications.resnet50 import preprocess_input
        preprocessing_function = preprocess_input

        # pooling="avg",
        base_model = ResNet50(weights='imagenet', include_top=False, pooling=pooling, input_shape=input_shape)

    elif model_name == "InceptionResNetV2":

        from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
        preprocessing_function = preprocess_input

        # pooling="avg",
        base_model = InceptionResNetV2(weights='imagenet', include_top=False, pooling=pooling, input_shape=input_shape)

    elif model_name == "InceptionV3":
        from tensorflow.keras.applications.inception_v3 import preprocess_input
        preprocessing_function = preprocess_input
        base_model = InceptionV3(weights='imagenet', include_top=False, pooling=pooling, input_shape=input_shape)

    elif model_name == "Xception":
        from tensorflow.keras.applications.xception import preprocess_input
        preprocessing_function = preprocess_input
        base_model = Xception(weights='imagenet', include_top=False, pooling=pooling, input_shape=input_shape)

    elif model_name == "InceptionResNetV2":
        from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
        preprocessing_function = preprocess_input
        base_model = InceptionResNetV2(weights='imagenet', include_top=False, pooling=pooling, input_shape=input_shape)

    elif model_name == "MobileNet":
        from tensorflow.keras.applications.mobilenet import preprocess_input
        preprocessing_function = preprocess_input
        base_model = MobileNet(weights='imagenet', include_top=False, pooling=pooling, input_shape=input_shape)


    elif model_name == "MobileNetV2":
        from tensorflow.keras.applications.mobilenet import preprocess_input
        preprocessing_function = preprocess_input
        base_model = MobileNetV2(weights='imagenet', include_top=False, pooling=pooling, input_shape=input_shape)

    elif model_name == "DenseNet121":
        from tensorflow.keras.applications.densenet import preprocess_input
        preprocessing_function = preprocess_input
        base_model = DenseNet121(weights='imagenet', include_top=False, pooling=pooling, input_shape=input_shape)

    elif model_name == "DenseNet169":
        from tensorflow.keras.applications.densenet import preprocess_input
        preprocessing_function = preprocess_input
        base_model = DenseNet169(weights='imagenet', include_top=False, pooling=pooling, input_shape=input_shape)

    elif model_name == "DenseNet201":
        from tensorflow.keras.applications.densenet import preprocess_input
        preprocessing_function = preprocess_input
        base_model = DenseNet201(weights='imagenet', include_top=False, pooling=pooling, input_shape=input_shape)

    elif model_name == "NASNetLarge":
        from tensorflow.keras.applications.nasnet import preprocess_input
        preprocessing_function = preprocess_input
        base_model = NASNetLarge(weights='imagenet', include_top=False, pooling=pooling, input_shape=input_shape)

    elif model_name == "NASNetMobile":
        from tensorflow.keras.applications.nasnet import preprocess_input
        preprocessing_function = preprocess_input
        base_model = NASNetMobile(weights='imagenet', include_top=False, pooling=pooling, input_shape=input_shape)

    elif model_name == "EfficientNet":
        from efficientnet.tfkeras import preprocess_input
        preprocessing_function = preprocess_input

        base_model = efn.EfficientNetB7(weights="imagenet", include_top=False, pooling=pooling, input_shape=input_shape)

    return preprocessing_function, base_model


if __name__ == '__main__':

    models = ['VGG16', 'ResNet50', 'InceptionV3', 'InceptionResNetV2', 'Xception', 'EfficientNet']

    DATASETS = ['UNREL']
    
    pooling = "max"

    for DATASET in DATASETS:

        for model_ in models:

            model_name = model_
            base_path = os.path.join(BASE_DIR, "examples")
            src_path = os.path.join(base_path, DATASET)
            base_dst_path = os.path.join(BASE_DIR, "feature_files_{}".format(pooling))
            data_set_dst_path = os.path.join(base_dst_path, DATASET)
            dst_path = os.path.join(data_set_dst_path, model_)

            if os.path.isdir(dst_path):
                pass
            else:
                os.mkdir(dst_path)

            image_width = 224
            image_height = 224
            image_channels = 3
            dim = (image_width, image_height, image_channels)

            preprocessing_function, model = get_model(model_, dim, pooling)

            EXTENSION_IMAGE = '.jpg'
            EXTENSION_TEXT = '.txt'

            labels = os.listdir(src_path)

            for label in labels:

                label_path = os.path.join(dst_path, label)

                print(label_path)

                if os.path.isdir(label_path):
                    pass
                else:
                    os.mkdir(label_path)

                lst_dst_files = []

                loading_counter = 0

                files = os.listdir(os.path.join(src_path, label))

                for file_ in files:
                    file_path = os.path.join(os.path.join(src_path, label), file_)

                    dst_final_path = os.path.join(label_path, file_)

                    # label_path = os.path.join(dst_path, label)

                    # if os.path.isdir(label_path):
                    # pass
                    # else:
                    # os.mkdir(label_path)

                    # file_name = os.path.join(label_path, raw_file_name)

                    img = image.load_img(file_path, target_size=(image_width, image_height))

                    image_array = image.img_to_array(img)

                    image_expanded = np.expand_dims(image_array, axis=0)

                    preprocessed_image = preprocessing_function(image_expanded)

                    feature_array = model.predict(preprocessed_image)

                    flatten_array = feature_array.flatten()

                    feature_array_string = str([value for value in flatten_array]).replace("'", "").replace("[",
                                                                                                            "").replace(
                        "]",
                        "")
                    with open(dst_final_path.replace('.jpg', '.txt'), "w") as file:
                        file.write(feature_array_string)

                    print('Image name:', file_, 'Model:', model_, 'Feature size:', len(flatten_array))



