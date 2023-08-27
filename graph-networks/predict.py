import torch as th
from keras.applications.imagenet_utils import decode_predictions
import numpy as np
from torch.autograd import Variable
import efficientnet.tfkeras as efn
import tensorflow as tf
from keras.models import load_model
from math import exp
import os


BASE_DIR = os.path.dirname(os.path.realpath(__file__))


class Predict(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.images_to_predict_path = BASE_DIR + f"/results_from_{self.dataset.lower()}/{self.dataset.lower()}/predictions/images/"
        self.labels_path = BASE_DIR + f"/results_from_{self.dataset.lower()}/{self.dataset.lower()}/predictions/labels.txt"
    
    def get_images_to_predict(self):
        if self.dataset == "VRD":
            return [
                {
                    "name":"14104516_947fb0cfbb_o.jpg",
                    "label":"has",
                    "predictions": []
                },
                {
                    "name":"8249063496_f6b85c3d88_b.jpg",
                    "label":"in-front-of",
                    "predictions": []
                },
                {
                    "name":"40471576_c3c16e1151_b.jpg",
                    "label":"on",
                    "predictions": []
                },
                {
                    "name":"3227335672_e420c7fd21_b.jpg",
                    "label":"wearing",
                    "predictions": []
                }
            ]
        if self.dataset == "MALARIA":
            return [
                {
                    "name":"716273ea-19c4-49af-8604-df587b295eca.png",
                    "label":"infected",
                    "predictions": []
                },
                {
                    "name":"e63e1607-f344-43d6-a0d8-e10cf2e5054a.png",
                    "label":"infected",
                    "predictions": []
                },
                {
                    "name":"540a5fee-175b-4b92-bf62-1ea9b59c86c6.png",
                    "label":"uninfected",
                    "predictions": []
                },
                {
                    "name":"de81bf7a-7a56-437f-a068-4d9c3e01134c.png",
                    "label":"uninfected",
                    "predictions": []
                }
            ]
        if self.dataset == "MIT67":
            return [
                {
                    "name":"00.jpg",
                    "label":"computerroom",
                    "predictions": []
                },
                {
                    "name":"elevator_google_0034.jpg",
                    "label":"elevator",
                    "predictions": []
                },
                {
                    "name":"10_GARAJE_3_JPG.jpg",
                    "label":"garage",
                    "predictions": []
                },
                {
                    "name":"44l.jpg",
                    "label":"grocerystore",
                    "predictions": []
                }
            ]
        if self.dataset == "UNREL":
            return [
                {
                    "name":"628.jpg",
                    "label":"bike-and-person",
                    "predictions": []
                },
                {
                    "name":"132.jpg",
                    "label":"dog-and-person",
                    "predictions": []
                },
                {
                    "name":"602.jpg",
                    "label":"dog-ride-motorcycle",
                    "predictions": []
                },
                {
                    "name":"185.jpg",
                    "label":"dog-wear-hat",
                    "predictions": []
                }
            ]

    def __get_labels(self):
        f = open(self.labels_path).read()
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

    def pre_process_and_model(self, model):

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

    def get_model(self, configuration):
        
        structure = configuration["structure"]
        if structure == "gcn":
            method = configuration["method"]
            walk = configuration["method_int"]
            extractor = configuration["extractor"]
            model_path = BASE_DIR + f"/results_from_{self.dataset.lower()}/{self.dataset.lower()}/models/gcn/{extractor}/{method}_model_{self.dataset.upper()}_{extractor}_{walk}.pth"
            device = "cuda" if th.cuda.is_available() else "cpu"
            try:
                model = th.load(model_path, map_location=device)
            except:
                raise Exception("Erro while loading the model GCN")

        else:
            extractor = configuration["extractor"]
            model_path = BASE_DIR + f"/results_from_{self.dataset.lower()}/{self.dataset.lower()}/models/cnn/{self.dataset.upper()}_CNN_MODEL_{extractor}.h5"
            try:
                model = load_model(model_path)
            except:
                raise Exception("Erro while loading the model CNN")

        return model

    def __get_decoder(self, model):
        if model == "ResNet50":
            return tf.keras.applications.resnet50.decode_predictions
        if model == "VGG16":
            return tf.keras.applications.vgg16.decode_predictions
        if model == "InceptionResNetV2":
            return tf.keras.applications.inception_resnet_v2.decode_predictions
        if model == "InceptionV3":
            return tf.keras.applications.inception_v3.decode_predictions
        if model == "Xception":
            return tf.keras.applications.xception.decode_predictions
        if model == "EfficientNet":
            return decode_predictions

    def pre_process_image(self, image, configuration):
        structure = configuration["structure"]
        if structure == "gcn":
            image_name = image["name"]
            image_path = os.path.join(self.images_to_predict_path, image_name)
            img = image.load_img(image_path, target_size=(244, 244))
            image_array = image.img_to_array(img)
            image_expanded = np.expand_dims(image_array, axis=0)
            pretrained_model, preprocess_input = self.pre_process_and_model(configuration["extractor"])
            preprocessed_image = preprocess_input(image_expanded)
            feature_array = pretrained_model.predict(preprocessed_image)
            flatten_array = feature_array.flatten()
            test_transforms = th.FloatTensor(flatten_array)
            image_tensor = test_transforms.unsqueeze_(0)
            variable = Variable(image_tensor)
            return variable                

        else:
            image_name = image["name"]
            image_path = os.path.join(self.images_to_predict_path, image_name)
            img = image.load_img(image_path, target_size=(244, 244))
            pretrained_model, preprocess_input = self.pre_process_and_model(configuration["extractor"])
            image_arr = image.img_to_array(img)
            image_reshaped = image_arr.reshape((1, image_arr.shape[0], image_arr.shape[1], image_arr.shape[2]))
            preprocessed_image = preprocess_input(image_reshaped)
            return preprocessed_image
                
    def predict(self, configuration, data_information):
        structure = configuration["structure"]
        if structure == "gcn":
            model = self.get_model(configuration)
            model.eval()
            for image in data_information:
                image_processed = self.pre_process_image(image, configuration)
                logits = model(image_processed)
                _, indices = th.max(logits, dim=1)
                classification = indices.data.cpu().numpy()
                top_1 = classification[0]
                top_2 = classification[1]
                top_3 = classification[2]
                probs = []
                for i in logits[0]:
                    probs.append(exp(i / 10000))

                top_1_prob = probs[top_1] / sum(probs)
                top_2_prob = probs[top_2] / sum(probs)
                top_3_prob = probs[top_3] / sum(probs)

                print("""Top 1 - {} - {:.4f} | 
                         Top 2 - {} - {:.4f}
                         Top 3 - {} - {:.4f}""".format(self.__get_labels()[top_1], top_1_prob,
                                                       self.__get_labels()[top_2], top_2_prob,
                                                       self.__get_labels()[top_3], top_3_prob))

                #image["predictions"].append((self.__get_labels()[top_1], top_1_prob))
                #image["predictions"].append((self.__get_labels()[top_2], top_2_prob))
                #image["predictions"].append((self.__get_labels()[top_3], top_3_prob))
            return data_information
        else:
            model = self.get_model(configuration)
            for image in data_information:
                image_processed = self.pre_process_image(image, configuration)
                yhat = model.predict(image_processed)
                decoder = self.__get_decoder(configuration["extractor"])
                if configuration["extractor"] == "EfficientNet":
                    pass
                else:
                    label = decoder(yhat)
                    label_1 = label[0][0]
                    print('%s (%.2f%%)' % (label_1[1], label_1[2]*100))
                    label_2 = label[0][1]
                    print('%s (%.2f%%)' % (label_2[1], label_2[2]*100))
                    label_3 = label[0][2]
                    print('%s (%.2f%%)' % (label_3[1], label_3[2]*100))

                    #image["predictions"].append((self.__get_labels()[top_1], top_1_prob))
                    #image["predictions"].append((self.__get_labels()[top_2], top_2_prob))
                    #image["predictions"].append((self.__get_labels()[top_3], top_3_prob))
            return data_information

class ImageMosaic(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.result_path_save = BASE_DIR + f"/results_from_{self.dataset.lower()}/{self.dataset.lower()}/predictions/mosaic_{self.dataset.lower()}.jpg"

    def __put_text(self, image, predictions):
        pass

    def __build_mosaic(self, images):
        pass

    def process_all_data(self, all_data_information):
        pass

def join_data_information(all_data_information):
    pass

def predict_unrel():

    dataset = "UNREL"
    prediction = Predict(dataset)
    data_information = prediction.get_images_to_predict()
    
    # CNN Xception
    configuration_cnn = {"extractor": "Xception",
                        "structure": "cnn"}
    data_information_cnn = prediction.predict(configuration_cnn, data_information)

    # GCN Best FC

    configuration_fc = {"extractor": "",
                        "structure": "gcn",
                        "method": "Full Connected",
                        "method_int": 0}
    data_information_fc = prediction.predict(configuration_fc, data_information)

    # GCN Best RW

    configuration_rw = {"extractor": "",
                        "structure": "gcn",
                        "method": "Random Walk",
                        "method_int": 0}
    data_information_rw = prediction.predict(configuration_rw, data_information)

    # GCN Best RC

    configuration_rc = {"extractor": "",
                        "structure": "gcn",
                        "method": "Random Cut",
                        "method_int": 0}
    data_information_rc = prediction.predict(configuration_rc, data_information)

    # GCN Best RE

    configuration_re = {"extractor": "",
                        "structure": "gcn",
                        "method": "Random Edge",
                        "method_int": 0}
    data_information_re = prediction.predict(configuration_re, data_information)

    # GCN Best RWE

    configuration_rwe = {"extractor": "",
                        "structure": "gcn",
                        "method": "Random Weighted",
                        "method_int": 0}
    data_information_rwe = prediction.predict(configuration_rwe, data_information)

    all_data_information = join_data_information([data_information_cnn, 
                                                    data_information_fc,
                                                    data_information_rw,
                                                    data_information_rc,
                                                    data_information_re,
                                                    data_information_rwe])
    mosaic = ImageMosaic(dataset)
    mosaic.process_all_data(all_data_information)

    print(f"========================= Predictions made for {dataset} ==========================")

def predict_mit67():
    
    dataset = "MIT67"
    prediction = Predict(dataset)
    data_information = prediction.get_images_to_predict()
    
    # CNN Xception
    configuration_cnn = {"extractor": "Xception",
                        "structure": "cnn"}
    data_information_cnn = prediction.predict(configuration_cnn, data_information)

    # GCN Best FC

    configuration_fc = {"extractor": "",
                        "structure": "gcn",
                        "method": "Full Connected",
                        "method_int": 0}
    data_information_fc = prediction.predict(configuration_fc, data_information)

    # GCN Best RW

    configuration_rw = {"extractor": "",
                        "structure": "gcn",
                        "method": "Random Walk",
                        "method_int": 0}
    data_information_rw = prediction.predict(configuration_rw, data_information)

    # GCN Best RC

    configuration_rc = {"extractor": "",
                        "structure": "gcn",
                        "method": "Random Cut",
                        "method_int": 0}
    data_information_rc = prediction.predict(configuration_rc, data_information)

    # GCN Best RE

    configuration_re = {"extractor": "",
                        "structure": "gcn",
                        "method": "Random Edge",
                        "method_int": 0}
    data_information_re = prediction.predict(configuration_re, data_information)

    # GCN Best RWE

    configuration_rwe = {"extractor": "",
                        "structure": "gcn",
                        "method": "Random Weighted",
                        "method_int": 0}
    data_information_rwe = prediction.predict(configuration_rwe, data_information)

    all_data_information = join_data_information([data_information_cnn, 
                                                    data_information_fc,
                                                    data_information_rw,
                                                    data_information_rc,
                                                    data_information_re,
                                                    data_information_rwe])
    mosaic = ImageMosaic(dataset)
    mosaic.process_all_data(all_data_information)

    print(f"========================= Predictions made for {dataset} ==========================")

def predict_malaria():
    
    dataset = "MALARIA"
    prediction = Predict(dataset)
    data_information = prediction.get_images_to_predict()
    
    # CNN Xception
    configuration_cnn = {"extractor": "Xception",
                        "structure": "cnn"}
    data_information_cnn = prediction.predict(configuration_cnn, data_information)

    # GCN Best FC

    configuration_fc = {"extractor": "",
                        "structure": "gcn",
                        "method": "Full Connected",
                        "method_int": 0}
    data_information_fc = prediction.predict(configuration_fc, data_information)

    # GCN Best RW

    configuration_rw = {"extractor": "",
                        "structure": "gcn",
                        "method": "Random Walk",
                        "method_int": 0}
    data_information_rw = prediction.predict(configuration_rw, data_information)

    # GCN Best RC

    configuration_rc = {"extractor": "",
                        "structure": "gcn",
                        "method": "Random Cut",
                        "method_int": 0}
    data_information_rc = prediction.predict(configuration_rc, data_information)

    # GCN Best RE

    configuration_re = {"extractor": "",
                        "structure": "gcn",
                        "method": "Random Edge",
                        "method_int": 0}
    data_information_re = prediction.predict(configuration_re, data_information)

    # GCN Best RWE

    configuration_rwe = {"extractor": "",
                        "structure": "gcn",
                        "method": "Random Weighted",
                        "method_int": 0}
    data_information_rwe = prediction.predict(configuration_rwe, data_information)

    all_data_information = join_data_information([data_information_cnn, 
                                                    data_information_fc,
                                                    data_information_rw,
                                                    data_information_rc,
                                                    data_information_re,
                                                    data_information_rwe])
    mosaic = ImageMosaic(dataset)
    mosaic.process_all_data(all_data_information)

    print(f"========================= Predictions made for {dataset} ==========================")

def predict_vrd():
    
    dataset = "VRD"
    prediction = Predict(dataset)
    data_information = prediction.get_images_to_predict()
    
    # CNN Xception
    configuration_cnn = {"extractor": "Xception",
                        "structure": "cnn"}
    data_information_cnn = prediction.predict(configuration_cnn, data_information)

    # GCN Best FC

    configuration_fc = {"extractor": "",
                        "structure": "gcn",
                        "method": "Full Connected",
                        "method_int": 0}
    data_information_fc = prediction.predict(configuration_fc, data_information)

    # GCN Best RW

    configuration_rw = {"extractor": "",
                        "structure": "gcn",
                        "method": "Random Walk",
                        "method_int": 0}
    data_information_rw = prediction.predict(configuration_rw, data_information)

    # GCN Best RC

    configuration_rc = {"extractor": "",
                        "structure": "gcn",
                        "method": "Random Cut",
                        "method_int": 0}
    data_information_rc = prediction.predict(configuration_rc, data_information)

    # GCN Best RE

    configuration_re = {"extractor": "",
                        "structure": "gcn",
                        "method": "Random Edge",
                        "method_int": 0}
    data_information_re = prediction.predict(configuration_re, data_information)

    # GCN Best RWE

    configuration_rwe = {"extractor": "",
                        "structure": "gcn",
                        "method": "Random Weighted",
                        "method_int": 0}
    data_information_rwe = prediction.predict(configuration_rwe, data_information)

    all_data_information = join_data_information([data_information_cnn, 
                                                    data_information_fc,
                                                    data_information_rw,
                                                    data_information_rc,
                                                    data_information_re,
                                                    data_information_rwe])
    mosaic = ImageMosaic(dataset)
    mosaic.process_all_data(all_data_information)

    print(f"========================= Predictions made for {dataset} ==========================")

def main():
    print("Predicting for UNREL")
    predict_unrel()

    print("Predicting for VRD")
    predict_vrd()

    print("Predicting for MALARIA")
    predict_malaria()

    print("Predicting for MIT67")
    predict_mit67()

if __name__ == '__main__':
    main()