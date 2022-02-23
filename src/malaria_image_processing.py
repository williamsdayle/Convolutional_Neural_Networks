import cv2 as cv
import os
import shutil as sh
from tqdm import tqdm

def global_images_change():

    images_path = 'all_images/malaria-dataset/images/'
    new_images_path = 'all_images/malaria-dataset/dataset/'
    file = open('config_arqs/malaria_subclass_bboxes.txt').read()
    lines = file.split('\n')
    for line in lines:
        line_metadata = line.split(' ')
        image_label = line_metadata[3]
        image_name = line_metadata[1]
        old_path = os.path.join(images_path, image_name)
        label_path = os.path.join(new_images_path, image_label)
        new_path = os.path.join(label_path, image_name)
        sh.copy2(old_path, new_path)
        print('Image {} moved to {}'.format(image_name, image_label))


def cut_image(image, r_min, c_min, r_max, c_max):
    pass

def process_image_and_save():

    dataset_path = 'all_images/malaria_dataset/images'
    config_file_path = 'config_arqs/malaria_subclass_bboxes.txt'
    config_file = open(config_file_path, "r")
    for line in tqdm(config_file.readlines(), desc ="Processing image"):
        line = line.replace("\n", "")
        line_metadata = line.split(" ")
        image_index = line_metadata[0]
        image_name = line_metadata[1]
        image_number_of_bbox = line_metadata[2]
        image_global_label = line_metadata[3]
        images_bbox_metadata = line_metadata[4:]

        image_path = os.path.join(dataset_path, image_name)
        image = cv.imread(image_path)
        bbox_index = 0
        for i in range(0, len(images_bbox_metadata), 5):
            bounding_box_metadata = images_bbox_metadata[i:i+5]
            try:
                bbox_label = bounding_box_metadata[0]
                r_min = int(bounding_box_metadata[1])
                c_min = int(bounding_box_metadata[2])
                r_max = int(bounding_box_metadata[3])
                c_max = int(bounding_box_metadata[4])
            except Exception as error:
                pass
            
            #image = cv.rectangle(image, (c_min, r_min), (c_max, r_max), (0, 0, 255), 3)
            if bbox_label != "":
                try:
                    cropped = image[r_min: r_max, c_min:c_max]
                    cropped = cv.resize(cropped, (224, 224))
                    image_save_name = f"Examples/MALARIA/{image_global_label}/{image_index}_{bbox_label}_{bbox_index}.jpg"
                    cv.imwrite(image_save_name, cropped)
                    bbox_index += 1
                except Exception as error:
                    print(error)

    
def main():

    #global_images_change()

    process_image_and_save()

if __name__ == '__main__':
    main()
