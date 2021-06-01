from skimage import io
import os
import shutil as sh

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

    dataset_path = 'all_images/malaria-dataset/dataset'

    config_file_path = 'config_arqs/malaria_subclass_bboxes.txt'

    config_file = open(config_file_path).read()

    lines = config_file.split('\n')

    for line in lines:

        line_metadata = line.split(' ')

        image_label = line_metadata[3]
        image_id = line_metadata[0]
        image_name = line_metadata[1]
        bounding_box_size = int(line_metadata[2])

        bounding_box_metadata = line_metadata[4:]

        bounding_box_metadata = [data for data in bounding_box_metadata if data != '']

        print(image_label)

        for data, index in zip(bounding_box_metadata, range(len(bounding_box_metadata))):

            if index%5 == 0 or index == 0:

                print(data)

        break








def main():

    #global_images_change()

    process_image_and_save()

if __name__ == '__main__':
    main()
