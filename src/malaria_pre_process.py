import json

def build_config_file():

    final_string = ''
    labels = []
    image_labels = []
    path = 'malaria/training.json'
    with open(path, 'r') as j:
        contents_train = json.loads(j.read())
    path = 'malaria/test.json'
    with open(path, 'r') as j:
        content_test = json.loads(j.read())
    contents = contents_train + content_test
    image_id = 0
    for content in contents:
        image = content['image']
        image_name = image['pathname'].split('/')[2]
        objects = content['objects']
        number_of_bounding_boxes = len(objects)
        temp_categorys = []
        for obj in objects:
            bounding_box_category = obj['category']
            temp_categorys.append(bounding_box_category)
        for cat in temp_categorys:
            if cat != 'red blood cell' and cat != 'leukocyte':
                image_label = 'infected'
            else:
                image_label = 'uninfected'
        if image_label == 'uninfected':
            pass
            #print(temp_categorys)
        if image_label not in labels:
            labels.append(image_label)
        image_labels.append(image_label)
        string_line = '{} {} {} {}'.format(image_id, image_name, number_of_bounding_boxes, image_label)
        bb_line = ''
        for obj in objects:
            bounding_box_category = obj['category'].replace(' ', '_')
            bounding_box_data = obj['bounding_box']
            bounding_box_metadata_min = bounding_box_data['minimum']
            bounding_box_metadata_max = bounding_box_data['maximum']
            r_min = bounding_box_metadata_min['r']
            c_min = bounding_box_metadata_min['c']
            r_max = bounding_box_metadata_max['r']
            c_max = bounding_box_metadata_max['c']
            bb_line = bb_line + '{} {} {} {} {} '.format(bounding_box_category, r_min, c_min, r_max, c_max)
        final_string = final_string + '{} {} \n'.format(string_line, bb_line)
        image_id = image_id + 1
    return final_string, labels, image_labels

string, labels, image_and_label = build_config_file()

print(string)

file = open('config_arqs/malaria_subclass_bboxes.txt', 'w')

file.write(string)

file.close()

file = open('config_arqs/malaria_subclass_globalClasses.txt', 'w')

for label in sorted(labels):

    file.write(label)
    file.write('\n')

file.close()

file = open('config_arqs/malaria_subclass_bboxesGlobalClasses.txt', 'w')

for image_label in image_and_label:

    file.write(image_label)
    file.write('\n')

file.close()











