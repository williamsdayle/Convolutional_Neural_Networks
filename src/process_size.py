import os

def get_process_size(MODEL, DATASET, WALK, CONJ):

    if walk > 10:

        path = 'gcn/data/{}/ind.mine_{}_conj_{}_step_{}.graph'.format(DATASET, MODEL, CONJ + 10, WALK)
    else:

        path = 'gcn/data/{}/ind.mine_{}_conj_{}_step_{}.graph'.format(DATASET, MODEL, CONJ, WALK)

    stats = os.stat(path)

    byte_size = stats.st_size

    return byte_size



walks = [i for i in range(21)]
models = ['VGG16', 'ResNet50', 'Xception']
conj = 0
datasets = ['UNREL']

for dataset in datasets:

    for model in models:

        for walk in walks:

            byte_size = get_process_size(model, dataset, walk, conj)

            if walk == 0:

                fc_size_file = open('process_size/{}/Full Connected graph size {}.txt'.format(dataset, model), 'w')

                fc_size_file.write('File size in bytes: ' + str(byte_size) + '\n')
                fc_size_file.write('File size in mbytes: ' + str(byte_size / (1024 * 1024)) + '\n')
                fc_size_file.close()

            if walk > 0 and walk <=10:

                rw_size_file = open('process_size/{}/Random Walk graph size {} {}.txt'.format(dataset, model, walk), 'w')

                rw_size_file.write('File size in bytes: ' + str(byte_size) + '\n')
                rw_size_file.write('File size in mbytes: ' + str(byte_size / (1024 * 1024)) + '\n')
                rw_size_file.close()

            if walk > 10 and walk <=20:

                rc_size_file = open('process_size/{}/Random Cut graph size {} {}.txt'.format(dataset, model, walk - 10), 'w')

                rc_size_file.write('File size in bytes: ' + str(byte_size) + '\n')
                rc_size_file.write('File size in mbytes: ' + str(byte_size / (1024 * 1024)) + '\n')
                rc_size_file.close()






