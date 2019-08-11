import os
import scipy.signal
import sklearn.utils
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import seaborn as sns


def importData(directory, x_range, y_range):
    # pull data into python, should be either for training set or eval set
    train_data_files = []
    for file in os.listdir(os.path.join(directory)):
        if file.endswith('.csv'):
            train_data_files.append(file)
    print(train_data_files)
    # get data
    ftr = []
    lbl = []
    for file_name in train_data_files:
        # import full arrays
        ftr_array = pd.read_csv(os.path.join(directory, file_name), delimiter=',',header = None, usecols=x_range)
        lbl_array = pd.read_csv(os.path.join(directory, file_name), delimiter=',',header = None, usecols=y_range)
        # append each data point to ftr and lbl
        for params, curve in zip(ftr_array.values, lbl_array.values):
            ftr.append(params)
            lbl.append(curve)
    ftr = np.array(ftr, dtype='float32')
    lbl = np.array(lbl, dtype='float32')
    return ftr, lbl


# check that the data we're using is distributed uniformly and generate some plots
def check_data(input_directory, col_range=range(2, 10), col_names=('h1','h2','h3','h4','r1','r2','r3','r4')):
    for file in os.listdir(input_directory):
        if file.endswith('.csv'):
            print('\n histogram for file {}'.format(os.path.join(input_directory, file)))
            with open(os.path.join(input_directory, file)) as f:
                data = pd.read_csv(f, header=None, delimiter=',', usecols=col_range,
                                   names=col_names)
                for name in col_names:
                    print('{} unique values for {}: {}'.format(len(data[name].unique()),
                                                               name,
                                                               np.sort(data[name].unique()))
                          )
                hist = data.hist(bins=13, figsize=(10, 5))
                plt.tight_layout()
                plt.show()
                print('done plotting column data')



# add columns of derived values to the input data
# for now, just ratios of the inputs
def addColumns(input_directory, output_directory, x_range, y_range):
    print('adding columns...')
    print('importing data')
    data_files = []
    for file in os.listdir(os.path.join(input_directory)):
        if file.endswith('.csv'):
            data_files.append(file)
    for file in data_files:
        ftr = pd.read_csv(os.path.join(input_directory, file), delimiter=',', usecols=x_range,
                          names=['id0', 'id1'] + ['ftr' + str(i) for i in range(8)])
        lbl = pd.read_csv(os.path.join(input_directory, file), delimiter=',', usecols=y_range, header=None)

        print('computing new columns')
        newCol = 0  # count the number of new columns added
        for i in range(2, 6):  # first four are heights
            for j in range(6, 10):  # second four are radii
                ftr['ftr{}'.format(j-2)+'/'+'ftr{}'.format(i-2)] = ftr.apply(lambda row: row.iloc[j]/row.iloc[i], axis=1)
                newCol += 1
        print('total new columns added is {}\n'.format(newCol))
        print('exporting')
        data_total = pd.concat([ftr, pd.DataFrame(lbl)], axis=1)
        # data_total['2010'] = data_total.str.replace('\n', ' ')
        with open(os.path.join(output_directory, file[:-4] + '_div01.csv'), 'a') as file_out:
            # for some stupid reason to_csv seems to insert a blank line between every single data line
            # maybe try to fix this issue later
            #data_total.to_csv(file_out, sep=',', index=False, header=False)
            data_out = data_total.values
            np.savetxt(file_out, data_out, delimiter=',', fmt='%f')
    print('done')

# finds simulation files in input_dir and finds + saves the subset that adhere to the geometry contraints r_bound
# and h_bound
def gridShape(input_dir, output_dir, shapeType, r_bounds, h_bounds):

    files_to_filter = []
    for file in os.listdir(input_dir):
        if file.endswith('.csv'):
            files_to_filter.append(os.path.join(input_dir, file))

    print('filtering through {} files...'.format(len(files_to_filter)))
    print('bounds on radii: [{}, {}], bounds on heights: [{}, {}]...'.format(r_bounds[0], r_bounds[1],
                                                                       h_bounds[0], h_bounds[1]))
    lengthsPreFilter = []
    lengthsPostFilter = []
    for file in files_to_filter:
        with open(file, 'r') as f:
            geom_specs = pd.read_csv(f, delimiter=',', header=None).values
            geoms_filt = []
            geoms_filtComp = []
            lengthsPreFilter.append(len(geom_specs))
            if shapeType=='corner':
                print('cutting a corner of the data...')
                for geom_spec in geom_specs:
                    hs = geom_spec[2:6]
                    rs = geom_spec[6:10]
                    if (np.all(hs >= h_bounds[0]) and np.all(hs <= h_bounds[1])) or \
                       (np.all(rs >= r_bounds[0]) and np.all(rs <= r_bounds[1])):
                        geoms_filt.append(geom_spec)
                    else:
                        geoms_filtComp.append(geom_spec)
            elif shapeType=='rCut':
                print('cutting based on r values only...')
                for geom_spec in geom_specs:
                    rs = geom_spec[6:10]
                    if (np.all(rs >= r_bounds[0]) and np.all(rs <= r_bounds[1])):
                        geoms_filt.append(geom_spec)
                    else:
                        geoms_filtComp.append(geom_spec)
            elif shapeType == 'hCut':
                print('cutting based on h values only...')
                for geom_spec in geom_specs:
                    hs = geom_spec[2:6]
                    if (np.all(hs >= h_bounds[0]) and np.all(hs <= h_bounds[1])):
                        geoms_filt.append(geom_spec)
                    else:
                        geoms_filtComp.append(geom_spec)
            else:
                print('shapeType {} is not valid.'.format(shapeType))
                return
            geoms_filt = np.array(geoms_filt)
            geoms_filtComp = np.array(geoms_filtComp)
            lengthsPostFilter.append(len(geoms_filt))
            print('{} reduced from {} to {}, ({}%)'.format(file, lengthsPreFilter[-1],
                                                           lengthsPostFilter[-1],
                                                           100*np.round(lengthsPostFilter[-1]/lengthsPreFilter[-1], 4)))

        save_file = os.path.join(output_dir, os.path.split(file)[-1][:-4] + '_filt')
        # save the filtered geometries, for training
        with open(save_file + '.csv', 'w+') as f:
            np.savetxt(f, geoms_filt, delimiter=',', fmt='%f')

        # save the all the goemetries filtered out, for evaluation
        with open(save_file + 'Comp.csv', 'w+') as f:
            np.savetxt(f, geoms_filtComp, delimiter=',', fmt='%f')


    print('\nAcross all files: of original {} combos, {} remain ({}%)'.format(sum(lengthsPreFilter),
                                                                              sum(lengthsPostFilter),
                                                                              100*np.round(sum(lengthsPostFilter)/ \
                                                                                       sum(lengthsPreFilter), 4)
                                                                              ))


def read_data(input_size, output_size, x_range, y_range, geoboundary, cross_val=5, val_fold=0, batch_size=100,
                 shuffle_size=100, data_dir=os.path.abspath(''), rand_seed=1234, normalize_input = False ):
    """
      :param input_size: input size of the arrays
      :param output_size: output size of the arrays
      :param x_range: columns of input data in the txt file
      :param y_range: columns of output data in the txt file
      :param cross_val: number of cross validation folds
      :param val_fold: which fold to be used for validation
      :param batch_size: size of the batch read every time
      :param shuffle_size: size of the batch when shuffle the dataset
      :param data_dir: parent directory of where the data is stored, by default it's the current directory
      :param rand_seed: random seed
      """
    """
    Read feature and label
    :param is_train: the dataset is used for training or not
    :param train_valid_tuple: if it's not none, it will be the names of train and valid files
    :return: feature and label read from csv files, one line each time
    """

    # get data files
    print('getting data files...')
    ftrTrain, lblTrain = importData(os.path.join(data_dir, 'dataIn'), x_range, y_range)
    ftrTest, lblTest = importData(os.path.join(data_dir, 'dataIn', 'eval'), x_range, y_range)

    print('total number of training samples is {}'.format(len(ftrTrain)))
    print('total number of test samples is {}'.format(len(ftrTest)),
          'length of an input spectrum is {}'.format(len(lblTest[0])))

    # print('downsampling output curves')
    # # resample via scipy method
    # lblTest = scipy.signal.resample(lblTest, output_size + 20, axis=1)
    # lblTrain = scipy.signal.resample(lblTrain, output_size + 20, axis=1)
    # lblTest = np.array([spec[10:-10] for spec in lblTest])
    # lblTrain = np.array([spec[10:-10] for spec in lblTrain])

    print('downsampling output curves')
    # resample the output curves so that there are not so many output points
    # drop the beginning of the curve so that we have a multiple of 300 points
    lblTrain = lblTrain[::, len(lblTest[0])-1800::6]
    lblTest = lblTest[::, len(lblTest[0])-1800::6]

    print('length of downsampled train spectra is {} for first, {} for final, '.format(len(lblTrain[0]),
                                                                                       len(lblTrain[-1])),
          'set final layer size to be compatible with this number')
    print('length of downsampled test spectra is {}, '.format(len(lblTest[0]),
                                                         len(lblTest[-1])),
          'set final layer size to be compatible with this number')

    # determine lengths of training and validation sets
    num_data_points = len(ftrTrain)
    #train_length = int(.8 * num_data_points)

    print('generating TF dataset')
    assert np.shape(ftrTrain)[0] == np.shape(lblTrain)[0]
    assert np.shape(ftrTest)[0] == np.shape(lblTest)[0]

    #Normalize the data if instructed using boundary
    if normalize_input:
		    ftrTrain[:,0:4] = (ftrTrain[:,0:4] - (geoboundary[0] + geoboundary[1]) / 2)/(geoboundary[1] - geoboundary[0]) * 2
		    ftrTest[:,4:] = (ftrTest[:,4:] - (geoboundary[2] + geoboundary[3]) / 2)/(geoboundary[3] - geoboundary[2]) * 2

    dataset_train = tf.data.Dataset.from_tensor_slices((ftrTrain, lblTrain))
    dataset_valid = tf.data.Dataset.from_tensor_slices((ftrTest, lblTest))
    
    # shuffle then split into training and validation sets
    dataset_train = dataset_train.shuffle(shuffle_size)

    dataset_train = dataset_train.repeat()
    dataset_train = dataset_train.batch(batch_size, drop_remainder=True)
    dataset_valid = dataset_valid.batch(batch_size, drop_remainder=True)

    iterator = tf.data.Iterator.from_structure(dataset_train.output_types, dataset_train.output_shapes)
    features, labels = iterator.get_next()
    train_init_op = iterator.make_initializer(dataset_train)
    valid_init_op = iterator.make_initializer(dataset_valid)

    return features, labels, train_init_op, valid_init_op

#if __name__ == '__main__':
    # addColumns(input_directory=os.path.join(".", "dataIn", "orig"),
    #            output_directory=os.path.join('.', "dataIn", "data_div"),
    #            x_range=[i for i in range(0, 10)],
    #            y_range=[i for i in range(10, 2011)]
    #            )

    # addColumns(input_directory=os.path.join(".", "dataIn", 'orig', 'outside_grid', 'set02Small'),
    #            output_directory=os.path.join('.', "dataIn", "data_div", 'outside_grid', 'set02Small'),
    #            x_range=[i for i in range(0, 10)],
    #            y_range=[i for i in range(10, 2011)]
    #            )


    # gridShape(input_dir=os.path.join('.', 'dataIn', 'data_div'),
    #           output_dir=os.path.join('.', 'dataIn', 'gridShapeData', 'shape010'),
    #           shapeType='corner',
    #           r_bounds=(42, 48.6), h_bounds=(30, 46))

    #check_data(input_directory=os.path.join('.', 'dataIn', 'orig'), col_range=range(2, 10))
    #X_RANGE = [i for i in range(2, 10)]
    #Y_RANGE = [i for i in range(10 , 2011)]
    #print('getting data files...')
#     #features, labels, train_init_op, valid_init_op = read_data(input_size=0,
#                                                                            output_size=0,
#                                                                            x_range=X_RANGE,
#                                                                            y_range=Y_RANGE,
#                                                                            cross_val=5,
#                                                                            val_fold=5,
#                                                                            batch_size=128,
#                                                                           shuffle_size=128)
    #ftrTrain, lblTrain = importData(os.path.join('./', 'dataIn'), X_RANGE, Y_RANGE)

