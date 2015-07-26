import generate_folds, os, sys, random, time, theano
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import theano.tensor as T


fold_paths = [
    "./folds/DUD-E",
    "./folds/MUV",
    "./folds/Tox21",
    "./folds/PCBA",
    ]



def get_fold_path(data_type):

    if(data_type == 'DUD-E'):
        return fold_paths[0]

    if(data_type == 'MUV'):
        return fold_paths[1]

    if(data_type == 'Tox21'):
        return fold_paths[2]

    if(data_type == 'PCBA'):
        return fold_paths[3]

    raise ValueError('data_type does not exist:' + str(data_type))



def get_target(fname, data_type):
    return generate_folds.get_target(fname, data_type)



def parse_line(line, data_type):

    # row format: [hash_id, is_active, native_id, fold, bitstring]
    parts = line.rstrip('\n').split(r' ')
    # hash_id = parts[0]

    # cast the string to int
    is_active = int(parts[1])

    # native_id = parts[2]
    fold = parts[3]
    bitstring = parts[4]

    return fold, [bitstring, is_active]



def build_targets(fold_path, data_type):

    # init targets
    targets = {}
    for dir_name, sub, files in os.walk(fold_path):
        for fname in files:
            if fname.startswith('.'):
                # ignore system files
                pass
            else:
                target = get_target(fname, data_type)
                targets[target] = []
                # print "file:" + fname + ", target:" + target

        for fname in files:
            if fname.startswith('.'):
                # ignore system files
                pass
            else:
                target = get_target(fname, data_type)
                targets[target].append(fname)
                # print "file:" + fname + ", target:" + target
    
    return targets



def oversample(data):
    # balance the number of actives / inactives in the dataset
    actives = []
    inactives = []
    for i in range(len(data)):
        if(int(data[i][1]) == 1):
            actives.append(data[i])
        else:
            inactives.append(data[i])

    total_inactives = len(inactives)
    total_actives = len(actives)
    ratio = total_inactives / total_actives

    ratio = ratio
    # oversample_total = ratio * total_actives

    oversamples = []
    for i in range(len(actives)):
        for j in range(ratio):
            oversamples.append(actives[i])

    # print len(oversamples)
    # print total_inactives
    # print len(oversamples + inactives)

    # combine oversampled actives + inactives into one list
    return oversamples + inactives



def get_folds(data_type, fold_path, target, fnames):
    # store folds by target
    folds = {}
    for i in range(5):
        # don't forget -- we are using strings & not integer keys!!!
        folds[i] = []

    #fnames contains all files for this target
    for fname in fnames:
        row = []
        with open(fold_path + '/' + fname) as f:
            lines = f.readlines()
            for line in lines:
                # put each row in it's respective fold
                fold, row = parse_line(line, data_type)
                folds[int(fold)].append(row)

    """ Debug """
    # print "length of all folds"
    # print len(folds)
    # print "length of respective folds"
    # print len(folds[0])
    # print len(folds[1])
    # print len(folds[2])
    # print len(folds[3])
    # print len(folds[4])
    
    # oversample the folds to balance actives / inactives
    for i in range(len(folds)):
        folds[i] = oversample(folds[i])


    # shuffle the folds once upfront
    for i in range(len(folds)):
        random.shuffle(folds[i])


    return folds


def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables
    The reason we store our dataset in shared variables is to allow Theano to copy it into the GPU memory
    (when code is run on GPU). Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
    # When storing data on the GPU it has to be stored as floats therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning  ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')


def build_data_set(fold):
    """ Featurizing 1024 bits is a slow process """
    """ ** Built for Theano ** """
    # build training data
    X = []
    Y = []
    for i in range(len(fold)):
        row = []
        for bit in fold[i][0]:
            row.append(int(bit))
        X.append(row)
        Y.append(int(fold[i][1]))

    X = np.array(X)
    Y = np.array(Y)

    return (X, Y)

def th_load_data(data_type, fold_path, target, fnames, fold_train, fold_test):
    """ Get just 1 test & 1 valid fold to avoid overloading memory """
    """The load_files_for_task module takes the input files for a single task"""
    """The module loads the data from these files into two dictionaries - foldsActive and foldsInactive"""
    """Each dictionary contains five lists: 0 to 4 corresponding to a fold."""

    # sanity checks
    if(fold_train < 0 or fold_train > 4):
        raise ValueError('fold_train = ' + str(fold_train) + \
            '. Oops! fold_train must be between 0 and 4!')

    if(fold_test < 0 or fold_test > 4):
        raise ValueError('fold_test = ' + str(fold_test) + \
            '. Oops! fold_test must be between 0 and 4!')

    if(fold_test  == fold_train):
        raise ValueError('fold_train ('+ str(fold_train) + \
            ') == fold_test ('+ str(fold_train) +')... oops!')

    #fnames contains all files for this target
    train_folds = []
    test_folds = []
    for fname in fnames:
        row = []
        with open(fold_path + '/' + fname) as f:
            lines = f.readlines()
            for line in lines:
                # put each row in it's respective fold
                curr_fold, row = parse_line(line, data_type)
                curr_fold = int(curr_fold)

                if(curr_fold == fold_train):
                    train_folds.append(row)

                if(curr_fold == fold_test):
                    test_folds.append(row)
    
    # oversample the folds to balance actives / inactives
    train_folds = oversample(train_folds)
    test_folds = oversample(test_folds)

    # shuffle the folds once upfront
    random.shuffle(train_folds)
    random.shuffle(test_folds)

    train_x, train_y = build_data_set(train_folds)
    test_x, test_y = build_data_set(test_folds)
    
    rval = [(train_x, train_y), (test_x, test_y)]

    return rval



def load_data(testFold, foldsActive, foldsInactive, multiplier):
    #print '... Loading for cross-validation ... ',testFold
    listTrainActive, listTrainInactive, listTestActive, listTestInactive = [],[],[],[]

    for fold in range(0,5):
        if (fold == testFold):
            for item in foldsActive[fold]:
                listTestActive.append(item)
            for item in foldsInactive[fold]:
                listTestInactive.append(item)
        else:
            for item in foldsActive[fold]:
                listTrainActive.append(item)
            for item in foldsInactive[fold]:
                listTrainInactive.append(item)

    trainList, trainFingerprintArrayList, trainClassLabelList, trainItemList = [], [], [], []

    for item in listTrainActive:
        compoundTuple = (item[0], item[2], item[3])
        tuple = (numpy.array(item[4]), item[3], compoundTuple)
        for w in range(0,multiplier):
            trainList.append(tuple)

    for item in listTrainInactive:
        compoundTuple = (item[0], item[2], item[3])
        tuple = (numpy.array(item[4]), item[3], compoundTuple)
        trainList.append(tuple)

    random.shuffle(trainList)
    random.shuffle(trainList)
    for item in trainList:
        trainFingerprintArrayList.append(item[0])
        trainClassLabelList.append(item[1])
        trainItemList.append(item[2])

    # numpy.vstack: Stack arrays in sequence vertically (row wise).
    trainFingerprints = numpy.vstack(trainFingerprintArrayList)
    trainClassLabels = numpy.asarray(trainClassLabelList)
    train_set = (trainFingerprints, trainClassLabels)

    testList, testClassLabelList, testFingerprintArrayList, testItemList = [], [], [], []
    for item in listTestActive:
        compoundTuple = (item[0], item[2], item[3])
        tuple = (numpy.array(item[4]), item[3], compoundTuple)
        for w in range(0,multiplier):
            testList.append(tuple)

    for item in listTestInactive:
        compoundTuple = (item[0], item[2], item[3])
        tuple = (numpy.array(item[4]), item[3], compoundTuple)
        testList.append(tuple)

    random.shuffle(testList)
    random.shuffle(testList)
    for item in testList:
        testFingerprintArrayList.append(item[0])
        testClassLabelList.append(item[1])
        testItemList.append(item[2])

    testFingerprints = numpy.vstack(testFingerprintArrayList)
    testClassLabels = numpy.asarray(testClassLabelList)
    test_set = (testFingerprints, testClassLabels)

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables
        The reason we store our dataset in shared variables is to allow Theano to copy it into the GPU memory
        (when code is run on GPU). Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
        # When storing data on the GPU it has to be stored as floats therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning  ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (test_set_x, test_set_y)]
    # rows_train = train_set_x.get_value(borrow=True).shape[0]
    # rows_test = test_set_x.get_value(borrow=True).shape[0]
    return rval, trainItemList, testItemList