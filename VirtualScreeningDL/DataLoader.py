import numpy
import theano
import theano.tensor as T
import random
import sys

# The load_files_for_task module takes the input files for a single task
# param: activeFile: File containing the Actives for the task
# param: inactiveFile: File containing the Inactives for the task
# return: foldsActive: Dictionary of five lists
# return: foldsInactive: Dictionary of five lists
# Each file contains five columns
# Col-0: Hashtag (Assigned by Jason)
# Col-1: ClassLabel (1 for Active and 0 for Inactive: from the original dataset)
# Col-2: CompoundName (from the original dataset)
# Col-3: Fold for Cross-Validation (assigned by Jason)
# Col-3: 1024 char fingerprint string (contains 0s and 1s: from the original dataset)
# The module loads the data from these files into two dictionaries - foldsActive and foldsInactive
# Each dictionary contains five lists: 0 to 4 corresponding to a fold.

def load_files_for_task(activeFile, inactiveFile):
    print "Entering load_files_for_task ..."
    # Initialize the two dictionaries
    foldsActive = {0:[],1:[],2:[],3:[],4:[]}
    foldsInactive = {0:[],1:[],2:[],3:[],4:[]}

    fileActive = open(activeFile, 'r')
    listOfActives = fileActive.readlines()
    fileActive.close()

    fileInactive = open(inactiveFile, 'r')
    listOfInactives = fileInactive.readlines()
    fileInactive.close()

    numActives = len(listOfActives)
    numInactives = len(listOfInactives)
    multiplier = int((numInactives-1)/numActives) + 1
    print "INITIAL: Actives: ", numActives, " Inactives: ", numInactives, " Multiplier: ", multiplier

    if (multiplier > 50):
        multiplier = 50
        print "Multiplier capped at: ", multiplier

    rowCount = 0
    print "Status for ", activeFile, ":"

    for line in listOfActives:
        rowCount += 1
        if (rowCount%10000==0):
            print rowCount, ", ",
        words = line.split()
        classLabel = int(words[1])
        compoundName = words[2]
        fold = int(words[3])
        fingerprint = words[4]
        tuple = (rowCount, fold, compoundName, classLabel, fingerprint)
        for m in range (0, multiplier):
            foldsActive[fold].append(tuple)
    print "Finished with Actives. Starting Inactives."

    print "Status for ", inactiveFile, ":"
    for line in listOfInactives:
        rowCount += 1
        if (rowCount%10000==0):
            print rowCount, ", ",
        words = line.split()
        classLabel = int(words[1])
        compoundName = words[2]
        fold = int(words[3])
        fingerprint = words[4]
        tuple = (rowCount, fold, compoundName, classLabel, fingerprint)
        foldsInactive[fold].append(tuple)

    print "Finished with Inactives."

    for x in range(0,5):
        print "In loadfiles_for_task: ", x, len(foldsActive[x]), len(foldsInactive[x])

    print "Exiting load_files_for_task ..."
    return foldsActive, foldsInactive, multiplier

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

def prepare_cv_datalists(foldSet, foldsActive, foldsInactive):
    print "Inside prepare_cv for: ", foldSet
    foldList, foldFingerprintArrayList, foldClassLabelList, foldItemList = [], [], [], []

    for fold in foldSet:
        foldList.extend(foldsActive[fold])
        foldList.extend(foldsInactive[fold])

    random.shuffle(foldList)

    for tuple in foldList:
        fingerprint = tuple[4]
        fingerprintArray = [float(int(fingerprint[i:i+1], 2)) for i in range(0, len(fingerprint), 1)]
        foldFingerprintArrayList.append(fingerprintArray)
        foldClassLabelList.append(tuple[3])
        compoundTuple = (tuple[0], tuple[2], tuple[3])
        foldItemList.append(compoundTuple)

    foldFingerprints = numpy.vstack(foldFingerprintArrayList)
    foldClassLabels = numpy.asarray(foldClassLabelList)
    rowsFold = len(foldClassLabelList)
    dataset = (foldFingerprints, foldClassLabels, foldItemList, rowsFold)

    return dataset

def create_mega_batches(dataset, mega_batch_size):
    datasetList = []
    num_rows = dataset[3]
    fingerprints = dataset[0]
    classLabels = dataset[1]
    itemList = dataset[2]
    num_mega_batches = (num_rows/mega_batch_size)
    for x in range(0,num_mega_batches):
        print "Creating megabatch: ", x
        fingerprintsMB = fingerprints[x*mega_batch_size:(x+1)*(mega_batch_size)]
        classLabelsMB = classLabels[x*mega_batch_size:(x+1)*(mega_batch_size)]
        itemListMB = itemList[x*mega_batch_size:(x+1)*(mega_batch_size)]
        datasetMB = (fingerprintsMB, classLabelsMB, itemListMB, mega_batch_size)
        datasetList.append(datasetMB)

    if (num_mega_batches*mega_batch_size < num_rows):
        num_mega_batches += 1
        x = num_mega_batches - 1
        print "Creating megabatch: ", x
        fingerprintsMB = fingerprints[x*mega_batch_size:num_rows]
        classLabelsMB = classLabels[x*mega_batch_size:num_rows]
        itemListMB = itemList[x*mega_batch_size:num_rows]
        datasetMB = (fingerprintsMB, classLabelsMB, itemListMB, (num_rows - (x*mega_batch_size)))
        datasetList.append(datasetMB)

    return datasetList, num_mega_batches

if __name__ == '__main__':
    numBatchesPerSet = 1
    batch_size = 128
    mega_batch_size = batch_size * numBatchesPerSet

    readPath = sys.argv[1]
    print readPath
    file = open(readPath, 'r')
    listOfLines = file.readlines()

    for line in listOfLines:
        print line,
        words = line.split('|')
        activeFile = words[2]
        inactiveFile = words[3]
        print "ActiveFile: ", activeFile
        print "InactiveFile: ", inactiveFile
        foldsActive, foldsInactive, multiplier = load_files_for_task(activeFile, inactiveFile)
        print "Multiplier: ", multiplier
        cvSet = set([0,1,2,3,4])
        for testFold in range (0,5):
            testFoldList = []
            print "Loading data for test fold: ", testFold
            testFoldList.append(testFold)
            testFoldSet = set(testFoldList)
            trainFoldSet = cvSet - testFoldSet
            print "TestFoldSet: ", testFoldSet, " TrainFoldSet: ", trainFoldSet

            trainDataset = prepare_cv_datalists(trainFoldSet, foldsActive, foldsInactive)
            testDataset = prepare_cv_datalists(testFoldSet, foldsActive, foldsInactive)

            trainDatasetList = create_mega_batches(trainDataset, mega_batch_size)
            testDatasetList = create_mega_batches(testDataset,mega_batch_size)

            trainDatasetMB = trainDatasetList[0]
            testDatasetMB = testDatasetList[0]

            train_set = (trainDatasetMB[0], trainDatasetMB[1])
            rowsTrain = trainDatasetMB[3]
            numTrainBatches = (rowsTrain / batch_size)
            if (numTrainBatches*batch_size < rowsTrain):
                numTrainBatches = numTrainBatches + 1

            test_set = (testDatasetMB[0], testDatasetMB[1])
            rowsTest = testDatasetMB[3]
            numTestBatches = (rowsTest / batch_size)
            if (numTestBatches*batch_size < rowsTest):
                numTestBatches = numTestBatches + 1

            print "Fold: ", testFold
            print " # of Train Rows: ", rowsTrain, " # of Train Batches: ", numTrainBatches
            print " # of Test Rows: ", rowsTest, " # of Train Batches: ", numTestBatches

            trainDatasets = shared_dataset(train_set)
            testDatasets = shared_dataset(test_set)

        print '-----------------------------------------------------------------------------------'
