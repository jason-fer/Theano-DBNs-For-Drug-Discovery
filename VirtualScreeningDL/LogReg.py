"""
This tutorial introduces logistic regression using Theano and stochastic gradient descent. Logistic regression is a
probabilistic, linear classifier. It is parametrized by a weight matrix :math:`W` and a bias vector :math:`b`.
Classification is done by projecting data points onto a set of hyperplanes, the distance to
which is used to determine a class membership probability. Mathematically, this can be written as:
.. math::
  P(Y=i|x, W,b) &= softmax_i(W x + b) \\
                &= \frac {e^{W_i x + b_i}} {\sum_j e^{W_j x + b_j}}

The output of the model or prediction is then done by taking the argmax of the vector whose i'th element is P(Y=i|x).
.. math::   y_{pred} = argmax_i P(Y=i|x,W,b)
This tutorial presents a stochastic gradient descent optimization method suitable for large datasets.
"""

import cPickle
import time
import sys
import numpy
import theano
import theano.tensor as T
from DataLoader import load_files_for_task, shared_dataset, prepare_cv_datalists, create_mega_batches

class LogisticRegression(object):
    """Multi-class Logistic Regression Class
    The logistic regression is fully described by a weight matrix :math:`W` and bias vector :math:`b`.
    Classification is done by projecting data points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression
        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the architecture (one minibatch)
        :type n_in: int
        :param n_in: number of input units, the dimension of the space in which the datapoints lie
        :type n_out: int
        :param n_out: number of output units, the dimension of the space in which the labels lie
        """
        # start-snippet-1

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared( value=numpy.zeros( (n_in, n_out), dtype=theano.config.floatX ), name='W', borrow=True )

        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(value=numpy.zeros( (n_out,), dtype=theano.config.floatX ), name='b', borrow=True )

        # symbolic expression for computing the matrix of class-membership probabilities where:
        # W is a matrix where column-k represent the separation hyper plain for class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of hyper plane-k
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction of this model under a given target distribution.
        .. math::
            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that the learning rate is less dependent on the batch size
        """
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e. number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain [0,1,2,... n-1]
        # T.log(self.p_y_given_x) is a matrix of Log-Probabilities (call it LP) with one row per example and one column per class
        # LP[T.arange(y.shape[0]),y] is a vector v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ..., LP[n-1,y[n-1]]] and
        # T.mean(LP[T.arange(y.shape[0]),y]) is the mean (across minibatch examples) of the elements in v, i.e., the mean log-likelihood across the minibatch.

        #print "y.ndim = ",y.ndim
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # end-snippet-2

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch
        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError( 'y should have the same shape as self.y_pred', ('y', y.type, 'y_pred', self.y_pred.type) )

        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


def perform_cv_onefold(taskId, testFold, foldsActive, foldsInactive, multiplier, modelDir, predDir):
    """
    Demonstrate stochastic gradient descent optimization of a log-linear model
    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic gradient)
    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer
    """

    learning_rate=0.1
    n_epochs=500
    batch_size = 128
    numBatchesPerSet = 100
    mega_batch_size = batch_size * numBatchesPerSet

    writeModelFile = modelDir + 'Model.' + taskId + '.' + str(testFold) +'.pkl'

    cvSet = set([0,1,2,3,4])
    testFoldList = []
    print "Loading data for test fold: ", testFold
    testFoldList.append(testFold)
    testFoldSet = set(testFoldList)
    trainFoldSet = cvSet - testFoldSet
    print "TestFoldSet: ", testFoldSet, " TrainFoldSet: ", trainFoldSet

    testDataset = prepare_cv_datalists(testFoldSet, foldsActive, foldsInactive)
    trainDataset = prepare_cv_datalists(trainFoldSet, foldsActive, foldsInactive)
    n_train_totalbatches = trainDataset[3]/batch_size
    if (n_train_totalbatches*batch_size < trainDataset[3]):
        n_train_totalbatches += 1

    print "Row Counts: ", trainDataset[3], testDataset[3]
    trainDatasetList, n_train_megabatches = create_mega_batches(trainDataset, mega_batch_size)
    print "Number of training megabatches = ", n_train_megabatches
    trainDatasetMB = trainDatasetList[0]

    test_set = (testDataset[0], testDataset[1])
    rows_test = testDataset[3]
    testItemList = testDataset[2]
    n_test_batches = (rows_test / batch_size)
    if (n_test_batches * batch_size < rows_test):
        n_test_batches = n_test_batches + 1

    train_set = (trainDatasetMB[0], trainDatasetMB[1])
    rows_train = trainDatasetMB[3]
    trainItemList = trainDatasetMB[2]
    n_train_minibatches = (rows_train / batch_size)
    if (n_train_minibatches * batch_size < rows_train):
        n_train_minibatches = n_train_minibatches + 1

    rows_valid = rows_train
    n_valid_batches = n_train_minibatches

    print "For Fold: ", testFold
    print "Number of Rows: Train, Valid, Test: ", rows_train, rows_valid, rows_test
    print "Number of Mini-batches: Train, Valid, Test: ", n_train_minibatches, n_valid_batches, n_test_batches

    # creating shared datasets for Theano.
    # each dataset has a set of mini-batches.
    trainSharedDataset = shared_dataset(train_set)
    testSharedDataset = shared_dataset(test_set)

    train_set_x, train_set_y = trainSharedDataset
    test_set_x, test_set_y = testSharedDataset
    valid_set_x, valid_set_y = trainSharedDataset
    print "Finished creating datasets."

    ####################### BUILD ACTUAL MODEL #######################
    #print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # generate symbolic variables for input (x and y represent a minibatch)
    x = T.matrix('x')  # data, presented as rasterized images
    y = T.ivector('y')  # labels, presented as 1D vector of [int] labels

    # construct the logistic regression class
    # n_in: Each fingerprint = 1x1024
    # n_out: 2 different classes (Single task LR)
    classifier = LogisticRegression(input=x, n_in=32 * 32, n_out=2)

    # the cost we minimize during training is the negative log likelihood of the model in symbolic format
    cost = classifier.negative_log_likelihood(y)

    # start-snippet-3
    # compute the gradient of cost with respect to theta = (W,b)
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function( inputs=[index], outputs=cost, updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # compiling a Theano function that computes the mistakes that are made by the model on a minibatch
    test_model = theano.function( inputs=[index], outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-3

    ################ TRAIN MODEL ################
    #print '... training the model'
    # early-stopping parameters
    patience = 5000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is found
    improvement_threshold = 0.995  # a relative improvement of this much is considered significant
    validation_frequency = min(n_train_totalbatches, patience / 2)
    print "Validation Frequency: ", validation_frequency
    # go through this many minibatches before checking the network on the validation set; in this case we check every epoch

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0
    countCumulMiniBatchesTrained = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        # this is the key point where he changed things; 
        # he's loading 10k to 15k tems into shared data each iteration
        # loading 300megs of data into shared data breaks the system; 
        # this prevents that issue
        for megabatch_index in xrange(n_train_megabatches):
            trainDatasetMB = trainDatasetList[megabatch_index]
            train_set = (trainDatasetMB[0], trainDatasetMB[1])
            trainSharedDataset = shared_dataset(train_set)
            train_set_x, train_set_y = trainSharedDataset

            nextMB = (megabatch_index + 1)% n_train_megabatches
            validDatasetMB = trainDatasetList[nextMB]
            valid_set = (validDatasetMB[0], validDatasetMB[1])
            validSharedDataset = shared_dataset(valid_set)
            valid_set_x, valid_set_y = trainSharedDataset

            for minibatch_index in xrange(n_train_minibatches):
                minibatch_avg_cost = train_model(minibatch_index)
                countCumulMiniBatchesTrained += 1
                actualMinibatch = (megabatch_index*numBatchesPerSet)+ minibatch_index
                # print "Megabatch: ", megabatch_index, " Minibatch: ", minibatch_index, " Actual MB: ", actualMinibatch, " countCumulMiniBatchesTrained: ", countCumulMiniBatchesTrained
                if (countCumulMiniBatchesTrained + 1) % validation_frequency == 0:
                    print "Validating for: ", countCumulMiniBatchesTrained,
                    # compute zero-one loss on validation set
                    validation_losses = [validate_model(i)
                                            for i in xrange(n_valid_batches)]
                    this_validation_loss = numpy.mean(validation_losses)

                    print( 'epoch %i, minibatch %i/%i, validation error %f %%' %
                        (epoch,  actualMinibatch%n_train_totalbatches, n_train_totalbatches, this_validation_loss * 100.) )

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
                        #improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss *  \
                            improvement_threshold:
                            patience = max(patience, countCumulMiniBatchesTrained * patience_increase)

                        best_validation_loss = this_validation_loss
                        # test it on the test set

                        test_losses = [test_model(i) for i in xrange(n_test_batches)]
                        test_score = numpy.mean(test_losses)

                        # save the best model
                        with open(writeModelFile, 'w') as f:
                            cPickle.dump(classifier, f)

                if patience <= countCumulMiniBatchesTrained:
                    done_looping = True
                    break

    end_time = time.clock()

    print( ('Optimization complete for %d with best validation score of %f %% with test performance %f %%')
        % (testFold, best_validation_loss * 100., test_score * 100.) )
    print 'The code ran for %d epochs, with %f epochs/sec' % (epoch, 1. * epoch / (end_time - start_time))
    # print >> sys.stderr, ('The code for file ' + os.path.split(__file__)[1] + ' ran for %.1fs' % ((end_time - start_time)))

    # end-snippet-4
    # Now we do the predictions

    # load the saved best model for this fold
    classifier = cPickle.load(open(writeModelFile))

    # compile a predictor function
    predict_model = theano.function( inputs=[classifier.input],  outputs=[classifier.y_pred,classifier.p_y_given_x])
    # compile a confidence predictor function
    # predict_conf_model = theano.function( inputs=[classifier.input], outputs=classifier.p_y_given_x)

    # We can test it on some examples from test test
    test_set_x = test_set_x.get_value()

    predicted_values, predicted_conf_values = predict_model(test_set_x[:(rows_test-1)])

    print ("Predicted values for test set:")
    print "# of rows in Test: ", rows_test
    writePredictionsFile = predDir + 'Predictions.' + taskId + '.' + str(testFold) +'.txt'

    countTP, countTN, countFP, countFN = 0,0,0,0
    accuracy, precision, recall = 0.0, 0.0, 0.0

    with open(writePredictionsFile, 'w') as f:
        f.write("rowId|prediction|probOf0|probOf1|compoundId|compoundName|Actual\n")
        for n in range(0,rows_test-1):
            printString = str(n) + "|" \
                          + str(predicted_values[n]) + "|" \
                          + str(round(predicted_conf_values[n][0],3)) + "|" + str(round(predicted_conf_values[n][1],3)) + "|" \
                          + str(testItemList[n][0]) + "|" + str(testItemList[n][1]) + "|" + str(testItemList[n][2]) + "\n"
            f.write(printString)
            if (predicted_values[n] == testItemList[n][2]):
                if (predicted_values[n] == 1):
                    countTP += 1
                else:
                    countTN += 1
            else:
                if (predicted_values[n] == 1):
                    countFP += 1
                else:
                    countFN += 1

        print "TruePos: ", countTP, " TrueNeg: ", countTN
        print "FalsePos: ", countFP, " FalseNeg: ", countFN
        totalPredictions = countTN + countTP + countFN + countFP
        accuracy = (100.0 * (countTP + countTN))/totalPredictions
        accuracyAct = (100.0 * countTP)/(countTP + countFN)
        accuracyInact = (100.0 * countTN)/(countTN + countFP)
        precision = (100.0*countTP) / (countTP + countFP)
        recall = (100.0*countTP) / (countTP + countFN)
        print "Accuracy: ", accuracy, " AccuracyActives: ", accuracyAct, " AccuracyInactives: ", accuracyInact
        print "Precision: ", precision, " Recall: ", recall

if __name__ == '__main__':

    filePath = sys.argv[1]
    start_time_main = time.clock()

    workingDir = sys.argv[2]
    modelDir = workingDir + 'Models/'
    predDir = workingDir + 'Predictions/'
    print filePath, workingDir, modelDir, predDir
    file = open(filePath, 'r')
    listOfLines = file.readlines()
    for line in listOfLines:
        start_time_file = time.clock()
        print "\n---------------------------------------------------------------\n", line,
        words = line.split('|')
        taskId = words[0]
        activeFile = words[2]
        inactiveFile = words[3]
        print "ActiveFile: ", activeFile
        print "InactiveFile: ", inactiveFile
        foldsActive, foldsInactive, multiplier = load_files_for_task(activeFile, inactiveFile)
        for fold in range(0,5):
            print 'inside fold', fold
            perform_cv_onefold(taskId, fold, foldsActive, foldsInactive, multiplier, modelDir, predDir)

        end_time_file = time.clock()
        print 'The full run for: ',words[0],' took: %f secs.' % (end_time_file - start_time_file)
    end_main_time = time.clock()
    print 'The complete program ran for: %f secs.' % (end_main_time - start_time_main)
