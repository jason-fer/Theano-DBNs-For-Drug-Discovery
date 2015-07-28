"""
**************************************************************************
Theano Logistic Regression
**************************************************************************

This version was just for local testing (Vee ran his version for our SBEL batch jobs)

@author: Jason Feriante <feriante@cs.wisc.edu>
@date: 10 July 2015

**************************************************************************
logistic regression using Theano and stochastic gradient descent. Logistic regression is a
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

import cPickle, time, os, sys, numpy, theano
from sklearn import metrics
import theano.tensor as T
from lib.theano import helpers

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



def sgd_optimization(data_type, target, model_dir, learning_rate=0.1, n_epochs=10, batch_size=100):
    """
    Demonstrate stochastic gradient descent optimization of a log-linear model
    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic gradient)
    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer
    """

    test_fold = 1 #xxxxxxxxxxxx TEMP XXXXXXXXXXXXXXXX
    write_model_file = model_dir + '/model.' + target + '.' + str(test_fold) +'.pkl'
    fold_path = helpers.get_fold_path(data_type)
    targets = helpers.build_targets(fold_path, data_type)
    fnames = targets[target]

    fold_accuracies = {}
    did_something = False

    # pct_ct = []
    # roc_auc = []
    # run 4 folds vs 1 fold with each possible scenario
    # for curr_fl in range(5):
    #     print 'Building data for target: ' + target + ', fold: ' + str(curr_fl)

    # loop through all folds, for now just do 1!
    datasets, test_set_labels = helpers.th_load_data(data_type, fold_path, target, fnames, 0, test_fold)

    train_set_x, train_set_y = datasets[0]
    test_set_x, test_set_y = datasets[1]
    valid_set_x = train_set_x
    valid_set_y = train_set_y

    # compute number of rows for training, validation and testing
    rows_train = train_set_x.get_value(borrow=True).shape[0]
    rows_valid = valid_set_x.get_value(borrow=True).shape[0]
    rows_test = test_set_x.get_value(borrow=True).shape[0]

    # compute number of minibatches for training, validation and testing
    n_train_batches = rows_train / batch_size
    n_valid_batches = rows_valid / batch_size
    n_test_batches = rows_test / batch_size

    ####################### BUILD ACTUAL MODEL #######################
    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # generate symbolic variables for input (x and y represent a minibatch)
    x = T.matrix('x')  # data, presented as rasterized images
    y = T.ivector('y')  # labels, presented as 1D vector of [int] labels

    # construct the logistic regression class
    # n_in: Each MNIST image has size 32*32 = 1024
    # n_out: 10 different digits - multi-task LR
    classifier = LogisticRegression(input=x, n_in=32 * 32, n_out=2)

    # the cost we minimize during training is the negative log likelihood of the model in symbolic format
    cost = classifier.negative_log_likelihood(y)

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

    # compute the gradient of cost with respect to theta = (W,b)
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    # start-snippet-3
    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(  inputs=[index], outputs=cost, updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-3

    ################ TRAIN MODEL ################
    # early-stopping parameters
    patience = 5000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is found
    improvement_threshold = 0.995  # a relative improvement of this much is considered significant
    validation_frequency = min(n_train_batches, patience / 2)
    # go through this many minibatches before checking the network on the validation set; in this case we check every epoch

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)

            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i)
                                     for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                # print( 'epoch %i, minibatch %i/%i, validation error %f %%' %
                #    (epoch,  minibatch_index + 1, n_train_batches, this_validation_loss * 100.) )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    # test it on the test set

                    test_losses = [test_model(i)
                                   for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    # print( ('     epoch %i, minibatch %i/%i, test error of best model %f %%' ) %
                    #   ( epoch,  minibatch_index + 1,  n_train_batches, test_score * 100. )  )

                    # save the best model
                    with open(write_model_file, 'w') as f:
                        cPickle.dump(classifier, f)

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()

    print( ('Optimization complete for %d with best validation score of %f %% with test performance %f %%')
        % (test_fold, best_validation_loss * 100., test_score * 100.) )
    print 'The code ran for %d epochs, with %f epochs/sec' % (epoch, 1. * epoch / (end_time - start_time))
    # print >> sys.stderr, ('The code for file ' + os.path.split(__file__)[1] + ' ran for %.1fs' % ((end_time - start_time)))

    # end-snippet-4
    # Now we do the predictions

    # load the saved best model for this fold
    classifier = cPickle.load(open(write_model_file))

    # compile a predictor function
    predict_model = theano.function(inputs=[classifier.input], outputs=[classifier.y_pred,classifier.p_y_given_x])
    # compile a confidence predictor function
    # predict_conf_model = theano.function( inputs=[classifier.input], outputs=classifier.p_y_given_x)
    # We can test it on some examples from test test

    """ *************** build AUC curve *************** """
    # get the probability of our predictions
    test_set = test_set_x.get_value()
    predicted_values, conf_preds = predict_model(test_set[:(rows_test)])

    conf_predictions = []
    for i in range(len(conf_preds)):
        # ignore the first column; this gives a lower score that seems wrong.
        conf_predictions.append(conf_preds[i][1])

    # determine ROC / AUC
    fpr, tpr, thresholds = metrics.roc_curve(test_set_labels, conf_predictions)
    auc = metrics.auc(fpr, tpr) # e.g. 0.855
    """ *********************************************** """

    num_correct = 0
    num_false = 0
    for i in range(len(predicted_values)):
        if predicted_values[i] == test_set_labels[i]:
            num_correct += 1
        else:
            num_false += 1

    total = len(predicted_values)
    percent_correct = num_correct / float(total)

    fold_results = ''
    fold_results += '####################  Results for ' + data_type + ' ####################' + '\n'
    fold_results += 'target:' + target + ' fold:' + str(test_fold) + ' predicted: ' + \
        str(total) + ' wrong: ' + \
        str(num_false) + ' pct correct: ' + str(percent_correct) + ', auc: ' + str(auc)

    print fold_results

    write_predictions_file = model_dir + '/predictions.' + target + '.' + str(test_fold) +'.txt'
    with open(write_predictions_file, 'w') as f:
        f.write(fold_results + "\n")



# def run_predictions(data_type, curr_target):

#     fold_path = get_fold_path(data_type)
#     targets = build_targets(fold_path, data_type)
#     # print "Found " + str(len(targets)) + " targets for " + data_type

#     fold_accuracies = {}
#     did_something = False
#     for target, fnames in targets.iteritems():
#         if (target != curr_target):
#             continue
#         else:
#             did_something = True

#         # retrieve our stratified folds
#         folds = get_folds(data_type, fold_path, target, fnames)

#         pct_ct = []
#         roc_auc = []
#         # run 4 folds vs 1 fold with each possible scenario
#         for curr_fl in range(5):
#             print 'Building data for target: ' + target + ', fold: ' + str(curr_fl)

#             # folds 1-4
#             temp_data = []
#             for i in range(len(folds)):
#                 if(i == curr_fl):
#                     # don't include the test fold
#                     continue
#                 else:
#                     temp_data += folds[i]

#             # vs current 5th test fold
#             test_data = folds[curr_fl]
            
#             """ Turning 1024 bits into features is a slow process """
#             # build training data
#             X = []
#             Y = []
#             for i in range(len(temp_data)):
#                 row = []
#                 for bit in temp_data[i][0]:
#                     row.append(int(bit))
#                 X.append(row)
#                 Y.append(int(temp_data[i][1]))

#             X = np.array(X)
#             Y = np.array(Y)

#             # build test data
#             X_test = []
#             Y_test = []
#             for i in range(len(test_data)):
#                 row = []
#                 for bit in test_data[i][0]:
#                     row.append(int(bit))
#                 X_test.append(row)
#                 Y_test.append(int(test_data[i][1]))

#             X_test = np.array(X_test)
#             Y_test = np.array(Y_test)

#             percent_correct, auc = random_forest(target, X, Y, X_test, Y_test, curr_fl)
#             pct_ct.append(percent_correct)
#             roc_auc.append(auc)


#             # now get the average fold results for this target
#             accuracy = sum(pct_ct) / float(len(pct_ct))
#             all_auc =  sum(roc_auc) / float(len(roc_auc))
#             print 'Results for '+ target + ': accuracy: ' + str(accuracy) + ', auc: ' + str(all_auc)
#             # update fold accuracies
#             fold_accuracies[target] = (accuracy, all_auc)


    if(did_something == False):
        print curr_target + ' not found in ' + data_type + '!'
        exit(0)
        
    print '####################  Results for ' + data_type + ' ####################'
    # output results
    accuracies = 0.00
    aucs = 0.00
    num_targets = 0.00
    for target, obj in fold_accuracies.iteritems():
        acc = obj[0]
        auc = obj[1]
        print target + ' accuracy: ' + str(acc) + ', auc:' + str(auc)
        accuracies += acc
        aucs += auc
        num_targets += 1

    # overall_acc = accuracies / num_targets
    # overall_auc = aucs / num_targets
    # print ' overall accuracy: ' + str(overall_acc) + ', overall auc: ' + str(overall_auc)
    print '############################################################'

def main(args):

    if(len(args) < 3 or len(args[2]) < 1):
        print 'usage: <tox21, dud_e, muv, or pcba> <target> '
        return

    dataset = args[1]
    target = args[2]
    # in case of typos
    if(dataset == 'dude'):
        dataset = 'dud_e'

    print "Running Theano Logistic Regression for " \
        + dataset + "........."

    is_numeric = helpers.is_numeric(target)
    if(is_numeric):
        target_list = helpers.get_target_list(dataset)
        target = target_list[int(target)]

    model_dir = 'theano_saved/logistic_regression'
    if(dataset == 'tox21'):
        sgd_optimization('Tox21', target, model_dir)

    elif(dataset == 'dud_e'):
        sgd_optimization('DUD-E', target, model_dir)

    elif(dataset == 'muv'):
        sgd_optimization('MUV', target, model_dir)

    elif(dataset == 'pcba'):
        sgd_optimization('PCBA', target, model_dir)
    else:
        print 'dataset param not found. options: tox21, dud_e, muv, or pcba'



if __name__ == '__main__':
    start_time = time.clock()

    main(sys.argv)

    end_time = time.clock()
    print 'runtime: %.2f secs.' % (end_time - start_time)


