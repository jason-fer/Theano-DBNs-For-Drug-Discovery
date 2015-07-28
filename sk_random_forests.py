"""
**************************************************************************
Scikit Learn Random Forests
**************************************************************************

@author Jason Feriante <feriante@cs.wisc.edu>
@date 10 July 2015
"""

import generate_folds, os, sys, random, time
import numpy as np
from sklearn import linear_model
from sklearn import metrics
from lib.theano import helpers
from sklearn.ensemble import RandomForestClassifier

get_fold_path = helpers.get_fold_path
get_target = helpers.get_target
parse_line = helpers.parse_line
build_targets = helpers.build_targets
oversample = helpers.oversample
get_folds = helpers.get_folds


def random_forest(target, X, Y, X_test, Y_test, fold_id):

    print 'Running random forests for target: ' + target
    clf = RandomForestClassifier(n_estimators=100, max_features=500)
    clf.fit(X, Y)

    # Z is our prediction
    Z = clf.predict(X_test)

    """ Debug """
    # print 'Training size:'
    # print X.shape
    # print 'Test size:'
    # print X_test.shape

    num_correct = 0
    num_false = 0
    for i in range(len(Z)):
        if Z[i] == Y_test[i]:
            num_correct += 1
        else:
            num_false += 1

    total = len(Z)
    percent_correct = num_correct / float(total)


    # get the probability of our predictions
    prob_preds = clf.predict_proba(X_test)[:, 1]
    
    # use that to determine the ROC curve
    fpr, tpr, thresholds = metrics.roc_curve(Y_test, prob_preds)
    auc = metrics.auc(fpr, tpr)


    print 'target:' + target + ' fold:' + str(fold_id) + ' predicted: ' + \
        str(total) + ' wrong: ' + \
        str(num_false) + ' pct correct: ' + str(percent_correct) + ', auc: ' + str(auc)

    return percent_correct, auc



def run_predictions(data_type, curr_target):

    fold_path = get_fold_path(data_type)
    targets = build_targets(fold_path, data_type)
    # print "Found " + str(len(targets)) + " targets for " + data_type

    fold_accuracies = {}
    did_something = False
    for target, fnames in targets.iteritems():
        if (target != curr_target):
            continue
        else:
            did_something = True

        # retrieve our stratified folds
        folds = get_folds(data_type, fold_path, target, fnames)

        pct_ct = []
        roc_auc = []
        # run 4 folds vs 1 fold with each possible scenario
        for curr_fl in range(len(folds)):

            print 'Building data for target: ' + target + ', fold: ' + str(curr_fl)

            # folds 1-4
            temp_data = []
            for i in range(len(folds)):
                if(i == curr_fl):
                    # don't include the test fold
                    continue
                else:
                    temp_data += folds[i]

            # vs current 5th test fold
            test_data = folds[curr_fl]
            
            """ Turning 1024 bits into features is a slow process """
            # build training data
            X = []
            Y = []
            for i in range(len(temp_data)):
                row = []
                for bit in temp_data[i][0]:
                    row.append(int(bit))
                X.append(row)
                Y.append(int(temp_data[i][1]))

            X = np.array(X)
            Y = np.array(Y)

            # build test data
            X_test = []
            Y_test = []
            for i in range(len(test_data)):
                row = []
                for bit in test_data[i][0]:
                    row.append(int(bit))
                X_test.append(row)
                Y_test.append(int(test_data[i][1]))

            X_test = np.array(X_test)
            Y_test = np.array(Y_test)

            percent_correct, auc = random_forest(target, X, Y, X_test, Y_test, curr_fl)
            pct_ct.append(percent_correct)
            roc_auc.append(auc)

            # now get the average fold results for this target
            accuracy = sum(pct_ct) / float(len(pct_ct))
            all_auc =  sum(roc_auc) / float(len(roc_auc))
            print 'Results for '+ target + ': accuracy: ' + str(accuracy) + ', auc: ' + str(all_auc)
            # update fold accuracies
            fold_accuracies[target] = (accuracy, all_auc)


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

    print "Running Scikit Learn Random Forests for " \
        + dataset + "........."

    is_numeric = helpers.is_numeric(target)
    if(is_numeric):
        target_list = helpers.get_target_list(dataset)
        target = target_list[int(target)]

    if(dataset == 'tox21'):
        run_predictions('Tox21', target)

    elif(dataset == 'dud_e'):
        run_predictions('DUD-E', target)

    elif(dataset == 'muv'):
        run_predictions('MUV', target)

    elif(dataset == 'pcba'):
        run_predictions('PCBA', target)
    else:
        print 'dataset param not found. options: tox21, dud_e, muv, or pcba'



if __name__ == '__main__':
    start_time = time.clock()

    main(sys.argv)

    end_time = time.clock()
    print 'runtime: %.2f secs.' % (end_time - start_time)

