import generate_folds, os, sys, random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier



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

    return folds



def random_forest(target, X, Y, X_test, Y_test, fold_id):

    print 'Running random forests for target: ' + target
    clf = RandomForestClassifier(n_estimators=100, max_features=500)
    clf.fit(X, Y)

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

    print 'target:' + target + ' fold:' + str(fold_id) + ' predicted: ' + \
        str(total) + ' correct: ' + str(num_correct) + ' wrong: ' + \
        str(num_false) + ' pct correct: ' + str(percent_correct)

    return percent_correct



def run_predictions(data_type):

    fold_path = get_fold_path(data_type)
    targets = build_targets(fold_path, data_type)
    print "Found " + str(len(targets)) + " targets for " + data_type

    fold_accuracies = {}
    for target, fnames in targets.iteritems():

        # retrieve our stratified folds
        folds = get_folds(data_type, fold_path, target, fnames)

        # shuffle the folds once upfront
        for i in range(len(folds)):
            random.shuffle(folds[i])

        pct_ct = []
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

            pct_ct.append(random_forest(target, X, Y, X_test, Y_test, curr_fl))

        # now get the average fold results for this target
        accuracy = sum(pct_ct) / float(len(pct_ct))
        print 'Results for '+ target + ':' + str(accuracy)
        # update fold accuracies
        fold_accuracies[target] = accuracy

    print '####################  Results for ' + data_type + ' ####################'
    # output results
    accuracies = 0.00
    num_targets = 0.00
    for target, acc in fold_accuracies.iteritems():
        print target + ' accuracy: ' + str(acc)
        accuracies += acc
        num_targets += 1

    overall_acc = accuracies / num_targets
    print ' overall accuracy: ' + str(overall_acc)
    print '############################################################'


def tox21():
    run_predictions('Tox21')



def dud_e():
    run_predictions('DUD-E')



def muv():
    run_predictions('MUV')



def pcba():
    run_predictions('PCBA')



def main(args):
    print "Running Scikit Learn Random Forest Classifier........."
    tox21()
    # dud_e()
    muv()
    pcba()



if __name__ == "__main__":
    main(sys.argv)


