import generate_folds, os, sys, random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from bitstring import BitArray

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
    hash_id = parts[0]
    is_active = parts[1]
    native_id = parts[2]
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



def get_folds(data_type):
    fold_path = get_fold_path(data_type)
    targets = build_targets(fold_path, data_type)

    print "Found " + str(len(targets)) + " targets for " + data_type

    # store folds by target
    all_folds = {}
    for target, fnames in targets.iteritems():
        # generate a fold set fo reach target
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

        # add the folds to our target
        # print 'target: ' + target
        # print len(folds)
        all_folds[target] =  folds

    """ Debug """
    # print "length of all folds"
    # print len(folds)
    # print "length of respective folds"
    # print len(folds['0'])
    # print len(folds['1'])
    # print len(folds['2'])
    # print len(folds['3'])
    # print len(folds['4'])

    return all_folds



def logistic_regression(target, X, Y, X_test, Y_test):
    print 'Running logistic regression for target: ' + target
    logreg = linear_model.LogisticRegression(C=1e5)
    logreg.fit(X, Y)

    Z = logreg.predict(X_test)

    # print X.shape
    # print Y.shape
    print 'Training size:'
    print X.shape
    print 'Test size:'
    print X_test.shape
    # print Y_test.shape

    # print Z
    # print Y_test
    # Z = Z.tolist()
    # Y_test = Y_test.tolist()

    num_correct = 0
    num_false = 0
    for i in range(len(Z)):
        if Z[i] == Y_test[i]:
            num_correct += 1
        else:
            num_false += 1

    total = len(Z)
    print 'Total predictions: ' + str(total)
    print 'Num correct: ' + str(num_correct)
    print 'Num false: ' + str(num_false)
    print 'Percent correct for ' + target + ': ' + \
        str(num_correct / float(total))


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



def tox21():
    # get our 5x folds...
    all_folds = get_folds('Tox21')

    for target, folds in all_folds.iteritems():
        # lets try a generic dataset with just 1 fold
        temp_data = []
        for i in range(len(folds) - 1):
            temp_data += folds[i]

        test_data = folds[4]
        
        print 'Building data for target: ' + target

        # oversample our datasets
        temp_data = oversample(temp_data)
        test_data = oversample(test_data)

        # randomize the ordering
        random.shuffle(temp_data)
        random.shuffle(test_data)

        # for i in range(10):
        #     print temp_data[i]

        # build training data
        X = []
        Y = []
        for i in range(len(temp_data)):
            string = BitArray(bin=str(temp_data[i][0]))
            row = []
            for bit in string:
                row.append(int(bit))
            X.append(row)
            Y.append(int(temp_data[i][1]))

        X = np.array(X)
        Y = np.array(Y)

        # build test data
        X_test = []
        Y_test = []
        for i in range(len(test_data)):
            string = BitArray(bin=str(test_data[i][0]))
            row = []
            for bit in string:
                row.append(int(bit))
            X_test.append(row)
            Y_test.append(int(test_data[i][1]))

        X_test = np.array(X_test)
        Y_test = np.array(Y_test)

        logistic_regression(target, X, Y, X_test, Y_test)

        # all done with 1 target....
        exit(0)



def main(args):
    # generate DUDE-E folds
    # dud_e()
    # muv()
    # pcba()
    tox21()


if __name__ == "__main__":
    main(sys.argv)


