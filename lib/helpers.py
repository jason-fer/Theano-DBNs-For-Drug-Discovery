import generate_folds, os, sys, random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model



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

    return folds
