"""
**************************************************************************
Generate Multitask Datasets
**************************************************************************

Randomly sample with replacement & generate multitask datasets

-Use stratified sampling across all tasks with replacement to fill the batches
-Target is about 10k items per file

@author: Jason Feriante <feriante@cs.wisc.edu>
@date: 26 July 2015
"""
import os, hashlib, sys, random, time
from lib.theano import helpers



def gen_multitask(data_type):
    hashmap = helpers.load_hashmap(data_type)
    rev_targets, target_columns = helpers.get_rev_targets(data_type)

    """Load data from the existing folds"""
    fold_path = helpers.get_fold_path(data_type)


    tasks = {}
    # build task object to contain the datasets
    for col_id in range(len(target_columns)):

        target = rev_targets[col_id]['target']
        tasks[target] = {'actives': [], 'inactives': [], 
            'active_count':0, 'inactive_count':0}


    # Load actives
    count_actives = 0
    """<target>_actives.fl"""
    for col_id in range(len(target_columns)):
        target = rev_targets[col_id]['target']
        fname = rev_targets[col_id]['fname']

        with open(fold_path + '/' + fname) as f:
            lines = f.readlines()

            for line in lines:
                # [hash_id, is_active, native_id, fold, bitstring]
                parts = line.rstrip('\n').split(r' ')
                bitstring = parts[4]
                is_active = int(parts[1])
                assert(is_active == 1)
                tasks[target]['actives'].append(bitstring)
                count_actives += 1


    # Load inactives
    count_inactives = 0
    """<target>_inactives.fl"""
    for col_id in range(len(target_columns)):
        target = rev_targets[col_id]['target']
        fname = target + '_inactives.fl'

        with open(fold_path + '/' + fname) as f:
            lines = f.readlines()

            for line in lines:
                # [hash_id, is_active, native_id, fold, bitstring]
                parts = line.rstrip('\n').split(r' ')
                bitstring = parts[4]
                is_active = int(parts[1])
                assert(is_active == 0)
                tasks[target]['inactives'].append(bitstring)
                count_inactives += 1



    for col_id in range(len(target_columns)):
        target = rev_targets[col_id]['target']
        tasks[target]['inactive_count'] = len(tasks[target]['inactives']) - 1
        tasks[target]['active_count'] = len (tasks[target]['actives']) - 1

    # now, we use tasks to "multitask"... 
    # build a set of 10k items
    # 1-draw evenly from actives / inactives
    # 2-draw evenly from each dataset
    # 3-sample randomly with replacment
    # 4-write to file with the 'truth' column in the hashmap
    # not in hashmap = all targets inactive
    #iterate through each task drawing 10 active & 10 inactive randomly
    

    # we will make random batches around the size of the original dataset
    multitask_size = count_inactives + count_actives

    # let's give each task an equal proportion for now 
    # (this is very naive, considering variations in batch size)
    # however, each target counts equal weight towards overall accuracy
    task_count = len(target_columns) 


    while(multitask_size > 0):
        # keep building batches until we make 'enough'
        



        multitask_size -= 10000
    
    

def main(args):
    """ Evenly draw from all datasets to create minibatches """
    """ Keep files to 30,000 lines or less (to avoid overloading memory) """
    if(len(args) < 2):
        print 'usage: <tox21, dud_e, muv, or pcba>'
        return

    dataset = args[1]

    # in case of typos
    if(dataset == 'dude'):
        dataset = 'dud_e'
        
    print "Generating multitask data for " \
        + dataset + "........."

    if(dataset == 'tox21'):
        gen_multitask('Tox21')

    elif(dataset == 'dud_e'):
        gen_multitask('DUD-E')

    elif(dataset == 'muv'):
        gen_multitask('MUV')

    elif(dataset == 'pcba'):
        gen_multitask('PCBA')
    else:
        print 'dataset param not found. options: tox21, dud_e, muv, or pcba'



if __name__ == '__main__':
    start_time = time.clock()

    main(sys.argv)

    end_time = time.clock()
    print 'runtime: %.2f secs.' % (end_time - start_time)

