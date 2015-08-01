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
import os, hashlib, sys, random, time, math
from lib.theano import helpers



def gen_multitask(data_type, size = False):

    print "Loading hashmap for " + str(data_type)
    hashmap = helpers.load_string_col_hashmap(data_type, size)
    rev_targets, target_columns = helpers.get_rev_targets(data_type)

    """Load data from the existing folds"""
    fold_path = helpers.get_fold_path(data_type)

    if(helpers.is_numeric(size) and size > 0 and size < 260):
        # Hardcode PCBA to 40 targets
        new_target_columns = []
        new_targets = {}
        # truncate things to contain 40 columns for PCBA
        for col_id in range(size):
            new_targets[col_id] = rev_targets[col_id]
            new_target_columns.append(target_columns[col_id])

        # wipe out the rev target data
        rev_targets.clear()
        rev_targets = new_targets
        target_columns = new_target_columns
        

    # build task object to contain the datasets
    tasks = {}
    for col_id in range(len(target_columns)):

        target = rev_targets[col_id]['target']
        tasks[target] = {'actives': [], 'inactives': [], 
            'active_count':0, 'inactive_count':0}


    print "Building active / inactive sets for " + str(data_type)

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

    # this is really 1/2 the ratio since we sample once from each data-type
    task_ratio = int(math.ceil( (10000.0 / task_count) / 2 ) )

    # build default set of inactive columns
    inactive_cols = [0] * task_count
    inactive_cols = ' '.join(str(v) for v in inactive_cols)

    """ where we will store our multitask batches """
    multitask_path = 'multitask/' + data_type + '/batch'

    print "Writing out multitask files for " + str(data_type)

    batch_count = 0
    # keep building batches until we make 'enough'
    while(multitask_size > 0):
        
        multitask = []
        #draw stratified samples based on the ratio
        for col_id in range(len(target_columns)):

            fold = 0
            for i in range(task_ratio):
                target = rev_targets[col_id]['target']

                # generate a foldID
                fold_id = ' fl' + str(fold) + ' '

                # insert an inactive
                inactive = random.choice(tasks[target]['inactives'])
                if inactive in hashmap:
                    multitask.append(inactive + fold_id + hashmap[inactive])
                else:
                    multitask.append(inactive + fold_id + inactive_cols)
                
                # insert an active
                active = random.choice(tasks[target]['actives'])
                if active in hashmap:
                    multitask.append(inactive + fold_id + hashmap[active])
                else:
                    # this should never happen
                    raise ValueError('active not in hashmap!!!')

                fold += 1
                if(fold >= 5):
                    fold = 0


        # pre-shuffle the data
        random.shuffle(multitask)

        # prevent the files from falling into a strange ordering
        batch_name = str(batch_count).zfill(5)

        # write the batch to disk
        filename = multitask_path + batch_name + '.fl'
        with open(filename, 'w') as file_obj:
            for row in multitask:
                file_obj.write(row + '\n')


        batch_count += 1
        multitask_size -= 10000
    

def main(args):
    """ Evenly draw from all datasets to create minibatches """
    """ Keep files to 30,000 lines or less (to avoid overloading memory) """
    """ The number of columns included can be truncated to generate a smaller """
    """ dataset """

    if(len(args) < 2):
        print 'usage: <tox21, dud_e, muv, or pcba>'
        print 'usage: <tox21, dud_e, muv, or pcba> <number of label columns>'
        print 'e.g. python generate_multitask.py pcba 10'
        print 'This will generate a PCBA multitask dataset that only includes'
        print 'the first 10 PCBA datasets'
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
        # PCBA is massive; this setting allows a smaller multitask dataset
        # to be created
        if(len(args) > 2):
            size = int(args[2])

        gen_multitask('PCBA', size)
    else:
        print 'dataset param not found. options: tox21, dud_e, muv, or pcba'



if __name__ == '__main__':
    start_time = time.clock()

    main(sys.argv)

    end_time = time.clock()
    print 'runtime: %.2f secs.' % (end_time - start_time)

