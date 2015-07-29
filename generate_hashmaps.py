"""
**************************************************************************
Generate Truth Sets
**************************************************************************

Use our active files to build sparse truth sets (items not in the set are 
implicitly negative across the board)

row format:
<hash_id> <binary_strings>



@author: Jason Feriante <feriante@cs.wisc.edu>
@date: 10 July 2015
"""

import generate_folds, os, sys, random, time, re
from lib.theano import helpers



def load_hashmap(data_type):
    """Load a hashmap into memory"""
    hashmap_path = 'hashmaps/' + data_type + '.hm'

    hashmap = {}
    with open(hashmap_path) as f:
        lines = f.readlines()
        for line in lines:
            # put each row in it's respective fold
            parts = line.rstrip('\n').split(r' ')

            bitstring = parts[0]
            row = parts[1:]

            hashmap[bitstring] = row

    return hashmap



def gen_hashmap(data_type):

    # place to store the completed hashmap
    hashmap_path = 'hashmaps/' + data_type + '.hm'
    fold_path = helpers.get_fold_path(data_type)

    # rev_targets lets us get information by col_id instead of target

    # each file is a column, build the dataset in the same order
    
    # we only include actives
    hashmap = {}

    rev_targets, target_columns = helpers.get_rev_targets(data_type)
    num_cols = len(target_columns)

    base_row = [0] * num_cols
    overlap = 0
    count = 0

    for col_id in range(num_cols):

        target = rev_targets[col_id]['target']
        fname = rev_targets[col_id]['fname']
        # print 'now on ' + str(col_id)
        # add only active file entries to the hashmap
        with open(fold_path + '/' + fname) as f:
            lines = f.readlines()
            for line in lines:
                # this is the structure 
                # [hash_id, is_active, native_id, fold, bitstring]
                parts = line.rstrip('\n').split(r' ')
                hash_id = parts[4]
                is_active = int(parts[1])
                assert(is_active == 1)

                if(hash_id in hashmap):
                    # there's overlap of bitstrings, even in the same dataset
                    # print hash_id 
                    # print target
                    # print fname
                    # exit(0)
                    # the row already exists
                    overlap += 1
                    row = hashmap[hash_id]
                    row[col_id] = 1
                    # add our new column
                else:
                    row = base_row[:]
                    row[col_id] = 1
                    hashmap[hash_id] = row

                count += 1
                # update the active in this row    

    # print 'dumped hashmap for dataset: ' + data_type
    # print str(count) + ' items found'
    # print str(overlap) + ' total overlap'

    new_count = 0
    for row in hashmap.iteritems():
        new_count += sum(row[1])
    print data_type + ': ' + str(new_count) + ' items in final hashmap. ' + \
        str(count - new_count) + ' bitstrings merged'


    confirmed_actives = 0
    print 'Dumping the hashmap to disk'
    with open(hashmap_path, 'w') as file_obj:
        for line in hashmap.iteritems():
            
            # build a row... 
            bitstring = line[0]
            is_actives = ' '.join(map(str, line[1]))  
            row = bitstring + ' ' + is_actives

            """row format: <1024 bitstring> <is_active, 1 per target>"""
            file_obj.write(row + '\n')

            confirmed_actives += len(re.findall('1', is_actives))

    print str(confirmed_actives) + ' confirmed_actives'



def main(args):
    if(len(args) < 2):
        print 'usage: <tox21, dud_e, muv, or pcba>'
        return

    dataset = args[1]

    # in case of typos
    if(dataset == 'dude'):
        dataset = 'dud_e'
        
    print "Generating hashmap for " \
        + dataset + "........."


    if(dataset == 'tox21'):
        gen_hashmap('Tox21')

    elif(dataset == 'dud_e'):
        gen_hashmap('DUD-E')

    elif(dataset == 'muv'):
        gen_hashmap('MUV')

    elif(dataset == 'pcba'):
        gen_hashmap('PCBA')
    else:
        print 'dataset param not found. options: tox21, dud_e, muv, or pcba'



if __name__ == '__main__':
    start_time = time.clock()

    main(sys.argv)

    end_time = time.clock()
    print 'runtime: %.2f secs.' % (end_time - start_time)

