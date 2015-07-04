# Traverse the data directories in ./data & generate the fold files in /folds
# 
# Details for generating stratified fold files: 
# 1-oversample the number of positives to be proportionate to the number of 
# negatives
# 2-use disparte folder structures to generate a single unified structure with
# 5x folds
# 3-the 5x folds will not change; this is what we will load for all experiments
# 
# @date 29 June 2015
# @author Jason Feriante
import os, hashlib, sys, random, errno, io

sha_1 = hashlib.sha1()

# datasets = ['DUD-E', 'MUV', 'PCBA', 'Tox21']
# DUD-E
# MUV
# PCBA
# Tox21


data_paths = [
    "./data/DUD-E",
    "./data/MUV",
    "./data/Tox21",
    "./data/PCBA/AIDS_fingerprints", # fingerprints
    "./data/PCBA/AIDs_PCassay_data", # CSVs
    ]


fold_paths = [
    "./folds/DUD-E",
    "./folds/MUV",
    "./folds/Tox21",
    "./folds/PCBA",
    ]



def mkdir_p(path):
    """ mkdir without errors """
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise ValueError('Failed to make directory for some reason.')



def get_target(fname, data_type):

    if(data_type == 'DUD-E'):
        parts = fname.split(r'_')
        target = parts[0]
        return target

    if(data_type == 'MUV'):
        fname = fname.replace('cmp_list_MUV_', '')
        parts = fname.split(r'_')
        target = parts[0]
        return target

    if(data_type == 'Tox21'):
        parts = fname.split(r'_')
        target = parts[0]
        return target



def parse_line(line, activity, data_type):
    if(data_type == 'DUD-E' or data_type == 'MUV'):

        parts = line.rstrip('\n').split(r' ')
        native_id = parts[0]
        bitstring = parts[1]
        # generate sha1 hash identity for bitstring
        sha_1.update(bitstring)
        hash_id = sha_1.hexdigest()

        # activity is based on the fname / argument passed in
        if(activity == '_actives'):
            activity = 1
        else:
            activity = 0

        # row format:
        # hash_id, native_id, activity, bitstring, fold
        row = [hash_id, native_id, activity, bitstring, 0]

        return row

    elif(data_type == 'Tox21'):

        parts = line.rstrip('\n').split(r' ')
        activity = parts[0]
        native_id = parts[1]
        bitstring = parts[2]
        # generate sha1 hash identity for bitstring
        sha_1.update(bitstring)
        hash_id = sha_1.hexdigest()

        # row format:
        # hash_id, native_id, activity, bitstring, fold
        row = [hash_id, native_id, activity, bitstring, 0]

        return row

    elif(data_type == 'PCBA'):
        raise ValueError('PCBA method not written yet')
    else:
        raise ValueError('Unknown data type:' + str(data_type))



def write_folds(filename, all_rows, is_active):
    """ each file only gets 2x folds: actives & inactive """
    num_rows = len(all_rows)
    fold_size = num_rows / 5
    last_fold_size = (num_rows % 5) + fold_size

    # sanity check for the fold math
    assert((last_fold_size + fold_size * 4) == num_rows)

    count = 0
    for fold_id in range(5):
        for row_id in range(fold_size):
            with open(filename, 'w') as file_obj:
                print all_rows[row_id]
                print all_rows.pop()
                exit(0)
                # row = all_rows.pop()
                # row[4] = is_active
                # print row
                # print activity
                # exit(0)
                # row = ' '.join(str(v) for v in all_rows.pop())
                # print 'row'
                # print row
                # exit(0)
                # file_obj.write(row + '\n')

    print 'wtf'
    exit(0)
    # confirm we built all the folds
    # assert(len(all_rows) == count)



def make_folds(filenames, activity, data_type, data_path, fold_path, csv_path = None):
    """ activity (string): actives or inactives """
    """ data_type (string): DUD-E, MUV, Tox21, PCBA """
    # parse the target names from the files
    for fname in filenames:

        target = get_target(fname, data_type)

        # filenames
        row = []
        all_rows = []
        with open(data_path + '/' + fname) as f:
            lines = f.readlines()
            for line in lines:
                row = parse_line(line, activity, data_type)
                all_rows.append(row)

        # generate folder for our folds:
        mkdir_p(fold_path)

        # randomize
        random.shuffle(all_rows)

        # write our files
        if(data_type == 'Tox21'):

            # build our active & inactive files respectively
            actives = []
            inactives = []
            for row in all_rows:
                is_active = row[4]
                if(is_active):
                    actives.append(row)
                else:
                    inactives.append(row)

            # release the rows from memory
            del all_rows

            #write actives
            filename = target + '_actives.fl'
            write_folds(filename, actives, 1)

            #write inactives
            filename = target + '_inactives.fl'
            write_folds(filename, inactives, 0)

        else:

            # It's not Tox21

            is_active = 0
            if(activity == '_actives'):
                is_active = 1

            filename = target + activity + '.fl'
            write_folds(filename, all_rows, is_active)


def dud_e():
    """ Generate folds for DUD-E """
    actives = []
    inactives = []

    data_path = data_paths[0]
    fold_path = fold_paths[0]

    for dir_name, sub, files in os.walk(data_path):
        for fname in files:
            if fname.startswith('.'):
                # ignore system files
                pass
            else:
                if 'inactives' in fname:
                    # print('inactive: %s' % fname)
                    inactives.append(fname)
                else:
                    # print('  active: %s' % fname)
                    actives.append(fname)

    # we should have an equal number of files
    assert len(actives) == len(inactives)
    print "DUD-E: " + str( len(inactives) + len(actives) ) + " files found"
    print "Now generating folds for DUD-E..."

    # pass along the file names
    make_folds(actives, '_actives', 'DUD-E', data_path, fold_path)
    make_folds(inactives, '_inactives', 'DUD-E', data_path, fold_path)



def muv():
    """ Generate folds for MUV """
    actives = []
    inactives = []

    data_path = data_paths[1]
    fold_path = fold_paths[1]

    for dir_name, sub, files in os.walk(data_path):
        for fname in files:
            if fname.startswith('.'):
                # ignore system files
                pass
            else:
                if 'decoys' in fname:
                    # print('inactive: %s' % fname)
                    inactives.append(fname)
                else:
                    # print('  active: %s' % fname)
                    actives.append(fname)

    # we should have an equal number of files
    assert len(actives) == len(inactives)
    print "MUV: " + str( len(inactives) + len(actives) ) + " files found"
    print "Now generating folds for MUV..."

    # pass along the file names
    make_folds(actives, '_actives', 'MUV', data_path, fold_path)
    make_folds(inactives, '_inactives', 'MUV', data_path, fold_path)



# Tox21 has classifications inline (unlike the rest of the system)
def tox21():
    """ Generate folds for Tox21 """
    all_files = []

    data_path = data_paths[2]
    fold_path = fold_paths[2]

    for dir_name, sub, files in os.walk(data_path):
        for fname in files:
            if '.log' in fname or fname.startswith('.'):
                pass
            else:
                all_files.append(fname)

    # we only have 1 group of files!
    print "Tox21: " + str(len(all_files)) + " files found"
    print "Now generating folds for Tox21..."

    make_folds(all_files, '', 'Tox21', data_path, fold_path)



def pcba():
    """ Generate folds for PCBA """
    actives = []
    inactives = []


    data_path = data_paths[3]
    csv_path  = data_paths[4]
    fold_path = fold_paths[3]

    for dir_name, sub, files in os.walk(data_path):
        for fname in files:
            if 'Log' in dir_name or '.list' in fname:
                pass
            else:
                if 'decoys' in fname:
                    # print('inactive: %s' % fname)
                    inactives.append(fname)
                else:
                    # print('  active: %s' % fname)
                    actives.append(fname)

    # we should have an equal number of files
    assert len(actives) == len(inactives)
    print "MUV: " + str( len(inactives) + len(actives) ) + " files found"
    print "Now generating folds for MUV..."

    # pass along the file names
    make_folds(actives, 'actives', 'MUV', data_path, csv_path)
    make_folds(inactives, 'inactives', 'MUV', data_path, csv_path)



def main(args):
    # generate DUDE-E folds
    # dud_e()
    # muv()
    tox21()
    # pcba()



if __name__ == "__main__":
    main(sys.argv)


