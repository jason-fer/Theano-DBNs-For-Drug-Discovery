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



def make_folds(filenames, activity, data_type):
    """ activity (string): actives or inactives """
    """ data_type (string): DUD-E, MUV, Tox21, PCBA """
    # parse the target names from the files
    for fname in filenames:
        target = get_target(fname, data_type)

        # filenames
        row = []
        all_inactives = []
        with open(data_path + '/' + fname) as f:
            lines = f.readlines()
            for line in lines:
                parts = line.rstrip('\n').split(r' ')
                native_id = parts[0]
                bitstring = parts[1]
                # generate sha1 hash identity for bitstring
                sha_1.update(bitstring)
                hash_id = sha_1.hexdigest()

                # row format:
                # hash_id, native_id, bitstring
                row = [hash_id, native_id, 0, bitstring]
                all_inactives.append(row)

        # generate folder for our folds:
        mkdir_p(fold_path)

        # randomize
        random.shuffle(all_inactives)
        num_rows = len(all_inactives)
        fold_size = num_rows / 5
        last_fold_size = (num_rows % 5) + fold_size

        # sanity check for the fold math
        assert((last_fold_size + fold_size * 4) == num_rows)

        # generate folds 1-4 for this dataset
        for i in range(4):
            # base the file name on the current fold
            filename = fold_path + '/' + target + '_' + activity +'_fold' + str(i) + '.txt'
            with open(filename, 'w') as file_obj:
                for j in range(fold_size):
                        row = ' '.join(str(v) for v in all_inactives.pop())
                        file_obj.write(row + '\n')

        # generate the 5th fold (which might be an odd number)
        filename = fold_path + '/' + target + '_' + activity +'_fold4.txt'
        with open(filename, 'w') as file_obj:
            for j in range(last_fold_size):
                row = ' '.join(str(v) for v in all_inactives.pop())
                file_obj.write(row + '\n')

        assert ( len(all_inactives) == 0)



def dud_e():
    """ Generate folds for DUD-E """
    actives = []
    inactives = []

    data_path = "./data/DUD-E/fingerprints"
    fold_path = "./folds/DUD-E"

    for dir_name, sub, files in os.walk(data_path):
        for fname in files:
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

    make_folds(actives, 'actives', 'DUD-E')
    make_folds(inactives, 'inactives', 'DUD-E')



def muv():
    """ Generate folds for MUV """
    actives = []
    inactives = []

    data_path = "./data/MUV/MUV-fingerprints"
    fold_path = "./folds/MUV"

    for dir_name, sub, files in os.walk(data_path):
        for fname in files:
            if 'Log' in dir_name:
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

    make_folds(actives, 'actives', 'MUV')
    make_folds(inactives, 'inactives', 'MUV')



def pcba():
    """ Generate folds for PCBA """
    actives = []
    inactives = []

    data_path = "./data/PCBA/AIDS_fingerprints"
    fold_path = "./folds/PCBA"

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

    make_folds(actives, 'actives', 'MUV')
    make_folds(inactives, 'inactives', 'MUV')



# Tox21 has classifications inline (unlike the rest of the system)
def tox21():
    """ Generate folds for Tox21 """
    actives = []
    inactives = []

    data_path = "./data/Tox21/fingerprints"
    fold_path = "./folds/Tox21"

    for dir_name, sub, files in os.walk(data_path):
        for fname in files:
            if '.log' in fname:
                pass
            else:
                inactives.append(fname)

    # we only have 1 group of files!
    print "Tox21: " + str(len(inactives)) + " files found"
    print "Now generating folds for Tox21..."

    # XXXXXXXXXXXXXXXXXXXXXXXXXXX This needs to be modified to work correctly XXXXXXXXXXXXXXXXXXXXXXX
    make_folds(inactives, 'inactives', 'Tox21')


def main(args):
    # generate DUDE-E folds
    # dud_e()
    muv()



if __name__ == "__main__":
    main(sys.argv)



file_obj.write("Look, ma, I'm writing to a new file!")