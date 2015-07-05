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
    "./data/PCBA/AIDs_fingerprints", # fingerprints
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
    """ Extract the target from the filename """
    if(data_type == 'DUD-E'):
        parts = fname.split(r'_')
        return parts[0]

    if(data_type == 'MUV'):
        fname = fname.replace('cmp_list_MUV_', '')
        parts = fname.split(r'_')
        return parts[0]

    if(data_type == 'Tox21'):
        parts = fname.split(r'_')
        return parts[0]

    if(data_type == 'PCBA'):
        fname = fname.replace('pcba_', '')
        parts = fname.split(r'_')
        return parts[0]



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
            is_active = 1
        else:
            is_active = 0

        fold = 0
        # row format:
        row = [hash_id, is_active, native_id, fold, bitstring]

        return row

    elif(data_type == 'Tox21'):

        parts = line.rstrip('\n').split(r' ')
        is_active = parts[0]
        native_id = parts[1]
        bitstring = parts[2]

        # generate sha1 hash identity for bitstring
        sha_1.update(bitstring)
        hash_id = sha_1.hexdigest()
        fold = 0

        # row format:
        row = [hash_id, is_active, native_id, fold, bitstring]

        return row

    elif(data_type == 'PCBA'):
        raise ValueError('PCBA method not written yet')
    else:
        raise ValueError('Unknown data type:' + str(data_type))



def write_folds(filename, all_rows, is_active):
    """ each file only gets 2x folds: actives & inactive """

    # there should be something to write.
    assert(len(all_rows) > 0)

    num_rows = len(all_rows)
    fold_size = num_rows / 5

    with open(filename, 'w') as file_obj:
        for fold_id in range(4):
            for row_id in range(fold_size):
                row = all_rows.pop()
                row[1] = is_active
                row[3] = fold_id
                row = ' '.join(str(v) for v in row)
                file_obj.write(row + '\n')

        # write leftovers to the last fold
        fold_id = 4
        while(all_rows):
            row = all_rows.pop()
            row[1] = is_active
            row[3] = fold_id
            row = ' '.join(str(v) for v in row)
            file_obj.write(row + '\n')

    # confirm we built all the folds
    assert(len(all_rows) == 0)



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

        base_path = fold_path + '/'

        # write our files
        if(data_type == 'Tox21'):

            # build our active & inactive files respectively
            actives = []
            inactives = []
            for row in all_rows:
                is_active = int(row[1])
                if(is_active == 1):
                    actives.append(row)
                else:
                    inactives.append(row)

            # release the rows from memory
            del all_rows

            #write actives
            filename = target + '_actives.fl'
            print filename
            write_folds(base_path + filename, actives, 1)

            #write inactives
            filename = target + '_inactives.fl'
            print filename
            write_folds(base_path + filename, inactives, 0)

        else:

            # It's not Tox21
            is_active = 0
            if(activity == '_actives'):
                is_active = 1

            filename = target + activity + '.fl'
            print filename
            write_folds(base_path + filename, all_rows, is_active)


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


def get_csv_index(first_line, csv, name):
    # make sure we have the right CID for this CSV file
    for i in range(len(first_line)):
        if(first_line[i] == name):
            return i

    # this should never happen
    raise ValueError('Failed to find ' + name + ' in ' + csv)



def pcba():
    """ Generate folds for PCBA """
    pcba_fps = []
    pcba_csv = []
    targets = {}

    data_path = data_paths[3]
    csv_path  = data_paths[4]
    fold_path = fold_paths[3]

    # init targets
    for dir_name, sub, files in os.walk(data_path):
        for fname in files:
            if 'Log' in dir_name or '.list' in fname or fname.startswith('.'):
                pass
            else:
                target = get_target(fname, 'PCBA')
                targets[target] = []

    # now, walk the data path again, appending each file name to each target list
    for dir_name, sub, files in os.walk(data_path):
        for fname in files:
            if 'Log' in dir_name or '.list' in fname or fname.startswith('.'):
                pass
            else:
                # pcba_fps is just for book-keeping.
                pcba_fps.append(fname)
                target = get_target(fname, 'PCBA')
                targets[target].append(fname)


    for dir_name, sub, files in os.walk(csv_path):
        for fname in files:
            if 'Log' in dir_name or '.list' in fname or fname.startswith('.'):
                pass
            else:
                pcba_csv.append(fname)

    """ this dumps all targets & associated files w/ each """
    # for k, vals in targets.iteritems():
    #     print k
    #     for v in vals:
    #         print '\t' + v
    # exit(0)

    # we should have an equal number of files
    num_csvs = len(pcba_csv)
    assert len(targets) == num_csvs
    print "PCBA: " + str(len(pcba_fps)) + " fps and " + str(num_csvs) + \
        " csv files found"
    print "Now generating folds for PCBA..."

    # now, for each target, generate a hashmap based on the CSV
    # the hashmap will tell us active / vs inactive based on CIDs
    for csv in pcba_csv:
        
        # extract target from CSV file name
        parts = csv.replace('.', '-').split(r'-')
        target = parts[1]

        # ground truth CID to {0,1} hashmap (based on CSV data)
        g_truth = {} # CID = compound ID.
        # now we can determine actives / inactives from each target file

        # now write actives & inactives to their respective folds for 1 file
        # rinse, repeat...
        all_rows = []
        with open(csv_path + '/' + csv) as f:
            lines = f.readlines()
            first_line = lines.pop(0).split(r',')

            CID_index = get_csv_index(first_line, csv, 'PUBCHEM_CID')
            outcome_index = get_csv_index(first_line, csv, 'PUBCHEM_ACTIVITY_OUTCOME')
            score_index = get_csv_index(first_line, csv, 'PUBCHEM_ACTIVITY_SCORE')
            pheno_index = get_csv_index(first_line, csv, '1^Phenotype^STRING^^^^')

            # now build our results based on the indexes
            for line in lines:
                line = line.split(r',')
              

                # skip the many annoying header rows that would corrupt our data
                """ Allows me to manually find errors (e.g. xml gateway """
                """ errors) and clear them out out by hand."""
                try:
                    if(line[CID_index] == 'PUBCHEM_CID'):
                        continue;
                except:
                    print 'line'
                    print line
                    print 'CID_index'
                    print CID_index
                    print csv
                    exit(0)

                # build the data
                row = [line[CID_index], line[outcome_index], line[score_index], line[pheno_index]]

                # map the PUBCHEM_CID to the PUBCHEM_ACTIVITY_OUTCOME
                is_active= 0
                if(line[outcome_index] == 'Active'):
                    is_active = 1

                g_truth[ line[CID_index] ] = is_active
                # for building debugging output
                # test_output[row[1]] = ''

                # Now that we have our  hashmap, we can build our respective sets
                # of actives & inactives
                

                # based on the target, we can retrieve the files; use the targets
                # hashmap we built earlier
                fps_files = targets[target]
                print 'fps_files'
                print fps_files
                print 'target'
                print target
                exit(0)
                # with open(data_path + '/' + csv) as f:

        print g_truth
        exit(0)
        # print all_rows
    print test_output
    exit(0)  



def gen_folds():
    dud_e()
    muv()
    tox21()
    

def main(args):
    # generate DUDE-E folds
    # gen_folds()

    pcba()


if __name__ == "__main__":
    main(sys.argv)


