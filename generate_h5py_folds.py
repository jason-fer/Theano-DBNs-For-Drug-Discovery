# Traverse the data directories in ./folds & generate the h5py fold files
# 
# @date 21 July 2015
# @author Jason Feriante
import os, hashlib, sys, random, errno, io, h5py
import numpy as np

sha_1 = hashlib.sha1()

data_paths = [
    "./folds/DUD-E",
    "./folds/MUV",
    "./folds/Tox21",
    "./folds/PCBA", 
    ]


fold_paths = [
    "./folds_h5py/DUD-E",
    "./folds_h5py/MUV",
    "./folds_h5py/Tox21",
    "./folds_h5py/PCBA",
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
    """ Allowed types: DUD-E MUV Tox21 PCBA"""
    # the function is the same for all types.
    parts = fname.split(r'_')
    return parts[0]



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
                if 'inactives' in fname:
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
    actives = []
    inactives = []

    data_path = data_paths[2]
    fold_path = fold_paths[2]

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
    print "Tox21: " + str( len(inactives) + len(actives) ) + " files found"
    print "Now generating folds for Tox21..."

    # pass along the file names
    make_folds(actives, '_actives', 'Tox21', data_path, fold_path)
    make_folds(inactives, '_inactives', 'Tox21', data_path, fold_path)


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

        # build the active / inactive sets 
        actives = []
        inactives = []
        # loop through all files in the set (usually not more than 2)
        for fname in targets[target]:
            with open(data_path + '/' + fname) as f:
                lines = f.readlines()
                for line in lines:
                    row = parse_line(line, 0, 'PCBA', g_truth, target)
                    """ row format: [hash_id, is_active, native_id, fold, bitstring] """
                    if(row[1] == 0):
                        inactives.append(row)
                    elif(row[1] == 1):
                        actives.append(row)
                    else:
                        # fingerprints we can't find a truth value for
                        # ... what to do?
                        pass

        # we are done with the hashmap
        del g_truth

        # we should always have SOMETHING active... (right?)
        assert( len(actives) > 0 )
        assert( len(inactives) > 0 )

        # now we can dump our data to the output folds
        mkdir_p(fold_path) # generate folder for our folds
        random.shuffle(actives)
        random.shuffle(inactives)
        base_path = fold_path + '/'

        #write actives
        filename = target + '_actives.fl'
        print filename
        write_folds(base_path + filename, actives, 1)

        #write inactives
        filename = target + '_inactives.fl'
        print filename
        write_folds(base_path + filename, inactives, 0)



def main(args):
    """ Traverse folder structures and convert inconstient raw data into """
    """ 2x files per target; 1 active file and 1 inactive file. """

    ################################################################################
    print "Remove this if you want to generate the folds... it could take a while!"
    exit(0)
    ################################################################################

    dud_e()
    muv()
    pcba()
    tox21()


if __name__ == "__main__":
    main(sys.argv)


