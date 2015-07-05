import generate_folds, os, sys



fold_paths = [
    "./folds/DUD-E",
    "./folds/MUV",
    "./folds/Tox21",
    "./folds/PCBA",
    ]



def get_target(fname, data_type):
    return generate_folds.get_target(fname, data_type)



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



def tox21():
    fold_path = fold_paths[2]
    targets = build_targets(fold_path, 'Tox21')

    print "Found " + str(len(targets)) + " targets for Tox21"
    
    data = []
    for target, fnames in targets.iteritems():
        #fnames contains all files for this target
        for fname in fnames:
            
        print key
        print val
        exit(0)
    exit(0)
    exit(0)



def main(args):
    # generate DUDE-E folds
    # dud_e()
    # muv()
    # pcba()
    tox21()


if __name__ == "__main__":
    main(sys.argv)


