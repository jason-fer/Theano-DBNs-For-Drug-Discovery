# Generate minibatch & truth sets from the folds
# 
# @date 26 July 2015
# @author Jason Feriante
import os, hashlib, sys, random, errno, io
from lib import helpers


get_fold_path = helpers.get_fold_path
get_target = helpers.get_target
parse_line = helpers.parse_line
build_targets = helpers.build_targets
oversample = helpers.oversample
get_folds = helpers.get_folds



def main(args):
    """ Traverse folder structures and convert inconstient raw data into """
    """ 2x files per target; 1 active file and 1 inactive file. """


    # dud_e()
    # muv()
    # pcba()
    # tox21()


if __name__ == "__main__":
    main(sys.argv)


