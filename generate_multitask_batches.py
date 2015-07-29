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
    """ Evenly draw from all datasets to create minibatches """
    """ Keep files to 30,000 lines or less (to avoid overloading memory) """


    # data type: d, m, p, t
    # target: tells us what file it came from
    
    # format: (of the output)
    # <data type> <target>

    # dud_e()
    # muv()
    # pcba()
    # tox21()


if __name__ == "__main__":
    main(sys.argv)


