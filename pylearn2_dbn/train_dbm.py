"""
1-get all singletask items done
2-get all multitask items done

q: how to make an efficient truth set?
1-it's sparse. 
2-filling a file with zeros is crazy
3-simple!

hashmap: by compound
1-python dict attached to each hash
2-only active hashes included
3-a dict object contains a list of keys where this item is active
4-if you can't find the key its not active

**one hashmap per compound
(to prevent a memory blowout)

#1 - Single Task Neural Network (STNN)
#2 - Pyramidal STNN 2 layers: (2000 neurons, 100 neurons) .25 dropout
#3 - 1-Hidden Layer (1200 neurons) Multitask Neural Net (MTNN)
#4 - Pyramidal (2000, 100) Multitask Neural Net (PMTNN)

"""
import os
from pylearn2.testing import skip
from pylearn2.testing import no_debug_mode
from pylearn2.config import yaml_parse

@no_debug_mode
def train_yaml(yaml_file):

    train = yaml_parse.load(yaml_file)
    train.main_loop()


def train(yaml_file_path, save_path):

    yaml = open("{0}/rbm.yaml".format(yaml_file_path), 'r').read()
    hyper_params = {'detector_layer_dim': 500,
                    'monitoring_batches': 10,
                    'train_stop': 50000,
                    'max_epochs': 100,
                    'save_path': save_path}

    yaml = yaml % (hyper_params)
    train_yaml(yaml)


def train_dbm():

    skip.skip_if_no_data()

    yaml_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ''))

    save_path = os.path.dirname(os.path.realpath(__file__)) + '/results'

    train(yaml_file_path, save_path)

if __name__ == '__main__':
    train_dbm()