import os,pdb,sys
sys.path.insert(0,'../../')

from models.wanglei4 import WangLei4
from controller import run_benchmark
from problems import load_config, pconfig

def run(''):
    configfile='config-sample.ini'
    folder = os.path.dirname(configfile)
    config = load_config(configfile)

    # definition of a neural network
    rbm = WangLei4(input_shape=tuple(config['hamiltonian']['size']), NF=8, K=4,num_features=[8], version='conv', dtype='complex128')
    optimizer, problem = pconfig(config, rbm)

    run_benchmark(problem, optimizer, do_plot_wf=False, token='')

if __name__=='__main__':
    run()
