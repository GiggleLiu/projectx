'''
Bentchmark utilities.
'''
import numpy as np
import sys, os, pdb
import matplotlib.pyplot as plt

from problems import load_config, pconfig
from utils import analyse_exact
from plotlib import scatter_vec_phase, compare_wf
from qstate.sampler.mpiutils import RANK

def run_bentchmark(configfile, bentch_id, monitors=[]):
    '''
    Parameters:
        :configfile: str, the location of configuration file.
        :bentch_id: number/str, specify the bentchmark item.
        :monitors: func, functions take (problem, optimizer) as parameters.
    '''
    config = load_config(configfile)
    # folder to store data, containing config.py
    folder = os.path.dirname(configfile)

    # modification to parameters
    sys.path.insert(0,folder)
    from config import modifyconfig_and_getnn
    rbm = modifyconfig_and_getnn(config, bentch_id)

    optimizer, problem = pconfig(config, rbm)
    h, sr, rbm, vmc = problem.hamiltonian, problem.sr, problem.rbm, problem.vmc

    if RANK==0:
        _save_net_and_show_config(rbm, config, folder, bentch_id)
        el=[] # to store energy
        print('\nRunning 0-th Iteration.')

    for info in optimizer:
        if RANK==0:  # plots
            for monitor in monitors:
                monitor(problem, optimizer)

            # append ei to el
            ei = problem.cache['opq_vals'][0]  
            el.append(ei)

        num_iter = info['n_iter']
        if num_iter>=1000:
            break

        if RANK==0: print('\nRunning %s-th Iteration.'%(num_iter+1))

    if RANK==0:
        # save energy and network variables.
        np.savetxt('%s/el-%s.dat'%(folder,bentch_id),np.real(el))
        np.savetxt('%s/rbm-%s.dat'%(folder,bentch_id),rbm.get_variables().view('float64'))


def _save_net_and_show_config(rbm, config, folder, bentch_id):
    # visualize network
    from poornn import viznn
    viznn(rbm, filename=folder+'/%s-%s.png'%(rbm.__class__.__name__,bentch_id))

    # now flush configuration to stdout
    print('#'*20+' Configuration '+'#'*20)
    for key, val in config.items():
        print('%s:'%(key,))
        for k, v in val.items():
            print('    %s = %s'%(k,v))
    print('-'*55+'\n')
    print('#'*22+' Network '+'#'*22)
    print(rbm)
    print('-'*55+'\n')

if __name__ == '__main__':
    np.random.seed(2)
    from monitors import Print_eng_with_exact, print_eng, show_wf
    configfile, bentch_id, e0 = sys.argv[1:]
    run_bentchmark(configfile, int(bentch_id), monitors = [Print_eng_with_exact(eval(e0))])
