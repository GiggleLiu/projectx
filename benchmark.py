'''
Bentchmark utilities.
'''
import numpy as np
import sys, os, pdb,time

from problems import load_config, pconfig
from utils import analyse_exact
from plotlib import scatter_vec_phase, compare_wf
from qstate.sampler.mpiutils import RANK
from profilehooks import profile

@profile
def run_benchmark(config, bench_id, monitors=[], folder='.'):
    '''
    Parameters:
        :configfile: str, the location of configuration file.
        :bench_id: number/str, specify the benchmark item.
        :monitors: func, functions take (problem, optimizer) as parameters.
    '''
    # modification to parameters
    sys.path.insert(0,folder)
    from config import modifyconfig_and_getnn
    rbm = modifyconfig_and_getnn(config, bench_id)

    optimizer, problem = pconfig(config, rbm)
    h, sr, rbm, vmc = problem.hamiltonian, problem.sr, problem.rbm, problem.vmc
    max_iter = config['optimize']['max_iter']

    if RANK==0:
        _save_net_and_show_config(rbm, config, folder, bench_id)
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
        if num_iter>=max_iter:
            break

        if RANK==0: print('\nRunning %s-th Iteration.'%(num_iter+1))

    if RANK==0:
        # save energy and network variables.
        np.savetxt('%s/el-%s.dat'%(folder,bench_id),np.real(el))
        np.savetxt('%s/rbm-%s.dat'%(folder,bench_id),rbm.get_variables().view('float64'))

def _save_net_and_show_config(rbm, config, folder, bench_id):
    # visualize network
    from poornn import viznn
    viznn(rbm, filename=folder+'/%s-%s.png'%(rbm.__class__.__name__,bench_id))

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

def load_rbm(configfile, bench_id, i_iter = None):
    '''
    Load a network, if i_iter specified, load variables at the same time.
    '''
    config = load_config(configfile)
    # folder to store data, containing config.py
    folder = os.path.dirname(configfile)

    # modification to parameters
    sys.path.insert(0,folder)
    from config import modifyconfig_and_getnn
    rbm = modifyconfig_and_getnn(config, bench_id)

    if i_iter is not None:
        rbm.set_variables(i_iter)
    return rbm

def main():
    from monitors import Print_eng_with_exact, print_eng, show_wf, DumpNetwork
    configfile, bench_id = sys.argv[1:]
    np.random.seed(3)

    config = load_config(configfile)
    # folder to store data, containing config.py
    folder = os.path.dirname(configfile)
    e0 = config['hamiltonian']['EG']

    import datetime
    print(datetime.datetime.now())
    run_benchmark(config, int(bench_id),\
            monitors = [
                Print_eng_with_exact(e0),\
                DumpNetwork(folder=os.path.dirname(configfile),token=bench_id,step=1000)\
                ], folder=folder)

if __name__ == '__main__':
    t0=time.time()
    main()
    t1=time.time()
    print('Elapse %s'%(t1-t0))
