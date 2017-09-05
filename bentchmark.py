'''
Bentchmark utilities.
'''
import numpy as np
import sys, os, pdb
import matplotlib.pyplot as plt

from problems import load_config, pconfig
from utils import scatter_vec_phase, compare_wf, analyse_exact, check_sample, sign_func_from_vec
from qstate.sampler.mpiutils import RANK

def run_bentchmark(configfile, bentch_id, do_plot_wf = True, compare_to_exact = True, e0=None):
    '''
    Parameters:
        :configfile: str, the location of configuration file.
        :bentch_id: number/str, specify the bentchmark item.
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

        # setup canvas
        if do_plot_wf:
            plt.ion()
            fig=plt.figure(figsize=(10,5))

        # Exact Results
        if (e0 is None and compare_to_exact) or do_plot_wf:
            H, e0, v0, configs = analyse_exact(h, do_printsign=False)

        el=[] # to store energy
        vv_pre = None
        print('\nRunning 0-th Iteration.')

    for info in optimizer:
        # `sampels` and `opq_vals` are cached!
        ei = problem.cache['opq_vals'][0]  

        if RANK==0:  # plots
            if do_plot_wf:
                vv = rbm.tovec(mag=h.mag)
                vv = vv/np.linalg.norm(vv)

                fig.clear()
                plt.subplot(121)
                compare_wf(vv, v0)
                plt.subplot(122)
                scatter_vec_phase(vv, vv_pre)
                D=0.8
                plt.xlim(-D,D)
                plt.ylim(-D,D)
                plt.pause(0.01)
                vv_pre = vv

            if compare_to_exact:
                err=abs(e0-ei)/(abs(e0)+abs(ei))*2
                print('E/site = %s (%s), Error = %.4f%%'%(ei/h.nsite,e0/h.nsite,err*100))
            else:
                print('E/site = %s'%(ei/h.nsite,))
            el.append(ei)

        num_iter = info['n_iter']
        if num_iter>=1000:
            break
        if RANK==0: print('\nRunning %s-th Iteration.'%(num_iter+1))

    if RANK==0:
        np.savetxt('%s/el-%s.dat'%(folder,bentch_id),np.real(el))
        np.savetxt('%s/rbm-%s.dat'%(folder,bentch_id),rbm.get_variables().view('float64'))

if __name__ == '__main__':
    np.random.seed(2)
    configfile, bentch_id, do_plot_wf, compare_to_exact, e0 = sys.argv[1:]
    run_bentchmark(configfile, int(bentch_id), eval(do_plot_wf), eval(compare_to_exact), eval(e0))
