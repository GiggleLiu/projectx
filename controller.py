import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps
from profilehooks import profile
import pdb

from problems import ModelProbDef, load_hamiltonian
from utils import scatter_vec_phase, compare_wf, analyse_exact, check_sample, sign_func_from_vec
from qstate.sampler import get_ground_toynn

def run_rtheta_toy(J2, nsite, version):
    from models.wanglei2 import WangLei2
    from models.toythnn import ToyTHNN
    if version=='2l':
        from models.psnn import PSNN
        thnn = PSNN((nsite,), nf=16, batch_wise=False, period=2, output_mode='theta', use_msr=False)
        pdb.set_trace()
    elif version=='1l':
        from qstate.classifier import PSNN
        thnn = PSNN((nsite,), batch_wise=False, period=2, output_mode='theta', use_msr=False)
    elif version=='toy':
        thnn=ToyTHNN(h)
    # definition of a problem
    h = load_hamiltonian('J1J2', size=(nsite,), J2=J2)
    H = h.get_mat()
    rbm = get_ground_toynn(h, thnn=thnn, train_amp=False, theta_period=nsite)
    problem = ModelProbDef(hamiltonian=h,rbm=rbm,reg_method='sd', optimize_method='gd', step_rate=1e-2)
    sr, rbm, optimizer, vmc = problem.sr, problem.rbm, problem.optimizer, problem.vmc
    sr.rtheta_training_ratio = 1
    optimizer.momentum=0.9

    do_plot_wf = True
    compare_to_exact = True

    # setup canvas
    if do_plot_wf:
        plt.ion()
        fig=plt.figure(figsize=(10,5))

    # Exact Results
    if compare_to_exact or compare_wf:
        H, e0, v0, configs = analyse_exact(h, do_printsign=False)

    el=[] # to store energy
    vv_pre = None
    print '\nRunning 0-th Iteration.'
    for info in optimizer:
        # `sampels` and `opq_vals` are cached!
        ei = problem.cache['opq_vals'][0]  

        if do_plot_wf:
            vv = rbm.tovec(mag=h.mag)
            vv = vv/np.linalg.norm(vv)

            plt.clf()
            plt.subplot(121)
            compare_wf(vv, v0)
            plt.subplot(122)
            scatter_vec_phase(vv, vv_pre)
            plt.pause(0.01)
            vv_pre = vv

        if compare_to_exact:
            err=abs(e0-ei)/(abs(e0)+abs(ei))*2
            print('E/site = %s (%s), Error = %.4f%%'%(ei/h.nsite,e0/h.nsite,err*100))
        else:
            print('E/site = %s'%(ei/h.nsite,))
        el.append(ei)

        if info['n_iter']>=300:
            plt.savefig('data/SIGN-N%s-J2%s-%s.png'%(nsite,J2,version))
            break
        print '\nRunning %s-th Iteration.'%(info['n_iter']+1)

    np.savetxt('data/el-%s%s.dat'%(h.nsite,'p' if h.periodic else 'o'),el)
    pdb.set_trace()

def run_ed_msr(J2, nsite):
    from qstate.classifier.rules import marshall_sign_rule
    h = load_hamiltonian('J1J2', size=(nsite,), J2=J2)
    H = h.get_mat()
    e0, v0 = sps.linalg.eigsh(H, which='SA', k=1)
    v0 = v0.ravel()
    marshall_signs = marshall_sign_rule(h.configs)
    plt.ion()
    scatter_vec_phase(v0[marshall_signs==1], color='r')
    scatter_vec_phase(v0[marshall_signs==-1], color='b')
    plt.legend([r'$+$', r'$-$'])
    pdb.set_trace()

@profile
def scale_ed_msr(size, J2MIN=0, J2MAX=1, NJ2=51, yscale='log'):
    from qstate.classifier.rules import marshall_sign_rule
    J2L = np.linspace(J2MIN, J2MAX, NJ2)
    e0l, el = [], []
    for i,J2 in enumerate(J2L):
        if len(size)==1:
            h = load_hamiltonian('J1J2', size=size, J2=J2)
        else:
            h = load_hamiltonian('J1J22D', size=size, J2=J2)
        H = h.get_mat()
        e0, v0 = sps.linalg.eigsh(H, which='SA', k=1)
        v0 = v0.ravel()
        marshall_signs = marshall_sign_rule(h.configs, size=size)
        v = abs(v0)*marshall_signs
        el.append(v.T.conj().dot(H.dot(v)))
        e0l.append(e0.item())
        print('%s'%i)
    plt.ion()
    plt.plot(J2L, np.array(el)-e0l)
    plt.xlabel(r'$J_2$')
    plt.ylabel(r'$E-E_0$')
    plt.yscale(yscale)
    pdb.set_trace()


def run_rtheta(J2, nsite):
    from models.wanglei2 import WangLei2
    # definition of a problem
    h = load_hamiltonian('J1J2', size=(nsite,), J2=J2)
    rbm = WangLei2(input_shape=(h.nsite,),num_feature_hidden=4, use_msr=False, theta_period=2, with_linear=False)
    problem = ModelProbDef(hamiltonian=h,rbm=rbm,reg_method='sd', optimize_method='gd', step_rate=3e-3)
    sr, rbm, optimizer, vmc = problem.sr, problem.rbm, problem.optimizer, problem.vmc
    sr.rtheta_training_ratio = 30

    do_plot_wf = False
    compare_to_exact = True
    do_check_sample =True

    # setup canvas
    if do_plot_wf or do_check_sample:
        plt.ion()
        fig=plt.figure(figsize=(10,5))

    # Exact Results
    if compare_to_exact or compare_wf:
        H, e0, v0, configs = analyse_exact(h, do_printsign=False)

    el=[] # to store energy
    vv_pre = None
    print '\nRunning 0-th Iteration.'
    for info in optimizer:
        # `sampels` and `opq_vals` are cached!
        ei = problem.cache['opq_vals'][0]  

        if do_plot_wf or do_check_sample:
            amp = []
            signs = []
            for config in h.configs:
                amp.append(rbm.forward(config)[-1])
                signs.append(rbm.get_sign(config))
            amp = np.asarray(amp)
            amp = amp/np.linalg.norm(amp)
            vv = amp*signs
            #vv = rbm.tovec(mag=h.mag)
            vv = vv/np.linalg.norm(vv)

        if do_plot_wf:
            fig.clear()
            plt.subplot(121)
            compare_wf(amp, v0)
            plt.subplot(122)
            scatter_vec_phase(vv, vv_pre)
            plt.pause(0.01)
            vv_pre = vv

        if do_check_sample:
            fig.clear()
            check_sample(rbm, h, problem.cache['samples'])
            plt.pause(0.01)

        if compare_to_exact:
            err=abs(e0-ei)/(abs(e0)+abs(ei))*2
            print('E/site = %s (%s), Error = %.4f%%'%(ei/h.nsite,e0/h.nsite,err*100))
        else:
            print('E/site = %s'%(ei/h.nsite,))
        el.append(ei)

        if info['n_iter']>=800:
            break
        print '\nRunning %s-th Iteration.'%(info['n_iter']+1)

    np.savetxt('data/el-%s%s.dat'%(h.nsite,'p' if h.periodic else 'o'),el)
    pdb.set_trace()

def run_rtheta_mlp(J2, nsite, mlp_shape):
    from models.rtheta_mlp import RTheta_MLP
    # definition of a problem
    h = load_hamiltonian('J1J2', size=(nsite, ), J2=J2)
    rbm = RTheta_MLP(input_shape=(h.nsite,),num_feature_hidden=4, mlp_shape=mlp_shape, use_msr=False, theta_period=2)
    #rbm = get_ground_toynn(h, mode='r-theta', train_amp=False, theta_period=nsite)
    #pdb.set_trace()
    problem = ModelProbDef(hamiltonian=h,rbm=rbm,reg_method='sd', optimize_method='gd', step_rate=3e-3)
    sr, rbm, optimizer, vmc = problem.sr, problem.rbm, problem.optimizer, problem.vmc
    sr.rtheta_training_ratio = 30

    # do_plot_wf = True
    compare_to_exact = True

    # setup canvas
    # if do_plot_wf:
    #     plt.ion()
    #     fig=plt.figure(figsize=(10,5))

    # Exact Results
    if compare_to_exact or compare_wf:
        H, e0, v0, configs = analyse_exact(h, do_printsign=False)

    el=[] # to store energy
    vv_pre = None
    print '\nRunning 0-th Iteration.'
    for info in optimizer:
        # `sampels` and `opq_vals` are cached!
        ei = problem.cache['opq_vals'][0]  

        # if do_plot_wf:
        #     # vv = rbm.tovec(mag=h.mag)
        #     amp = []
        #     signs = []
        #     for config in h.configs:
        #         amp.append(rbm.forward(config)[-1])
        #         signs.append(rbm.get_sign(config))
        #     amp = np.asarray(amp)
        #     amp = amp/np.linalg.norm(amp)
        #     vv = amp*signs

        #     plt.clf()
        #     plt.subplot(121)
        #     compare_wf(amp, v0)
        #     plt.subplot(122)
        #     scatter_vec_phase(vv, vv_pre)
        #     plt.pause(0.1)
        #     vv_pre = vv

        if compare_to_exact:
            err=abs(e0-ei)/(abs(e0)+abs(ei))*2
            print('E/site = %s (%s), Error = %.4f%%'%(ei/h.nsite,e0/h.nsite,err*100))
        else:
            print('E/site = %s'%(ei/h.nsite,))
        el.append(ei)

        if info['n_iter']>=1000:
            break
        print '\nRunning %s-th Iteration.'%(info['n_iter']+1)

    number = ''
    for i in mlp_shape:
        number += str(i)
        number += '-'
    np.savetxt('data/rtheta-%sel-%s%s.dat'%(number, h.nsite,'p' if h.periodic else 'o'),el,fmt='%.10f%+.10fj')
    #pdb.set_trace()

def run_wanglei(J2, nsite):
    from models.wanglei import WangLei
    # definition of a problem
    h = load_hamiltonian('J1J2', size=(nsite,), J2=J2)
    rbm = WangLei(input_shape=(h.nsite,),num_features=[16, 8], version='linear', use_conv=True, dtype='complex128')
    problem = ModelProbDef(hamiltonian=h,rbm=rbm,reg_method='delta', optimize_method='gd', step_rate=1e-1)
    sr, rbm, optimizer, vmc = problem.sr, problem.rbm, problem.optimizer, problem.vmc
    vmc.inverse_rate = 0.05

    do_plot_wf = True
    compare_to_exact = True

    # setup canvas
    if do_plot_wf:
        plt.ion()
        fig=plt.figure(figsize=(10,5))

    # Exact Results
    if compare_to_exact or compare_wf:
        H, e0, v0, configs = analyse_exact(h, do_printsign=False)

    el=[] # to store energy
    vv_pre = None
    print '\nRunning 0-th Iteration.'
    for info in optimizer:
        # `sampels` and `opq_vals` are cached!
        ei = problem.cache['opq_vals'][0]  

        if do_plot_wf:
            vv = rbm.tovec(mag=h.mag)
            vv = vv/np.linalg.norm(vv)

            fig.clear()
            plt.subplot(121)
            compare_wf(vv, v0)
            plt.subplot(122)
            scatter_vec_phase(vv, vv_pre)
            plt.xlim(-0.3,0.3)
            plt.ylim(-0.3,0.3)
            plt.pause(0.01)
            vv_pre = vv

        if compare_to_exact:
            err=abs(e0-ei)/(abs(e0)+abs(ei))*2
            print('E/site = %s (%s), Error = %.4f%%'%(ei/h.nsite,e0/h.nsite,err*100))
        else:
            print('E/site = %s'%(ei/h.nsite,))
        el.append(ei)

        num_iter = info['n_iter']
        #optimizer.step_rate *= 0.995
        if num_iter>=2000:
            break
        print '\nRunning %s-th Iteration.'%(num_iter+1)

    np.savetxt('data/el-%s%s.dat'%(h.nsite,'p' if h.periodic else 'o'),el)
    pdb.set_trace()

def rbm_given_sign(J2, nsite):
    from models.poorrbm import RBM
    do_plot_wf = True
    compare_to_exact = True

    # definition of a problem
    h = load_hamiltonian('J1J2', size=(nsite,), J2=J2)
    # Exact Results
    if compare_to_exact or compare_wf:
        H, e0, v0, configs = analyse_exact(h, do_printsign=False)

    rbm = RBM(input_shape=(h.nsite,),num_feature_hidden=4, dtype='float64', sign_func = sign_func_from_vec(h.configs, v0))
    problem = ModelProbDef(hamiltonian=h,rbm=rbm,reg_method='delta', optimize_method='gd', step_rate=1e-2)
    sr, rbm, optimizer, vmc = problem.sr, problem.rbm, problem.optimizer, problem.vmc
    vmc.inverse_rate = 0.05

    # setup canvas
    if do_plot_wf:
        plt.ion()
        fig=plt.figure(figsize=(10,5))


    el=[] # to store energy
    vv_pre = None
    print '\nRunning 0-th Iteration.'
    for info in optimizer:
        # `sampels` and `opq_vals` are cached!
        ei = problem.cache['opq_vals'][0]  

        if do_plot_wf:
            vv = rbm.tovec(mag=h.mag)
            vv = vv/np.linalg.norm(vv)

            fig.clear()
            plt.subplot(121)
            compare_wf(vv, v0)
            plt.subplot(122)
            scatter_vec_phase(vv, vv_pre)
            plt.xlim(-0.3,0.3)
            plt.ylim(-0.3,0.3)
            plt.pause(0.01)
            vv_pre = vv

        if compare_to_exact:
            err=abs(e0-ei)/(abs(e0)+abs(ei))*2
            print('E/site = %s (%s), Error = %.4f%%'%(ei/h.nsite,e0/h.nsite,err*100))
        else:
            print('E/site = %s'%(ei/h.nsite,))
        el.append(ei)

        num_iter = info['n_iter']
        #optimizer.step_rate *= 0.995
        if num_iter>=2000:
            break
        print '\nRunning %s-th Iteration.'%(num_iter+1)

    np.savetxt('data/el-%s%s.dat'%(h.nsite,'p' if h.periodic else 'o'),el)
    pdb.set_trace()
