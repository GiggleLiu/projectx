import numpy as np
import matplotlib.pyplot as plt
from profilehooks import profile
import scipy.sparse as sps
import pdb, os

from problems import ModelProbDef, load_hamiltonian, get_optimizer, load_config, pconfig
from utils import scatter_vec_phase, compare_wf, analyse_exact, check_sample, sign_func_from_vec
from qstate.sampler import get_ground_toynn
from qstate.sampler.mpiutils import RANK

def run_rtheta_toy(J2, nsite, version, rtheta_training_ratio, momentum=0.):
    from models.wanglei2 import WangLei2
    from models.toythnn import ToyTHNN
    h = load_hamiltonian('J1J2', size=(nsite,), J2=J2)
    if version=='2l':
        from models.psnn_leo import PSNN
        thnn = PSNN((nsite,), nf=16, batch_wise=False, period=2, output_mode='theta')
    elif version=='1l':
        from qstate.classifier import PSNN
        thnn = PSNN((nsite,), batch_wise=False, period=2, output_mode='theta', use_msr=False)
    elif version=='toy':
        thnn=ToyTHNN(h)
    # definition of a problem
    H = h.get_mat()
    rbm = get_ground_toynn(h, thnn=thnn, train_amp=True, theta_period=2)
    problem = ModelProbDef(hamiltonian=h,rbm=rbm,reg_method='sd', sr_layerwise=False if version=='toy' else True)
    sr, rbm, vmc = problem.sr, problem.rbm, problem.vmc
    sr.rtheta_training_ratio = rtheta_training_ratio

    optimizer = get_optimizer(wrt=rbm.get_variables(), fprime=problem.compute_gradient,
            optimize_method='gd', step_rate=3e-3, momentum=momentum)

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
    print('\nRunning 0-th Iteration.')
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
        print('\nRunning %s-th Iteration.'%(info['n_iter']+1))

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


def run_rtheta(J2, nsite, rtheta_training_ratio, momentum=0.):
    from models.wanglei2 import WangLei2
    # definition of a problem
    h = load_hamiltonian('J1J2', size=(nsite,), J2=J2)
    rbm = WangLei2(input_shape=(h.nsite,),num_feature_hidden=4, use_msr=False, theta_period=2, with_linear=False, dtype='float64')
    problem = ModelProbDef(hamiltonian=h,rbm=rbm,reg_method='sd')
    sr, rbm, vmc = problem.sr, problem.rbm, problem.vmc
    sr.rtheta_training_ratio = rtheta_training_ratio

    optimizer = get_optimizer(wrt=rbm.get_variables(), fprime=problem.compute_gradient,
            optimize_method='gd', step_rate=1e-2, momentum=momentum)

    do_plot_wf = True
    compare_to_exact = True
    do_check_sample = False

    # setup canvas
    if do_plot_wf or do_check_sample:
        plt.ion()
        fig=plt.figure(figsize=(10,5))

    # Exact Results
    if compare_to_exact or compare_wf:
        H, e0, v0, configs = analyse_exact(h, do_printsign=False, num_eng=10)

    el=[] # to store energy
    vv_pre = None
    print('\nRunning 0-th Iteration.')
    for info in optimizer:
        # `sampels` and `opq_vals` are cached!
        ei = problem.cache['opq_vals'][0]  

        if do_plot_wf or do_check_sample:
            amps = []
            thetas = []
            signs = []
            for config in h.configs:
                amps.append(rbm.forward(config)[-1])
                thetas.append(rbm.thnn.forward(config)[-1])
                signs.append(np.exp(1j*thetas[-1]))
            amps = np.asarray(amps)
            amps = amps/np.linalg.norm(amps)
            vv = amps*signs
            #vv = rbm.tovec(mag=h.mag)

        if do_plot_wf:
            fig.clear()
            plt.subplot(121)
            #compare_wf(amps, v0)
            compare_wf(vv, v0)
            plt.subplot(122)
            scatter_vec_phase(vv, vv_pre, winding=np.int32(np.floor(np.array(thetas)/2/np.pi)))
            plt.xlim(-0.8,0.8)
            plt.ylim(-0.8,0.8)
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
        print('\nRunning %s-th Iteration.'%(info['n_iter']+1))

    np.savetxt('data/el-%s%s.dat'%(h.nsite,'p' if h.periodic else 'o'),el)
    pdb.set_trace()

def run_rtheta_mlp(J2, nsite, mlp_shape):
    from models.rtheta_mlp import RTheta_MLP
    # definition of a problem
    h = load_hamiltonian('J1J2', size=(nsite, ), J2=J2)
    rbm = RTheta_MLP(input_shape=(h.nsite,),num_feature_hidden=4, mlp_shape=mlp_shape, use_msr=False, theta_period=2)
    #rbm = get_ground_toynn(h, mode='r-theta', train_amp=False, theta_period=nsite)
    #pdb.set_trace()
    problem = ModelProbDef(hamiltonian=h,rbm=rbm,reg_method='sd')
    sr, rbm, vmc = problem.sr, problem.rbm, problem.vmc
    sr.rtheta_training_ratio = [1.,30.]

    optimizer = get_optimizer(wrt=rbm.get_variables(), fprime=problem.compute_gradient,
            optimize_method='adam', step_rate=3e-3, momentum=momentum)

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
    print('\nRunning 0-th Iteration.')
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

        if info['n_iter']>=1:
            break
        print('\nRunning %s-th Iteration.'%(info['n_iter']+1))

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
    rbm = WangLei(input_shape=(h.nsite,),num_features=[16, 64, 16], version='linear', use_conv=True, dtype='complex128')
    problem = ModelProbDef(hamiltonian=h,rbm=rbm,reg_method='delta', sr_layerwise=True)
    sr, rbm, vmc = problem.sr, problem.rbm, problem.vmc
    vmc.inverse_rate = 0.05

    optimizer = get_optimizer(wrt=rbm.get_variables(), fprime=problem.compute_gradient,
            optimize_method='gd', step_rate=1e-1, momentum=momentum)

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
    print('\nRunning 0-th Iteration.')
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
        if num_iter>=200:
            break
        print('\nRunning %s-th Iteration.'%(num_iter+1))

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
    problem = ModelProbDef(hamiltonian=h,rbm=rbm,reg_method='delta', sr_layerwise=False)
    sr, rbm, vmc = problem.sr, problem.rbm, problem.vmc
    vmc.inverse_rate = 0.05

    optimizer = get_optimizer(wrt=rbm.get_variables(), fprime=problem.compute_gradient,
            optimize_method='gd', step_rate=3e-2, momentum=momentum)

    # setup canvas
    if do_plot_wf:
        plt.ion()
        fig=plt.figure(figsize=(10,5))


    el=[] # to store energy
    vv_pre = None
    print('\nRunning 0-th Iteration.')
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
        print('\nRunning %s-th Iteration.'%(num_iter+1))

    np.savetxt('data/el-%s%s.dat'%(h.nsite,'p' if h.periodic else 'o'),el)
    pdb.set_trace()

def run_rtheta_switch(J2, nsite, rtheta_training_ratio, switch_step, momentum=0., \
        do_plot_wf=True, compare_to_exact=True, do_check_sample=False):
    from models.wanglei2 import WangLei2
    # definition of a problem
    h = load_hamiltonian('J1J2', size=(nsite,), J1=1.,J2=J2)
    rbm = WangLei2(input_shape=(h.nsite,),num_feature_hidden=4, use_msr=False, theta_period=2, with_linear=False, dtype='float64')
    #rbm.thnn = get_exact_thnn4(fixed_var=True)
    problem = ModelProbDef(hamiltonian=h,rbm=rbm,reg_method='sd')
    sr, rbm, vmc = problem.sr, problem.rbm, problem.vmc
    vmc.inverse_rate = 0.05

    optimizer = get_optimizer(wrt=rbm.get_variables(), fprime=problem.compute_gradient,
            optimize_method='gd', step_rate=3e-3, momentum=momentum)

    # setup canvas
    if do_plot_wf or do_check_sample:
        plt.ion()
        fig=plt.figure(figsize=(10,5))

    # Exact Results
    if compare_to_exact or compare_wf:
        H, e0, v0, configs = analyse_exact(h, do_printsign=False, num_eng=5)

    el=[] # to store energy
    vv_pre = None
    print('\nRunning 0-th Iteration.')
    sr.rtheta_training_ratio = [rtheta_training_ratio[0], 0]
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

        if do_plot_wf:
            fig.clear()
            plt.subplot(121)
            #compare_wf(amp, v0)
            compare_wf(vv, v0)
            plt.subplot(122)
            scatter_vec_phase(vv, vv_pre)
            plt.xlim(-0.8,0.8)
            plt.ylim(-0.8,0.8)
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

        k = info['n_iter']
        if k>=800:
            break
        if k%(2*switch_step)<switch_step:
            print('\nRunning %s-th Iteration (optimize amplitudes).'%(k+1))
            sr.rtheta_training_ratio = [rtheta_training_ratio[0], 0]
        else:
            print('\nRunning %s-th Iteration (optimize signs).'%(k+1))
            sr.rtheta_training_ratio = [0, rtheta_training_ratio[1]]

    np.savetxt('data/el-%s%s.dat'%(h.nsite,'p' if h.periodic else 'o'),el)
    pdb.set_trace()

def run_target_sign(J2, nsite):
    '''Given Sign train amplitude, the arbituary state version.'''
    from models.wanglei import WangLei
    # definition of a problem
    h = load_hamiltonian('J1J2', size=(nsite,), J2=J2)
    rbm = WangLei(input_shape=(h.nsite,), version='linear', use_conv=True, dtype='complex128')
    problem = ModelProbDef(hamiltonian=h,rbm=rbm,reg_method='delta', sr_layerwise=True)
    sr, rbm, vmc = problem.sr, problem.rbm, problem.vmc
    vmc.inverse_rate = 0.05

    optimizer = get_optimizer(wrt=rbm.get_variables(), fprime=problem.compute_gradient,
            optimize_method='gd', step_rate=1e-1, momentum=momentum)

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
    print('\nRunning 0-th Iteration.')
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
        num_iter = info['n_iter']
        #optimizer.step_rate *= 0.995
        if num_iter>=200:
            break
        print('\nRunning %s-th Iteration.'%(num_iter+1))

    np.savetxt('data/el-%s%s.dat'%(h.nsite,'p' if h.periodic else 'o'),el)
    pdb.set_trace()

def get_exact_thnn4(fixed_var=True):
    '''
    Number of site = 4.
    expect outputs
        ++-- => pi
        +--+ => 0
        -++- => 0
        --++ => pi
    '''
    inputs = np.array([[1,1,-1,-1],
            [1,-1,-1,1],
            [-1,1,1,-1],
            [-1,-1,1,1]])
    outputs = np.array([np.pi,0,0,np.pi])
    # we construct the following convolution as the first layer.
    # in order to get an XOR gate.
    W0 = np.array([[1,1,0,0],  #add first two bits
            [1,-1,0,0]]).T #subtract first two bits.
    b0 = np.array([0,0])
    y1a = inputs.dot(W0) + b0
    y1b = inputs.dot(np.roll(W0,2,axis=0)) + b0
    y1 = y1a**2+y1b**2
    #y1 = np.log(2*np.cosh(inputs.dot(W0) + b0)) + np.log(2*np.cosh(inputs.dot(np.roll(W0,2,axis=0)) + b0))

    # we wish outputs == y1.dot(W1)
    print('Solving %sx = %s'%(y1,outputs))
    W1 = np.linalg.solve(y1[:2],outputs[:2])
    print('Get %s'%W1)

    # construct THNN
    from models.psnn_leo import PSNN
    thnn = PSNN((4,), nf=2, batch_wise=False, period=2, output_mode='theta')
    thnn.set_variables(np.concatenate([W0.T.ravel(order='F'), b0,W1.T.ravel(order='F')]))
    if fixed_var:
        thnn.layers[1].var_mask=(0,0)
        thnn.layers[-2].var_mask=(0,0)
    return thnn

def run_rtheta_mlp_exp(J2, nsite, mlp_shape):
    from models.rtheta_mlp_exp import RTheta_MLP_EXP
    # definition of a problem
    h = load_hamiltonian('J1J2', size=(nsite, ), J2=J2)
    rbm = RTheta_MLP_EXP(input_shape=(h.nsite,),num_feature_hidden=4, mlp_shape=mlp_shape, use_msr=False, theta_period=2)
    #rbm = get_ground_toynn(h, mode='r-theta', train_amp=False, theta_period=nsite)
    #pdb.set_trace()
    problem = ModelProbDef(hamiltonian=h,rbm=rbm,reg_method='sd')
    sr, rbm, vmc = problem.sr, problem.rbm, problem.vmc
    sr.rtheta_training_ratio = 30

    optimizer = get_optimizer(wrt=rbm.get_variables(), fprime=problem.compute_gradient,
            optimize_method='adam', step_rate=3e-3, momentum=momentum)

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
    print('\nRunning 0-th Iteration.')
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
        print('\nRunning %s-th Iteration.'%(info['n_iter']+1))

    number = ''
    for i in mlp_shape:
        number += str(i)
        number += '-'
    np.savetxt('data/rtheta-mlp-exp-%sel-%s%s.dat'%(number, h.nsite,'p' if h.periodic else 'o'),el,fmt='%.10f%+.10fj')
    #pdb.set_trace()

def run_wanglei3(J2, size, optimize_method='adam', momentum=0., do_plot_wf = True, compare_to_exact = True, learning_rate=1e-2):
    from models.wanglei3 import WangLei3
    # definition of a problem
    h = load_hamiltonian('J1J2', size=size, J2=J2)
    rbm = WangLei3(input_shape=size,num_features=[8], version='conv', dtype='complex128')

    # visualize network
    from poornn import viznn
    viznn(rbm, filename='data/%s.pdf'%rbm.__class__.__name__)

    problem = ModelProbDef(hamiltonian=h,rbm=rbm,reg_method='delta', sr_layerwise=False)
    sr, rbm, vmc = problem.sr, problem.rbm, problem.vmc
    vmc.inverse_rate = 0.05

    optimizer = get_optimizer(wrt=rbm.get_variables(), fprime=problem.compute_gradient,
            optimize_method=optimize_method, step_rate=learning_rate, momentum=momentum)

    # setup canvas
    if do_plot_wf:
        plt.ion()
        fig=plt.figure(figsize=(10,5))

    # Exact Results
    if compare_to_exact or compare_wf:
        H, e0, v0, configs = analyse_exact(h, do_printsign=False)

    el=[] # to store energy
    vv_pre = None
    print('\nRunning 0-th Iteration.')
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
        #optimizer.step_rate *= 0.995
        if num_iter>=1000:
            break
        print('\nRunning %s-th Iteration.'%(num_iter+1))

    np.savetxt('data/el-%s%s.dat'%(h.nsite,'p' if h.periodic else 'o'),el)
    pdb.set_trace()


def run_wanglei4(J2, size, optimize_method='adam', momentum=0., do_plot_wf = True, compare_to_exact = True, learning_rate=1e-2):
    from models.wanglei4 import WangLei4
    # definition of a problem
    h = load_hamiltonian('J1J2', size=size, J2=J2)
    rbm = WangLei4(input_shape=size,NF=8, K=3,num_features=[8], version='conv', dtype='complex128')

    # visualize network
    from poornn import viznn
    viznn(rbm, filename='data/%s.png'%rbm.__class__.__name__)

    problem = ModelProbDef(hamiltonian=h,rbm=rbm,reg_method='delta', sr_layerwise=False)
    sr, rbm, vmc = problem.sr, problem.rbm, problem.vmc
    vmc.inverse_rate = 0.05

    optimizer = get_optimizer(wrt=rbm.get_variables(), fprime=problem.compute_gradient,
            optimize_method=optimize_method, step_rate=learning_rate, momentum=momentum)


    # setup canvas
    if do_plot_wf:
        plt.ion()
        fig=plt.figure(figsize=(10,5))

    # Exact Results
    if compare_to_exact or compare_wf:
        H, e0, v0, configs = analyse_exact(h, do_printsign=False)

    el=[] # to store energy
    vv_pre = None
    print('\nRunning 0-th Iteration.')
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
        #optimizer.step_rate *= 0.995
        if num_iter>=1000:
            break
        print('\nRunning %s-th Iteration.'%(num_iter+1))

    np.savetxt('data/el-%s%s.dat'%(h.nsite,'p' if h.periodic else 'o'),el)
    pdb.set_trace()

def show_el44(window=None):
    datafile = 'data/el44-J20.8.dat'
    EG=-0.627335103992*16
    # prepair data
    el=np.loadtxt(datafile)
    steps=np.arange(len(el))

    plt.ion()
    plt.plot(steps,el, lw=2, color='k')
    plt.xlabel('Step')
    plt.ylabel('$E$')
    plt.xscale('log')
    if EG is not None: plt.axhline(y=EG, ls='--', color='#666666')
    if window is not None: plt.ylim(*window)
    plt.show()
    pdb.set_trace()
    plt.savefig('data/ENG44%s-J20.8.png'%('[%s,%s]'%tuple(window) if window is not None else  ''))

def show_el44(window=None):
    datafile = 'data/el44-J20.8.dat'
    EG=-0.627335103992*16
    # prepair data
    el=np.loadtxt(datafile)
    steps=np.arange(len(el))

    plt.ion()
    plt.plot(steps,el, lw=2, color='k')
    plt.xlabel('Step')
    plt.ylabel('$E$')
    plt.xscale('log')
    if EG is not None: plt.axhline(y=EG, ls='--', color='#666666')
    if window is not None: plt.ylim(*window)
    plt.show()
    pdb.set_trace()
    plt.savefig('data/ENG44%s-J20.8.png'%('[%s,%s]'%tuple(window) if window is not None else  ''))

def show_kernel(rbm):
    kernel = rbm.layers[2]
