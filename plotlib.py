try:
    from matplotlib import pyplot as plt
except:
    import matplotlib
    matplotlib.rcParams['backend'] = 'TkAgg'
    from matplotlib import pyplot as plt
import numpy as np
import sys,pdb,os

from problems import load_config

def show_kernel_bentch(configfile, bentch_id):
    config = load_config(configfile)
    # folder to store data, containing config.py
    folder = os.path.dirname(configfile)

    # modification to parameters
    sys.path.insert(0,folder)
    from config import modifyconfig_and_getnn
    rbm = modifyconfig_and_getnn(config, bentch_id)
    show_kernel(rbm)

def show_kernel(rbm):
    kernel = rbm.layers[2]
    #strength = abs(kernel.weight).mean(axis=0)[0]
    strength = kernel.weight[0,0]
    plt.pcolormesh(strength.real)
    plt.axis('equal')

def show_el(datafiles, EG=None, xlogscale=True, window=None, legends=None,\
        show_err=False, smooth_step=1):
    if legends is None: legends = np.arange(len(datafiles))
    # prepair data
    for datafile in datafiles:
        el=np.loadtxt(datafile)
        if smooth_step!=1: el=el.reshape([-1,smooth_step]).mean(axis=1)
        steps=np.arange(len(el))*smooth_step
        if show_err:
            plt.plot(steps,abs((el-EG)/EG), lw=2)
        else:
            plt.plot(steps,el, lw=2)

    plt.xlabel('Step')
    plt.ylabel(r'$\Delta E/E_G$' if show_err else r'$E$')
    if xlogscale: plt.xscale('log')
    if show_err: plt.yscale('log')
    if EG is not None and not show_err: plt.axhline(y=EG, ls='--', color='#666666')
    if window is not None: plt.ylim(*window)
    plt.legend(legends)

def scatter_vec_phase(v, v0=None, color='r', color0='b', winding=None):
    '''
    show the amplitude-phase graph in complex plane.
    '''
    x,y = v.real, v.imag
    plt.scatter(x,y,s=20, color=color)
    if v0 is not None:
        x0,y0 = v0.real, v0.imag
        plt.quiver(x0,y0,x-x0,y-y0, angles='xy', units='xy', scale=1)
        plt.scatter(x0,y0,s=20, color=color0)
    if winding is not None:
        for xi, yi, wi in zip(x,y,winding):
            plt.text(xi,yi,'%s'%wi, ha='left', va='bottom')
    plt.xlabel('$\Re[\Psi]$')
    plt.ylabel('$\Im[\Psi]$')

def compare_wf(vv, v0):
    #if vv.dot(v0)<0: vv=-vv

    # pivot
    overlap = v0.dot(vv)
    vv = vv*np.exp(-1j*np.angle(overlap))

    print('|<Psi_0|Psi>|^2 = %s'%abs(overlap)**2)
    plt.title('Wave Function')
    plt.plot(v0, lw=1, color='k')
    plt.plot(vv, lw=1, color='r')
    plt.legend([r'$\Psi_0$', r'$\Psi$'])

def check_sample(rbm, h, samples):
    nsite = h.nsite
    # get rbm wave function
    v=rbm.tovec(mag=h.mag)
    v=v/np.linalg.norm(v)
    H=h.get_mat(dense=False)
    print('ExactVMC E/site = %s'%(v.conj().T.dot(H.dot(v)).item()/nsite))

    configs = h.configs
    hndim=len(configs)
    wf_sample = np.zeros(hndim)
    wf_sample[h.config_indexer[samples.config_inds]] = np.array(samples.counts, dtype='float64')/samples.num_sample

    plt.plot(abs(v)**2, color ='k')
    plt.plot(wf_sample, color ='r')
    plt.xticks(np.arange(hndim), packnbits_pm(configs))
    plt.legend(['exact', 'vmc'])

############# not supported any more ###################
def plot_wf_distri(h, v0):
    # BUGGY
    v0_ = sort(abs(v0))[::-1]
    plt.plot(v0_**2)
    plt.yscale('log')
    #ylabel(r'$\log(\Psi(x)^2)$')
    plt.ylabel(r'$\Psi(x)^2$')
    plt.ylim(1e-11,1)
    plt.pdb.set_trace()

def plot_sign_mat(sign_classifier):
    plt.subplot(121)
    plt.title('Raw  Sign')
    sign_classifier.plot_mat_sign(signs = np.ones(v0.shape[0]))
    plt.subplot(122)
    plt.title('True Sign')
    sign_classifier.plot_mat_sign(signs = sign(v0))
    plt.tight_layout()


