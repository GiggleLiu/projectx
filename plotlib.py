from matplotlib.pyplot import *
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
    pcolormesh(strength.real)
    axis('equal')

def show_el(datafiles, EG=None, xlogscale=True, window=None, legends=None,\
        show_err=False, smooth_step=1):
    if legends is None: legends = np.arange(len(datafiles))
    # prepair data
    for datafile in datafiles:
        el=np.loadtxt(datafile)
        if smooth_step!=1: el=el.reshape([-1,smooth_step]).mean(axis=1)
        steps=np.arange(len(el))*smooth_step
        if show_err:
            plot(steps,abs((el-EG)/EG), lw=2)
        else:
            plot(steps,el, lw=2)

    xlabel('Step')
    ylabel(r'$\Delta E/E_G$' if show_err else r'$E$')
    if xlogscale: xscale('log')
    if show_err: yscale('log')
    if EG is not None and not show_err: axhline(y=EG, ls='--', color='#666666')
    if window is not None: ylim(*window)
    legend(legends)

def scatter_vec_phase(v, v0=None, color='r', color0='b', winding=None):
    '''
    show the amplitude-phase graph in complex plane.
    '''
    x,y = v.real, v.imag
    scatter(x,y,s=20, color=color)
    if v0 is not None:
        x0,y0 = v0.real, v0.imag
        quiver(x0,y0,x-x0,y-y0, angles='xy', units='xy', scale=1)
        scatter(x0,y0,s=20, color=color0)
    if winding is not None:
        for xi, yi, wi in zip(x,y,winding):
            text(xi,yi,'%s'%wi, ha='left', va='bottom')
    xlabel('$\Re[\Psi]$')
    ylabel('$\Im[\Psi]$')

def compare_wf(vv, v0):
    #if vv.dot(v0)<0: vv=-vv

    # pivot
    overlap = v0.dot(vv)
    vv = vv*np.exp(-1j*np.angle(overlap))

    print('|<Psi_0|Psi>|^2 = %s'%abs(overlap)**2)
    title('Wave Function')
    plot(v0, lw=1, color='k')
    plot(vv, lw=1, color='r')
    legend([r'$\Psi_0$', r'$\Psi$'])

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

    plot(abs(v)**2, color ='k')
    plot(wf_sample, color ='r')
    xticks(np.arange(hndim), packnbits_pm(configs))
    legend(['exact', 'vmc'])

############# not supported any more ###################
def plot_wf_distri(h, v0):
    # BUGGY
    v0_ = sort(abs(v0))[::-1]
    plot(v0_**2)
    yscale('log')
    #ylabel(r'$\log(\Psi(x)^2)$')
    ylabel(r'$\Psi(x)^2$')
    ylim(1e-11,1)
    pdb.set_trace()

def plot_sign_mat(sign_classifier):
    subplot(121)
    title('Raw  Sign')
    sign_classifier.plot_mat_sign(signs = np.ones(v0.shape[0]))
    subplot(122)
    title('True Sign')
    sign_classifier.plot_mat_sign(signs = sign(v0))
    tight_layout()


