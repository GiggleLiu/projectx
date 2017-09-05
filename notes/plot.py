from matplotlib.pyplot import *
from numpy import *
import pdb, os

def show_el(datafiles, EG=None, xlogscale=True, window=None, legends=None, token='', show_err=False):
    if legends is None: legends = arange(len(datafiles))
    ion()
    # prepair data
    for datafile in datafiles:
        el=np.loadtxt(datafile)
        steps=np.arange(len(el))
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
    pdb.set_trace()
    savefig('img/%s-%s%s.png'%('errl' if show_err else 'el', token, '[%s,%s]'%tuple(window) if window is not None else  ''))

def bentch44K():
    ntask = 4
    for show_err in [True, False]:
        show_el(datafiles = ['../bentchmarks/wanglei4K/el-%i.dat'%i for i in range(ntask)],
                EG = -8.45792335,
                legends = ['K = %s'%(i+1) for i in range(ntask)],
                token = 'K',
                show_err=show_err,
                xlogscale=not show_err)
        cla()

def bentch44dn():
    ntask = 5
    for show_err in [True, False]:
        show_el(datafiles = ['../bentchmarks/wanglei4dn/el-%i.dat'%i for i in range(ntask)],
                EG = -8.45792335,
                legends = ['structure = %s'%(i+1) for i in range(ntask)],
                token = 'dn',
                show_err=show_err,
                xlogscale=not show_err)
        cla()

if __name__ == '__main__':
    #bentch44K()
    bentch44dn()
