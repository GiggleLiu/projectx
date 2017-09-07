import sys, os
sys.path.insert(0,'../')
from plotlib import *

def show_bentch44K():
    ntask = 4
    token = 'K'
    ion()
    for show_err in [True, False]:
        cla()
        show_el(datafiles = ['../benchmarks/wanglei4K/el-%i.dat'%i for i in range(ntask)],
                EG = -8.45792335,
                legends = ['K = %s'%(i+1) for i in range(ntask)],
                show_err=show_err,
                xlogscale=not show_err)
        pdb.set_trace()
        savefig('img/%s-%s%s.png'%('errl' if show_err else 'el', token, ''))

def show_bentch44dn():
    '''Deep network.'''
    token = 'dn'
    ntask = 5
    ion()
    for show_err in [True, False]:
        cla()
        show_el(datafiles = ['../benchmarks/wanglei4dn/el-%i.dat'%i for i in range(ntask)],
                legends = ['structure = %s'%(i+1) for i in range(ntask)],
                show_err=show_err,
                xlogscale=not show_err)
        pdb.set_trace()
        savefig('img/%s-%s%s.png'%('errl' if show_err else 'el', token, ''))

def show_bentch44dncr():
    '''Compare complex and real.'''
    token = 'dn'
    ids = [1,6,7]
    show_err = True
    ion()
    from problems import load_config
    folder = '../benchmarks/wanglei4dn'
    config = load_config(folder+'/config-sample.ini')
    legends = []
    for id in ids:
        for version in ['c', 'r']:
            show_el(datafiles = [folder+'/el-%i%s.dat'%(id, 'c' if version=='c' else '')],
                    EG = config['hamiltonian']['EG'],
                    show_err=show_err,
                    xlogscale=not show_err, smooth_step=20)
            legends.append('%s-%s'%(version, id+1))
    legend(legends)
    ylim(1e-5,1)
    pdb.set_trace()
    savefig('img/errcr-%s.png'%token)

def show_kernel44K(bentch_id):
    configfile = '../benchmarks/wanglei4K/config-sample.ini'
    ion()
    show_kernel_bentch(configfile, bentch_id = bentch_id)
    colorbar()
    pdb.set_trace()

if __name__ == '__main__':
    #show_bentch44K()
    #show_bentch44dn()
    #show_kernel44K(1)
    show_bentch44dncr()
