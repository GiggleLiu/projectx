import sys, os
sys.path.insert(0,'../')
from plotlib import *

def show_bentch44K():
    ntask = 4
    token = 'K'
    plt.ion()
    for show_err in [True, False]:
        plt.cla()
        show_el(datafiles = ['../benchmarks/wanglei4K/el-%i.dat'%i for i in range(ntask)],
                EG = -8.45792335,
                legends = ['K = %s'%(i+1) for i in range(ntask)],
                show_err=show_err,
                xlogscale=not show_err)
        pdb.set_trace()
        plt.savefig('img/%s-%s%s.png'%('errl' if show_err else 'el', token, ''))

def show_bentch44dn():
    '''Deep network.'''
    token = 'dn'
    ntask = 5
    plt.ion()
    for show_err in [True, False]:
        plt.cla()
        show_el(datafiles = ['../benchmarks/wanglei4dn/el-%i.dat'%i for i in range(ntask)],
                legends = ['structure = %s'%(i+1) for i in range(ntask)],
                show_err=show_err,
                xlogscale=not show_err)
        pdb.set_trace()
        plt.savefig('img/%s-%s%s.png'%('errl' if show_err else 'el', token, ''))

def show_bentch44dncr():
    '''Compare complex and real.'''
    token = 'dn'
    ids = [1,6,7]
    show_err = True
    plt.ion()
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
    plt.legend(legends)
    plt.ylim(1e-5,1)
    pdb.set_trace()
    plt.savefig('img/errcr-%s.png'%token)

def show_kernel44K(bentch_id):
    configfile = '../benchmarks/wanglei4K/config-sample.ini'
    plt.ion()
    show_kernel_bentch(configfile, bentch_id = bentch_id)
    plt.colorbar()
    pdb.set_trace()

def show_mpiacc(task='tianhe'):
    tdata = np.loadtxt('mpiacc-%s.tbl'%task)
    ncores, t_tots, t_fors, t_cnvs, t_mpis, accs = tdata.T
    tmpis = t_tots-t_fors
    plt.ion()
    plt.figure(figsize=(6,4))
    plt.title('Component Analysis')
    plt.ylabel('niter/sec')
    for data, color in zip(tdata[:,1:4].T, ['r','g','b']):
        plt.plot(ncores, 1./data, color=color)
        plt.plot(ncores, 1./data[0]*ncores, color=color, ls='--')
    plt.legend(['total', '%.4f'%(tdata[0,1]/tdata[-1,1]/ncores[-1]),'forward',
        '%.4f'%(tdata[0,2]/tdata[-1,2]/ncores[-1]),
        'conv','%.4f'%(tdata[0,3]/tdata[-1,3]/ncores[-1])])
    plt.xlim(0,24)
    plt.tight_layout()
    pdb.set_trace()
    plt.savefig('img/mpiacc-%s.png'%task)

def show_mpi_err():
    task='tianhe'
    tdata = np.loadtxt('mpiacc-%s.tbl'%task)
    ncores, t_tots, t_fors, t_cnvs, t_mpis, accs = tdata.T
    #plt.plot(ncores, 1./ncores/t_tots[:]*t_tots[0])
    plt.ion()
    plt.figure(figsize=(6,4))
    plt.plot(ncores, accs, '-o')
    plt.xlim(0,24)
    plt.title('Error in 100 steps')
    plt.xlabel('#  of cores')
    pdb.set_trace()
    plt.savefig('img/mpiacc-error-%s.png'%task)

def show_mpiparts():
    task='tianhe'
    tdata = np.loadtxt('mpiacc-%s.tbl'%task)
    ncores, t_tots, t_fors, t_cnvs, t_mpis, accs = tdata.T
    tcals = t_tots-t_mpis
    a, b = np.polyfit(ncores, tcals*ncores,1)  #resource = a*x+b
    label = 'parallel = %s, sequencial = %s'%(b, a)

    plt.ion()
    plt.figure(figsize=(6,4))
    #plt.plot(ncores, tcals, color='k')
    #plt.plot(ncores, b/ncores)
    #plt.plot(ncores, a*np.ones(len(ncores)))
    #plt.plot(ncores, t_mpis)
    plt.plot(ncores, tcals*ncores, color='k')
    plt.plot(ncores, b*np.ones(len(ncores)))
    plt.plot(ncores, a*ncores)
    plt.plot(ncores, t_mpis*ncores)

    plt.ylabel('Resources',fontsize=14)
    plt.legend(['Total','Parallel','Sequencial','Transimision'])
    plt.xlim(0,24)
    plt.tight_layout()

    pdb.set_trace()
    plt.savefig('img/mpiacc-parts-tianhe.png')

if __name__ == '__main__':
    #show_bentch44K()
    #show_bentch44dn()
    #show_kernel44K(1)
    #show_bentch44dncr()
    show_mpiacc('delta')
    #show_mpiparts()
    #show_mpi_err()
