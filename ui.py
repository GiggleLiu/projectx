#!/usr/bin/env python

import fire, os, sys, pdb
import numpy as np

from utils import analyse_exact
from problems import load_hamiltonian

class UI(object):
    ###################  BENTCHMARK  ######################
    def bentchmark(self, configfile, id, e0=None, interactive=False):
        from problems import load_config
        config = load_config(configfile)
        folder = os.path.dirname(configfile)
        n = config['mpi']['num_core']

        logfile = '%s/log-%s.log'%(folder,id)
        options = ' '.join(['%s'%item for item in [configfile, id, e0]])
        exec_code = 'python bentchmark.py %s'%(options,)
        if not interactive:
            exec_code = 'nohup mpirun -n %s %s > %s &'%(n,exec_code,logfile)
        print('Excuting: %s'%exec_code)
        os.system(exec_code)

    def bbb(self, subfolder, id, e0=None):
        '''Quick bentchmark, not using mpi or nohup.'''
        self.bentchmark('bentchmarks/%s/config-sample.ini'%subfolder, id, e0, interactive=True)

    def bdn(self, id):
        '''shortcut for bentchmark depth of wanglei4 model.'''
        self.bentchmark(configfile='bentchmarks/wanglei4dn/config-sample.ini', id=id, e0=None)

    def bK(self, id):
        '''shortcut for bentchmark filter size of wanglei4 model.'''
        self.bentchmark(configfile='bentchmarks/wanglei4K/config-sample.ini', id=id, e0=None)

    def b6(self, id):
        self.bentchmark(configfile='bentchmarks/wanglei6/config-sample.ini', id=id, e0=-0.503810*36)

    def test(self, arg1, arg2):
        print('GET: arg1 = %s, arg2 = %s'%(arg1, arg2))

    def bestep(self, subfolder, id, istep):
        '''show energy at specific step for a bentchmark instance.'''
        # get configuration and foler
        from problems import load_config, pconfig
        configfile = 'bentchmarks/%s/config-sample.ini'%subfolder
        config = load_config(configfile)

        # modification to parameters
        folder = os.path.dirname(configfile)
        sys.path.insert(0,folder)
        from config import modifyconfig_and_getnn
        rbm = modifyconfig_and_getnn(config, id)
        varfile = os.path.join(folder,'variables-%s%s.npy'%(id,istep))
        rbm.set_variables(np.load(varfile))
        optimizer, problem = pconfig(config, rbm)
        print('Energy at step %s = %s'%(istep, problem.get_energy()))
     

    ######################  EXACT  #########################

    def ed(self, J2, size, *tasks):
        h = load_hamiltonian(model='J1J2',J1=1, J2=J2, size=size)
        H, e, v, configs = analyse_exact(h, do_printsign=False, num_eng=1)
        if 'v' in tasks: print('v = %s'%v)
        if 'H' in tasks: print('H = %s'%H)
        if 'sign' in tasks:
            np.set_printoptions(threshold='nan')
            print('sign = %s'%np.sign(v))
            np.set_printoptions(threshold=1000)
    

def main():
    fire.Fire(UI)

if __name__ == '__main__':
    main()
