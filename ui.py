#!/usr/bin/env python

import fire, os, sys

class UI(object):
    def bentchmark(self, configfile, id, e0=None):
        from problems import load_config
        config = load_config(configfile)
        folder = os.path.dirname(configfile)
        n = config['mpi']['num_core']

        logfile = '%s/log-%s.log'%(folder,id)
        options = ' '.join(['%s'%item for item in [configfile, id, e0]])
        exec_code = 'python bentchmark.py %s'%(options,)
        exec_code = 'nohup mpirun -n %s %s > %s &'%(n,exec_code,logfile)
        print('Excuting: %s'%exec_code)
        os.system(exec_code)

    def bdn(self, id):
        self.bentchmark(configfile='bentchmarks/wanglei4dn/config-sample.ini', id=id, e0=None)

    def bK(self, id):
        self.bentchmark(configfile='bentchmarks/wanglei4K/config-sample.ini', id=id, e0=None)

    def b6(self, id):
        self.bentchmark(configfile='bentchmarks/wanglei6/config-sample.ini', id=id, e0=-0.503810*36)

    def test(self, arg1, arg2):
        print('GET: arg1 = %s, arg2 = %s'%(arg1, arg2))

def main():
    fire.Fire(UI)

if __name__ == '__main__':
    main()
