import re, pdb
import numpy as np

floatstr = r'[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?'

def match(filename, outfile):
    el = []
    with open(filename,'r') as f:
        for i,line in enumerate(f):
            res = re.match(r'E/site = \((%s)[+-]%sj\)'%(floatstr, floatstr), line)
            if res:
                el.append(float(res.group(1))*16)
    np.savetxt(outfile,el)
    print(len(el))

for i in xrange(5):
    match(filename='log-%s.log'%i,outfile = 'el-%s.dat'%i)
