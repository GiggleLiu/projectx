from numpy import *
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
from utils import *

def test_mvar():
    print('Testing mvar')
    a = random.random(100)+1j*random.random(100)
    weights = random.randint(0,10,100)
    wmean, wvar = mvar(a,weights)

    aa = repeat(a, weights)
    assert_almost_equal(mean(aa), wmean)
    assert_almost_equal(var(aa), wvar)

if __name__=='__main__':
    test_mvar()
