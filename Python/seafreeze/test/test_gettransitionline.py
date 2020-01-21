import warnings, unittest as ut
import numpy as np
import seafreeze.seafreeze as sf
from time import time


class TestGetTransitionLine(ut.TestCase):
    def setup(self):
        warnings.simplefilter('ignore', category=ImportWarning)
    def tearDown(self):
        pass
    def test_get_transition_line_border_defaultprec(self):
        st = time()
        out = sf.get_transition_line(['III', 'V'], path='../../../Matlab/SeaFreeze_Gibbs.mat')
        print('elapsed: '+str(time()-st)+' secs')
        print(str(out))


