import warnings, unittest as ut
import numpy as np
import seafreeze.seafreeze as sf
from time import time

tpath = '../../../Matlab/SeaFreeze_Gibbs.mat'


class TestGetTransitionLine(ut.TestCase):
    def setup(self):
        warnings.simplefilter('ignore', category=ImportWarning)
    def tearDown(self):
        pass
    def test_get_transition_line_unsupported_precision(self):
        try:
            sf.get_transition_line(['Ih','III'], prec=7, path=tpath)
        except ValueError as ve:
            pass
    def test_get_transition_line_border_defaultprec(self):
        st = time()
        out = sf.get_transition_line(['III', 'V'], path=tpath)
        print('elapsed: '+str(time()-st)+' secs')
        print(str(out))


