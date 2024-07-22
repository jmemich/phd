import numpy as np

from auction import BelloniAuctionApproximation

V=[[0,1],[0,1]]

runs = []
for T in [20,15,10,5]:
    print('------- T=%s...' % T)
    approx = BelloniAuctionApproximation(
        n_buyers=2,
        n_grades=2,
        costs=[0,0],
        V=V,
        T=T,
        check_local_ic='star',
        border_type='belloni',
        force_symmetric=True,
        log_level='warning')
    approx.run()
    approx.to_file('uniform_01_N2_T%s.pkl' % T)
    runs.append(approx)
