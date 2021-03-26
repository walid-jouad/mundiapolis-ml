  
#!/usr/bin/env python3
'''   neuroneclass  '''

import numpy as np


class Neuron:
    ''' neuroneclass initialized '''

    def __init__(self, nx):
        ''' neuroneclass fonction  '''

        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.b = 0
        self.A = 0
        self.W = np.random.randn(nx).reshape(1, nx)
