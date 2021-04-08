class Constant:
    def __init__(self):

        self.m = 1.                 # Mass
        self.e = 1.                 # Charge
        self.h_bar = 1.             # Reduced Planck's constant

        self.x0 = -.5               # Initial  position
        self.L = 1                  # Length of both sides
        self.N = 512                # Number of steps in space
        self.dx = self.L / self.N   # Space stepsize
        self.dt = 0.00001           # Time stepsize

        self.dy = self.dx

        self._scale = (128 / self.N) * 5e5

    def _get_constants(self):
        return self.m, self.h_bar, self.e, self.L, self.N, self.dx, self.dt
