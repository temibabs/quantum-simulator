from typing import Any

import numpy as np

from qm.constants import Constant


class WaveFunction2D(Constant):
    """
    Wavefunction in 2D
    """

    def __init__(self, x_waveform: Any, y_waveform):
        super().__init__()

        if callable(x_waveform) and callable(y_waveform):
            try:
                lin = np.linspace(self.x0, self.L + self.x0, self.N)
                self.x_wave = x_waveform(lin)
                self.y_wave = y_waveform(lin)
            except TypeError:
                lin = np.linspace(self.x0, self.L + self.x0, self.N)
                self.x_wave = np.array([x_waveform(tmp) for tmp in lin])
                self.y_wave = np.array([y_waveform(tmp) for tmp in lin])

            try:
                len(self.x_wave)
                len(self.y_wave)
            except TypeError as e:
                print(e)

        elif (isinstance(x_waveform, np.ndarray)
              and isinstance(y_waveform, np.ndarray)):
            self.x_wave = x_waveform
            self.y_wave = y_waveform

        self.wave_function = self.x_wave * self.y_wave

    def normalize(self):
        try:
            self.x_wave = (self.x_wave /
                           np.sqrt(np.trapz(np.conj(self.x_wave) * self.x_wave,
                                            dx=self.dx)))
            self.y_wave = (self.y_wave /
                           np.sqrt(np.trapz(np.conj(self.y_wave) * self.y_wave,
                                            dx=self.dy)))
        except FloatingPointError as E:
            print(E)

    @property
    def p_x(self):
        """
        Wavefunction in the momentum basis
        """
        return np.fft.fftshift(np.fft.fft(self.x_wave) / (self.N / 10))

    @property
    def p_y(self):
        """
        Wavefunction in the momentum basis
        """
        return np.fft.fftshift(np.fft.fft(self.y_wave) / (self.N / 10))

    def set_to_eigenstate(self, eigenvalues, eigenvectors, smear=False):
        prob = np.abs(np.dot(self.wave_function, eigenvectors)) ** 2
        if np.max(prob) != 0.:
            prob = prob / np.sum(prob)
        a = [i for i in range(len(prob))]
        choice = np.random.choice(a, p=prob, replace=False)[0]
        self.wave_function = eigenvectors.T[[choice]][0]
        self.normalize()

        if smear:
            for i in range(1, 3):
                if choice - i >= 0 and choice + i <= self.N:
                    self.wave_function += (eigenvectors.T[[choice + i]][0]
                                           * np.exp(-i ** 2. / 2.))
                    self.wave_function += (eigenvectors.T[[choice - i]][0]
                                           * np.exp(-i ** 2. / 2.))

        return eigenvalues[choice]

    def expectation_value(self, energy_eigenvalues, energy_eigenstates):
        try:
            probs = np.abs(np.dot(self.x_wave, energy_eigenstates)) ** 2
            if np.max(probs) != 0.:
                probs = probs / np.sum(probs)
        except FloatingPointError:
            return 0.
        return np.sum(np.dot(np.real(energy_eigenvalues), probs))


class UnitaryOperator2D(Constant):

    def __init__(self, potential):
        """Initialize the unitary operator.
        """

        # The unitary operator can be found by applying the Cranck-Nicholson
        # method. This method is outlined in great detail in exercise 9.8 of
        # Mark Newman's Computational Physics. Here is a link to a
        # page containing all problems in his book:
        # http://www-personal.umich.edu/~mejn/cp/exercises.html

        # Newman, M. (2013). Partial differential equations.
        # In Computational Physics, chapter 9.
        # CreateSpace Independent Publishing Platform.

        # TODO: This can be optimized.

        super().__init__()

        if isinstance(potential, np.ndarray):
            v = potential

        elif callable(potential):
            x = np.linspace(self.x0, (self.L + self.x0), self.N)
            v = np.array([potential(xi) for xi in x])

        else:
            raise TypeError("Invalid Type")

        v *= self._scale

        # Get constants
        m, hbar, e, L, N, dx, dt = self._get_constants()

        # Initialize A and B matrices
        a = np.zeros([N, N], np.complex64)
        b = np.zeros([N, N], np.complex64)

        # \Delta t \frac{i \hbar}{2m \Delta x^2}
        k = (dt * 1.0j * hbar) / (4 * m * dx ** 2)

        # \frac{\Delta t i \hbar}{2}
        j = (dt * 1.0j) / (2 * hbar)

        # Initialize the constant,
        # nonzero elements of the A and B matrices
        a1 = 1 + 2 * k
        a2 = -k
        b1 = 1 - 2 * k
        b2 = k

        # Construct the A and B matrices
        for i in range(N - 1):
            a[i][i] = a1 + j * v[i]
            b[i][i] = b1 - j * v[i]
            a[i][i + 1] = a2
            a[i + 1][i] = a2
            b[i][i + 1] = b2
            b[i + 1][i] = b2
        a[N - 1][N - 1] = a1 + j * v[N - 1]
        b[N - 1][N - 1] = b1 - j * v[N - 1]

        # Obtain U
        self.U = np.dot(np.linalg.inv(a), b)
        # self.U = np.matmul(np.linalg.inv(A), B)

        # The identity operator is what the unitary matrix
        # reduces to at time zero. Also,
        # since the wavefunction and all operators are
        # in the position basis, the identity matrix
        # is the position operator.
        self.id = np.identity(len(self.U[0]), complex)

        # Get the Hamiltonian from the unitary operator
        # and aquire the energy eigenstates.
        # self.set_energy_eigenstates()

        self.energy_eigenvalues = None
        self.energy_eigenstates = None

    def __call__(self, wavefunction: WaveFunction2D):
        """Call this class on a wavefunction to time-evolve it."""
        try:
            wavefunction.x_wave = np.matmul(self.U, wavefunction.x_wave)
            wavefunction.y_wave = np.matmul(self.U, wavefunction.y_wave)
        except FloatingPointError:
            pass

    def _set_HU(self):
        """Set HU (the Hamiltonian times the unitary operator).
        Note that HU is not Hermitian.
        """
        # The Hamiltonian H is proportional to the
        # time derivative of U times its inverse
        self._HU = ((0.5 * 1.0j * self.h_bar / self.dt) *
                    (self.U - np.conj(self.U.T)))
        # self._HU =(1.0j*self.hbar/self.dt)*(self.U - self.id)

    def set_energy_eigenstates(self):
        """Set the eigenstates and energy eigenvalues.
        """
        self._set_HU()

        eigvals, eigvects = np.linalg.eigh(self._HU)
        eigvects = eigvects.T
        eigvals = np.sign(np.real(eigvals)) * abs(eigvals)

        tmp_dict = {}
        for i in range(len(eigvals)):
            e = round(eigvals[i], 6)
            if e in tmp_dict:
                tmp_dict[e] = np.add(eigvects[i], tmp_dict[e])
            else:
                tmp_dict[e] = eigvects[i]

        eigvals, eigvects = tmp_dict.keys(), tmp_dict.values()
        self.energy_eigenvalues = np.array(list(eigvals))
        self.energy_eigenstates = np.array(list(eigvects), np.complex128).T
