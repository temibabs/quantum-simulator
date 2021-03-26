from time import perf_counter
from typing import Union, Any

from function import Function
from qm.constants import Constant
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import interactive, animation

from qm.qm import WaveFunction2D, UnitaryOperator2D


def scale(_x: np.ndarray, scale_val: float) -> np.ndarray:
    absmaxposval = np.abs(np.amax(_x))
    absmaxnegval = np.abs(np.amin(_x))
    if absmaxposval > scale_val or absmaxnegval > scale_val:
        _x = scale_val * _x / absmaxposval \
            if absmaxposval > absmaxnegval else \
            scale_val * _x / absmaxnegval
    return _x


def ordinate(number_string: str) -> str:
    """
    Turn numbers of the form '1' into '1st',
    '2' into '2nd', and so on.
    """
    if (len(number_string) >= 2) and (number_string[-2:] == "11"):
        return number_string + "th"
    elif (len(number_string) >= 2) and (number_string[-2:] == "12"):
        return number_string + "th"
    elif (len(number_string) >= 2) and (number_string[-2:] == "13"):
        return number_string + "th"
    elif number_string[-1] == "1":
        return number_string + "st"
    elif number_string[-1] == "2":
        return number_string + "nd"
    elif number_string[-1] == "3":
        return number_string + "rd"
    else:
        return number_string + "th"


class QuantumAnimation(Constant):
    def __init__(self, function="np.exp(-0.5())", potential="(x)**2/2"):
        super().__init__()

        # String attributes
        self.v_x = None
        self._KE_ltx = r"-\frac{\hbar^2}{2m} \frac{d^2}{dx^2}"
        self._lmts_str = r"  %s$ \leq x \leq $%s" % \
                         (str(np.round(self.x0, 2)),
                          str(np.round(self.L + self.x0, 2)))

        self._msg = ""  # Temporary messages in the textbox in the corner
        self._main_msg = ""  # Primary messages in this same text box.
        self._main_msg_store = ""  # Store the primary message
        self.psi_name = ""  # Name of the wavefunction
        self.psi_latex = ""  # LaTEX name of the wavefunction
        self.V_name = ""  # Name of the potential
        self.v_latex = ""  # LaTEX name of the potential
        self.identity_matrix = np.identity(self.N, np.complex128)

        # Ticking int attributes
        self.fpi = 1  # Set the number of time evolutions per animation frame
        self._t = 0  # Time that has passed
        self._msg_i = 0  # Message counter for displaying temporary messagesT
        self.fps = 30  # frames per second
        self.fps_total = 0  # Total number of fps
        self.avg_fps = 0  # Average fps
        self.ticks = 0  # total number of ticks

        self.x_ticks = []

        self.t_perf = [1.0, 0.]

        self._dpi = 120

        # Display the probability function?
        self._display_probs = False

        self._scale_y = 1.0

        # Show position or momentum?
        self._show_p = False

        # Show energy levels?
        self._show_energy_levels = False

        # Show expectation values?
        self.show_exp_val = False

        # Position of the message
        self._msg_pos = (0, 0)

        # Positions
        self.x = np.linspace(self.x0, (self.L + self.x0), self.N)
        self.y = np.linspace(self.x0, (self.L + self.x0), self.N)

        # Parameters
        self.psi_base = None
        self.psi_params = {}
        self.V_base = None
        self.V_params = None

        Function.add_function('arg', lambda theta: np.exp(2.0j * np.pi * theta))
        Function.add_function('ees', lambda n, _x: self.get_energy_eigenstate(
            int(n)) if np.array_equal(self.x, _x) else
        rescale_array(_x, self.x, np.real(self.get_energy_eigenstate(int(n)))))

        self.set_wavefunction(function, function)

        self.V_x = None
        self.set_unitary(potential)

        self._init_plots()

        self.psi: Union[None, WaveFunction2D] = None
        self.U_t: Union[None, UnitaryOperator2D] = None
        self.V: Union[None, np.ndarray, str] = None
        self.V_latex: Union[None, str] = None

        self.lines = None
        self.line10 = None
        self.line11 = None

    def set_unitary(self, potential):
        if isinstance(potential, str):
            try:
                if potential.strip().replace(".", "").replace(
                        "-", "").replace("e", "").isnumeric():
                    self.V_name = ""
                    self.V_latex = str(np.round(float(potential), 2))
                    if float(potential) == 0:
                        potential = 1e-30
                        v_f = float(potential) * np.ones([self.N])
                        self.U_t = UnitaryOperator2D(np.copy(v_f))
                        self.V_x = 0.0 * v_f
                    else:
                        v_f = scale(float(potential) * np.ones([self.N]), 15)
                        self.V_x = v_f
                        self.U_t = UnitaryOperator2D(np.copy(v_f))
                        self.V_latex = "%sk" % self.V_latex if v_f[0] > 0 \
                            else " %sk" % self.V_latex
                    self.V_params = {}
                    self.V_base = None
                else:
                    potential = potential.replace("^", "**")
                    f = Function(potential, "x")
                    self.V = lambda _x: f(_x, *f.get_tupled_default_values())
                    self.V_x = scale(self.V(self.x), 15)
                    self.V_name = str(f)
                    self.V_latex = "$" + f.multiply_latex_string("k") + "$"
                    self.U_t = UnitaryOperator2D(self.V)
                    self.V_base = f
                    self.V_params = f.get_enumerated_default_values()
            except (TypeError, AttributeError,
                    SyntaxError, ValueError, NameError) as E:
                print(E)
        elif isinstance(potential, np.ndarray):
            self.V_params = {}
            self.V_base = None
            self.V_x = scale(V, 15)
            self.V_name = "V(x)"
            self.V_latex = "$V(x)$"
            self.U_t = UnitaryOperator2D(V)
        else:
            print('Unable to parse input')

        if hasattr(self, "lines"):
            self.update_draw_potential()

    def measure_position(self, *args):
        """
        Measure the position. This collapses the wavefunction
        to the most probable position eigenstate.
        """
        x = self.psi.set_to_eigenstate(self.x, self.identity_matrix, smear=True)
        self._msg = "Position x = %s" % (str(np.round(x, 3)))
        self._msg_i = 50
        self.update_expected_energy_level()

    def measure_momentum(self, *args):
        """
        Measure the momentum. This collapses the wavefunction
        to the most probable momentum eigenstate.
        """
        p = self.psi.set_to_momentum_eigenstate()
        freq = str(int(p * (1 / (2 * np.pi * self.hbar / self.L))))
        self._msg = "Momentum p = %s\n(k = %s)" % (
            str(np.round(p, 3)), freq)
        self._msg_i = 50
        self.update_expected_energy_level()

    def measure_energy(self, *args):
        if self.U_t.energy_eigenstates is not None:
            self.U_t.set_energy_eigenstates()
        ee = np.sort(np.real(self.U_t.energy_eigenvalues))
        ee_dict = {e: (i + 1) for i, e in enumerate(ee)}
        e = self.psi.set_to_eigenstate(self.U_t.energy_eigenvalues,
                                       self.U_t.energy_eigenstates)
        self._msg = ('Energy E = %s\n(%s energy level)'
                     % (str(np.round(np.real(e), 1)),
                        ordinate(str(ee_dict[np.real(e)]))))
        self._msg_i = 50
        self.update_expected_energy_level()

    def update_expected_energy_level(self):
        if self._show_energy_levels:
            exp_energy = self.psi.expectation_value(
                self.U_t.energy_eigenvalues, self.U_t.energy_eigenstates)
            exp_energy_show = exp_energy / (self._scale_y * self.U_t._scale)
            self.line11.set_ydata([exp_energy_show, exp_energy_show])

    def toggle_energy_levels(self) -> None:
        self.set_scale_y()
        if self._show_p:
            alpha = 0. if self.lines[9] == 1.0 else 1.0
            self.lines[9].set_alpha(alpha)
            self.lines[10].set_alpha(alpha)
            self.lines[11].set_alpha(alpha)

        if not self._show_energy_levels:
            if not hasattr(self.U_t, "_nE"):
                self._set_eigenstates()
            q = np.array([self.x[0] if (((i - 1) // 2) % 2 == 0) else self.x[-1]
                          for i in
                          range(2 * len(self.U_t.energy_eigenvalues) - 1)])
            e = np.array([self.U_t.energy_eigenvalues[i // 2]
                          for i in
                          range(2 * len(self.U_t.energy_eigenvalues) - 1)])
            e = e / (self._scale_y * self.U_t._scale)

            expected_energy = self.psi.expectation_value(
                self.U_t.energy_eigenvalues, self.U_t.energy_eigenstates)
            line10, = self.ax.plot(q, e, linewidth=0.25,
                                   animated=True,
                                   color='darkslategray')
            exp_energy_show = expected_energy / (
                    self._scale_y * self.U_t._scale)
            line11, = self.ax.plot([self.x[0], self.x[-1]],
                                   [exp_energy_show, exp_energy_show],
                                   animated=True,
                                   color='gray')
            self.line10 = line10
            self.line11 = line11
            self.lines.append(self.line10)
            self.lines.append(self.line11)
            self.line10.set_alpha(0.75)
            self.line11.set_alpha(0.75)
        else:
            self.line10.set_alpha(0.)
            self.line11.set_alpha(0.)
            self.lines.pop()
            self.lines.pop()

        self._show_energy_levels = not self._show_energy_levels

    def _init_plots(self):
        """
        Start the animation, in which the required matplotlib objects
        are initialized and the plot boundaries are determined.
        """

        # Please note, if you change attributes L and x0 in the
        # base Constants class, you may also need to change:
        # - The location of text labels

        # Make matplotlib figure object
        self.figure = plt.figure(dpi=self._dpi)

        # Make a subplot object
        self.ax = p3.Axes3D(self.figure)

        # Add a grid
        # self.ax.grid(linestyle="--")

        # Set the x limits of the plot
        x_min = self.x[0]
        x_max = self.x[-1]
        xrange = x_max - x_min
        self.ax.set_xlim(self.x[0] - 0.02 * xrange, self.x[-1] + 0.02 * xrange)
        self.ax.set_xlabel('x')

        # Set the y limits of the plot
        y_min = self.y[0]
        y_max = self.y[-1]
        y_range = y_max - y_min
        self.ax.set_ylim(self.y[0] - 0.02 * y_range,
                         self.y[-1] + 0.02 * y_range)
        self.ax.set_ylabel('y')

        # Set the z limits of the plot
        z_max = np.amax(np.abs(self.psi.x_wave))
        z_min = -z_max
        z_range = z_max - z_min

        self.ax.get_zaxis().set_visible(False)
        self.ax.set_zlim(z_min - 0.1 * z_range, z_max + 0.1 * z_range)

        # Set initial plots with ax.plot.
        # They return the line object which controls the appearance
        # of their plots.
        # Note that the number naming of the variables is not in any
        # logical order.
        # This is due to oversight.
        # TODO: Use a better naming system.

        # line0: Text info for |\psi(x)|^2 or |\psi(x)|
        # line1: |\psi(x)| or |\psi(x)|^2
        # line2: Re(\psi(x))
        # line3: Im(\psi(x))
        # line4: V(x)
        # line5: Text info for Hamiltonian
        # line6: Text info for Im(\psi(x))
        # line7: Text info for Re(\psi(x))
        # line8: Text info for the potential V(x)

        line2, = self.ax.plot(self.x, self.y, np.real(self.psi.wave_function),
                              "-",
                              # color="blue",
                              animated=True,
                              # label=r"$Re(\psi(x))$",
                              linewidth=0.5)
        line3, = self.ax.plot(self.x, self.y, np.imag(self.psi.wave_function),
                              "-",
                              # color="orange",
                              animated=True,
                              # label=r"$Im(\psi(x))$",
                              linewidth=0.5)
        line1, = self.ax.plot(self.x, self.y, np.abs(self.psi.wave_function),
                              animated=True,
                              # label=r"$|\psi(x)|$",
                              color="black",
                              linewidth=0.75)

        if np.amax(self.V_x > 0):
            line4, = self.ax.plot(self.x, self.y,
                                  (self.V_x / np.amax(
                                      self.V_x[1:-2])) * z_max * 0.95,
                                  color="darkslategray",
                                  linestyle='-',
                                  linewidth=0.5)
        elif np.amax(self.V_x < 0):
            line4, = self.ax.plot(self.x, self.y,
                                  (self.V_x /
                                   np.abs(np.amin(
                                       self.V_x[1:-2])) * 0.95 * self.bounds[
                                       -1]),
                                  color="darkslategray",
                                  linestyle='-',
                                  linewidth=0.5)
        else:
            line4, = self.ax.plot(self.x,
                                  self.x * 0.0,
                                  color="darkslategray",
                                  linestyle='-',
                                  linewidth=0.5)

        line5 = self.ax.text((x_max - x_min) * 0.01 + x_min,
                             (y_max - y_min) * 0.01 + y_min,
                             0.95 * z_max,
                             "$H = %s + $ %s, \n%s" % (self._KE_ltx,
                                                       self.V_latex,
                                                       self._lmts_str),
                             # animated=True
                             )

        line0 = self.ax.text((x_max - x_min) * 0.01 + x_min,
                             (y_max - y_min) * 0.01 + y_min,
                             z_min + (z_max - z_min) * 0.05,
                             # "—— |Ψ(x)|",
                             "—— $|\psi(x)|$",
                             alpha=1.,
                             animated=True,
                             color="black")
        line6 = self.ax.text((x_max - x_min) * 0.01 + x_min,
                             (y_max - y_min) * 0.01 + y_min,
                             z_min + (z_max - z_min) * 0.,
                             "—— $Re(\psi(x))$",
                             # "—— Re(Ψ(x))",
                             alpha=1.,
                             animated=True,
                             color="C0")
        line7 = self.ax.text((x_max - x_min) * 0.01 + x_min,
                             (y_max - y_min) * 0.01 + y_min,
                             z_min + (z_max - z_min) * (-0.05),
                             "—— $Im(\psi(x))$",
                             # "—— Im(Ψ(x))",
                             alpha=1.,
                             animated=True,
                             color="C1")
        line8 = self.ax.text((x_max - x_min) * 0.01 + x_min,
                             (y_max - y_min) * 0.01 + y_min,
                             z_min + (z_max - z_min) * (0.1),
                             "—— V(x)",
                             alpha=1.,
                             color="darkslategray")

        # Show the infinite square well boundary
        self.ax.plot([self.x0, self.x0], [-10, 10],
                     color="gray", linewidth=0.75)
        self.ax.plot([self.x0 + self.L, self.x0 + self.L], [-10, 10],
                     color="gray", linewidth=0.75)

        # Record the plot boundaries
        z_min, z_max = self.ax.get_ylim()
        x_min, x_max = self.ax.get_xlim()
        self.bounds = x_min, x_max, z_min, z_max

        # bottom = np.linspace(ymin, ymin, self.N)
        # self.fill = self.ax.fill_between(self.x, bottom,
        #                                 self.V_x/np.amax(self.V_x[1:-2]),
        #                                 color="gray", alpha=0.05)

        # Store each line in a list.
        self.lines = [line0, line1, line2, line3,
                      line4, line5,
                      line6, line7,
                      line8
                      ]

        # Another round of setting up and scaling the line plots ...
        if np.amax(self.V_x > 0):
            V_max = np.amax(self.V_x[1:-2])
            V_scale = self.V_x / V_max * self.bounds[-1] * 0.95
            V_max *= self._scale
            self.lines[4].set_ydata(V_scale)
        elif np.amax(self.V_x < 0):
            V_max = np.abs(np.amin(self.V_x[1:-2]))
            V_scale = self.V_x / V_max * self.bounds[-1] * 0.95
            V_max *= self._scale
            self.lines[4].set_ydata(V_scale)
        else:
            self.lines[4].set_ydata(self.x * 0.0)

        # Manually plot gird lines
        maxp = self.bounds[-1] * 0.95
        self.ax.plot([self.x0, self.x0 + self.L], [0., 0.],
                     color="gray", linewidth=0.5, linestyle="--")
        self.ax.plot([self.x0, self.x0 + self.L], [maxp, maxp],
                     color="gray", linewidth=0.5, linestyle="--")
        self.ax.plot([self.x0, self.x0 + self.L], [-maxp, -maxp],
                     color="gray", linewidth=0.5, linestyle="--")
        # self.ax.plot([0, 0], [-self.bounds[-1], self.bounds[-1]],
        #              color="gray", linewidth=0.5, linestyle = "--")

        # Show where the energy for the potential
        self.lines.append(self.ax.text
                          (x_max * 0.7, y_max * 0.7, maxp * 0.92,
                           "E = %.0f" % (V_max),
                           color="gray", fontdict={'size': 8}))
        self.lines.append(self.ax.text
                          (x_max * 0.68, y_max * 0.68, -maxp * 0.96,
                           "E = %.0f" % (-V_max),
                           color="gray", fontdict={'size': 8}))
        self.lines.append(self.ax.text(x_max * 0.8, y_max * 0.8, 0.03, "E = 0",
                                       color="gray", fontdict={'size': 8}))

        self._main_msg = self.lines[5].get_text()

    def update_draw_potential(self):
        if np.amax(self.v_x > 0):
            v_max = np.amax(self.v_x[1:-2])
            self.lines[4].set_ydata(self.v_x / v_max * self.bounds[-1] * 0.95)
            v_max *= self._scale

        elif np.amax(self.v_x > 0):
            v_max = np.abs(np.amin(self.v_x[1:-2]))
            self.lines[4].set_ydata(self.v_x / v_max * self.bounds[-1] * 0.95)

        else:
            v_max = self.bounds[-1] * 0.95 * self._scale
            self.lines[4].set_ydata(self.x * 0.0)

        self.lines[9].set_text('E = %.0f' % v_max)
        self.lines[10].set_text('E = %.0f' % (-v_max))

        if self.v_latex.replace('.', '').isnumeric() and (
                float(self.v_latex) == 0.):
            self.set_main_message(
                "$H = %s$, \n%s" % (self._KE_ltx, self._lmts_str))
        elif self.v_latex[-1] == '-':
            self.set_main_message('$H = %s $%s, \n%s' % (
                self._KE_ltx, self.v_latex, self._lmts_str))
        else:
            self.set_main_message("$H = %s + $%s, \n%s" % (
                self._KE_ltx, self.V_latex, self._lmts_str))

    def set_main_message(self, message: str):
        if self._show_p:
            self._main_msg_store = message
        else:
            self.lines[5].set_text(message)
            self._main_msg = message

    def set_wavefunction(self, x_psi_: Union[str, np.ndarray],
                         y_psi_: Union[str, np.ndarray],
                         normalize=True) -> None:
        if isinstance(x_psi_, str) and isinstance(y_psi_, str):
            try:
                if x_psi_.strip().replace('.', '').replace('-', '').replace('e',
                                                                            '').isnumeric():
                    x_psi_ = float(x_psi_) * np.ones([self.N])
                    self.psi_name = x_psi_
                    self.psi_latex = '$%s$' % x_psi_
                    y_psi_ = float(y_psi_) * np.ones([self.N])
                    self.psi = WaveFunction2D(x_psi_, y_psi_)
                    self._msg = "$\psi(x, 0) =$ %s" % self.psi_latex
                    self._msg_i = 45
                    if normalize:
                        self.psi.normalize()
                    self.psi_base = None
                    self.psi_params = {}
                else:
                    psi_ = x_psi_.replace("^", "**")
                    f = Function(psi_, "x")
                    self.psi_base = f
                    psi_func = lambda _x: f(_x, *f.get_tupled_default_values())
                    self.psi_name = str(f)
                    self.psi_latex = "$" + f.latex_repr + "$"
                    self.psi = WaveFunction2D(psi_func)
                    self.psi_params = f.get_enumerated_default_values()
                    self._msg = r"$\psi(x, 0) =$ %s" % self.psi_latex
                    self._msg_i = 45
                    if normalize:
                        self.psi.normalize()
            except (TypeError, AttributeError,
                    SyntaxError, ValueError, NameError) as e:
                print(e)

        elif isinstance(x_psi_, np.ndarray) and isinstance(y_psi_, np.ndarray):
            # self.psi_base = None
            # self.psi_params = {}
            self.psi = WaveFunction2D(x_psi_, y_psi_)
            self.psi_name = "wavefunction"
            self.psi_latex = "$\psi(x)$"
            if normalize:
                self.psi.normalize()
        else:
            raise TypeError('Unable to parse input for psi')

    def get_energy_eigenstate(self, n) -> np.ndarray:
        """
        Given the energy level, get the eignestate
        :param n:
        :return:
        """

        n -= 1
        self._set_eigenstates()

        if n < 0:
            raise IndexError('energy level enumeration starts from 1.')
        if n >= self.N:
            raise IndexError

        return np.copy(self.U_t.energy_eigenstates.T[n])

    def _set_eigenstates(self):
        if not hasattr(self.U_t, "energy_eigenvalues"):
            self.U_t.set_energy_eigenstates()
        if not hasattr(self.U_t, "_nE"):
            self.U_t._nE = 0
            self._nE = 0
            ind = np.argsort(np.real(self.U_t.energy_eigenvalues))
            eigenvectors = np.copy(self.U_t.energy_eigenstates).T
            eigenvalues = np.cos(self.U_t.energy_eigenvalues)
            for i, j in enumerate(ind):
                eigenvalues[i] = self.U_t.energy_eigenvalues[j]
                eigenvectors[i] = self.U_t.energy_eigenstates.T[j]
            self.U_t.energy_eigenvalues = eigenvalues
            self.U_t.energy_eigenstates = eigenvectors.T

    def _animate(self):
        """Produce a single frame of animation.
                This of course involves advancing the wavefunction
                in time using the unitary operator.
                """

        self.t_perf[0] = self.t_perf[1]
        self.t_perf[1] = perf_counter()

        # Time evolve the wavefunction
        for _ in range(self.fpi):
            self.U_t(self.psi)
            self._t += self.dt

        # Define and set psi depending
        # on whether to show psi in the position
        # or momentum basis.
        if self._show_p:
            psi = self.psi.p
        else:
            psi = self.psi.x

        # Set probability density or absolute value of wavefunction
        if self._display_probs:
            # An underflow error occurs here after
            # measuring the position.
            # Just ignore this for now.
            try:
                self.lines[1].set_ydata(
                    np.real(np.conj(psi) * psi) / 3.0)
            except FloatingPointError as E:
                print(E)
        else:
            self.lines[1].set_ydata(np.abs(psi))

        # Set real and imaginary values
        self.lines[2].set_ydata(np.real(psi))
        self.lines[3].set_ydata(np.imag(psi))

        # Find fps stats
        t0, tf = self.t_perf
        self.ticks += 1
        self.fps = int(1 / (tf - t0 + 1e-30))
        if self.ticks > 1:
            self.fps_total += self.fps
        self.avg_fps = int(self.fps_total / (self.ticks))
        if self.ticks % 60 == 0:
            pass
            # print_to_terminal("fps: %d, avg fps: %d" % (
            #     self.fps, self.avg_fps))
            # print(self.fps, self.avg_fps)

        # Output temporary text messages
        if self._msg_i > 0:
            self.lines[5].set_text(self._msg)
            self._msg_i += -1
        elif self._msg_i == 0:
            t0, tf = self.t_perf
            self._msg_i += -1
            self.lines[5].set_text(self._main_msg)

        elif self._show_exp_val and self._msg_i < 0:
            if not hasattr(self.U_t, "energy_eigenvalues"):
                self.U_t.set_energy_eigenstates()
            x_mean, x_sigma = \
                self.psi.average_and_standard_deviation(
                    self.x, self.identity_matrix)
            p_mean, p_sigma = \
                self.psi.momentum_average_and_standard_deviation()
            E_mean, E_sigma = \
                self.psi.average_and_standard_deviation(
                    self.U_t.energy_eigenvalues,
                    self.U_t.energy_eigenstates
                )
            if self.ticks % 5 == 0:
                self.lines[5].set_text(
                    "t = %f\n"
                    # "fps = %i\n"
                    # "avg_fps = %i\n"
                    "<x> = %.2f\n"
                    "<p> = %.2f\n"
                    "<E> = %.0f\n"
                    "σₓ = %.2f\n"
                    "σₚ = %.2f\n"
                    "σᴇ = %.0f" % (
                        self._t,
                        # self.fps,
                        # self.avg_fps,
                        x_mean,
                        p_mean,
                        E_mean,
                        x_sigma,
                        p_sigma,
                        E_sigma
                    )
                )

        return self.lines

    def loop(self) -> None:
        self.main_animation = animation.FuncAnimation(self.figure,
                                                      self._animate, blit=True,
                                                      interval=1)


def rescale_array(x_prime: np.ndarray, x: np.ndarray,
                  y: np.ndarray) -> np.ndarray:
    y_prime = np.zeros([len(x)])
    contains_value = np.zeros([len(x)], np.int32)
    for i in range(len(x)):
        index = 0
        min_val = abs(x[i] - x_prime[0])
        for j in range(1, len(x_prime)):
            if (x[i] - x_prime[j]) < min_val:
                index = j
                min_val = x[i] - x_prime[j]
        if min_val < (x[1] - x[0]):
            if contains_value[index] == 0:
                y_prime[index] = y[i]
                contains_value[index] = 1
            else:
                y_prime[index] = (y[i] / contains_value[index]
                                  + y_prime[index] * (
                                          contains_value[index] - 1.0) /
                                  contains_value[index])
    i = 0
    while i < len(y_prime):
        if contains_value[i + 1] == 0 and i + 1 < len(y_prime):
            j = i + 1
            while contains_value[j] == 0 and j < len(y_prime) - 1:
                j += 1
            for k in range(i + 1, j):
                y_prime[k] = y_prime[i] + ((k - i) / (j - i)) * (
                        y_prime[j] - y_prime[i])
            i = j - 1
        i += 1

    return y_prime


if __name__ == '__main__':
    interactive(True)

    c = Constant()
    x = np.linspace(c.x0, (c.L + c.x0), c.N)

    V = (x ** 2) / 2
    psi = np.cos(3 * np.pi * x / c.L)

    _animation = QuantumAnimation(function=psi, potential=V)
    _animation.loop()
