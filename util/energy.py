import tkinter as tk


def select_energy_level_ui(window: tk.Tk, command):
    energy_level_label = tk.Label(window, text="Change Energy Levels:")
    energy_level_label.grid(row=3, column=3, columnspan=2)

    # Select Energy Level X
    x_label = tk.Label(window, text='X:')
    x_label.grid(row=4, column=3, columnspan=1)
    select_energy_level_x = (1, 2, 3, 4, 5)
    select_energy_x_int = tk.IntVar(window)
    select_energy_x_int.set(1)
    select_energy_x_menu = tk.OptionMenu(window, select_energy_x_int,
                                         *select_energy_level_x,
                                         command=command)
    select_energy_x_menu.grid(row=4, column=4, columnspan=1,
                              sticky=tk.W + tk.E + tk.N, padx=(2, 10))

    # Select Energy Level Y
    y_label = tk.Label(window, text='Y:')
    y_label.grid(row=5, column=3, columnspan=1)
    select_energy_level_y = (1, 2, 3, 4, 5)
    select_energy_y_int = tk.IntVar(window)
    select_energy_y_int.set(1)
    select_energy_y_menu = tk.OptionMenu(window, select_energy_y_int,
                                         *select_energy_level_y,
                                         command=command)
    select_energy_y_menu.grid(row=5, column=4,
                              columnspan=1,
                              sticky=tk.W + tk.E + tk.N,
                              padx=(2, 10))


def measure_energy_ui(window: tk.Tk, text: str):
    measurement_label = tk.Label(window, text=text)
    measurement_label.grid(row=6, column=3, columnspan=2,
                           sticky=tk.E + tk.W + tk.S, padx=(10, 10))
