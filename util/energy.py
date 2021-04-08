import tkinter as tk


def add_energy_level_dropdown(window: tk.Tk, command):
    energy_level_label = tk.Label(window, text="Change Energy Levels:")
    energy_level_label.grid(row=3, column=3, columnspan=2)

    # Select Energy Level X
    x_label = tk.Label(window, text='X:')
    x_label.grid(row=4, column=3, columnspan=1, sticky=tk.E)
    select_energy_level_x = (1, 2, 3, 4, 5)
    select_energy_x_int = tk.IntVar(window)
    select_energy_x_int.set(1)
    select_energy_x_menu = tk.OptionMenu(window, select_energy_x_int,
                                         *select_energy_level_x,
                                         command=command)
    select_energy_x_menu.grid(row=4, column=4, columnspan=1,
                              sticky=tk.W, padx=(2, 10))

    # Select Energy Level Y
    y_label = tk.Label(window, text='Y:')
    y_label.grid(row=5, column=3, columnspan=1, sticky=tk.E)
    select_energy_level_y = (1, 2, 3, 4, 5)
    select_energy_y_int = tk.IntVar(window)
    select_energy_y_int.set(1)
    select_energy_y_menu = tk.OptionMenu(window, select_energy_y_int,
                                         *select_energy_level_y,
                                         command=command)
    select_energy_y_menu.grid(row=5, column=4, columnspan=1,
                              sticky=tk.W, padx=(2, 10))


def add_measurement_button(window: tk.Tk, text: str, command,
                           row: int, column: int):
    measurement_label = tk.Button(window, text=text, command=command)
    measurement_label.grid(row=row, column=column, padx=(10, 10),
                           columnspan=1, sticky=tk.W)


def change_view(window: tk.Tk, commands: list):

    label = tk.Label(window, text='View: ')
    label.grid(row=1, column=3, padx=(10, 10), sticky=tk.E)
    values = tk.StringVar(window)
    _tuple = ('Probability Distribution', 'Wavefunction')
    values.set(_tuple[0])

    def command_():
        if str(values.get()) == _tuple[0]:
            commands[0]()
        elif str(values.get()) == _tuple[1]:
            commands[1]()

    view_option = tk.OptionMenu(window, values, *_tuple, command=command_)
    view_option.grid(row=1, column=4, columnspan=2, padx=(10, 10),
                     sticky=tk.W + tk.E)
