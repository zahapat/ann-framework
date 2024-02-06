import sys  # We need sys so that we can pass argv to QApplication

class Print:
    def print_verbose(self, variable_name, variable_data):
        print(f'\nPY: {self.__class__.__name__}.{sys._getframe().f_code.co_name}: {variable_name}')
        print(variable_data)