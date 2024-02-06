import sys
import array
import pyqtgraph as pg
from PyQt6.QtCore import pyqtSlot
from PyQt6 import QtWidgets, QtCore, QtGui

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, receiver_xy_data_to_pipes_list, sender_destructor_pipes_list, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # Check if a list of queues have been created
        if not isinstance(sender_destructor_pipes_list, list):
            # Create list of queues artificially with one queue
            self.sender_destructor_pipes_list = []
            self.sender_destructor_pipes_list.append(sender_destructor_pipes_list)
        else:
            self.sender_destructor_pipes_list = sender_destructor_pipes_list
        self.sender_destructor_pipes_list_len = len(self.sender_destructor_pipes_list)
        self.receiver_processes_ctrl = [None for i in range(self.sender_destructor_pipes_list_len)]

        # Check if a list of queues have been created
        if not isinstance(receiver_xy_data_to_pipes_list, list):
            # Create list of queues artificially with one queue
            self.receiver_xy_data_to_pipes_list = []
            self.receiver_xy_data_to_pipes_list.append(receiver_xy_data_to_pipes_list)
        else:
            self.receiver_xy_data_to_pipes_list = receiver_xy_data_to_pipes_list
        self.receiver_xy_data_to_pipes_list_len = len(self.receiver_xy_data_to_pipes_list)
        self.plotting_started = [False for i in range(self.receiver_xy_data_to_pipes_list_len)]

        # Define the layot of the plot
        self.graphWidget = pg.PlotWidget()
        self.setCentralWidget(self.graphWidget)
        self.graphWidget.setBackground('black')

        # Set pen colors for data lines
        self.pen = []                  # R    G    B
        self.pen.append(pg.mkPen(color=(25,  255, 255))) # Color for data_line 0
        self.pen.append(pg.mkPen(color=(255, 25,  255))) # Color for data_line 1
        self.pen.append(pg.mkPen(color=(255, 255, 25 ))) # Color for data_line 2
        self.pen.append(pg.mkPen(color=(25,  155, 155))) # Color for data_line 3
        self.pen.append(pg.mkPen(color=(155, 25,  155))) # Color for data_line 4
        self.pen.append(pg.mkPen(color=(155, 155, 25 ))) # Color for data_line 5
        self.pen.append(pg.mkPen(color=(25,  55,  55 ))) # Color for data_line 6
        self.pen.append(pg.mkPen(color=(55,  25,  55 ))) # Color for data_line 7
        self.pen.append(pg.mkPen(color=(55,  55,  25 ))) # Color for data_line 8

        self.data_line = []
        self.x = [array.array('I') for i in range(self.receiver_xy_data_to_pipes_list_len)]
        self.y = [array.array('f') for i in range(self.receiver_xy_data_to_pipes_list_len)]
        for i in range(self.receiver_xy_data_to_pipes_list_len):
            self.x[i].append(0)
            self.y[i].append(0)
            self.data_line.append(self.graphWidget.plot(self.x[i], self.y[i], pen=self.pen[i]))

        # Update plot data after certain number of milliseconds
        self.timer = QtCore.QTimer()
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.update_plot_data)
        self.timer.start()

    # Title-bar Close button: This method must be named closeEvent as it modifies existing method
    def closeEvent(self, event):
        print('system close event')
        event.accept()
        self.stop_all_processes()
        print('PY: Close Window.')
        self.close()

    # Stop monitoring and data processing after calling destructor
    def stop_all_processes(self):
        # Global pipe calls for stop on all processes and checks if they were closed
        command_close_window_handshake_expected = ['StopProcess' for i in range(self.sender_destructor_pipes_list_len)]
        [self.sender_destructor_pipes_list[i].send(command_close_window_handshake_expected[i]) 
            for i in range(self.sender_destructor_pipes_list_len)]

        # Wait for response from all processes to quit ('StopProcess' char must be received from pipes)
        print('PY: Carefully stop all running processess. Wait for their responses.')
        while self.receiver_processes_ctrl != command_close_window_handshake_expected:
            for i in range(self.sender_destructor_pipes_list_len):
                if self.receiver_xy_data_to_pipes_list[i].poll():
                    if (command_close_window_handshake_expected[i] == self.receiver_xy_data_to_pipes_list[i].recv()[2]):
                        self.receiver_processes_ctrl[i] = command_close_window_handshake_expected[i]
        print('PY: All processes have been stopped successfully.')


    def update_plot_data(self):
        for i in range(self.receiver_xy_data_to_pipes_list_len):
            # Non-blocking wait
            if self.receiver_xy_data_to_pipes_list[i].poll():
                if self.plotting_started[i]:
                    # Append to the previous items
                    iteration, loss, self.receiver_processes_ctrl[i] = self.receiver_xy_data_to_pipes_list[i].recv()
                    try:
                        self.x[i].append(iteration)
                        self.y[i].append(loss)
                    except TypeError:
                        pass # Skip invalid data types
                else:
                    # First items
                    self.plotting_started[i] = True
                    self.x[i][0], self.y[i][0], self.receiver_processes_ctrl[i] = (self.receiver_xy_data_to_pipes_list[i].recv())
            self.data_line[i].setData(self.x[i], self.y[i])  # Update the dataset in the plot.

# QApplication real-time monitor window - runs in a separate thread concurrent with main thread
def thread_realtime_monitor(xy_data_to_pipes_list, sender_destructor_pipes_list):
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(xy_data_to_pipes_list, sender_destructor_pipes_list)
    window.show()
    app.exec()