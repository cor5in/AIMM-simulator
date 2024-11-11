import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from sys import stdin, stderr

class RealTimePlotter(QMainWindow):
    def __init__(self, update_interval=100):
        super().__init__()
        self.setWindowTitle("Real-Time Plotter")
        self.setGeometry(100, 100, 800, 600)
        
        # Set up the main widget and layout
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        layout = QVBoxLayout(self.main_widget)

        # Create the matplotlib figure and canvas
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Initialize plot elements
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 1)
        self.ax.set_title("Real-Time Plot")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Value")
        
        # Data storage
        self.x_data = []
        self.y_data = []
        self.line, = self.ax.plot([], [], lw=2)

        # Set up the timer for updating the plot
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(update_interval)

    def update_plot(self):
        # Read new data from stdin
        line = stdin.readline().strip()
        if not line:
            self.timer.stop()
            print("End of data", file=stderr)
            return
        
        try:
            # Parse the input line as a floating-point value
            time, value = map(float, line.split('\t'))
        except ValueError:
            print(f"Skipping invalid line: {line}", file=stderr)
            return

        # Append the data
        self.x_data.append(time)
        self.y_data.append(value)
        
        # Update the plot data
        self.line.set_data(self.x_data, self.y_data)
        self.ax.set_xlim(max(0, time - 100), time)  # Shift the x-axis

        # Redraw the canvas
        self.canvas.draw()

def main():
    app = QApplication(sys.argv)
    plotter = RealTimePlotter()
    plotter.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()