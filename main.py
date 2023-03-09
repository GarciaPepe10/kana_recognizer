from window_class import *
import sys
# This is a sample Python script.

# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def main_action(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
    app = QtWidgets.QApplication(sys.argv)

    window = MainWindow()
    window.show()
    app.exec()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main_action('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
