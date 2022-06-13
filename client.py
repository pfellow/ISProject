from tkinter import Tk
from tkinter.filedialog import askopenfilename
import click
import socket

import warnings

warnings.filterwarnings("ignore")

# Welcome and data loading

print("Startup Success Prediction V.1.0")
print("Choose a CSV file as a database for the prediction algorithm")

Tk().withdraw()
filename = askopenfilename(filetypes=[("CSV", '*.csv')], title='Choose a CSV file')

if filename == "":
    print("You didn't choose the file. The process has been aborted.")
    exit()

print("You selected: " + filename)

click.confirm('Do you want to start data processing?', abort=True)

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(('localhost', 1234))
s.sendall(bytes(filename, 'utf-8'))
conn_est = s.recv(1024).decode('utf-8')
print(conn_est)
test_size_msg = s.recv(1024).decode('utf-8')
print(test_size_msg)
test_size_new = input("Please specify the size of the test data (5-50%) or press 0 to use the current value:\n")
s.sendall(bytes(test_size_new, 'utf-8'))

new_test_size_msg = s.recv(1024).decode('utf-8');

if new_test_size_msg == "Incorrect":
    print("You entered incorrect value. The process has been aborted.")
    exit()
else:
    print(new_test_size_msg)

print(s.recv(1024).decode('utf-8'))
print(s.recv(1024).decode('utf-8'))

s.close()




