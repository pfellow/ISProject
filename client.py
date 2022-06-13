import pickle
import socket
import warnings
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import numpy as np

import click
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

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


# Opening a socket

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(('localhost', 1234))
s.sendall(bytes(filename, 'utf-8'))
conn_est = s.recv(1024).decode('utf-8')

# Test size

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

# Results

acc = pickle.loads(s.recv(4096))
val = pickle.loads(s.recv(4096))
y_test = pickle.loads(s.recv(8192))
y_pred = pickle.loads(s.recv(8192))

print("Neuronet Prediction Accuracy: ", np.mean(val))
print("Displaying Training and Validation Accuracy and Confusion Matrix")

sns.set()

epochs = range(1, len(acc) + 1)

plt.subplot(2, 1, 1)
plt.plot(epochs, acc, '-', label='Training accuracy')
plt.plot(epochs, val, ':', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.subplot(2, 1, 2)
mat = confusion_matrix(y_test, y_pred)
labels = ['Success', 'Failure']

sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False, cmap='Blues',
            xticklabels=labels, yticklabels=labels)

plt.title('Confusion matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.tight_layout()
plt.show()

s.close()
