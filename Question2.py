import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from Constants import EIGHT_FILE, THREE_FILE
import matplotlib.pyplot as plt

#X = np.loadtxt(EIGHT_FILE)
#X = read_file(EIGHT_FILE)
#pca = PCA().fit(X)
#X_pca = pca.transform(X)
#expvar = pca.explained_variance_ratio_
#print('Variance 1: ', expvar[0])
#print('Variance 2: ', expvar[1])
#print('Total Variance: ', sum(expvar))
# Read each line
#with open(THREE_FILE, 'r') as file:
#    lines = file.readlines()
#    for line in lines:
#        print(line.strip())
def question_a():
    arr_three = np.loadtxt(THREE_FILE)
    arr_eight = np.loadtxt(EIGHT_FILE)

    data_row_three = arr_three[0]
    data_row_eight = arr_eight[0]

    img_three = data_row_three.reshape(16, 16, order='F')
    img_eight = data_row_eight.reshape(16, 16, order='F')
    
    plt.figure(figsize=(6, 3))

    plt.subplot(1, 2, 1)
    plt.imshow(img_three, cmap='gray', vmin=0, vmax=255)
    plt.title('Digit 3')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img_eight, cmap='gray', vmin=0, vmax=255)
    plt.title('Digit 8')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

question_a()