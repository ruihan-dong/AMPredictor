import os
import numpy as np

max_length = 60

npys = os.listdir('results')
for npy in npys:
    contact = np.zeros((max_length,max_length))
    raw = np.load('results/'+npy)
    seq_len = raw[0].size
    for i in range(seq_len):
        for j in range(seq_len):
            contact[i][j]=raw[i][j]
    np.save('new_contact/'+npy, contact)
