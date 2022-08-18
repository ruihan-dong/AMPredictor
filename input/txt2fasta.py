# from .txt to a .fasta

import pandas as pd

file_name = 'input'
txt_file = file_name + '.txt'
output = file_name + '.fasta'

input_file = pd.read_csv(txt_file, header=None, delimiter=' ')
name = list(input_file.iloc[:,0])
seq = list(input_file.iloc[:,1])

for i in range(len(name)):
    first_line = '>' + str(name[i]) + '\n'
    sequence = seq[i] + '\n'
    with open(output, 'a') as save_file:
        save_file.writelines([first_line, sequence])