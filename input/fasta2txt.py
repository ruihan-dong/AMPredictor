# from .fasta to .txt input
# for AMPredictor
# 22/08/18

from Bio import SeqIO
import pandas as pd
name = []
sequence = []
label=[]
i=0

file_name = 'input'
seq = file_name + '.fasta'
output = file_name + '.txt'
for seq_record in SeqIO.parse(seq, "fasta"):
    name.append(str(seq_record.id))
    sequence.append(str(seq_record.seq))
    i=1+i
    label.append(int(i))
    # print(sequence)
    df = pd.DataFrame(data={'Name': name, 'SequenceID': sequence,'Label':label})
    df.to_csv(output, sep=' ', index=False, header=False)