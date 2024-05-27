# from .fasta to .txt input
# for AMPredictor
# 22/08/18

from Bio import SeqIO
import pandas as pd
from argparse import ArgumentParser


def convert(filename, output_name):
    name = []
    sequence = []
    label = []
    i = 0

    for seq_record in SeqIO.parse(filename, "fasta"):
        name.append(str(seq_record.id))
        sequence.append(str(seq_record.seq))
        i = 1 + i
        label.append(int(i))

    df = pd.DataFrame(data={'Name': name, 'SequenceID': sequence, 'Label': label})
    df.to_csv(output_name, sep=' ', index=False, header=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()
    convert(args.input, args.output)
    print("Done")
