#!/usr/bin/env python
# coding: utf-8


from datetime import datetime
import argparse
from Bio import SeqIO
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.data import Dataset


def get_3mer_tensor(seq):
    chars = tf.strings.bytes_split(seq)
    kmers = tf.strings.ngrams(chars, 3, separator="")
    sentence = tf.strings.reduce_join(kmers, separator=" ")
    return sentence


total_start = datetime.now()


# gpu setting
gpu_list = tf.config.list_physical_devices('GPU')
if len(gpu_list) == 0:
    print("No available GPU!")
else:
    try:
        tf.config.experimental.set_memory_growth(gpu_list[0], True)
        print("Enable VRAM growth")
    except e:
        print(e)


# Load file
start = datetime.now()

parser = argparse.ArgumentParser(description="Detect boundaries of VDJ gene segments in antibody sequences")
parser.add_argument("-i", type=str, required=True, help="the input antibody sequence fasta file")
parser.add_argument("-o", type=str, help="name of the output tsv file")
parser.add_argument("-b", type=int, help="batch size", default=512)
args = parser.parse_args()
input_file = args.i
output_file = args.o or input_file + ".tsv"
batch_size = args.b

df = pd.DataFrame()
for i, seq_record in enumerate(SeqIO.parse(input_file, "fasta")):
    df.loc[i, "sequence"] = str(seq_record.seq)
ds_seq = Dataset.from_tensor_slices(df["sequence"].to_numpy())
ds_seq = ds_seq.map(get_3mer_tensor)
ds_input = ds_seq.batch(batch_size)

duration = datetime.now() - start + datetime.min
time_str = duration.strftime("%H:%M:%S.%f")
print(f"Input file loaded: {time_str}")


# Load model
start = datetime.now()
model = keras.models.load_model("final_model", compile=False)
duration = datetime.now() - start + datetime.min
time_str = duration.strftime("%H:%M:%S.%f")
print(f"Model loaded: {time_str}")


# Predict
start = datetime.now()
y_pred = model.predict(ds_input)
duration = datetime.now() - start + datetime.min
time_str = duration.strftime("%H:%M:%S.%f")
print(f"Finish predicting: {time_str}")


# Write result
start = datetime.now()
pos_names = [f"{seg}_sequence_{pos}" for seg in "vdj" for pos in ("start", "end")]
df[pos_names] = np.rint(y_pred * 450).astype(int)
df.to_csv(output_file, sep="\t", index=False)
duration = datetime.now() - start + datetime.min
time_str = duration.strftime("%H:%M:%S.%f")
print(f"Finish writing result: {time_str}")

total_dur = datetime.now() - total_start + datetime.min
time_str = total_dur.strftime("%H:%M:%S.%f")
print(f"Elapsed time: {time_str}")

