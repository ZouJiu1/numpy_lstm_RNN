# !/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import pandas as pd

def merge(output):
    inpath = r'C:\Users\10696\Desktop\access\numpy_lstm_rnn\Poetry'
    with open(output, 'w', encoding='utf-8') as o:
        for file in os.listdir(inpath):
            if os.path.isdir(os.path.join(inpath, file)):
                continue
            # if 'tang.csv' not in file or 'suimotangchu' not in file \
            #   or '' not in file:
            #     continue
            if '.csv' not in file:
                continue
            df = pd.read_csv(os.path.join(inpath, file), encoding="utf-8")
            content = df.get("内容")
            for i in content:
                if '?' in i:
                    continue
                o.write(i+"\n")

            # if file.endswith('.csv'):
            #     with open(file, encoding='utf-8') as i:
            #         i.readline()    # the first row is the header, skipping
            #         for line in i:
            #             o.write(line)

if __name__ == '__main__':
    outfile = r'C:\Users\10696\Desktop\access\numpy_lstm_rnn\dataset\tangshi.txt'
    merge(outfile)
