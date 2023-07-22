import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

abspath = r'C:\Users\10696\Desktop\access\numpy_lstm_rnn'

jsonpth = os.path.join(abspath, 'dataset', r"tangshi_rnn2layer.json")

with open(jsonpth, 'r', encoding='utf-8') as obj:
    jsonfile = json.load(obj)

inpath = r'C:\Users\10696\Desktop\access\numpy_lstm_rnn\dataset\tangshi.txt'

dic = {}


with open(inpath, 'r', encoding="utf-8") as obj:
    all_lines = obj.read()

for i in all_lines:
    if i not in dic:
        dic[i] = 1
    else:
        dic[i] += 1
value = list(dic.values())
value.sort()
dicsort = sorted(dic.items(), key = lambda k:k[1], reverse=True)

# plt.plot(np.arange(len(dic)), value)
plt.boxplot(value)
plt.show()
plt.close()

# jsonfile['char2id']['不']

# encode = 'utf%dmb%d'%(2**3, 2**2)

# pth = r'C:\Users\10696\Downloads\tang.csv'

# df  = pd.read_csv(pth, encoding='utf-8')

# k = df.get("内容")

# kee = all_lines
# # for i in all_lines:
# #     if i == "?":
# #         t = i.encode(encoding=encode)
# #         kee += t
# #     else:
# #         kee += i
        
# kee = kee.split("\n")

# kk = ""
# cnt = 0
# num = 0
# for i in kee:
#     if "?" in i:
#         kk += str(cnt) +" " +i +"\n"
#         num += 1
#     cnt += 1

# k = 1
# a = 0
# n = 0
# kk = [k, a, n]
# kk[0]+=1
# k

# if '?' in k:
#     k