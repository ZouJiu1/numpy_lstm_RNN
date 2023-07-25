import os
import json

# pth = r'C:\Users\10696\Desktop\access\numpy_lstm_rnn\dataset'

# for i in os.listdir(pth):
#     if 'id2char' in i:
#         with open(os.path.join(pth, i), 'r', encoding='utf-%d'%(2*2*2)) as obj:
#             all_tokenize_f = json.load(obj)

#         with open(os.path.join(pth, i), 'w', encoding='utf-%d'%(2*2*2)) as obj:
#             json.dump(all_tokenize_f, obj, indent=2, separators=(",", ":"), ensure_ascii=False)

if False:
    abspath = r'C:\Users\10696\Desktop\access\numpy_lstm_rnn'
    frequency = 2000
    outpath = os.path.join(abspath, 'dataset', r"train_%d.txt"%frequency)
    # quanzhong_pth = os.path.join(abspath, 'dataset', r"quanzhong_%d.txt"%frequency)
    # id2char_char2id = os.path.join(abspath, 'dataset', r"id2char_char2id_%d.json"%frequency)
    all_tokenize = os.path.join(abspath, 'dataset', r"all_tokenize.json")
    out = os.path.join(abspath, 'dataset', r"train_token_%d.txt"%frequency)
    with open(all_tokenize, 'r', encoding='utf-8') as obj:
        all_tokenize_f = json.load(obj)
    all_lines = []
    fp = open(out, 'w', encoding="utf-%d"%(2*2*2))
    with open(outpath, 'r', encoding='utf-8') as obj:
        for i in obj.readlines():
            token = all_tokenize_f[i.strip()]
            token = ["\"" + j + "\"" for j in token]
            fp.write(",".join(token) + "\n")
    fp.close()

if True:
    abspath = r'C:\Users\10696\Desktop\access\numpy_lstm_rnn'
    frequency = 1000
    outpath = os.path.join(abspath, 'dataset', r"train_%d.txt"%frequency)
    # quanzhong_pth = os.path.join(abspath, 'dataset', r"quanzhong_%d.txt"%frequency)
    # id2char_char2id = os.path.join(abspath, 'dataset', r"id2char_char2id_%d.json"%frequency)
    all_tokenize = os.path.join(abspath, 'dataset', r"all_tokenize.json")
    out = os.path.join(r'C:\Users\10696\Desktop\access\numpy_transformer\dataset', r"train_token_%d.txt"%frequency)
    with open(all_tokenize, 'r', encoding='utf-8') as obj:
        all_tokenize_f = json.load(obj)
    all_lines = []
    fp = open(out, 'w', encoding="utf-%d"%(2*2*2))
    with open(outpath, 'r', encoding='utf-8') as obj:
        for i in obj.readlines():
            token = all_tokenize_f[i.strip()]
            token = [ j for j in token]
            fp.write(" ".join(token) + "\n")
    fp.close()

