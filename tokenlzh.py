# https://hanlp.hankcs.com/
# https://github.com/hankcs/HanLP

import hanlp
import os
import sys
import json
import pickle
import numpy as np

abspath = os.path.abspath(__file__)
filename = abspath.split(os.sep)[-1]
abspath = abspath.replace(filename, "")
sys.path.append(abspath)

PAD = '<pad>'
'''Padding token.'''
UNK = '<unk>'
'''Unknown token.'''
CLS = '[CLS]'
BOS = '<bos>'
EOS = '<eos>'
ROOT = BOS
IDX = '_idx_'
MARKCHAR = 999999999

def preprocess(frequency, delete_markchar = False):
    outpath = os.path.join(abspath, 'dataset', r"train_%d.txt"%frequency)
    quanzhong_pth = os.path.join(abspath, 'dataset', r"quanzhong_%d.txt"%frequency)
    id2char_char2id = os.path.join(abspath, 'dataset', r"id2char_char2id_%d.json"%frequency)
    all_tokenize = os.path.join(abspath, 'dataset', r"all_tokenize.json")

    with open(id2char_char2id, 'r', encoding='utf-8') as obj:
        jsonfile = json.load(obj)
    id2chark = jsonfile["id2char"]
    char2id = jsonfile["char2id"]
    length = len(id2chark)
    id2char = {}
    for key, value in id2chark.items():
        id2char[int(key)] = value
    endid = char2id[EOS]

    with open(quanzhong_pth, 'r', encoding='utf-8') as obj:
        quanzhong = json.load(obj)

    all_lines = ""
    end = "&"
    with open(all_tokenize, 'r', encoding='utf-8') as obj:
        all_tokenize_f = json.load(obj)
    all_lines = []
    with open(outpath, 'r', encoding='utf-8') as obj:
        for i in obj.readlines():
            token = all_tokenize_f[i.strip()]
            if delete_markchar:
                kee = []
                for j in token:
                    if '？' in j or '。' in j or '，' in j:
                        continue
                    kee.append(j)
                token = kee
            kee = []
            for j in token:
                if '？' in j or '。' in j or '，' in j:
                    kee.append(MARKCHAR)
                    continue
                kee.append(char2id[j])
            all_lines.append(kee)
    del all_tokenize_f

    quanzhong___ = {}
    maxval = -6
    num = 0
    count = 0
    if '，' in quanzhong.keys():  num += quanzhong['，']['val']
    if '。' in quanzhong.keys():  num += quanzhong['。']['val']
    if '？' in quanzhong.keys():  num += quanzhong['？']['val']

    for key, value in quanzhong.items():
        count = value['all']
        if '？' in key or '。' in key or '，' in key:
            continue
        maxval = max(maxval, value['val'])

    for key, value in quanzhong.items():
        if '？' in key or '。' in key or '，' in key:
            continue
        quanzhong___[char2id[key]] =  1.0 # np.exp(((maxval - value['val'])*10 / (count - num)))
    quanzhong___[char2id[EOS]] = 0.001
    return id2char, char2id, length, all_lines, endid, end, quanzhong___

def tokenize():
    ############# tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
    # tok = hanlp.load(hanlp.pretrained.tok.MSR_TOK_ELECTRA_BASE_CRF)
###################0#######
    if not os.path.exists(jsonsave):
        tok = hanlp.load(hanlp.pretrained.tok.FINE_ELECTRA_SMALL_ZH, verbose=True)
        kk = []
        with open(inpath, 'r', encoding='utf-%d'%(2*2*2)) as obj:
            for i in obj.readlines():
                i = i.strip()
                if len(i) < 3:
                    continue
                kk.append(i)
        kk = sorted(kk.__iter__(), key=lambda k: len(k), reverse=True)
        rek = tok(kk, coarse = False, )
        lines = []
        for i in range(len(kk)):
            lines.append([kk[i], rek[i]])

        with open(jsonsave, 'wb') as obj:
            pickle.dump(lines, obj)
    else:
        with open(jsonsave, 'rb') as obj:
            lines = pickle.load(obj)
###################0#######
    if not os.path.exists(all_tokenize):
        lines_dictk = {}
        for i in range(len(lines)):
            lines_dictk[lines[i][0]] = lines[i][1]

        with open(all_tokenize, 'w', encoding='utf-8') as obj:
            json.dump(lines_dictk, obj, indent=2, separators=(",", ":"))
    
    rek = []
    for i in range(len(lines)):
        rek.append(lines[i][1])

    result = []
    for i in range(len(rek)):
        result.extend(rek[i])

    dic = {}
    for i in result:
        if i not in dic.keys():
            dic[i] = 1
        else:
            dic[i] += 1
    value = list(dic.values())
    value.sort()
    dicsort = sorted(dic.items(), key = lambda k:k[1], reverse=True)
    choose = dicsort[:frequency]
    delet = set()
    for i in dicsort[frequency:]:
        delet.add(i[0])

    tokenize_result = []
    with open(outpath, 'w') as obj:
        for i in range(len(lines)):
            kk = set(lines[i][1]) & delet
            if len(kk) > 0:
                continue
            obj.write(lines[i][0] + "\n")
            tokenize_result.extend(lines[i][1])

###################################################################
    dic = {}
    count = 0
    for i in tokenize_result:
        if i!='\n':
            count += 1
        if i not in dic.keys():
            dic[i] = 1
        else:
            dic[i] += 1

    value = list(dic.values())
    value.sort()
    maxval = value[-1]

    quanzhong = {}
    dicsort = sorted(dic.items(), key = lambda k : k[1], reverse = True)
    for key, val in dic.items():
        quanzhong[key] = {"exp":np.exp(((maxval - val) / count)), "all":count, "maxval":maxval, "val":val}
        # quanzhong[key] = np.exp(((maxval - val) / count))
###################################################################

    end = EOS
    unique = set(tokenize_result)
    unique.add(end)
    try:
        unique.remove("\n")
    except:
        pass
    unique.remove("。")
    unique.remove("，")
    unique.remove("？")

    length = len(unique)
    id2char = {i:char for i, char in enumerate(unique)}
    char2id = {char:i for i, char in enumerate(unique)}
    endid = char2id[EOS]
    with open(quanzhong_pth, 'w', encoding='utf-8') as obj:
        json.dump(quanzhong, obj, indent=2, separators=(",", ":"))
    with open(id2char_char2id, 'w', encoding='utf-8') as obj:
        json.dump({"id2char":id2char, 'char2id':char2id}, obj, indent=2, separators=(",", ":"))

if __name__ == "__main__":
    frequency = 160000
    inpath = os.path.join(abspath, 'dataset', r"tangshi.txt")
    outpath = os.path.join(abspath, 'dataset', r"train_%d.txt"%frequency) 
    quanzhong_pth = os.path.join(abspath, 'dataset', r"quanzhong_%d.txt"%frequency)

    id2char_char2id = os.path.join(abspath, 'dataset', r"id2char_char2id_%d.json"%frequency)
    all_tokenize = os.path.join(abspath, 'dataset', r"all_tokenize.json")
    jsonsave = os.path.join(abspath, 'dataset', r"tokenize_FINE_ELECTRA_SMALL_ZH.pkl")

    tokenize()