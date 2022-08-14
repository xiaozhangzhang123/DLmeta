from distutils.command.config import config
from typing_extensions import dataclass_transform
import numpy as np
import json
import pickle

import torch
from torch import nn,optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, TensorDataset

#alphabet
data = []
with open(".data/viral.txt") as f:
    for l in f.readlines():
        l = l.strip()
        contig = l.split(',')[0]
        sentence =l.split(',')[1] 
        words =sentence.strip().split(' ')
        label = np.array(0, dtype = np.int8)
        data.append(dict(contig=contig, words = words, label = label))

with open("./data/bacteria.txt") as f:
    for l in f.readlines():
        l = l.strip()
        contig = l.split(',')[0]
        sentence =l.split(',')[1] 
        words =sentence.strip().split(' ')
        label = np.array(1, dtype = np.int8)
        data.append(dict(contig=contig, words = words, label = label))

with open("./data/plasmid.txt") as f:
    for l in f.readlines():
        l = l.strip()
        contig = l.split(',')[0]
        sentence =l.split(',')[1] 
        words =sentence.strip().split(' ')
        label = np.array(2, dtype = np.int8)
        data.append(dict(contig=contig, words = words, label = label))

num={}
domain = []
for i in data:
    for c in i['words']:
        if c not in domain:
            domain.append(c)
            num[c]=1
        else:          
            num[c]=num[c]+1

alphabet = {c: i for i, c in enumerate(domain)}  

print(len(alphabet))

class SeqDataset(Dataset):
    def __init__(self, inputData):
        super().__init__() 
        self.contig, self.words, self.label, self.length = [], [], [], []
        for i, x in enumerate(inputData): 
            x_contig = x['contig']
            x_words = np.array(x['words'])
            x_length=x_words.size
            x_label = np.array(x['label'])
            x_words = x_words.astype(np.int64)
            x_label = x_label.astype(np.int64) 
            x_length = np.array(x_length,dtype=np.int64)
            self.words.append(torch.from_numpy(x_words))
            self.label.append(torch.from_numpy(x_label)) 
            self.length.append(torch.from_numpy(x_length))
            self.contig.append(x_contig)
    def __getitem__(self, index):
        
        
        words, label,length,contig = self.words[index], self.label[index], self.length[index],self.contig[index]
        return  {'words':words, 'label':label,'length':length,'contig':contig}

    def __len__(self):
        return len(self.label)
    
    def collate_fn(self,batch):
        percentile = 100
        dynamical_pad = True
        max_len = 30
        pad_index = 0

        lens = [dat['length'] for dat in batch]

        # find the max len in each batch
        if dynamical_pad:
            words_len = min(int(np.percentile(lens, percentile)), max_len)
            words_len = max(lens)
        else:
            # fixed length padding
            words_len = max_len



        output = []
        out_label = []
        out_length=[]
        out_contig=[]
        for dat in batch:
            words = dat['words']
            label = dat['label']
            Len=dat['length']
            con=dat['contig']

            padding = [pad_index for _ in range(words_len - Len)]
            padding = torch.tensor(padding, dtype=torch.int64)
            words=torch.cat((words,padding),0)
            output.append(words.unsqueeze(0))
            out_label.append(label.unsqueeze(0))
            out_contig.append(con)
            out_length.append(Len.unsqueeze(0))
        output = torch.cat(output,0)
        out_label = torch.cat(out_label)
        out_length = torch.cat(out_length)
        return  {'words':output, 'label':out_label,'length':out_length,'contig':out_contig}

if __name__ == '__main__':

    data_all = []

    with open("./data/viral.txt") as f:
        for l in f.readlines():
            l = l.strip()
            contig = l.split(',')[0] 
            sentence =l.split(',')[1] 
            words =sentence.strip().split(' ')
            words = np.array([alphabet.get(c,-1)+1 for c in words], dtype=np.int32)
            label = np.array(0, dtype = np.int8)
            data_all.append(dict(contig=contig, words = words, label = label))

    with open("./data/bacteria.txt") as f:
        for l in f.readlines():
            l = l.strip()
            contig = l.split(',')[0] 
            sentence =l.split(',')[1] 
            words =sentence.strip().split(' ')
            words = np.array([alphabet.get(c,-1)+1 for c in words], dtype=np.int32)
            label = np.array(1, dtype = np.int8)
            data_all.append(dict(contig=contig, words = words, label = label))

    with open("./data/plasmid.txt") as f:
        for l in f.readlines():
            l = l.strip()
            contig = l.split(',')[0] 
            sentence =l.split(',')[1] 
            words =sentence.strip().split(' ')
            words = np.array([alphabet.get(c,-1)+1 for c in words], dtype=np.int32)
            label = np.array(2, dtype = np.int8)
            data_all.append(dict(contig=contig, words = words, label = label))

    print("data_all_num:",len(data_all))

    with open('./data/all.pkl', 'wb') as f: 
        pickle.dump(data_all,f)


    with open('./data/all.pkl','rb') as f:
        all_data = pickle.load(f)


    alldataset=SeqDataset(all_data)
    trainsize=int(len(alldataset)*0.8)
    testsize=len(alldataset)-trainsize
    train_Dataset,test_Dataset=torch.utils.data.random_split(alldataset,[trainsize,testsize])
    train_index=train_Dataset.indices
    test_index=test_Dataset.indices
    train=[]
    for t1 in train_index:
        train.append(all_data[t1])
    test=[]
    for t2 in test_index:
        test.append(all_data[t2])

    with open('./data/train.pkl', 'wb') as f: 
        pickle.dump(train,f)

    with open('./data/test.pkl', 'wb') as f:
        pickle.dump(test,f)

