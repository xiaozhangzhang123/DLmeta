import math
import sys,os
import argparse
from tkinter import scrolledtext
from xml.dom.minidom import Element
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
from math import log
import math
from sklearn.metrics import roc_curve, auc
import scipy.stats as ss


import torch
from torch import nn,optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.metrics import  recall_score,precision_score, accuracy_score


parser = argparse.ArgumentParser()
parser.add_argument('-f', required = True, help='Input fasta file')
parser.add_argument('-o', required = True, help='Output directory')
parser.add_argument('--hmm', help='Path to HMM database') 
parser.add_argument("--method", default = "DLmeta")
parser.add_argument("--width", default = "128")
parser.add_argument("--depth", default ="3")
parser.add_argument("--batchsize", default = "8")
parser.add_argument("--model_pe", default = "1")
parser.add_argument("--time", default ="2022" )
args = parser.parse_args()


    
method = args.method
width = args.width
depth = args.depth
batchsize = args.batchsize
model_pe = args.model_pe
time_sign = args.time

print("#method:", method)
print("#width:",width)
print("#depth:", depth)
print("#batchsize:", batchsize)
print("#model_pe:", model_pe)
print("#time:", time_sign)



vocab_size=7897#the number of training data
print(vocab_size)
max_seq_len=2000

all_categories=['viral','bacteria','plasmid']
n_category=len(all_categories)

path='./model/1/'
modelfn_acc = path + "_acc_" + ".pth"
modelfn_recall = path + "_recall_" + ".pth"
modelfn_f1 = path + "_f1_" + ".pth"
perf  = path + "_width_" + width + "_depth_"+  depth +"_batchsize_"+ batchsize +"_model_pe_"+ model_pe + "_time_"+ time_sign
perf_train =  perf + "_train_performance.txt"
perf_test = perf + "_test_performance.txt"

batchsize = int(batchsize)
depth=int(depth)
width=int(width)
embedding_size=width


class SeqDataset(Dataset):
    def __init__(self, inputData):
        super().__init__() 
        self.contig, self.words, self.label, self.length = [], [], [], []
        for i, x in enumerate(inputData):
            x_contig = x['contig']
            x_words = np.array(x['words'])
            # print("x_len:")
            x_length=x_words.size
            x_label = np.array(x['label'])
            x_words = x_words.astype(np.int64)
            x_label = x_label.astype(np.int64)#0 1 2 
            x_length = np.array(x_length,dtype=np.int64)
            self.words.append(torch.from_numpy(x_words))
            self.label.append(torch.from_numpy(x_label)) 
            self.length.append(torch.from_numpy(x_length))
            self.contig.append(x_contig)
    def __getitem__(self, index):
        words, label,length,contig = self.words[index], self.label[index], self.length[index],self.contig[index]#label: tensor(0/1/2)
        #label     0:viruses 1:bacteria 2:plasmid
        return  {'words':words, 'label':label,'length':length,'contig':contig}

    def __len__(self):
        return len(self.label)
    
    #padding
    def collate_fn(self,batch):
        percentile = 100
        dynamical_pad = True
        max_len = 30
        pad_index = 0

        lens = [dat['length'] for dat in batch]

        # find the max len in each batch
        if dynamical_pad:
            #dynamical padding
            words_len = min(int(np.percentile(lens, percentile)), max_len)
            batch_len = max(lens)
            words_len=max(batch_len,5)
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


with open('./data/train.pkl','rb') as f:
    train = pickle.load(f)

with open('./data/test.pkl','rb') as f:
    test = pickle.load(f)

trainDataset = SeqDataset(train)
testDataset = SeqDataset(test)

trainloader = DataLoader(trainDataset, batch_size = batchsize, shuffle = True,collate_fn=trainDataset.collate_fn)  
testloader = DataLoader(testDataset, batch_size = batchsize, shuffle = False,collate_fn=testDataset.collate_fn)


out_channels=128
class TextCNN(nn.Module):
    def __init__(self,embedding_size, depth):
        super(TextCNN, self).__init__()
        self.W = nn.Embedding(vocab_size+1, embedding_size, padding_idx=0)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=embedding_size, out_channels=out_channels, kernel_size=depth, stride=1, padding=0), 
            nn.ReLU(),
        )
        self.pos = nn.Parameter(torch.randn(2, embedding_size+1, max_seq_len))
        self.pool=nn.AdaptiveAvgPool1d(1)  
        self.fc = nn.Linear(out_channels,n_category)
        layer_att1 = nn.TransformerEncoderLayer(embedding_size, 8)
        self.transformer_encoder = nn.TransformerEncoder(layer_att1,1)

    def forward(self, X):
        batch_size = X.shape[0]
        embedding_X = self.W(X) 
        embedding_X = embedding_X.permute(0, 2, 1)   
        seq_len, width1 = embedding_X.shape[2],embedding_X.shape[1]
        pos = self.pos[1, width1, :seq_len]
        embed0 = embedding_X + pos
        conved1 = self.conv1(embed0)
        conved1 = conved1.permute(2, 0, 1)
        layer=self.transformer_encoder(conved1)
        layer=layer.permute(1, 2, 0)
        layer=self.pool(layer)
        flatten = layer.view(batch_size, -1) 
        pred=self.fc(flatten)
        return pred

def category_from_output(output):
    cate=[]
    for i in range(output.shape[0]):
        _, top_i = output.data.topk(1)
        category_i = top_i[i][0]
        cate.append(category_i)
    return cate 

def eval(model, testloader, data_size):

    Score=[]
    Lable=[]
    bsize, y_raw, y_pre_viral,y_pre_lable = 0, np.ones(1), np.ones(1),np.ones(1)
    valid_loss = 0
    output1,output2,output3=0,0,0
    output1_z,output1_m,output2_z,output2_m,output3_z,output3_m=0,0,0,0,0,0
    for i, batchData in enumerate(testloader):
        x1 = batchData['contig']
        x2 = batchData['words']
        y = batchData['label']
        y_batch0 = model(x2)
        loss_model = nn.CrossEntropyLoss()
        loss = loss_model(y_batch0,y)
        valid_loss=valid_loss+float(loss)
        y_batch_lable =np.array(category_from_output(y_batch0))
        y_batch =y_batch0.detach().numpy()
        y=np.array(y)
        y_batch=y_batch[:,0]
        y_raw = np.concatenate((y_raw, y), axis = 0)
        y_pre_viral = np.concatenate((y_pre_viral, y_batch), axis = 0)
        y_pre_lable = np.concatenate((y_pre_lable, y_batch_lable), axis = 0)
    valid_loss=valid_loss/(i+1)
    lables = y_raw[1:]
    scores_lable = y_pre_lable[1:]
    scores=y_pre_viral[1:]

    precision =precision_score(lables, scores_lable,average=None)
    precision_0=precision[0]
    recall =recall_score(lables, scores_lable,average=None) 
    recall_0=recall[0]
    f1=(2*precision_0*recall_0)/(precision_0+recall_0)
    
    for i in range(len(lables)):
        if lables[i]==0:
            output1_m+=1
            if scores_lable[i]==0:
                output1_z+=1
    output1=float(output1_z/output1_m)


    for i in range(len(lables)):
        if lables[i]==1:
            output2_m+=1
            if scores_lable[i]==0:
                output2_z+=1
    output2=float(output2_z/output2_m)

    
    for i in range(len(lables)):
        if lables[i]==2:
            output3_m+=1
            if scores_lable[i]==0:
                output3_z+=1
    output3=float(output3_z/output3_m)

    # Compute ROC curve and area the curve
    test_label = [ 2 if label ==0 else 1 for label in lables]
    fpr, tpr, thresholds = roc_curve(test_label,scores,pos_label=2,drop_intermediate=False)
    roc_auc = auc(fpr, tpr)


    #Compute the accuracy

    accurate = [1 if scores_lable[i] == lables[i] else 0 for i in range(len(lables))]
    acc = np.sum(accurate)/float(len(accurate))
    return (valid_loss,roc_auc,output1*100,output2*100,output3*100, recall[1]*100,recall[2]*100,acc*100,f1)


if __name__ == '__main__':

    model = TextCNN(embedding_size,depth)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    all_losses = []  
    print_every = 1 
    plot_every = 2  
    epoch_loss_list =[]
    best_pcc, best_epoch = 0, 0
    best_acc=0
    best_f1=0
    best_recall=0
    save_acc=0
    save_recall=0
    save_f1=0
    epoch_train,  epoch_test = [], []
    for epoch in range(200):
        epoch_loss = 0
        model.train()
        for i, batchData in enumerate(trainloader):
    
            x1 = batchData['contig']
            x2 = batchData['words']
            x3 =batchData['length']
            y = batchData['label']
            pred = model(x2)
            loss = criterion(pred, y)
            epoch_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

      
        loss_plot=epoch_loss.detach().numpy()
        epoch_loss_list.append(loss_plot)
       

        model.eval()
        test_result = eval(model, testloader, len(test))
        train_result = eval(model, trainloader, len(train))
        
        summary_test = [optimizer.param_groups[0]['lr'], *test_result]
        summary_train = [optimizer.param_groups[0]['lr'], *train_result]
        print('#test[%d]:\t%.1e\tloss:%.5f\tauc:%.4f\tviral:%.2f%%\tbacteria:%.2f%%\tplasmid:%.2f%%\tbacteria_recall:%.2f%%\tplasmid_recall:%.2f%%\tacc:%.2f%%\tf1:%.4f%%\t*' % (epoch + 1, *summary_test))
        print('#train[%d]:\t%.1e\tloss:%.5f\tauc:%.4f\tviral:%.2f%%\tbacteria:%.2f%%\tplasmid:%.2f%%\tbacteria_recall:%.2f%%\tplasmid_recall:%.2f%%\tacc:%.2f%%\tf1:%.4f%%\t*' % (epoch + 1, *summary_train))
        if summary_test[-2] > best_acc >= 0:
            torch.save(model.state_dict(), modelfn_acc)
            save_acc=save_acc+1
            best_acc = summary_test[-2]
            print("best_acc",best_acc)
        if summary_test[-1] > best_f1 >= 0:
            torch.save(model.state_dict(), modelfn_f1)
            save_f1=save_f1+1
            best_f1= summary_test[-1]
            print("best_f1:",best_f1)
        if summary_test[-7] > best_recall >= 0:
            torch.save(model.state_dict(), modelfn_recall)
            save_recall=save_recall+1
            best_recall = summary_test[-7]
            print("best_recall:",best_recall)
        epoch_test.append(summary_test)
        epoch_train.append(summary_train)
        perf_test_value = np.array(epoch_test)
        perf_train_value = np.array(epoch_train)
        np.savetxt(perf_test, perf_test_value,fmt='%0.4f',delimiter = "\t") 
        np.savetxt(perf_train, perf_train_value,fmt='%0.4f',delimiter = "\t" )

    plt.figure()
    plt.plot(epoch_loss_list)
    plt.show()