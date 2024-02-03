#coding=utf-8
import os.path as osp
import argparse

import torch
import torch.nn.functional as F
from torch_geometric.datasets import ShapeNet
from torch_geometric.data import Data
import torch_geometric.transforms as T
#from rgcn_conv import RGCNConv
from torch_geometric.nn import GCNConv, DynamicEdgeConv,knn_graph, EdgeConv, radius_graph, RGCNConv# noqa
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
#import h5py
import numpy as np
from collections import Counter
from tqdm import tqdm
from EigenGraph import eigen_Graph
from data import ShapeNetPart
from scipy.stats import mode
random.seed(100)
torch.manual_seed(7)
np.random.seed(100)
from torch.nn import Sequential as Seq, Dropout, Linear as Lin,BatchNorm1d as BN
def MLP(channels, bias=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i], bias=bias),BN(channels[i]),torch.nn.LeakyReLU(negative_slope=0.2),Lin(channels[i], channels[i], bias=bias))
        for i in range(1, len(channels))
    ])

class RelationGraphConvNet(torch.nn.Module):
    def __init__(self,num_points, k = 20,num_bases = 5):
        super(RelationGraphConvNet, self).__init__()
        self.conv1 = RGCNConv(3, 16, num_relations = 3, num_bases = num_bases)
        self.conv2 = RGCNConv(16, num_classes, num_relations = 3, num_bases = num_bases)
        
        self.k = k
        self.num_points = num_points;

    def forward(self):
        x = data.x
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)

        x = self.conv2(x, edge_index, edge_weight)

        return F.log_softmax(x, dim=1)

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(point_features, 16, cached=True)
        self.conv2 = GCNConv(16, num_classes, cached=True)


    def forward(self):
        x= data.x
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


import os
import sys
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset


def center_data(pcs):
	for pc in pcs:
		centroid = np.mean(pc, axis=0)
		pc[:,0]-=centroid[0]
		pc[:,1]-=centroid[1]
		pc[:,2]-=centroid[2]
	return pcs

def normalize_data(pcs):
	for pc1 in pcs:

		d = max(np.sum(np.abs(pc1)**2,axis=-1)**(1./2))
		pc1 /= d



	return pcs


def load_data(partition):

    with h5py.File("..//data//s3dis//"+'%s.h5' % partition,'r')  as f:

        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        mask = f['mask'][:].astype('int64')
        f.close()

    data = center_data(data)
    data = normalize_data(data)

    print("load real data and center and normalize data")
    print(data.shape, label.shape, mask.shape)
    return data, label,mask


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud


class RealPointCloudRaw(Dataset):
    def __init__(self, num_points, partition='test', shuffle = True):
        self.data, self.label,self.mask= load_data(partition)
        self.num_points = num_points
        self.partition = partition
        self.shuffle = shuffle
    def __getitem__(self, item):

        label = self.label[item]
        mask = self.mask[item][:self.num_points]
        pointcloud = self.data[item][:self.num_points]
        if self.shuffle and self.partition == 'train':
            indices = list(range(pointcloud.shape[0]))
            np.random.shuffle(indices)
            pointcloud = pointcloud[indices]
            mask = mask[indices]
        return pointcloud, label,mask

    def __len__(self):
        return self.data.shape[0]

def draw_Point_Cloud(Points, Lables=None, axis=True, save_name = "default", **kags):

    x_axis = Points[:, 1]
    y_axis = Points[:, 2]
    z_axis = Points[:, 0]
    fig = plt.figure()
    ax = Axes3D(fig)

    ax.scatter(x_axis, y_axis, z_axis, c=Lables)
 
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(elev=25, azim=0)
    if not axis:
       
        plt.axis('off')

    plt.savefig( save_name + ".png")

def compute_mIoU(pred_label, gt_label):
    minl, maxl = np.min(gt_label), np.max(gt_label)
    ious = []
    for l in range(minl, maxl+1):
        I = np.sum(np.logical_and(pred_label == l, gt_label == l))
        U = np.sum(np.logical_or(pred_label == l, gt_label == l))
        if U == 0: iou = 1
        else: iou = float(I) / U
        ious.append(iou)
    return np.mean(ious)

parser = argparse.ArgumentParser()
parser.add_argument('--use_gdc', action='store_true',
                    help='Use GDC preprocessing.')
args = parser.parse_args()


save2txt = open("result.txt", 'a+')
save2txt_log = open("logs.txt",'a+')
ShapeNetPseudoLabeling = None
cat = 0
processed_sample = 0
import h5py

f = h5py.File("testing_s3dis.h5", 'w')
f_2 = h5py.File("testing_s3dis.h5", 'w')
target_names = ["ceiling", "floor", "wall", "beam", "column", "window", "chair", "door", "table", "bookcase", "sofa","board", "clutter"]
sample_cls = []
oAcc = []
refineoAcc = []

res = open('s3dis.txt','a+')
prev_label = 0
count_label = 0
mious,mtrain_acc, mtest_acc = [], [], []

for seg_class in [1]: 
    catAcc = []
    refinecatAcc=[]
    


    def train():
        model.train()
        optimizer.zero_grad()
        F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
        optimizer.step()


    @torch.no_grad()
    def test():
        model.eval()
        logits, accs = model(), []
        for _, mask in data('train_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)
        return accs,logits.max(1)[1]

    bestIoUs = []
    for dataseti in range(len(shapenet_dataset)): 
        pos, origilabel, seg  = shapenet_dataset[dataseti]
        if prev_label != origilabel:
            category_miou, mtrainacc, mtestacc = 0.0, 0.0, 0.0
            for i in range(len(mious)):
                category_miou+=mious[i]
                mtrainacc+=mtrain_acc[i]
                mtestacc += mtest_acc[i]
            category_miou/=len(mious)
            mtrainacc /= len(mious)
            mtestacc /= len(mious)

            res.write(target_names[prev_label] + " " + str(len(mious)) + " miou : {:.5f}  train acc : {:.5f}  test acc : {:.5f}\n".format(category_miou, mtrainacc, mtestacc))

            prev_label = origilabel
            count_label = 0
            mious = []
            mtrain_acc = []
            mtest_acc = []
        else:
            count_label += 1
        category = target_names[origilabel]
        
        seg_processed = [0 if i == -1  else 1 for i in seg]
        print(Counter(seg_processed))

        numpoints  = pos.shape[0]

        label_ratio = 0.1
        batch = torch.tensor([0]*numpoints)

        dic_label_idx = {}
        for idx, label_ in enumerate(seg_processed):
            if label_ not in dic_label_idx:
                dic_label_idx[label_] = [idx]
            else:
                dic_label_idx[label_].append(idx)

        traininglabelIndex = []
        for dic_key in dic_label_idx.keys():
            dic_key_list_len = len(dic_label_idx[dic_key])
            sample_idxs = int(dic_key_list_len * label_ratio)
            if (sample_idxs == 0):
                sample_idxs = 1
            idx2choose = np.random.choice(dic_key_list_len, sample_idxs, replace=False)
            dic_key_label = np.array(dic_label_idx[dic_key])[idx2choose]
            traininglabelIndex.extend(dic_key_label.tolist())

        train_mask, test_mask ,val_mask = [],[],[]
        for i in range(numpoints):
            if i in traininglabelIndex:
                train_mask.append(True)
                test_mask.append(False)
            else:
                train_mask.append(False)
                test_mask.append(True)
            val_mask.append(False)

        print("train mask count: " , Counter(train_mask)[True])
        print("test_mask mask count: " , Counter(test_mask)[True])




        mydata = Data(x = torch.tensor(pos))
        mydata.train_mask = torch.tensor(train_mask).bool();



        mydata.test_mask = torch.tensor(test_mask).bool();

        mydata.val_mask = torch.tensor(val_mask).bool();
        mydata.y = torch.tensor(seg_processed).long();
        data = mydata

        device = 'cuda'
        data = data.to(device)
        batch = batch.to(device)
        best_val_acc = test_acc = 0
        epoch = 250
        best_miou = 0.0
        # rgcn
        model = RelationGraphConvNet(numpoints,num_bases = 7).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0135)



        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epoch, eta_min=1e-4)



        
        
        k = 20
        x_, idx_eu, idx_ei = eigen_Graph(data.x.unsqueeze(0), k, device=device)
        idx_radius = radius_graph(data.x, r = 0.2, batch = batch)
        print(idx_eu)
        print(idx_eu.shape)
       
        edge_index = torch.cat([idx_eu, idx_ei,idx_radius], dim=1)
        edge_weight_eu = [0] * len(idx_eu[0])
        edge_weight_ei = [1] * len(idx_ei[0])
        edge_weight_er = [2] * len(idx_radius[0])
        edge_weight = torch.tensor(edge_weight_eu + (edge_weight_ei) + edge_weight_er).long().to(device)
        
     
        print(data.x.shape)
        #input()
        for epoch in tqdm(range(1, epoch+1),ascii = True):

            train()
            scheduler.step()

        accs, pred = test()
        train_acc, tmp_test_acc = accs[0], accs[1]
        acc_log = (" {} train acc: {:.5f} test acc: {:.5f}").format(dataseti, train_acc,tmp_test_acc)
        print(acc_log)
        curr_miou = compute_mIoU(pred.cpu().numpy(), data.y.cpu().numpy())
        print("mIoU : ", curr_miou)
        save2txt_log.write(acc_log)
        save2txt_log.write(" mIoU : {:.5f} \n".format(curr_miou))
        bestIoUs.append(curr_miou)
        catAcc.append(tmp_test_acc)
        for idx in traininglabelIndex:
            pred[idx] = data.y[idx];
        curr_miou = compute_mIoU(pred.cpu().numpy(), data.y.cpu().numpy())
        print("new miou:", curr_miou)
        mious.append(curr_miou)
        mtrain_acc.append(train_acc)
        mtest_acc.append(tmp_test_acc)


        points = data.x.cpu().numpy()
        predseg = pred.cpu().numpy().reshape(-1,1) #+ seg_classes[category][0]
        #print(seg)
        mask2save = mydata.train_mask.cpu().numpy().reshape(-1,1)

        

        idx_eu = knn_graph(data.x, 50, batch, loop=True).to(device)

        neighborPred1 = predseg[idx_eu[0].cpu()].reshape(numpoints,-1)

        marjortiyPred1 = mode(neighborPred1, axis = 1)[0].ravel()

        downSampleMask = (predseg.reshape(-1) == ( marjortiyPred1))

        seg_label = np.max(seg)
        predseg = np.array([-1 if i == 0  else seg_label for i in predseg])
        #seg = np.array([0 if i == -1  else seg_label for i in seg])
        data2save = np.concatenate([points, predseg.reshape(-1,1),seg.reshape(-1,1), mask2save.reshape(-1,1),downSampleMask.reshape(-1,1)], axis=1)
        """
        print(data2save[:,3])
        print(data2save[:,4])
        print(Counter(predseg))
        print(Counter(seg))
        print(Counter(mask2save.reshape(-1).tolist()))
        print(Counter(downSampleMask.reshape(-1).tolist()))
        """
        f.create_dataset(str(processed_sample).zfill(5), data = data2save)

        sample_cls.append(origilabel)
        processed_sample += 1
        seg = np.array(seg)

        
        predseg = predseg.reshape(-1)
        seg = seg.reshape(-1)
        loss_on_noisy_labels_mask = (~mask2save.reshape(-1)) & (downSampleMask)
        mask2save = mask2save.reshape(-1)
        refineAcc = ((predseg[mask2save] == seg[mask2save]).sum() + (predseg[loss_on_noisy_labels_mask] == seg[loss_on_noisy_labels_mask]).sum() ) / (mask2save.sum() + (loss_on_noisy_labels_mask).sum())
       
        refinecatAcc.append(refineAcc)
        print("refine new acc : ", refineAcc)

    cat += 1
    print(bestIoUs)
    sum = 0.0
    i = 0
    for iou in bestIoUs:
        sum += iou
        i+=1
    print(" mean IoU : " , sum/i)
    save2txt.write(category + " : {:.5f} \n".format(sum/i))

    sum = 0.0
    for acc in catAcc:
        sum+=acc
    sum/=len(catAcc)
    oAcc.append(sum)
    
    sum = 0.0
    for acc in refinecatAcc:
        sum += acc
    sum /= len(refinecatAcc)
    refineoAcc.append(sum)
    
    
sum1 = 0.0
for acc in oAcc:
    sum1+=acc
sum1/=len(oAcc)

sum2 = 0.0
for acc in refineoAcc:
    sum2+=acc
sum2/=len(refineoAcc)
s = "cat acc : " + str(oAcc) + " oAcc: " + str(sum1) + "\n"
s += "refine cat acc : " + str(refineoAcc) + "refine oAcc: " + str(sum2) + "\n" 

save2txt.write(s)

category_miou, mtrainacc, mtestacc = 0.0, 0.0, 0.0
for i in range(len(mious)):
    category_miou += mious[i]
    mtrainacc += mtrain_acc[i]
    mtestacc += mtest_acc[i]
category_miou /= len(mious)
mtrainacc /= len(mious)
mtestacc /= len(mious)

res.write(
    target_names[prev_label] + " " + str(len(mious)) + " miou : {:.5f}  train acc : {:.5f}  test acc : {:.5f}\n".format(category_miou, mtrainacc,
                                                                                               mtestacc))
res.close()
f_2.create_dataset("cls", data = np.array(sample_cls))
#


f.close()
f_2.close()


save2txt.close()
