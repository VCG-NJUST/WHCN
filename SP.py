
import os.path
import sys
import numpy as np
import argparse
from timeit import default_timer as timer
sys.path.append("./cut-pursuit/build/src")
sys.path.append("./ply_c")
import libcp
import libply_c
from graphs import *
from provider import *

parser = argparse.ArgumentParser(description='superpoint')
parser.add_argument('--ROOT_PATH', default='datasets/s3dis')
parser.add_argument('--dataset', default='s3dis', help='s3dis')
parser.add_argument('--k_nn_geof', default=45, type=int, help='number of neighbors for the geometric features')
parser.add_argument('--k_nn_adj', default=10, type=int, help='adjacency structure for the minimal partition')
parser.add_argument('--lambda_edge_weight', default=1., type=float, help='parameter determine the edge weight for minimal part.')
parser.add_argument('--reg_strength', default=0.1, type=float, help='regularization strength for the minimal partition')
parser.add_argument('--d_se_max', default=0, type=float, help='max length of super edges')
parser.add_argument('--voxel_width', default=0.03, type=float, help='voxel size when subsampling (in m)')
parser.add_argument('--ver_batch', default=0, type=int, help='Batch size for reading large files, 0 do disable batch loading')
parser.add_argument('--overwrite', default=0, type=int, help='Wether to read existing files or overwrite them')
args = parser.parse_args()


root = args.ROOT_PATH+'/'
if args.dataset == 's3dis':
    folders = ["Area_1/", "Area_2/", "Area_3/", "Area_4/", "Area_5/", "Area_6/"]
    n_labels = 13

else:
    raise ValueError('%s is an unknown data set' % dataset)

times = [0,0,0]

if not os.path.isdir(root + "clouds"):
    os.mkdir(root + "clouds")
if not os.path.isdir(root + "features"):
    os.mkdir(root + "features")
if not os.path.isdir(root + "superpoint_graphs"):
    os.mkdir(root + "superpoint_graphs")

for folder in folders:
    print("=================\n   "+folder+"\n=================")
    
    data_folder = root   + "data/"              + folder
    cloud_folder  = root + "clouds/"            + folder
    fea_folder  = root   + "features/"          + folder
    spg_folder  = root   + "superpoint/" + folder
    if not os.path.isdir(data_folder):
        raise ValueError("%s does not exist" % data_folder)
        
    if not os.path.isdir(cloud_folder):
        os.mkdir(cloud_folder)
    if not os.path.isdir(fea_folder):
        os.mkdir(fea_folder)
    if not os.path.isdir(spg_folder):
        os.mkdir(spg_folder)
    
    if args.dataset=='s3dis':    
        files = [os.path.join(data_folder, o) for o in os.listdir(data_folder) 
                if os.path.isdir(os.path.join(data_folder,o))]

        files = glob.glob(data_folder+"*.txt")
        
    if (len(files) == 0):
        raise ValueError('%s is empty' % data_folder)
        
    n_files = len(files)
    i_file = 0
    for file in files:
        file_name   = os.path.splitext(os.path.basename(file))[0]
        
        if args.dataset=='s3dis':
            data_file   = data_folder      + file_name + '/' + file_name + ".txt"
            cloud_file  = cloud_folder     + file_name
            fea_file    = fea_folder       + file_name + '.h5'
            spg_file    = spg_folder       + file_name + '.h5'

        
        i_file = i_file + 1
        print(str(i_file) + " / " + str(n_files) + "---> "+file_name)

        if os.path.isfile(fea_file) and not args.overwrite:
            print("    reading the existing feature file...")
            geof, xyz, rgb, graph_nn, labels = read_features(fea_file)
        else :
            print("    creating the feature file...")
            if args.dataset=='s3dis':
                xyz, rgb, labels, objects = read_s3dis_format(data_file)
                if args.voxel_width > 0:
                    xyz, rgb, labels, dump = libply_c.prune(xyz.astype('f4'), args.voxel_width, rgb.astype('uint8'), labels.astype('uint8'), np.zeros(1, dtype='uint8'), n_labels, 0)


            start = timer()

            graph_nn, target_fea = compute_graph_nn_2(xyz, args.k_nn_adj, args.k_nn_geof)
            geof = libply_c.compute_geof(xyz, target_fea, args.k_nn_geof).astype('float32')
            end = timer()
            times[0] = times[0] + end - start
            del target_fea
            write_features(fea_file, geof, xyz, rgb, graph_nn, labels)
