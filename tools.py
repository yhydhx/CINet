import numpy as np 
import json
import pandas as pd
import torch
import os

from tqdm import tqdm
import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
def get_avg(dataset, seed):
    if dataset == "adni_mh1":
        length = 120 
    elif dataset == "adni_mh2":
        length = 202
    elif dataset == "adni_mh_norm":
        length = 369
    elif "all" in dataset:
        length = 321

    total = [0]* length

    for i in range(68):
        a = torch.load(f"results/{dataset}/{seed}/ADRF{i}.pt").numpy()
        total += a
    print(total)
    total = total / 69
    import matplotlib.pyplot as plt
    # plot
    font1 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 22,
    }
    font_legend = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 22,
    }
    plt.figure(figsize=(5, 5))

    # b = 'C0'
    # o = 'C1'
    # g = 'C2'

    c1 = '#F87664'
    c1 = 'red'
    c1 = 'gold'
    c2 = '#00BA38'
    c2 = 'red'
    c3 = '#619CFF'
    c3 = 'dodgerblue'

    # truth_grid = t_grid[:,t_grid[0,:].argsort()]
    # x = truth_grid[0, :]
    # y = truth_grid[1, :]
    # plt.plot(x, y, marker='', ls='-', label='Truth', linewidth=4, color=c1)


    # x = grid[1][0, :]
    # y = grid[1][1, :]
    # plt.scatter(x, y, marker='h', label='Vcnet', alpha=1, zorder=2, color=c2, s=20)

    x = [(i+1)/length for i in range(length)]
    y = total

    plt.scatter(x, y, marker='H', label='Drnet', alpha=1, zorder=3, color=c3, s=20)

    # x = t_grid[0, :]
    # y = t_grid[1, :]

    plt.yticks(np.arange(0., 1, 0.1), fontsize=10, family='Times New Roman')
    plt.xticks(np.arange(0, 1.1, 0.2), fontsize=10, family='Times New Roman')
    plt.grid()
    #plt.legend(prop=font_legend, loc='upper left')
    plt.xlabel('Treatment', font1)
    plt.ylabel('Response', font1)
    print(total)
    plt.savefig(f"results/{dataset}/{seed}Vc_adni_avg.jpg", bbox_inches='tight')
    plt.show()


get_avg("adni_mh1", 6)

# for i in range(10):
#     get_avg("adni_mhall_AD", i)
#     get_avg("adni_mhall_NC", i)

def redraw(i):

    
    y = torch.load("results/ADRF{}.pt".format(i)).numpy()
    
    import matplotlib.pyplot as plt
    # plot
    font1 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 22,
    }
    font_legend = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 22,
    }
    plt.figure(figsize=(5, 5))

    # b = 'C0'
    # o = 'C1'
    # g = 'C2'

    c1 = '#F87664'
    c1 = 'red'
    c1 = 'gold'
    c2 = '#00BA38'
    c2 = 'red'
    c3 = '#619CFF'
    c3 = 'dodgerblue'

    # truth_grid = t_grid[:,t_grid[0,:].argsort()]
    # x = truth_grid[0, :]
    # y = truth_grid[1, :]
    # plt.plot(x, y, marker='', ls='-', label='Truth', linewidth=4, color=c1)


    # x = grid[1][0, :]
    # y = grid[1][1, :]
    # plt.scatter(x, y, marker='h', label='Vcnet', alpha=1, zorder=2, color=c2, s=20)

    x = [(i+1)/119 for i in range(119)]


    plt.scatter(x, y, marker='H', label='Drnet', alpha=1, zorder=3, color=c3, s=20)

    # x = t_grid[0, :]
    # y = t_grid[1, :]

    plt.yticks(np.arange(0., 1.0, 0.1), fontsize=10, family='Times New Roman')
    plt.xticks(np.arange(0, 1.1, 0.2), fontsize=10, family='Times New Roman')
    plt.grid()
    #plt.legend(prop=font_legend, loc='upper left')
    plt.xlabel('Treatment', font1)
    plt.ylabel('Response', font1)

    plt.savefig("results/Vc_adni_{}.jpg".format(i), bbox_inches='tight')
    #plt.show()


def get_command():

    for i in range(69):
        print("python adni_mh.py --key_brain {} --dataset adni_mh2 --seed 1".format(i))
        #print("python adni_generate_data.py --key_brain {} ".format(i))



def draw_loss():
    train_loss = np.array([0.]*2000)
    test_loss = np.array([0.]*2000)


    for i in range(69):
        a = np.array(torch.load("results/train_loss{}.pt".format(i)))
        b = np.array(torch.load("results/test_loss{}.pt".format(i)))
        train_loss += a
        test_loss += b
    
    train_loss = train_loss/69
    test_loss = test_loss/69

    import matplotlib.pyplot as plt
    # plot
    font1 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 22,
    }
    font_legend = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 22,
    }
    plt.figure(figsize=(5, 5))

    # b = 'C0'
    # o = 'C1'
    # g = 'C2'

    c1 = '#F87664'
    c1 = 'red'
    c1 = 'gold'
    c2 = '#00BA38'
    c2 = 'red'
    c3 = '#619CFF'
    c3 = 'dodgerblue'

    # truth_grid = t_grid[:,t_grid[0,:].argsort()]
    # x = truth_grid[0, :]
    # y = truth_grid[1, :]
    # plt.plot(x, y, marker='', ls='-', label='Truth', linewidth=4, color=c1)


    # x = grid[1][0, :]
    # y = grid[1][1, :]
    # plt.scatter(x, y, marker='h', label='Vcnet', alpha=1, zorder=2, color=c2, s=20)

    x = [i for i in range(2000)]
    y = train_loss

    plt.scatter(x, y, marker='H', label='train loss', alpha=1, zorder=3, color=c3, s=20)
    y = test_loss
    plt.scatter(x, y, marker='h', label='test loss', alpha=1, zorder=2, color=c2, s=20)
    
    # x = t_grid[0, :]
    # y = t_grid[1, :]

    plt.yticks(np.arange(0., 0.5, 0.1), fontsize=10, family='Times New Roman')
    plt.xticks(np.arange(0, 2000, 400), fontsize=10, family='Times New Roman')
    plt.grid()
    #plt.legend(prop=font_legend, loc='upper left')
    plt.xlabel('epoch', font1)
    plt.ylabel('loss', font1)
    plt.legend()
    plt.savefig("results/Vc_adni_avg.jpg", bbox_inches='tight')
    plt.show()


def treat_distribution():
    import matplotlib.pyplot as plt

    dataset = "adni_mh1"
    if not os.path.exists(f"results/{dataset}/treat_dist"):
        os.makedirs(f"results/{dataset}/treat_dist")
   
    # load data
    mat1 = np.load(f"dataset/{dataset}/mat69_1.npy")
    label1 = np.load(f"dataset/{dataset}/label69_1.npy")
    print("label1: ", label1, len(label1))

    mat = mat1
    for _ in range(mat.shape[1]):
        max_freq = max(mat[:,_])
        min_freq = min(mat[:,_])
        mat[:,_] = (mat[:,_] - min_freq) / (max_freq - min_freq)

        # the histogram of the data
        n, bins, patches = plt.hist(mat[:,_], 20, density=True, facecolor='g', alpha=0.75)

        plt.xlabel('Treatment')
        plt.ylabel('Probability')
        plt.title(f'The Distribution of ROI{_} Treatment')
        #plt.axis([40, 160, 0, 0.03])
        plt.grid(True)
        #plt.show()
        plt.savefig(f"results/{dataset}/treat_dist/ROI{_}.jpg")
        plt.clf()
    
    data  = mat.reshape((46368))
    print(data.reshape((46368)))
    n, bins, patches = plt.hist(data, 20, density=True, facecolor='g', alpha=0.75)

    plt.xlabel('Treatment')
    plt.ylabel('Probability')
    plt.title(f'The Distribution of Whole Brain')
    #plt.axis([40, 160, 0, 0.03])
    plt.grid(True)
    #plt.show()
    plt.savefig(f"results/{dataset}/treat_dist/WholeBrain.jpg")

#treat_distribution()

Z = np.array([[1,2,3],[1,2,3],[2,5,4],[2,3,4]])
y = np.array([0,0,1,1])

def coding_rate(Z, eps = 1e-4):
    n, d = Z.shape 
    (_, rate) = np.linalg.slogdet((np.eye(d) + 1/ (n*eps) * Z.transpose() @ Z))
    return 0.5 * rate 


def transrate(Z, y, eps= 1e-4):
    Z = Z- np.mean(Z, axis = 0, keepdims = True)
    RZ = coding_rate(Z, eps)
    RZY = 0
    K = int(y.max() + 1) 
    for i in range(K):
        RZY += coding_rate(Z[(y == i).flatten()], eps)
    return RZ - RZY/K 

print(transrate(Z,y))


