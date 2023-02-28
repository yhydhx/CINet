import numpy as np
import json
import pandas as pd
import torch
import os

from tqdm import tqdm
import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
def get_avg():
    dataset = "adni_mh_norm"
    total = [0]*369

    for i in range(69):
        a = torch.load(f"results/{dataset}/ADRF{i}.pt").numpy()
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

    x = [(i+1)/369 for i in range(369)]
    y = total

    plt.scatter(x, y, marker='H', label='Drnet', alpha=1, zorder=3, color=c3, s=20)

    # x = t_grid[0, :]
    # y = t_grid[1, :]

    plt.yticks(np.arange(0., 0.5, 0.1), fontsize=10, family='Times New Roman')
    plt.xticks(np.arange(0, 1.1, 0.2), fontsize=10, family='Times New Roman')
    plt.grid()
    #plt.legend(prop=font_legend, loc='upper left')
    plt.xlabel('Treatment', font1)
    plt.ylabel('Response', font1)
    print(total)
    plt.savefig(f"results/{dataset}/Vc_adni_avg.jpg", bbox_inches='tight')
    plt.show()


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
        print("python adni_mh.py --key_brain {} --dataset adni_mh_norm".format(i))
        #print("python manhua_generate_data.py --key_brain {} --norm True ".format(i))

#get_avg()
#get_command()

def draw_loss():
    train_loss = np.array([0.]*1000)
    test_loss = np.array([0.]*1000)
    #acc = np.array([0.]*1000)
    dataset = "adni_mh_norm"
    for i in range(69):
        a = np.array(torch.load(f"results/{dataset}/train_loss{i}.pt"))
        b = np.array(torch.load(f"results/{dataset}/test_loss{i}.pt"))
        #c = np.array(torch.load(f"results/{dataset}/acc{i}.pt"))
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

    x = [i for i in range(1000)]
    y = train_loss
    plt.plot(x, y, marker='', ls='-', label='Truth', linewidth=2, color=c2)
    y = test_loss
    
    plt.plot(x, y, marker='', ls='-', label='Truth', linewidth=4, color=c3)
    
    # x = t_grid[0, :]
    # y = t_grid[1, :]

    plt.yticks(np.arange(0., 0.5, 0.1), fontsize=10, family='Times New Roman')
    plt.xticks(np.arange(0, 1000, 400), fontsize=10, family='Times New Roman')
    plt.grid()
    #plt.legend(prop=font_legend, loc='upper left')
    plt.xlabel('epoch', font1)
    plt.ylabel('loss', font1)
    plt.legend()
    plt.savefig("results/Vc_adni_avg.jpg", bbox_inches='tight')
    plt.show()

#draw_loss()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate adni data')
    parser.add_argument('--save_dir', type=str, default='dataset/adni_mh/', help='dir to save generated data')
    parser.add_argument('--num_eval', type=int, default=10, help='num of dataset for evaluating the methods')
    parser.add_argument('--num_tune', type=int, default=2, help='num of dataset for tuning the parameters')
    parser.add_argument('--roi', type=int, default=69, help='num of dataset for evaluating the methods')
    parser.add_argument('--key_brain', type=int, default=0, help='num of dataset for evaluating the methods')
    parser.add_argument('--norm', type=bool, default=False, help='Normalization or not ')
    args = parser.parse_args()


    dataset = "manhua1"
    if dataset == "manhua1":
        save_path = "dataset/adni_mh1/"

        args = parser.parse_args()
        save_path = save_path + str(args.key_brain)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # load data

        mat1 = np.load("dataset/adni_mh1/mat69_1.npy")
        label1 = np.load("dataset/adni_mh1/label69_1.npy")
        print("label1: ", label1, len(label1))
        label1[label1<2] = 0
        label1[label1>=2] = 1
        mat = mat1
        label = label1

        index = list(range(69))
        index.remove(55)
        #print(index)
        mat = mat[:,index].squeeze()

    elif dataset == "manhua2":
        save_path = "dataset/adni_mh2/"

        args = parser.parse_args()
        save_path = save_path + str(args.key_brain)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        mat2 = np.load("dataset/adni_mh2/mat69_2.npy")
        label2 = np.load("dataset/adni_mh2/label69_2.npy")
        print("label2: ", label2, len(label2))
        label2[label2<3] = 0
        label2[label2>=3] = 1

        mat = mat2
        label = label2

        index = list(range(69))
        index.remove(55)
        #print(index)
        mat = mat[:,index].squeeze()
    elif dataset == "manhuaall":
        save_path = "dataset/adni_mhall_AD/"

        args = parser.parse_args()
        save_path = save_path + str(args.key_brain)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # load data

        mat1 = np.load("dataset/adni_mh1/mat69_1.npy")

        label1 = np.load("dataset/adni_mh1/label69_1.npy")
        print("label1: ", label1, len(label1))
        label1[label1<2] = 0
        label1[label1>=2] = 1


        mat2 = np.load("dataset/adni_mh2/mat69_2.npy")

        label2 = np.load("dataset/adni_mh2/label69_2.npy")
        print("label2: ", label2, len(label2))
        label2[label2<3] = 0
        label2[label2>=3] = 1

        
        index = list(range(69))
        index.remove(55)
        #print(index)
        mat = np.concatenate((mat1,mat2),axis=0)
        #delete brain stem 
        mat = mat[:,index].squeeze()

        label = np.concatenate((label1,label2),axis=0)
    elif dataset == "manhuaallnc":
        save_path = "dataset/adni_mhall_NC/"

        args = parser.parse_args()
        save_path = save_path + str(args.key_brain)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # load data

        mat1 = np.load("dataset/adni_mh1/mat69_1.npy")

        label1 = np.load("dataset/adni_mh1/label69_1.npy")
        print("label1: ", label1, len(label1))
        label1[label1<1] = 0
        label1[label1>=1] = 1


        mat2 = np.load("dataset/adni_mh2/mat69_2.npy")

        label2 = np.load("dataset/adni_mh2/label69_2.npy")
        print("label2: ", label2, len(label2))
        label2[label2<1] = 0
        label2[label2>=1] = 1

        
        index = list(range(69))
        index.remove(55)
        #print(index)
        mat = np.concatenate((mat1,mat2),axis=0)
        #delete brain stem 
        mat = mat[:,index].squeeze()

        label = np.concatenate((label1,label2),axis=0)


    print(mat.shape,label.shape)
    
    print(save_path)

    #  normalize data
    
    for _ in range(mat.shape[1]):
        max_freq = max(mat[:,_])
        min_freq = min(mat[:,_])
        mat[:,_] = (mat[:,_] - min_freq) / (max_freq - min_freq)
    

    #print(mat)
    #after swap 
    for i in range(len(mat)):
        mat[i][0], mat[i][args.key_brain] = mat[i][args.key_brain], mat[i][0]
    #print(mat)
    num_data = mat.shape[0]
    num_feature = mat.shape[1]

    def news_matrix():
        data_matrix = torch.zeros(num_data, num_feature+1)
        # get data matrix
        for _ in range(num_data):
            x = mat[_, :]
            y = torch.from_numpy(np.array([label[_]]))
            x = torch.from_numpy(x)
            t = x[0]
            

            data_matrix[_, 0] = t
            data_matrix[_, -1] = y
            data_matrix[_, 0: num_feature] = x
        
        t_grid = torch.zeros(2, num_data)
        t_grid[0, :] = data_matrix[:, 0].squeeze()

        return data_matrix, t_grid

    dm, tg= news_matrix()

    torch.save(dm, save_path + '/data_matrix.pt')
    torch.save(tg, save_path + '/t_grid.pt')
    training_size = int(np.floor(len(dm) *0.7))
    print("total, ",len(dm)," training seize", training_size)
    # generate eval splitting
    for _ in range(args.num_eval):
        print('generating eval set: ', _)
        data_path = os.path.join(save_path, 'eval', str(_))
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        idx_list = torch.randperm(num_data)
        idx_train = idx_list[0:training_size]
        idx_test = idx_list[training_size:]

        torch.save(idx_train, data_path + '/idx_train.pt')
        torch.save(idx_test, data_path + '/idx_test.pt')

        np.savetxt(data_path + '/idx_train.txt', idx_train.numpy())
        np.savetxt(data_path + '/idx_test.txt', idx_test.numpy())

    # generate tuning splitting
    for _ in range(args.num_tune):
        print('generating tuning set: ', _)
        data_path = os.path.join(save_path, 'tune', str(_))
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        idx_list = torch.randperm(num_data)
        idx_train = idx_list[0:training_size]
        idx_test = idx_list[training_size:]

        torch.save(idx_train, data_path + '/idx_train.pt')
        torch.save(idx_test, data_path + '/idx_test.pt')

        np.savetxt(data_path + '/idx_train.txt', idx_train.numpy())
        np.savetxt(data_path + '/idx_test.txt', idx_test.numpy())



