import torch
import math
import numpy as np
from models.dynamic_net import Vcnet, Drnet, TR
from data.data import get_iter
from utils.eval import curve, curve_smooth, get_acc
import torch.nn as nn
import os
import argparse

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def adjust_learning_rate(optimizer, init_lr, epoch):
    if lr_type == 'cos':  # cos without warm-up
        lr = 0.5 * init_lr * (1 + math.cos(math.pi * epoch / num_epoch))
    elif lr_type == 'exp':
        step = 1
        decay = 0.96
        lr = init_lr * (decay ** (epoch // step))
    elif lr_type == 'fixed':
        lr = init_lr
    else:
        raise NotImplementedError
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def save_checkpoint(state, checkpoint_dir='.', brain_id=0):
    model_name = "VCNet"
    filename = os.path.join(checkpoint_dir, model_name + '{}_ckpt.pth.tar'.format(brain_id))
    print('=> Saving checkpoint to {}'.format(filename))
    torch.save(state, filename)

# criterion
def criterion(out, y, alpha=0.5, epsilon=1e-6):
    crt = nn.BCELoss()

    return crt(out[1].squeeze(), y.squeeze())- alpha * torch.log(out[0] + epsilon).mean()

def criterion_TR(out, trg, y, beta=1., epsilon=1e-6):
    # out[1] is Q
    # out[0] is g
    return beta * ((y.squeeze() - trg.squeeze() / (out[0].squeeze() + epsilon) - out[1].squeeze()) ** 2).mean()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train with news data')

    # i/o
    # parser.add_argument('--data_dir', type=str, default='dataset/adni', help='dir of data matrix')
    # parser.add_argument('--data_split_dir', type=str, default='dataset/adni/eval/0', help='dir data split')
    parser.add_argument('--dataset', type=str, default='adni_mh1', help='dir data split')
    parser.add_argument('--save_dir', type=str, default='logs/adni/eval', help='dir to save result')

    # training
    parser.add_argument('--n_epochs', type=int, default=1000, help='num of epochs to train')

    # print train info
    parser.add_argument('--verbose', type=int, default=100, help='print train info freq')

    # plot adrf
    parser.add_argument('--plt_adrf', type=bool, default=True, help='whether to plot adrf curves. (only run two methods if set true; '
                                                                    'the label of fig is only for drnet and vcnet in a certain order)')
    parser.add_argument('--key_brain', type=int, default=0, help='num of dataset for evaluating the methods')


    args = parser.parse_args()
    #args.key_brain = key_brain

    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Parameters

    # optimizer
    lr_type = 'cos'
    wd = 5e-3
    momentum = 0.9

    # targeted regularization optimizer
    tr_wd = 1e-3

    # epoch: 800!
    num_epoch = args.n_epochs

    # check val loss
    verbose = args.verbose

    args.data_dir = f"dataset/{args.dataset}/{args.key_brain}"
    print(args.data_dir )

    args.data_split_dir = f"dataset/{args.dataset}/{args.key_brain}/eval/0"
    if not os.path.exists(f"results/{args.dataset}"):
        os.makedirs(f"results/{args.dataset}")
    # get data
    data_matrix = torch.load(args.data_dir + '/data_matrix.pt')
    t_grid_all = torch.load(args.data_dir + '/t_grid.pt')

    idx_train = torch.load(args.data_split_dir + '/idx_train.pt')
    idx_test = torch.load(args.data_split_dir + '/idx_test.pt')

    train_matrix = data_matrix[idx_train, :]
    test_matrix = data_matrix[idx_test, :]
    t_grid = t_grid_all[:, idx_test]

    n_data = data_matrix.shape[0]

    # train_matrix, test_matrix, t_grid = simu_data1(500, 200)
    train_loader = get_iter(data_matrix[idx_train,:], batch_size=700, shuffle=True)
    test_loader = get_iter(data_matrix[idx_test,:], batch_size=data_matrix[idx_test,:].shape[0], shuffle=False)

    grid = []
    MSE = []
    acc = []
    save_path = f'./results/{args.dataset}/'
    # for model_name in ['Tarnet', 'Tarnet_tr', 'Drnet', 'Drnet_tr', 'Vcnet', 'Vcnet_tr']:
    #for model_name in ['Drnet_tr', 'Vcnet_tr']:
    for model_name in ['Vcnet']:
        # import model
        if model_name == 'Vcnet' or model_name == 'Vcnet_tr':
            cfg_density = [(68, 128, 1, 'relu'), (128, 128, 1, 'relu')]
            num_grid = 10
            cfg = [(128, 128, 1, 'relu'), (128, 1, 1, 'sigmoid')]
            degree = 2
            knots = [0.33, 0.66]
            model = Vcnet(cfg_density, num_grid, cfg, degree, knots)
            model._initialize_weights()

        elif model_name == 'Drnet' or model_name == 'Drnet_tr':
            cfg_density = [(68, 128, 1, 'relu'), (128, 128, 1, 'relu')]
            num_grid = 10
            cfg = [(128, 128, 1, 'relu'), (128, 1, 1, 'sigmoid')]
            isenhance = 1
            model = Drnet(cfg_density, num_grid, cfg, isenhance=isenhance)
            model._initialize_weights()

        elif model_name == 'Tarnet' or model_name == 'Tarnet_tr':
            cfg_density = [(25, 50, 1, 'relu'), (50, 50, 1, 'relu')]
            num_grid = 10
            cfg = [(50, 50, 1, 'relu'), (50, 1, 1, 'sigmoid')]
            isenhance = 0
            model = Drnet(cfg_density, num_grid, cfg, isenhance=isenhance)
            model._initialize_weights()

        # use Target Regularization?
        if model_name == 'Vcnet_tr' or model_name == 'Drnet_tr' or model_name == 'Tarnet_tr':
            isTargetReg = 1
        else:
            isTargetReg = 0

        #if isTargetReg:
        tr_knots = list(np.arange(0.05, 1, 0.05))
        tr_degree = 2
        TargetReg = TR(tr_degree, tr_knots)
        TargetReg._initialize_weights()

        # best cfg for each model
        if model_name == 'Tarnet':
            init_lr = 0.02
            alpha = 1.0
            tr_init_lr = 0.001
            beta = 1.
        elif model_name == 'Tarnet_tr':
            init_lr = 0.02
            alpha = 0.5
            tr_init_lr = 0.001
            beta = 1.
        elif model_name == 'Drnet':
            init_lr = 0.02
            alpha = 1.
            tr_init_lr = 0.001
            beta = 1.
        elif model_name == 'Drnet_tr':
            init_lr = 0.02
            alpha = 0.5
            tr_init_lr = 0.001
            beta = 1.
        elif model_name == 'Vcnet':
            init_lr = 0.001
            alpha = 0.5
            tr_init_lr = 0.001
            beta = 1.
        elif model_name == 'Vcnet_tr':
            # init_lr = 0.0001
            init_lr = 0.001
            alpha = 0.5
            tr_init_lr = 0.001
            beta = 1.

        optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=momentum, weight_decay=wd, nesterov=True)

        if isTargetReg:
            tr_optimizer = torch.optim.SGD(TargetReg.parameters(), lr=tr_init_lr, weight_decay=tr_wd)

        print('model = ', model_name)
        train_loss = []
        test_loss = [] 
        for epoch in range(num_epoch):

            for idx, (inputs, y) in enumerate(train_loader):
                t = inputs[:, 0]
                x = inputs[:, 1:]
                

                if isTargetReg:
                    optimizer.zero_grad()
                    out = model.forward(t, x)
                    trg = TargetReg(t)
                    loss = criterion(out, y, alpha=alpha) + criterion_TR(out, trg, y, beta=beta)
                    loss.backward()
                    optimizer.step()

                    tr_optimizer.zero_grad()
                    out = model.forward(t, x)
                    trg = TargetReg(t)
                    tr_loss = criterion_TR(out, trg, y, beta=beta)
                    tr_loss.backward()
                    tr_optimizer.step()
                else:
                    optimizer.zero_grad()
                    out = model.forward(t, x)
                    #criterion = nn.BCELoss()
                    #print(out[1].shape,y.shape)
                   # y_bar = out[1].squeeze(dim=-1)
                    #no t setimation
                    loss = criterion(out, y,alpha=alpha)
                    train_loss.append(loss.data.numpy())
                    loss.backward()
                    optimizer.step()

            #eval 
            model.eval()
            for idx, (inputs, y) in enumerate(test_loader):
                t = inputs[:, 0]
                x = inputs[:, 1:]
                out = model.forward(t, x)

                #criterion = nn.BCELoss()
                loss = criterion(out,y,alpha=alpha)
                test_loss.append(loss.data.numpy())
                acc.append(get_acc(out[1],y))

            model.train()
            if epoch % verbose == 0:
                print('current epoch: ', epoch)
                print('loss: ', loss)

        if isTargetReg:
            t_grid_hat, mse = curve(model, test_matrix, t_grid, targetreg=TargetReg)
            mse = float(mse)
            print('current loss: ', float(loss.data))
            print('current test loss: ', mse)
        else:
            t_grid_hat, mse = curve_smooth(model, test_matrix, t_grid)
            mse = float(mse)
            print('current loss: ', float(loss.data))
            print('current test loss: ', mse)

        print('-----------------------------------------------------------------')

        save_checkpoint({
            'model': model_name,
            'best_test_loss': mse,
            'model_state_dict': model.state_dict(),
            'TR_state_dict': TargetReg.state_dict(),
        }, checkpoint_dir=save_path, brain_id =args.key_brain)

        print('-----------------------------------------------------------------')

        grid.append(t_grid_hat)
        MSE.append(mse)


    import matplotlib.pyplot as plt
    # plot
    font1 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 17,
    }
    font_legend = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 8,
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

    x = [(i+1)/len(test_matrix) for i in range(len(test_matrix))]
    y = grid[0][1, :]

    
    torch.save(y, f'results/{args.dataset}/ADRF{args.key_brain}.pt')
    torch.save(train_loss, f'results/{args.dataset}/train_loss{args.key_brain}.pt')
    torch.save(test_loss, f'results/{args.dataset}/test_loss{args.key_brain}.pt')

    #plt.scatter(x, y, marker='H', label='Drnet', alpha=1, zorder=3, color=c3, s=20)
    plt.scatter(x, y, marker='H', alpha=1, zorder=3, color=c3, s=20)

    # x = t_grid[0, :]
    # y = t_grid[1, :]

    plt.yticks(np.arange(0., 1.0, 0.1), fontsize=10, family='Times New Roman')
    plt.xticks(np.arange(0, 1.1, 0.2), fontsize=10, family='Times New Roman')
    plt.grid()
    #plt.legend(prop=font_legend, loc='upper left')
    plt.xlabel('Treatment', font1)
    plt.ylabel('Response', font1)

    plt.savefig(f"results/{args.dataset}/Vc_adni_{args.key_brain}.jpg", bbox_inches='tight')
    plt.show()

    x = [i+1 for i in range(args.n_epochs)]
    #print(loss_curve)
    y = train_loss
    #print(len(x),len(train_loss), len(test_loss))
    plt.scatter(x, y, marker='H', label='train loss', alpha=1, zorder=3, color=c3, s=20)

    y = test_loss
    plt.scatter(x, y, marker='h', label='test loss', alpha=1, zorder=2, color=c2, s=20)
    
    y = acc
    plt.plot(x, y, marker='', ls='-', label='Accuracy', linewidth=2, color=c1)

    #print(train_loss,test_loss)
    # x = t_grid[0, :]
    # y = t_grid[1, :]

    plt.yticks(np.arange(0., 1.0, 0.1), fontsize=10, family='Times New Roman')
    plt.xticks(np.arange(0, args.n_epochs, 400), fontsize=10, family='Times New Roman')
    plt.grid()
    plt.legend(prop=font_legend, loc='lower right')
    plt.xlabel('Epoch', font1)
    plt.ylabel('loss', font1)
    plt.title("Final AUC: %.4f"%(mse))
    plt.savefig(f"results/{args.dataset}/Vc_adni_loss_{args.key_brain}.jpg", bbox_inches='tight')
    #plt.show()

# if __name__ == "__main__":
#     for i in range(69):
#         main(i)