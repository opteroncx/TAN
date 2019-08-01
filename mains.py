# -*- coding:utf-8 -*-
import argparse, os
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model6b import Net, L1_Charbonnier_loss,ScaleLayer
from datasetD2K3 import DatasetFromHdf5


# Training settings
parser = argparse.ArgumentParser(description="TAN")
parser.add_argument("--batchSize", type=int, default=35, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=100, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=10, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="weight decay, Default: 1e-4")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")

def main():

    global opt, model
    opt = parser.parse_args()
    print(opt)

    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print("===> Loading datasets")
    train_set = DatasetFromHdf5("./data/ntire.h5")
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

    print("===> Building model")
    model = Net()
    criterion = L1_Charbonnier_loss()


    print("===> Setting GPU")
    if cuda:
        model=nn.DataParallel(model,device_ids=[0,1,2,3,4]).cuda()
        criterion = criterion.cuda()
    else:
        model = model.cpu()

    loadmultiGPU = True
    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            saved_state = checkpoint["model"].state_dict()
            # multi gpu loader
            if loadmultiGPU:
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in saved_state.items():
                    namekey = 'module.'+k # remove `module.`
                    new_state_dict[namekey] = v
                    # load params
                model.load_state_dict(new_state_dict)
            else: 
                model.load_state_dict(saved_state)
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained)
            pretrained_dict = weights['model'].state_dict()

            model_dict = model.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict) 
            # 3. load the new state dict
            model.load_state_dict(model_dict)
            # model.load_state_dict(state)
        else:
            print("=> no model found at '{}'".format(opt.pretrained))

    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    print("===> Training")
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        train(training_data_loader, optimizer, model, criterion, epoch)
        save_checkpoint(model, epoch)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 100 epochs"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr

def train(training_data_loader, optimizer, model, criterion, epoch):

    lr = adjust_learning_rate(optimizer, epoch-1)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    print("epoch =", epoch,"lr =",optimizer.param_groups[0]["lr"]) 
    model.train()

    for iteration, batch in enumerate(training_data_loader, 1):

        input=Variable(batch[0])
        label_x2 = Variable(batch[1], requires_grad=False)
        label_x4 = Variable(batch[2], requires_grad=False)
        label_x8 = Variable(batch[3], requires_grad=False)

        if opt.cuda:
            input = input.cuda()
            label_x2 = label_x2.cuda()
            label_x4 = label_x4.cuda()
            label_x8 = label_x8.cuda()

        HR,ui,ur= model(label_x4)
        # Supervise residual
        # ui ---> upscale(LR)
        # ur ---> pridected residual
        SLoss = multiFuseLoss(HR, label_x8 ,ui,ur,criterion)

        loss = SLoss

        optimizer.zero_grad()

        SLoss.backward()

        optimizer.step()

        if iteration%100 == 0:
            print("===> Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch, iteration, len(training_data_loader), loss.item()))

def save_checkpoint(model, epoch):
    model_folder = "model_20180716_2x/"
    model_out_path = model_folder + "model_epoch_{}.pth".format(epoch)
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))

def multiFuseLoss(imglist,img,ui,ur,criterion):
    # ui ---> upscale(LR)
    # ur ---> pridected residual
    # img --> label img
    lossi=0
    lossr=0
    true_res = img-ui
    for i in range(len(imglist)):
        l=criterion(imglist[i], img)
        r=criterion(ur[i],true_res)
        lossi+=l
        lossr+=r
    loss_i = lossi/len(imglist)    
    loss_r = lossr/len(imglist)
    loss_all = loss_i+loss_r
    return loss_all

def genWeights(num):
    scales=[]
    for i in range(num):
        scale = ScaleLayer()
        scale.cuda()
        scales.append(scale)
    return scales

if __name__ == "__main__":
    main()