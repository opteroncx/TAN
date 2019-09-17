# -*- coding:utf-8 -*-
import argparse
import torch
import os
import cv2
import pyssim
import codecs
from scipy.ndimage import gaussian_filter
from numpy.lib.stride_tricks import as_strided as ast
from PIL import Image
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import time, math
import scipy.io as sio
from skimage import measure, io
from functools import partial
import pickle
from model8b8xt import Net
parser = argparse.ArgumentParser(description="TAN Test")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="model/model_epoch_100.pth", type=str, help="model path")
parser.add_argument("--image", default="butterfly_GT", type=str, help="image name")
parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")
parser.add_argument("--testdir", default='all', type=str, help="")  #"testdir/Urban100a"
parser.add_argument("--mode", default="evaluate", type=str, help="")

opt = parser.parse_args()
cuda = opt.cuda

if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

def savelog(path,psnr,ssim):
    log_path='./log/'
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    test_time=time.time()
    test_time=str(int(test_time))
    log=codecs.open(log_path+'test_log'+'.txt','a+','utf-8')
    log.writelines("=======================================\n")
    log.writelines(test_time+'\n')
    log.writelines(path+'\n')
    log.writelines('PSNR==>%f  \n'%psnr)
    log.writelines('SSIM==>%f  \n'%ssim)
    log.close()

def eval():
    if opt.testdir == 'all':
        # run all tests
        testdirs=["testdir/Set5b","testdir/Set14b","testdir/bsd100a","testdir/Urban100a"]
        for t in testdirs:
            evaluate_by_path('../lapsrn/'+t)
    else:
        t=opt.testdir
        evaluate_by_path(t)

def data_trans(im,num):
    org_image = im
    if num ==0:    
        ud_image = np.flipud(org_image)
        tranform = ud_image
    elif num ==1:      
        lr_image = np.fliplr(org_image)
        tranform = lr_image
    elif num ==2:
        lr_image = np.fliplr(org_image)
        lrud_image = np.flipud(lr_image)        
        tranform = lrud_image
    elif num ==3:
        rotated_image1 = np.rot90(org_image)        
        tranform = rotated_image1
    elif num ==4: 
        rotated_image2 = np.rot90(org_image, -1)
        tranform = rotated_image2
    elif num ==5: 
        rotated_image1 = np.rot90(org_image) 
        ud_image1 = np.flipud(rotated_image1)
        tranform = ud_image1
    elif num ==6:        
        rotated_image2 = np.rot90(org_image, -1)
        ud_image2 = np.flipud(rotated_image2)
        tranform = ud_image2
    else:
        tranform = org_image
    return tranform

def data_trans_inv(im,num):
    org_image = im
    if num ==0:    
        ud_image = np.flipud(org_image)
        tranform = ud_image
    elif num ==1:      
        lr_image = np.fliplr(org_image)
        tranform = lr_image
    elif num ==2:
        lr_image = np.fliplr(org_image)
        lrud_image = np.flipud(lr_image)        
        tranform = lrud_image
    elif num ==3:
        rotated_image1 = np.rot90(org_image,-1)        
        tranform = rotated_image1
    elif num ==4: 
        rotated_image2 = np.rot90(org_image)
        tranform = rotated_image2
    elif num ==5: 
        rotated_image1 = np.rot90(org_image) 
        ud_image1 = np.flipud(rotated_image1)
        tranform = ud_image1
    elif num ==6:        
        rotated_image2 = np.rot90(org_image, -1)
        ud_image2 = np.flipud(rotated_image2)
        tranform = ud_image2
    else:
        tranform = org_image
    return tranform

def evaluate_by_path(path):
    pimages=os.listdir(path)
    s_psnr=0
    s_ssim=0
    s_time=0
    save=True
    eva=True
    convert=True
    for pimg in pimages:
        img = io.imread(path+'/'+pimg)
        im_list = []
        for i in range(8):
            tmp = data_trans(img,i)
            seim1=predict(tmp,save,convert,eva,pimg)
            seim2=data_trans_inv(seim1,i)
            print('i===',i,'shape==',seim2.shape)
            im_list.append(seim2)
        for i in range(len(im_list)):
            if i == 0:
                sum = im_list[0]
            else:
                sum += im_list[i]
        avg = sum/len(im_list)
        psnr,ssim = eva_se(avg,img,pimg)
        s_psnr+=psnr
        s_ssim+=ssim

    avg_psnr=s_psnr/len(pimages)
    avg_ssim=s_ssim/len(pimages)
    avg_time=s_time/len(pimages)
    print_summary(avg_psnr,avg_ssim,avg_time)
    savelog(path,avg_psnr,avg_ssim)

def predict(img_read, save, convert, eva, name):
    if convert:
        if eva:
            h, w, _ = img_read.shape
            im_gt_y = convert_rgb_to_y(img_read)
            gt_yuv = convert_rgb_to_ycbcr(img_read)
            im_gt_y = im_gt_y.astype("float32")

            sc = 1.0 / opt.scale
            img_y = resize_image_by_pil(im_gt_y, sc)
            img_y = img_y[:, :, 0]
            im_gt_y = im_gt_y[:, :, 0]
        else:
            sc = opt.scale
            tmp = resize_image_by_pil(img_read, sc)
            gt_yuv = convert_rgb_to_ycbcr(tmp)
            img_y = convert_rgb_to_y(img_read)
            img_y = img_y.astype("float32")
    else:
        im_gt_y, img_y = img_read
        im_gt_y = im_gt_y.astype("float32")
    im_input = img_y / 255.
    im_input = Variable(torch.from_numpy(im_input).float()).view(1, -1, im_input.shape[0], im_input.shape[1])
    img_y = np.uint8(img_y)

    UseCPU = True
    model = Net()
    weights = torch.load(opt.model)
    saved_state = weights['model'].state_dict()
    if UseCPU:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in saved_state.items():
            namekey = k[7:] # remove `module.`
            new_state_dict[namekey] = v
            # load params
        model.load_state_dict(new_state_dict)
    if cuda:        
        model = model.cuda()
        im_input = im_input.cuda()
    else:
        model = model.cpu()

    start_time = time.time()
    # model=nn.DataParallel(model,device_ids=[0,1], output_device=1)
    
    if opt.scale ==2:
        HR_2x,a,b = model(im_input)
    elif opt.scale ==4:
        HR_4x,a,b = model(im_input)
    else:
        HR_8x,a,b = model(im_input)
    # elapsed_time = time.time() - start_time
    if opt.scale == 2:
        HR_2x = HR_2x[-1].cpu()
        im_h_y = HR_2x.data[0].numpy().astype(np.float32)
    elif opt.scale == 4:
        HR_4x = HR_4x[-1].cpu()
        im_h_y = HR_4x.data[0].numpy().astype(np.float32)
    else:
        HR_8x = HR_8x[-1].cpu()
        im_h_y = HR_8x.data[0].numpy().astype(np.float32)

    im_h_y = im_h_y * 255.
    im_h_y[im_h_y < 0] = 0
    im_h_y[im_h_y > 255.] = 255.
    im_h_y = im_h_y[0, :, :]
    
    return im_h_y

def eva_se(im_h_y,img_read,name):
    convert = True
    eva = True
    save = True
    if convert:
        if eva:
            h, w, _ = img_read.shape
            im_gt_y = convert_rgb_to_y(img_read)
            gt_yuv = convert_rgb_to_ycbcr(img_read)
            im_gt_y = im_gt_y.astype("float32")

            sc = 1.0 / opt.scale
            img_y = resize_image_by_pil(im_gt_y, sc)
            img_y = img_y[:, :, 0]
            im_gt_y = im_gt_y[:, :, 0]
        else:
            sc = opt.scale
            tmp = resize_image_by_pil(img_read, sc)
            gt_yuv = convert_rgb_to_ycbcr(tmp)
            img_y = convert_rgb_to_y(img_read)
            img_y = img_y.astype("float32")
    else:
        im_gt_y, img_y = img_read
        im_gt_y = im_gt_y.astype("float32")    
    
    if save:
        # recon= im_h_y
        recon = convert_y_and_cbcr_to_rgb(im_h_y, gt_yuv[:, :, 1:3])
        save_figure(recon, name)
    if eva:
        # PSNR and SSIM
        psnr_predicted = PSNR(im_gt_y, im_h_y, shave_border=opt.scale)
        ssim_predicted = pyssim.compute_ssim(im_gt_y, im_h_y)
        print("test psnr/ssim=%f/%f" % (psnr_predicted, ssim_predicted))
        return psnr_predicted, ssim_predicted

def print_summary(psnr,ssim,time):
    print("Scale=",opt.scale)
    print("PSNR=", psnr)
    print("SSIM=",ssim)
    print("time=",time)

def save_figure(img,name):
    out_path='./save_img/1107se8x/'
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    print('saved '+name)
    # rgb -> bgr
    tmp = np.zeros([img.shape[0],img.shape[1],img.shape[2]])
    tmp[:,:,0] = img[:,:,2]
    tmp[:,:,1] = img[:,:,1]
    tmp[:,:,2] = img[:,:,0]
    cv2.imwrite(out_path+name[:-4]+'.png',tmp)

def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

def convert_rgb_to_y(image, jpeg_mode=False, max_value=255.0):
    if len(image.shape) <= 2 or image.shape[2] == 1:
        return image

    if jpeg_mode:
        xform = np.array([[0.299, 0.587, 0.114]])
        y_image = image.dot(xform.T)
    else:
        xform = np.array([[65.738 / 256.0, 129.057 / 256.0, 25.064 / 256.0]])
        y_image = image.dot(xform.T) + (16.0 * max_value / 256.0)

    return y_image


def convert_rgb_to_ycbcr(image, jpeg_mode=False, max_value=255):
    if len(image.shape) < 2 or image.shape[2] == 1:
        return image

    if jpeg_mode:
        xform = np.array([[0.299, 0.587, 0.114], [-0.169, - 0.331, 0.500], [0.500, - 0.419, - 0.081]])
        ycbcr_image = image.dot(xform.T)
        ycbcr_image[:, :, [1, 2]] += max_value / 2
    else:
        xform = np.array(
            [[65.738 / 256.0, 129.057 / 256.0, 25.064 / 256.0], [- 37.945 / 256.0, - 74.494 / 256.0, 112.439 / 256.0],
             [112.439 / 256.0, - 94.154 / 256.0, - 18.285 / 256.0]])
        ycbcr_image = image.dot(xform.T)
        ycbcr_image[:, :, 0] += (16.0 * max_value / 256.0)
        ycbcr_image[:, :, [1, 2]] += (128.0 * max_value / 256.0)

    return ycbcr_image


def convert_y_and_cbcr_to_rgb(y_image, cbcr_image, jpeg_mode=False, max_value=255.0):

    if len(y_image.shape) == 3 and y_image.shape[2] == 3:
        y_image = y_image[:, :, 0:1]

    ycbcr_image = np.zeros([y_image.shape[0], y_image.shape[1], 3])
    ycbcr_image[:, :, 0] = y_image
    ycbcr_image[:, :, 1:3] = cbcr_image[:, :, 0:2]

    return convert_ycbcr_to_rgb(ycbcr_image)


def convert_ycbcr_to_rgb1(ycbcr_image, jpeg_mode=False, max_value=255.0):
    rgb_image = np.zeros([ycbcr_image.shape[0], ycbcr_image.shape[1], 3])  # type: np.ndarray

    if jpeg_mode:
        rgb_image[:, :, [1, 2]] = ycbcr_image[:, :, [1, 2]] - (128.0 * max_value / 256.0)
        xform = np.array([[1, 0, 1.402], [1, - 0.344, - 0.714], [1, 1.772, 0]])
        rgb_image = rgb_image.dot(xform.T)
    else:
        rgb_image[:, :, 0] = ycbcr_image[:, :, 0] - (16.0 * max_value / 256.0)
        rgb_image[:, :, [1, 2]] = ycbcr_image[:, :, [1, 2]] - (128.0 * max_value / 256.0)
        xform = np.array(
            [[max_value / 219.0, 0, max_value * 0.701 / 112.0],
             [max_value / 219, - max_value * 0.886 * 0.114 / (112 * 0.587), - max_value * 0.701 * 0.299 / (112 * 0.587)],
             [max_value / 219.0, max_value * 0.886 / 112.0, 0]])
        rgb_image = rgb_image.dot(xform.T)

    return rgb_image

def convert_ycbcr_to_rgb(ycbcr_image):
    rgb_image = np.zeros([ycbcr_image.shape[0], ycbcr_image.shape[1], 3])  # type: np.ndarray

    rgb_image[:, :, 0] = ycbcr_image[:, :, 0] - 16.0
    rgb_image[:, :, [1, 2]] = ycbcr_image[:, :, [1, 2]] - 128.0
    xform = np.array(
        [[298.082 / 256.0, 0, 408.583 / 256.0],
         [298.082 / 256.0, -100.291 / 256.0, -208.120 / 256.0],
         [298.082 / 256.0, 516.412 / 256.0, 0]])
    rgb_image = rgb_image.dot(xform.T)

    return rgb_image

def resize_image_by_pil(image, scale, resampling_method="bicubic"):
    width, height = image.shape[1], image.shape[0]
    new_width = int(width * scale)
    new_height = int(height * scale)

    if resampling_method == "bicubic":
        method = Image.BICUBIC
    elif resampling_method == "bilinear":
        method = Image.BILINEAR
    elif resampling_method == "nearest":
        method = Image.NEAREST
    else:
        method = Image.LANCZOS

    if len(image.shape) == 3 and image.shape[2] == 3:
        image = Image.fromarray(image, "RGB")
        image = image.resize([new_width, new_height], resample=method)
        image = np.asarray(image)
    elif len(image.shape) == 3 and image.shape[2] == 4:
        # RGBA
        image = Image.fromarray(image, "RGB")
        image = image.resize([new_width, new_height], resample=method)
        image = np.asarray(image)
    else:
        image = Image.fromarray(image.reshape(height, width))
        image = image.resize([new_width, new_height], resample=method)
        image = np.asarray(image)
        image = image.reshape(new_height, new_width, 1)
    return image

def main():
    if opt.mode=="evaluate":
        eval()

if __name__ == '__main__':
    main()
