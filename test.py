import os
import time
import socket
import argparse
import torch
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn as cudnn



from models.net import Net
from utils.dirs2save import *
from utils.model_util import *
from models.vmamba import VSSM
from collections import OrderedDict
from utils.calculate_PSNR_SSIM import *
from data.data_load import load_data_rgb, load_data_multi

from robustness import *


parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='test', help='train or test')
parser.add_argument('--num_workers', type=int, default=4,
                    help='number of data loading workers')
parser.add_argument('--imageSize', type=int, default=256,
                    help='the number of frames')
parser.add_argument('--norm', default='instance', 
                    help='batch or instance')
parser.add_argument('--loss', default='l2', help='l1 or l2')
parser.add_argument('--beta_R', type=float, default=0.75, help='hyper parameter beta of reveal')
parser.add_argument('--cuda', type=bool, default=True,
                    help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1,
                    help='number of GPUs to use')
parser.add_argument('--trainpics', default='./training/',
                    help='folder to output training images')
parser.add_argument('--validationpics', default='./training/',
                    help='folder to output validation images')
parser.add_argument('--testPics', default='./training/',
                    help='folder to output test images')
parser.add_argument('--outckpts', default='./training/',
                    help='folder to output checkpoints')
parser.add_argument('--outlogs', default='./training/',
                    help='folder to output images')
parser.add_argument('--outcodes', default='./training/',
                    help='folder to save the experiment codes')
parser.add_argument('--Model_dir', default='', help='the path of model')
parser.add_argument('--Hmodel', default='checkpointH.pt', help='the filename of the checkpoint')
parser.add_argument('--Rmodel', default='checkpointR.pt', help='the filename of the checkpoint')
parser.add_argument('--hostname', default=socket.gethostname(), help='the host name of the running server')
parser.add_argument('--num_secret', type=int, default=1, help='How many secret images are hidden in one cover image?')
parser.add_argument('--num_cover', type=int, default=1, help='How many secret images are hidden in one cover image?')
parser.add_argument('--batch_stegs', type=int, default=32, help='How many stegs for one batch')
parser.add_argument('--num_training', type=int, default=1, help='During training, how many cover images are used for one secret image')
parser.add_argument('--channel_cover', type=int, default=3, help='1: gray; 3: color')
parser.add_argument('--channel_secret', type=int, default=3, help='1: gray; 3: color')


def main():

    args = parser.parse_args()
    args.ngpu = torch.cuda.device_count()
    if torch.cuda.is_available() and not args.cuda:
        print("You are not using GPU, it is recommended to use CUDA to run")

    cudnn.benchmark = True

    args = file(args)
    args = result_analysis(args)

    logPath = args.outlogs + '/log.txt' 


    ##################  Model initialize  ##################
    Hnet = Net(in_chans=args.channel_cover+args.channel_secret, out_chans=args.channel_cover, imgsize=args.imageSize, hidden_dim=32, depths=[4, 4, 4, 4], mode='ssdf', use_att=True)
    Rnet = Net(in_chans=args.channel_cover, out_chans=args.channel_secret, imgsize=args.imageSize, hidden_dim=32, depths=[4, 4, 4, 4], mode='ssdf', use_att=True)
    
    Hnet.cuda()
    Rnet.cuda()

    if args.ngpu > 1:
        Hnet = nn.DataParallel(Hnet).cuda()
        Rnet = nn.DataParallel(Rnet).cuda()


    checkpoint = "./training/" + args.Model_dir + "/checkPoints/"
    checkpointH = torch.load(checkpoint + args.Hmodel)
    checkpointR = torch.load(checkpoint + args.Rmodel)
    Hnet.load_state_dict(checkpointH['state_dict'])
    Rnet.load_state_dict(checkpointR['state_dict'])


    # print_network(Net, logPath)

    # Loss and Metric
    if args.loss == 'l1':
        criterion = nn.L1Loss().cuda()
    if args.loss == 'l2':
        criterion = nn.MSELoss().cuda()


    ##################  Datasets  ##################
    # DATA_DIR = '/home/user-yc/imagenet'
    # coverdir = os.path.join(DATA_DIR, 'testdir')
    # secretdir = os.path.join(DATA_DIR, 'testdir')
    # DATA_DIR = '/home/user-yc/datasets/COCO'
    # coverdir = os.path.join(DATA_DIR, 'test')
    # secretdir = os.path.join(DATA_DIR, 'test')

    COVER_DIR = '/home/user-yc/test_rgb'
    SECRET_DIR = '/home/user-yc/test_rgb'
    coverdir = os.path.join(COVER_DIR, 'cover')
    secretdir = os.path.join(SECRET_DIR, 'secret')

    batch_size_C=args.batch_stegs*args.num_cover
    batch_size_S=args.batch_stegs*args.num_secret

    cover_loader = load_data_rgb(args, coverdir, batch_size_C)
    secret_loader = load_data_rgb(args, secretdir, batch_size_S)

    testLoader = zip(cover_loader, secret_loader)


    print("#################################################### test begin ########################################################")
    start_time = time.time()

    Hnet.eval()
    Rnet.eval()

    Hlosses = AverageMeter() 
    Rlosses = AverageMeter()
    H_low_losses = AverageMeter()
    R_low_losses = AverageMeter() 
    Hdiff = AverageMeter()
    Rdiff = AverageMeter()
    SumLosses = AverageMeter()

    test_results = OrderedDict()
    test_results['apd_cover'] = []
    test_results['mse_cover'] = []
    test_results['psnr_cover'] = []
    test_results['ssim_cover'] = []

    test_results['apd_secret'] = []
    test_results['mse_secret'] = []
    test_results['psnr_secret'] = []
    test_results['ssim_secret'] = []

    epoch = 0

    with torch.no_grad():

        for i, ((cover_img, cover_target), (secret_img, secret_target)) in enumerate(testLoader, 0):
            Hnet.zero_grad()
            Rnet.zero_grad()

            batch_size_cover, channel_cover, _, _ = cover_img.size()
            batch_size_secret, channel_secret, _, _ = secret_img.size()

            if args.cuda:
                cover_img = cover_img.cuda()
                secret_img = secret_img.cuda()

            secret_img = secret_img.cuda()          

            # Adjust the tensor shape based on the number of cover and secret images
            cover_imgv = cover_img.view(batch_size_cover // args.num_cover, 
                                        channel_cover * args.num_cover, args.imageSize, args.imageSize)
            secret_imgv = secret_img.view(batch_size_secret // args.num_secret, 
                                            channel_secret * args.num_secret, args.imageSize, args.imageSize)        
            secret_imgv_r = secret_imgv.repeat(1,1,1,1)

            # input_guass = gauss_noise(cover_input.shape)
            # concat_img = torch.cat((cover_input, input_guass), dim=1)

            concat_img = torch.cat((cover_imgv, secret_imgv), dim=1)
            steg_img = Hnet(concat_img)

            # rev_guass = gauss_noise(steg_img.shape)

            # rev_in = torch.cat((steg_img, rev_guass), 1)

            # steg_img = right_crop(steg_img, 80)
            # steg_img = jpeg_compress(steg_img, 100)

            rev_img = Rnet(steg_img)
            
            # cover_dwt = dwt(cover_imgv)
            # secret_dwt = dwt(secret_imgv)
            # steg_dwt = dwt(steg_img)
            # rev_dwt = dwt(rev_img)

            # loss between cover/steg and secret/reveal image 
            errH = criterion(steg_img, cover_imgv)   
            errR = criterion(rev_img, secret_imgv_r)

            # # loss between the low-frequency of steg and cover image
            # steg_low_loss = steg_dwt.narrow(1, 0, 3)
            # cover_low_loss = cover_dwt.narrow(1, 0, 3)
            # hiding_low_err = criterion(steg_low_loss, cover_low_loss)

            # # loss between the low-frequency of reveal and secret image
            # rev_low_loss =  rev_dwt.narrow(1, 0, 3)
            # secret_low_loss = secret_dwt.narrow(1, 0, 3)
            # revealing_low_err = criterion(rev_low_loss, secret_low_loss)

            # L1 metric
            diffH = (steg_img - cover_imgv).abs().mean()*255
            diffR = (rev_img - secret_imgv_r).abs().mean()*255

            # Loss, backprop, and optimization step
            # total_loss = errH + args.beta_R * errR + args.beta_hl * hiding_low_err + args.beta_rl * revealing_low_err
            total_loss = errH + args.beta_R * errR

            Hlosses.update(errH.item(), args.batch_stegs*args.num_cover*args.num_training)
            Rlosses.update(errR.item(), args.batch_stegs*args.num_cover*args.num_training)
            Hdiff.update(diffH.item(), args.batch_stegs*args.num_cover*args.num_training)
            Rdiff.update(diffR.item(), args.batch_stegs*args.num_secret*args.num_training)
            # H_low_losses.update(hiding_low_err.item(), args.batch_stegs*args.num_cover)
            # R_low_losses.update(revealing_low_err.item(), args.batch_stegs*args.num_secret)  
            SumLosses.update(total_loss.item(), args.batch_stegs*args.num_cover*args.num_training)


            apd, mse, psnr, ssim = tensor_psnr_ssim(cover_imgv, steg_img)
            test_results['apd_cover'].append(apd)
            test_results['mse_cover'].append(mse)
            test_results['psnr_cover'].append(psnr)
            test_results['ssim_cover'].append(ssim)

            apd, mse, psnr, ssim = tensor_psnr_ssim(secret_imgv_r, rev_img)
            test_results['apd_secret'].append(apd)
            test_results['mse_secret'].append(mse)
            test_results['psnr_secret'].append(psnr)
            test_results['ssim_secret'].append(ssim)

            save_result_pic_test(args, cover_imgv, steg_img.data, secret_imgv_r, rev_img.data, epoch, i,
                            args.testPics)
            # save_result_pic(args, cover_imgv, steg_img.data, secret_imgv_r, rev_img.data, epoch, i,
            #                 args.testPics)
            

            if i > 134:
                break
            
    logPath = 'result_analysis/result/result_logs.txt'
    log_apdCover = 'apd of cover images: ' + ', '.join(['%.5f' % num for num in test_results['apd_cover']])
    log_mseCover = 'mse of cover images: ' + ', '.join(['%.5f' % num for num in test_results['mse_cover']])
    log_psnrCover = 'psnr of cover images: ' + ', '.join(['%.5f' % num for num in test_results['psnr_cover']])
    log_ssimCover = 'ssim of cover images: ' + ', '.join(['%.5f' % num for num in test_results['ssim_cover']])
    print_log(log_apdCover, logPath)
    print_log(log_mseCover, logPath)
    print_log(log_psnrCover, logPath)
    print_log(log_ssimCover, logPath)

    log_apdSecret = 'apd of secret images: ' + ', '.join(['%.5f' % num for num in test_results['apd_secret']])
    log_mseSecret = 'mse of secret images: ' + ', '.join(['%.5f' % num for num in test_results['mse_secret']])
    log_psnrSecret = 'psnr of secret images: ' + ', '.join(['%.5f' % num for num in test_results['psnr_secret']])
    log_ssimSecret = 'ssim of secret images: ' + ', '.join(['%.5f' % num for num in test_results['ssim_secret']])
    print_log(log_apdSecret, logPath)
    print_log(log_mseSecret, logPath)
    print_log(log_psnrSecret, logPath)
    print_log(log_ssimSecret, logPath)

    ave_apd_cover = sum(test_results['apd_cover']) / len(test_results['apd_cover'])
    ave_mse_cover = sum(test_results['mse_cover']) / len(test_results['mse_cover'])
    ave_psnr_cover = sum(test_results['psnr_cover']) / len(test_results['psnr_cover'])
    ave_ssim_cover = sum(test_results['ssim_cover']) / len(test_results['ssim_cover'])

    ave_apd_secret = sum(test_results['apd_secret']) / len(test_results['apd_secret'])
    ave_mse_secret = sum(test_results['mse_secret']) / len(test_results['mse_secret'])
    ave_psnr_secret = sum(test_results['psnr_secret']) / len(test_results['psnr_secret'])
    ave_ssim_secret = sum(test_results['ssim_secret']) / len(test_results['ssim_secret'])
    
    log_ave = 'ave_apd_cover = %.4f\nave_mse_cover = %.4f\nave_psnr_cover = %.4f\nave_ssim_cover = %.4f\nave_apd_secret = %.4f\nave_mse_secret = %.4f\nave_psnr_secret = %.4f\nave_ssim_secret = %.4f' % (
        ave_apd_cover, ave_mse_cover, ave_psnr_cover, ave_ssim_cover, ave_apd_secret, ave_mse_secret, ave_psnr_secret, ave_ssim_secret)
    print_log(log_ave, logPath)


    h_loss = Hlosses.avg
    r_loss = Rlosses.avg
    h_low_loss = H_low_losses.avg
    r_low_loss = R_low_losses.avg
    total_loss = SumLosses.avg

    val_time = time.time() - start_time
    val_log = "validation[%d] H_loss = %.6f\t R_loss = %.6f\t h_low_loss = %.6f\t r_low_loss = %.6f\t total_loss = %.6f\t validation time=%.2f" % (
        epoch, h_loss, r_loss, h_low_loss, r_low_loss, total_loss, val_time)
    print_log(val_log, logPath)

    print("#################################################### test end ########################################################")




if __name__ == '__main__':
    main()