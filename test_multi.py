import os
import time
import socket
import argparse
import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import torch.backends.cudnn as cudnn

from models.net import Net
from utils.model_util import *
from collections import OrderedDict
from utils.calculate_PSNR_SSIM import *
from utils.dirs2save import file, result_analysis
from data.data_load import load_data_rgb, load_data_multi


from robustness import *


parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='test', help='train or test')
parser.add_argument('--num_workers', type=int, default=2,
                    help='number of data loading workers')
parser.add_argument('--imageSize', type=int, default=256,
                    help='the size of image')
parser.add_argument('--norm', default='instance', help='batch or instance')
parser.add_argument('--loss', default='l2', help='l1 or l2')

parser.add_argument('--epochs', type=int, default=500, help='number of epochs to train for')
parser.add_argument('--iters_epochs', type=int, default=20000, help='train numbers of each epoch')
parser.add_argument('--warmup_epoch', type=int, default=5, help='number of epochs to train for warmup')

parser.add_argument('--lr_H', type=float, default=0.001,
                    help='Hnet learning rate, default=0.0001')
parser.add_argument('--lr_R', type=float, default=0.001,
                    help='Hnet learning rate, default=0.0001')
parser.add_argument('--min_lr', type=float, default=6e-5, help='Hnet learning rate, default=0.0001')
parser.add_argument('--warmup_lr', type=float, default=6e-6, help='Hnet learning rate, default=0.0001')

parser.add_argument('--beta_R', type=float, default=0.75, help='hyper parameter beta of reveal')
parser.add_argument('--beta_hl', type=float, default=0.75, help='hyper parameter beta of the low-frequency hiding')
parser.add_argument('--beta_rl', type=float, default=0.75, help='hyper parameter beta of the low-frequency revealing')

parser.add_argument('--cuda', type=bool, default=True,
                    help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1,
                    help='number of GPUs to use')

parser.add_argument('--dyt', type=bool, default=False, help='whether to choose layernorm')


parser.add_argument('--trainpics', default='./training_MSI/',
                    help='folder to output training images')
parser.add_argument('--validationpics', default='./training_MSI/',
                    help='folder to output validation images')
parser.add_argument('--testPics', default='./training_MSI/',
                    help='folder to output test images')
parser.add_argument('--outckpts', default='./training_MSI/',
                    help='folder to output checkpoints')
parser.add_argument('--outlogs', default='./training_MSI/',
                    help='folder to output images')
parser.add_argument('--outcodes', default='./training_MSI/',
                    help='folder to save the experiment codes')
# whether to continue training
parser.add_argument('--train_continue', type=bool, default=False, 
                    help='continue to training')
parser.add_argument('--trained_epochs', type=int, default=0, help='the epochs of the train')
parser.add_argument('--Model_dir', default='', help='the dir of model')
parser.add_argument('--Hmodel', default='checkpointH.pt', help='the filename of the checkpoint')
parser.add_argument('--Rmodel', default='checkpointR.pt', help='the filename of the checkpoint')

parser.add_argument('--save_freq', default=10, help='How many epochs to save model')
parser.add_argument('--log_freq', type=int, default=10, help='the frequency of print the log on the console')
parser.add_argument('--resultPicFrequency', type=int, default=500, help='the frequency of save the resultPic')
parser.add_argument('--num_secret', type=int, default=1, help='How many secret images are hidden in one cover image?')
parser.add_argument('--num_cover', type=int, default=1, help='How many secret images are hidden in one cover image?')
parser.add_argument('--batch_stegs', type=int, default=1, help='How many stegs for one batch')
parser.add_argument('--channel_cover', type=int, default=3, help='1: gray; 3: color')
parser.add_argument('--channel_secret', type=int, default=5, help='1: gray; 3: color')




def main():
    args = parser.parse_args()
    args.ngpu = torch.cuda.device_count()
    if torch.cuda.is_available() and not args.cuda:
        print("You are not using GPU, it is recommended to use CUDA to run")

    cudnn.benchmark = True

    args = file(args)
    args = result_analysis(args)

    logPath = args.outlogs + '/log.txt'
    print_log(str(args), logPath)

    ##################  Model initialize  ##################
    Hnet = Net(in_chans=args.channel_cover+args.channel_secret, out_chans=args.channel_cover, imgsize=args.imageSize, hidden_dim=32, depths=[4, 4, 4, 4], mode='ssdf', use_att=True)
    Rnet = Net(in_chans=args.channel_cover, out_chans=args.channel_secret, imgsize=args.imageSize, hidden_dim=32, depths=[4, 4, 4, 4], mode='ssdf', use_att=True)
    
    Hnet.cuda()
    Rnet.cuda()

    if args.ngpu > 1:
        Hnet = nn.DataParallel(Hnet).cuda()
        Rnet = nn.DataParallel(Rnet).cuda()


    # load model
    checkpoint = "./training_MSI/" + args.Model_dir + "/checkPoints/"
    checkpointH = torch.load(checkpoint + args.Hmodel)
    checkpointR = torch.load(checkpoint + args.Rmodel)
    Hnet.load_state_dict(checkpointH['state_dict'])
    Rnet.load_state_dict(checkpointR['state_dict'])

    # Loss and Metric
    if args.loss == 'l1':
        criterion = nn.L1Loss().cuda()
    if args.loss == 'l2':
        criterion = nn.MSELoss().cuda()


    ##################  Datasets  ##################
    # COVER_DIR = '/home/user-yc/imagenet'
    # SECRET_DIR = '/home/user-yc/datasets/moscow'
    # # SECRET_DIR = '/home/user-yc/datasets/Sanjuan'
    # test_cover = os.path.join(COVER_DIR, 'testdir')
    # test_secret = os.path.join(SECRET_DIR, 'test')
    # # test_secret = os.path.join(SECRET_DIR, 'MS')

    # COVER_DIR = '/home/user-yc/test_mis'
    # SECRET_DIR = '/home/user-yc/test_mis'
    # test_cover = os.path.join(COVER_DIR, 'cover')
    # test_secret = os.path.join(SECRET_DIR, 'secret/1')

    COVER_DIR = 'data/analysis_fig'
    SECRET_DIR = 'data/analysis_fig'
    test_cover = os.path.join(COVER_DIR, 'cover')
    test_secret = os.path.join(SECRET_DIR, 'secret/1')

    batch_size_C=args.batch_stegs*args.num_cover
    batch_size_S=args.batch_stegs*args.num_secret
    test_loader_cover = load_data_rgb(args, test_cover, batch_size_C)
    test_loader_secret = load_data_multi(args, test_secret, batch_size_S)

    testLoader = zip(test_loader_cover, test_loader_secret)


    print("#################################################### test begin ########################################################")
    Hlosses = AverageMeter()
    Rlosses = AverageMeter()
    TotalLosses = AverageMeter()
    Hdiff = AverageMeter()
    Rdiff = AverageMeter()

    test_results = OrderedDict()
    test_results['apd_cover'] = []
    test_results['mse_cover'] = []
    test_results['psnr_cover'] = []
    test_results['ssim_cover'] = []

    test_results['apd_secret'] = []
    test_results['mse_secret'] = []
    test_results['psnr_secret'] = []
    test_results['ssim_secret'] = []

    # start_time = time.time()
    epoch = 0

    with torch.no_grad():
        Hnet.eval()
        Rnet.eval()
        for i, ((cover_img, cover_target), (secret_img)) in enumerate(testLoader, 0):
            batch_size_cover, channel_cover, _, _ = cover_img.size()
            batch_size_secret, channel_secret, _, _ = secret_img.size()

            start_time = time.time()
            if args.cuda:
                cover_img = cover_img.cuda()
                secret_img = secret_img.cuda()

            # Adjust the tensor shape based on the number of cover and secret images
            cover_imgv = cover_img.view(batch_size_cover // args.num_cover, 
                                        channel_cover * args.num_cover, args.imageSize, args.imageSize)
            secret_imgv = secret_img.view(batch_size_secret // args.num_secret, 
                                        channel_secret * args.num_secret, args.imageSize, args.imageSize)        
            secret_imgv_r = secret_imgv.repeat(1,1,1,1)

            concat_img = torch.cat((cover_imgv, secret_imgv), dim=1)

            # hiding and revealing
            steg_img = Hnet(concat_img)
            # print('steg_img shape:', steg_img.shape)


            # steg_attack = add_gaussian_noise(steg_img)
            # steg_attack = add_multiplicative_noise(steg_img)
            # steg_attack = add_salt_and_pepper_noise(steg_img)
            # steg_attack = add_rayleigh_noise(steg_img)
            # steg_attack = add_gamma_noise(steg_img)
            # steg_attack = jpeg_compress(steg_img, 90)
            # steg_attack = rotate(steg_img, 45)


            # steg_attack = right_crop(steg_img, 70)
            # steg_attack = center_crop(steg_img, 0.3)
            # steg_attack_Name = 'result_analysis/result/steg/1/steg_attack.png'
            # vutils.save_image(steg_attack, steg_attack_Name, padding=1, normalize=True)


            # rev_img = Rnet(steg_attack)
            rev_img = Rnet(steg_img)


            # logPath = 'result_analysis/result/result_logs.txt'
            # test_time = time.time() - start_time
            # val_log = "test time=%.2f" % (test_time)
            # ######### calculate flops #########
            # from thop import profile
            # input = torch.randn(1, 11, 256, 256).cuda()
            # input1 = torch.randn(1, 3, 256, 256).cuda()
            # flops, params = profile(Hnet, inputs=(input, ))
            # flops1, params1 = profile(Rnet, inputs=(input1, ))

            # val_log = "flops:%d, params:%d, flops:%d, params:%d, test time=%.2f" % (flops, params, flops1, params1, test_time)
            # print_log(val_log, logPath)


            # loss between cover/steg and secret/reveal image 
            errH = criterion(steg_img, cover_imgv)   
            errR = criterion(rev_img, secret_imgv_r)
            # L1 metric
            diffH = (steg_img - cover_imgv).abs().mean()*255
            diffR = (rev_img - secret_imgv_r).abs().mean()*255

            total_loss = errH + args.beta_R * errR

            Hlosses.update(errH.item(), args.batch_stegs*args.num_cover)
            Rlosses.update(errR.item(), args.batch_stegs*args.num_cover)
            Hdiff.update(diffH.item(), args.batch_stegs*args.num_cover)
            Rdiff.update(diffR.item(), args.batch_stegs*args.num_secret)  
            TotalLosses.update(total_loss.item(), args.batch_stegs*args.num_cover)


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

            save_testpic_multi(args, cover_imgv, steg_img.data, secret_imgv_r, rev_img.data, epoch, i, args.testPics)


            for band in range(secret_img.shape[1]):  # 遍历每个波段

                # 生成文件名，保存每个波段
                rev_bandImgName = 'result_analysis/result/rev/rev_img%04d_band%02d.png' % (i, band)
                secret_bandImgName = 'result_analysis/result/secret/1/secret_img%04d_band%02d.png' % (i, band)            
                # 提取每个波段
                secret_i_band = secret_img[:, band, :, :]
                rev_secret_i_band = rev_img[:, band, :, :]

                # 保存每个波段的图像
                vutils.save_image(secret_i_band, secret_bandImgName, padding=1, normalize=True)
                vutils.save_image(rev_secret_i_band, rev_bandImgName, padding=1, normalize=True)


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
    total_loss = TotalLosses.avg

    val_time = time.time() - start_time
    val_log = "validation[%d] H_loss = %.6f\t R_loss = %.6f\t total_loss = %.6f\t validation time=%.2f" % (
        epoch, h_loss, r_loss, total_loss, val_time)
    print_log(val_log, logPath)

    print("#################################################### test end ########################################################")



if __name__ == '__main__':
    main()