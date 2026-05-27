import os
import time
import socket
import argparse
import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

from models.net import Net
from utils.dirs2save import *
from utils.model_util import *
from utils.calculate_PSNR_SSIM import *
from utils.cosine_lr import CosineLRScheduler 
from data.data_load import load_data_rgb, load_data_multi



parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train', help='train or test')
parser.add_argument('--num_workers', type=int, default=2,
                    help='number of data loading workers')
parser.add_argument('--imageSize', type=int, default=144,
                    help='the size of image')
parser.add_argument('--norm', default='instance', help='batch or instance')
parser.add_argument('--loss', default='l2', help='l1 or l2')

parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--train_iters_epochs', type=int, default=20000, help='train numbers of each epoch')
parser.add_argument('--val_iters_epochs', type=int, default=2000, help='val numbers of each epoch')
parser.add_argument('--warmup_epoch', type=int, default=5, help='number of epochs to train for warmup')

parser.add_argument('--lr_H', type=float, default=0.001,
                    help='Hnet learning rate, default=0.0001')
parser.add_argument('--lr_R', type=float, default=0.001,
                    help='Hnet learning rate, default=0.0001')
parser.add_argument('--min_lr', type=float, default=6e-5, help='Hnet learning rate, default=0.0001')
parser.add_argument('--warmup_lr', type=float, default=6e-6, help='Hnet learning rate, default=0.0001')

parser.add_argument('--beta_R', type=float, default=0.75, help='hyper parameter beta of reveal')

parser.add_argument('--cuda', type=bool, default=True,
                    help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1,
                    help='number of GPUs to use')

parser.add_argument('--dyt', type=bool, default=False, help='whether to choose layernorm')


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
# whether to continue training
parser.add_argument('--train_continue', type=bool, default=False, 
                    help='continue to training')
parser.add_argument('--trained_epochs', type=int, default=0, help='the epochs of the train')
parser.add_argument('--Model_dir', default='', help='the dir of model')

parser.add_argument('--save_freq', default=10, help='How many epochs to save model')
parser.add_argument('--log_freq', type=int, default=10, help='the frequency of print the log on the console')
parser.add_argument('--resultPicFrequency', type=int, default=1000, help='the frequency of save the resultPic')
parser.add_argument('--num_secret', type=int, default=1, help='How many secret images are hidden in one cover image?')
parser.add_argument('--num_cover', type=int, default=1, help='How many secret images are hidden in one cover image?')
parser.add_argument('--batch_stegs', type=int, default=1, help='How many stegs for one batch')
parser.add_argument('--channel_cover', type=int, default=3, help='1: gray; 3: color')
parser.add_argument('--channel_secret', type=int, default=3, help='1: gray; 3: color')




def main():
    ############### define global parameters ###############
    global args, optimizerH, optimizerR, writer, logPath, schedulerH, schedulerR, valLoader, smallestLoss

    args = parser.parse_args()
    args.ngpu = torch.cuda.device_count()
    if torch.cuda.is_available() and not args.cuda:
        print("You are not using GPU, it is recommended to use CUDA to run")

    cudnn.benchmark = True

    args = file(args)

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

    # Loss and Metric
    if args.loss == 'l1':
        criterion = nn.L1Loss().cuda()
    if args.loss == 'l2':
        criterion = nn.MSELoss().cuda()


    optimizerH = optim.AdamW(Hnet.parameters(), eps=1e-8, betas=(0.9, 0.999), lr=args.lr_H, weight_decay=0.05)
    optimizerR = optim.AdamW(Rnet.parameters(), eps=1e-8, betas=(0.9, 0.999), lr=args.lr_R, weight_decay=0.05)

    schedulerH = CosineLRScheduler(
        optimizerH,
        t_initial=(args.epochs - args.warmup_epoch),
        t_mul=1.,
        lr_min=args.min_lr,
        warmup_lr_init=args.warmup_lr,
        warmup_t=args.warmup_epoch,
        cycle_limit=1,
        t_in_epochs=True,
        warmup_prefix=True,
    )
    schedulerR = CosineLRScheduler(
        optimizerR,
        t_initial=(args.epochs - args.warmup_epoch),
        t_mul=1.,
        lr_min=args.min_lr,
        warmup_lr_init=args.warmup_lr,
        warmup_t=args.warmup_epoch,
        cycle_limit=1,
        t_in_epochs=True,
        warmup_prefix=True,
    )

    # load model
    if args.train_continue:
        checkpoint_h = f"./training/{args.Model_dir}/checkPoints/checkpointH.pt"
        checkpoint_r = f"./training/{args.Model_dir}/checkPoints/checkpointR.pt"
        checkpointH = torch.load(checkpoint_h)
        checkpointR = torch.load(checkpoint_r)
        Hnet.load_state_dict(checkpointH['state_dict'])
        Rnet.load_state_dict(checkpointR['state_dict'])

        optimizerH.load_state_dict(checkpointH['optimizer'])
        optimizerR.load_state_dict(checkpointR['optimizer'])

        schedulerH.load_state_dict(checkpointH['scheduler'])
        schedulerR.load_state_dict(checkpointR['scheduler'])

        # schedulerH.step(args.trained_epochs + 1)
        # schedulerR.step(args.trained_epochs + 1)        
    else:
        save_current_codes(args.outcodes)

        print_network(Hnet, logPath)
        print_network(Rnet, logPath)

    # write the networks
    writer = SummaryWriter(log_dir='runs/' + args.experiment_dir)


    ##################  Datasets  ##################
    DATA_DIR = '/home/datasets/imagenet'
    traindir = os.path.join(DATA_DIR, 'train')
    valdir = os.path.join(DATA_DIR, 'validation')

    batch_size_C=args.batch_stegs*args.num_cover
    batch_size_S=args.batch_stegs*args.num_secret

    train_loader_cover = load_data_rgb(args, traindir, batch_size_C)
    train_loader_secret = load_data_rgb(args, traindir, batch_size_S)
    val_loader_cover = load_data_rgb(args, valdir, batch_size_C)
    val_loader_secret = load_data_rgb(args, valdir, batch_size_S)


    ##################  start to training  ##################
    print_log("............................ start to training ...........................", logPath)
    smallestLoss = 10000
    for epoch in range(args.trained_epochs, args.epochs):
        trainLoader = zip(train_loader_cover, train_loader_secret)
        valLoader = zip(val_loader_cover, val_loader_secret)

        train(trainLoader, epoch, Hnet=Hnet, Rnet=Rnet, criterion=criterion)
        val_hloss, val_rloss, val_hdiff, val_rdiff = validation(valLoader, epoch, Hnet=Hnet, Rnet=Rnet, criterion=criterion)

        total_loss = val_hloss + args.beta_R * val_rloss

        schedulerH.step(epoch)
        schedulerR.step(epoch)

        # save the best model parameters
        is_best = total_loss < globals()["smallestLoss"]
        globals()["smallestLoss"] = total_loss

        save_model(args, {
            'epoch': epoch,
            'state_dict': Hnet.state_dict(),
            'optimizer' : optimizerH.state_dict(),
            'scheduler' : schedulerH.state_dict(),
        }, is_best, epoch, '%s/epoch_%d_Hloss_%.4f_Hdiff=%.4f_Rloss_%.4f_Rdiff=%.4f_Total_loss_%.4f' 
        % (args.outckpts, epoch, val_hloss, val_hdiff, val_rloss, val_rdiff, total_loss), 'H')

        save_model(args, {
            'epoch': epoch,
            'state_dict': Rnet.state_dict(),
            'optimizer' : optimizerR.state_dict(),
            'scheduler' : schedulerR.state_dict(),
        }, is_best, epoch, '%s/epoch_%d_Hloss_%.4f_Hdiff=%.4f_Rloss_%.4f_Rdiff=%.4f_Total_loss_%.4f' 
        % (args.outckpts, epoch, val_hloss, val_hdiff, val_rloss, val_rdiff, total_loss), 'R')

    writer.close()


def train(trainLoader, epoch, Hnet, Rnet, criterion):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    Hlosses = AverageMeter()
    Rlosses = AverageMeter()
    TotalLosses = AverageMeter()  # Hloss + β*Rloss
    Hdiff = AverageMeter()
    Rdiff = AverageMeter()
    PSNR_C = AverageMeter()
    PSNR_S = AverageMeter()
    SSIM_C = AverageMeter()
    SSIM_S = AverageMeter()

    # switch to train mode
    Hnet.train()
    Rnet.train()

    start_time = time.time()
    for i, ((cover_img, cover_target), (secret_img, secret_target)) in enumerate(trainLoader, 0):

        # print('cover_img shape:', cover_img.shape, 'secret_img shape:', secret_img.shape)
        data_time.update(time.time() - start_time)

        cover_imgv, steg_img, secret_imgv_r, rev_img, errH, errR, diffH, diffR = steg(args, cover_img, secret_img, Hnet, Rnet, criterion)

        # Loss function
        err_total = errH + args.beta_R * errR

        Hlosses.update(errH.item(), args.batch_stegs*args.num_cover)
        Rlosses.update(errR.item(), args.batch_stegs*args.num_secret)
        Hdiff.update(diffH.item(), args.batch_stegs*args.num_cover)
        Rdiff.update(diffR.item(), args.batch_stegs*args.num_secret)
        TotalLosses.update(err_total.item(), args.batch_stegs*(args.num_cover+args.num_secret))

        optimizerH.zero_grad()
        optimizerR.zero_grad()
        err_total.backward()
        optimizerH.step()
        optimizerR.step()


        # Time spents on one batch
        batch_time.update(time.time() - start_time)
        start_time = time.time()

        log = '[%d/%d][%d/%d]\tLoss_H: %.6f Loss_R: %.6f L1_H_diff: %.4f L1_R_diff: %.4f Loss_sum: %.6f \tdatatime: %.6f \tbatchtime: %.6f' % (
            epoch, args.epochs, i, args.train_iters_epochs,
            Hlosses.val, Rlosses.val, Hdiff.val, Rdiff.val, TotalLosses.val, data_time.val, batch_time.val)

        if i % args.log_freq == 0:
            print(log)

        # genereate a picture every resultPicFrequency steps
        if i % args.resultPicFrequency == 0:
            save_result_pic(args, cover_imgv, steg_img.data, secret_imgv_r, rev_img.data, epoch, i, args.trainpics)

            
        if i == args.train_iters_epochs-1:
            apd, ssim, psnr_c, ssim_c = tensor_psnr_ssim(cover_imgv, steg_img)
            apd, ssim, psnr_s, ssim_s = tensor_psnr_ssim(secret_imgv_r, rev_img)        
            PSNR_C.update(psnr_c)
            PSNR_S.update(psnr_s)
            SSIM_C.update(ssim_c)
            SSIM_S.update(ssim_s)

            break


    train_log = "Training[%d] Hloss= %.6f\tRloss= %.6f\tHdiff= %.4f\tRdiff= %.4f\tpsnr_ssim_c= %.4f/%.4f\tpsnr_ssim_s= %.4f/%.4f\tlr_H= %.6f\tlr_R= %.6f\t Epoch time= %.4f" % (
        epoch, Hlosses.avg, Rlosses.avg, Hdiff.avg, Rdiff.avg, PSNR_C.avg, SSIM_C.avg, PSNR_S.avg, SSIM_S.avg, optimizerH.param_groups[0]['lr'], optimizerR.param_groups[0]['lr'], batch_time.sum)
    print_log(train_log, logPath)


    writer.add_scalar("lr/lr_H", optimizerH.param_groups[0]['lr'], epoch)
    writer.add_scalar("lr/lr_R", optimizerR.param_groups[0]['lr'], epoch)
    writer.add_scalar('train/R_loss', Rlosses.avg, epoch)
    writer.add_scalar('train/H_loss', Hlosses.avg, epoch)
    writer.add_scalar('train/total_loss', TotalLosses.avg, epoch)    
    writer.add_scalar('train/H_diff', Hdiff.avg, epoch)
    writer.add_scalar('train/R_diff', Rdiff.avg, epoch)
    writer.add_scalar('train/psnr_c', PSNR_C.avg, epoch)
    writer.add_scalar('train/ssim_c', SSIM_C.avg, epoch)
    writer.add_scalar('train/psnr_s', PSNR_S.avg, epoch)
    writer.add_scalar('train/ssim_s', SSIM_S.avg, epoch)

    return Hdiff.avg, Rdiff.avg, TotalLosses.avg


def validation(valLoader, epoch, Hnet, Rnet, criterion):
    print("#################################################### validation begin ########################################################")

    Hlosses = AverageMeter()
    Rlosses = AverageMeter()
    TotalLosses = AverageMeter()
    Hdiff = AverageMeter()
    Rdiff = AverageMeter()
    PSNR_C = AverageMeter()
    PSNR_S = AverageMeter()
    SSIM_C = AverageMeter()
    SSIM_S = AverageMeter()


    start_time = time.time()

    with torch.no_grad():
        Hnet.eval()
        Rnet.eval()

        for i, ((cover_img, cover_target), (secret_img, secret_target)) in enumerate(valLoader, 0):

            cover_imgv, steg_img, secret_imgv_r, rev_img, errH, errR, diffH, diffR = steg(args, cover_img, secret_img, Hnet, Rnet, criterion)

            # Loss function
            err_total = errH + args.beta_R * errR

            Hlosses.update(errH.item(), args.batch_stegs*args.num_cover)
            Rlosses.update(errR.item(), args.batch_stegs*args.num_secret)
            Hdiff.update(diffH.item(), args.batch_stegs*args.num_cover)
            Rdiff.update(diffR.item(), args.batch_stegs*args.num_secret)  
            TotalLosses.update(err_total.item(), args.batch_stegs*(args.num_cover+args.num_secret))

            log = '[%d/%d][%d/%d]\tLoss_H: %.6f Loss_R: %.6f L1_H_diff: %.4f L1_R_diff: %.4f Loss_sum: %.6f' % (
                epoch, args.epochs, i, args.val_iters_epochs, Hlosses.val, Rlosses.val, Hdiff.val, Rdiff.val, TotalLosses.val)
            if i % args.log_freq == 0:
                print(log)

            if i == 0:
                save_result_pic(args, cover_imgv, steg_img.data, secret_imgv_r, rev_img.data, epoch, i, args.validationpics)

                apd, ssim, psnr_c, ssim_c = tensor_psnr_ssim(cover_imgv, steg_img)
                apd, ssim, psnr_s, ssim_s = tensor_psnr_ssim(secret_imgv_r, rev_img)        
                PSNR_C.update(psnr_c)
                PSNR_S.update(psnr_s)
                SSIM_C.update(ssim_c)
                SSIM_S.update(ssim_s)

            if i == args.val_iters_epochs-1:
                break    

    val_time = time.time() - start_time
    
    val_log = "validation[%d] val_Hloss = %.6f\t val_Rloss = %.6f\t val_Hdiff = %.4f\t val_Rdiff=%.4f\t psnr_ssim_c= %.4f/%.4f\tpsnr_ssim_s= %.4f/%.4f\t validation time=%.2f" % (
        epoch, Hlosses.avg, Rlosses.avg, Hdiff.avg, Rdiff.avg, PSNR_C.avg, SSIM_C.avg, PSNR_S.avg, SSIM_S.avg, val_time)
    print_log(val_log, logPath)


    writer.add_scalar('validation/H_loss', Hlosses.avg, epoch)
    writer.add_scalar('validation/R_loss', Rlosses.avg, epoch)
    writer.add_scalar('validation/total_loss', TotalLosses.avg, epoch) 
    writer.add_scalar('validation/H_diff', Hdiff.avg, epoch)
    writer.add_scalar('validation/R_diff', Rdiff.avg, epoch)
    writer.add_scalar('validation/psnr_c', PSNR_C.avg, epoch)
    writer.add_scalar('validation/ssim_c', SSIM_C.avg, epoch)
    writer.add_scalar('validation/psnr_s', PSNR_S.avg, epoch)
    writer.add_scalar('validation/ssim_s', SSIM_S.avg, epoch)

    print("#################################################### validation end ########################################################")
    return Hlosses.avg, Rlosses.avg, Hdiff.avg, Rdiff.avg


def steg(args, cover_img, secret_img, Hnet, Rnet, criterion):
    batch_size_cover, channel_cover, _, _ = cover_img.size()
    batch_size_secret, channel_secret, _, _ = secret_img.size()

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
    rev_img = Rnet(steg_img)

    # loss between cover/steg and secret/reveal image 
    errH = criterion(steg_img, cover_imgv)   
    errR = criterion(rev_img, secret_imgv_r)

    # L1 metric
    diffH = (steg_img - cover_imgv).abs().mean()*255
    diffR = (rev_img - secret_imgv_r).abs().mean()*255

    return cover_imgv, steg_img, secret_imgv_r, rev_img, errH, errR, diffH, diffR


# Code saving
def save_current_codes(des_path):
    main_file_path = os.path.realpath(__file__)
    cur_work_dir, mainfile = os.path.split(main_file_path)

    new_main_path = os.path.join(des_path, mainfile)
    shutil.copyfile(main_file_path, new_main_path)

    data_dir = cur_work_dir + "/data/"
    new_data_dir_path = des_path + "/data/"
    shutil.copytree(data_dir, new_data_dir_path)

    model_dir = cur_work_dir + "/models/"
    new_model_dir_path = des_path + "/models/"
    shutil.copytree(model_dir, new_model_dir_path)

    utils_dir = cur_work_dir + "/utils/"
    new_utils_dir_path = des_path + "/utils/"
    shutil.copytree(utils_dir, new_utils_dir_path)


if __name__ == '__main__':
    main()