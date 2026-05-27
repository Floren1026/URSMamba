import os
import csv
import shutil
import rasterio
import torch
import torch.optim
import torchvision.utils as vutils


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    # return {'Total': total_num, 'Trainable': trainable_num}
    return total_num


def print_log(log_info, log_path):
    print(log_info)
    if not os.path.exists(log_path):
        fp = open(log_path, "w")
        fp.writelines(log_info + "\n")
    else:
        with open(log_path, 'a+') as f:
            f.writelines(log_info + '\n')


# Print the structure and parameters number of the net
def print_network(net, logPath):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print_log(str(net), logPath)
    print_log('Total number of parameters: %d' % num_params, logPath)


# Code saving
def save_current_codes(des_path):
    main_file_path = os.path.realpath(__file__)
    cur_work_dir = os.path.dirname(os.path.split(main_file_path)[0])

    data_dir = cur_work_dir + "/data/"
    new_data_dir_path = des_path + "/data/"
    shutil.copytree(data_dir, new_data_dir_path)

    model_dir = cur_work_dir + "/models/"
    new_model_dir_path = des_path + "/models/"
    shutil.copytree(model_dir, new_model_dir_path)

    utils_dir = cur_work_dir + "/utils/"
    new_utils_dir_path = des_path + "/utils/"
    shutil.copytree(utils_dir, new_utils_dir_path)

    train_file = cur_work_dir + '/train.py'
    new_train_file_path = des_path + '/train.py'
    shutil.copyfile(train_file, new_train_file_path)

    test_file = cur_work_dir + '/test.py'
    new_test_file_path = des_path + '/test.py'
    shutil.copyfile(test_file, new_test_file_path)


def gauss_noise(shape):
    noise = torch.zeros(shape).cuda()
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).cuda()

    return noise


def save_model(config, state, is_best, epoch, prefix, net):

    checkpoint_epoch='%s/checkpoint%s_%.3i.pt'% (config.outckpts, net, (epoch+1))
    checkpoint='%s/checkpoint%s.pt'% (config.outckpts, net)
    torch.save(state, checkpoint)

    if (epoch+1) % config.save_freq == 0:
        torch.save(state, checkpoint_epoch)
    if is_best:
        shutil.copyfile(checkpoint, '%s/best_checkpoint%s.pt'% (config.outckpts, net))
    if epoch == config.epochs-1:
        with open(prefix + '.csv', 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([state])



def save_result_pic(args, cover_img, steg_img, secret_img, rev_img, epoch, i, save_path):

    resultImgName = '%s/ResultPics_epoch%03d_batch%04d.png' % (save_path, epoch, i)

    if args.cuda:
        cover_img = cover_img.cuda()
        steg_img = steg_img.cuda()
        rev_img = rev_img.cuda()
        secret_img = secret_img.cuda()

    cover_gap = steg_img - cover_img
    secret_gap = rev_img - secret_img
    cover_gap = (cover_gap*10 + 0.5).clamp_(0.0, 1.0)
    secret_gap = (secret_gap*10 + 0.5).clamp_(0.0, 1.0)

    for i in range(args.num_cover):
        cover_i = cover_img[:,i*args.channel_cover:(i+1)*args.channel_cover,:,:]
        steg_i = steg_img[:,i*args.channel_cover:(i+1)*args.channel_cover,:,:]
        cover_gap_i = cover_gap[:,i*args.channel_cover:(i+1)*args.channel_cover,:,:]

        if i == 0:
            showCover = torch.cat((cover_i, steg_i, cover_gap_i),0)
        else:
            showCover = torch.cat((showCover, cover_i, steg_i, cover_gap_i),0)

    for i_secret in range(args.num_secret):
        secret_i = secret_img[:,i_secret*args.channel_secret:(i_secret+1)*args.channel_secret,:,:]
        rev_secret_i = rev_img[:,i_secret*args.channel_secret:(i_secret+1)*args.channel_secret,:,:]
        secret_gap_i = secret_gap[:,i_secret*args.channel_secret:(i_secret+1)*args.channel_secret,:,:]


        if i_secret == 0:
            showSecret = torch.cat((secret_i, rev_secret_i, secret_gap_i),0)
        else:
            showSecret = torch.cat((showSecret, secret_i, rev_secret_i, secret_gap_i),0)

    if args.channel_secret == args.channel_cover:
        showAll = torch.cat((showCover, showSecret),0)
        vutils.save_image(showAll, resultImgName, nrow=9, padding=1, normalize=True)
    else:
        ContainerImgName = '%s/ContainerPics_epoch%03d_batch%04d.png' % (save_path, epoch, i)
        SecretImgName = '%s/SecretPics_epoch%03d_batch%04d.png' % (save_path, epoch, i)
        vutils.save_image(showCover, ContainerImgName, nrow=3*(args.num_cover+args.num_secret), padding=1, normalize=True)
        vutils.save_image(showSecret, SecretImgName, nrow=3*(args.num_cover+args.num_secret), padding=1, normalize=True)


def save_result_pic_test(args, cover_img, steg_img, secret_img, rev_img, epoch, i, save_path):

    resultImgName = '%s/ResultPics_epoch%03d_batch%04d.png' % (save_path, epoch, i)

    if args.cuda:
        cover_img = cover_img.cuda()
        steg_img = steg_img.cuda()
        rev_img = rev_img.cuda()
        secret_img = secret_img.cuda()


    cover_gap = steg_img - cover_img
    secret_gap = rev_img - secret_img
    cover_gap = (cover_gap*10 + 0.5).clamp_(0.0, 1.0)
    secret_gap = (secret_gap*10 + 0.5).clamp_(0.0, 1.0)


    coverImgName = '%s/cover_img%04d.png' % (args.cover, i)
    stegImgName = '%s/steg_img%04d.png' % (args.steg, i)
    res_steg = '%s/res_steg_img%04d.png' % (args.res_steg, i)
    revImgName = '%s/rev_img%04d.png' % (args.rev, i)
    secretImgName = '%s/secret_img%04d.png' % (args.secret, i)
    res_rev = '%s/res_rev_img%04d.png' % (args.res_rev, i)

    vutils.save_image(cover_img, coverImgName, padding=1, normalize=True)
    vutils.save_image(steg_img, stegImgName, padding=1, normalize=True)
    vutils.save_image(cover_gap, res_steg, padding=1, normalize=True)
    vutils.save_image(rev_img, revImgName, padding=1, normalize=True)
    vutils.save_image(secret_img, secretImgName, padding=1, normalize=True)
    vutils.save_image(secret_gap, res_rev, padding=1, normalize=True)
    


    for i in range(args.num_cover):
        cover_i = cover_img[:,i*args.channel_cover:(i+1)*args.channel_cover,:,:]
        steg_i = steg_img[:,i*args.channel_cover:(i+1)*args.channel_cover,:,:]
        cover_gap_i = cover_gap[:,i*args.channel_cover:(i+1)*args.channel_cover,:,:]

        if i == 0:
            showCover = torch.cat((cover_i, steg_i, cover_gap_i),0)
        else:
            showCover = torch.cat((showCover, cover_i, steg_i, cover_gap_i),0)

    for i_secret in range(args.num_secret):
        secret_i = secret_img[:,i_secret*args.channel_secret:(i_secret+1)*args.channel_secret,:,:]
        rev_secret_i = rev_img[:,i_secret*args.channel_secret:(i_secret+1)*args.channel_secret,:,:]
        secret_gap_i = secret_gap[:,i_secret*args.channel_secret:(i_secret+1)*args.channel_secret,:,:]

        if i_secret == 0:
            showSecret = torch.cat((secret_i, rev_secret_i, secret_gap_i),0)
        else:
            showSecret = torch.cat((showSecret, secret_i, rev_secret_i, secret_gap_i),0)


    if args.channel_secret == args.channel_cover:
        showAll = torch.cat((showCover, showSecret),0)
        vutils.save_image(showAll, resultImgName, nrow=3*(args.num_cover+args.num_secret), padding=1, normalize=True)
    else:
        ContainerImgName = '%s/ContainerPics_epoch%03d_batch%04d.png' % (save_path, epoch, i)
        SecretImgName = '%s/SecretPics_epoch%03d_batch%04d.png' % (save_path, epoch, i)
        vutils.save_image(showCover, ContainerImgName, nrow=3*(args.num_cover+args.num_secret), padding=1, normalize=True)
        vutils.save_image(showSecret, SecretImgName, nrow=3*(args.num_cover+args.num_secret), padding=1, normalize=True)




def save_pic(config, cover_img, steg_img, secret_img, rev_img, epoch, i, save_path):

    resultImgName = '%s/ResultPics_epoch%03d_batch%04d.png' % (save_path, epoch, i)

    cover_img = cover_img.cuda()
    steg_img = steg_img.cuda()
    rev_img = rev_img.cuda()
    secret_img = secret_img.cuda()

    cover_gap = steg_img - cover_img
    secret_gap = rev_img - secret_img
    cover_gap = (cover_gap*10 + 0.5).clamp_(0.0, 1.0)
    secret_gap = (secret_gap*10 + 0.5).clamp_(0.0, 1.0)

    for i in range(config.NUM_COVER):
        cover_i = cover_img[:,i*config.COVER_CHANS:(i+1)*config.COVER_CHANS,:,:]
        steg_i = steg_img[:,i*config.COVER_CHANS:(i+1)*config.COVER_CHANS,:,:]
        cover_gap_i = cover_gap[:,i*config.COVER_CHANS:(i+1)*config.COVER_CHANS,:,:]

        if i == 0:
            showCover = torch.cat((cover_i, steg_i, cover_gap_i),0)
        else:
            showCover = torch.cat((showCover, cover_i, steg_i, cover_gap_i),0)

    for i_secret in range(config.NUM_SECRET):
        secret_i = secret_img[:,i_secret*config.SECRET_CHANS:(i_secret+1)*config.SECRET_CHANS,:,:]
        rev_secret_i = rev_img[:,i_secret*config.SECRET_CHANS:(i_secret+1)*config.SECRET_CHANS,:,:]
        secret_gap_i = secret_gap[:,i_secret*config.SECRET_CHANS:(i_secret+1)*config.SECRET_CHANS,:,:]


        if i_secret == 0:
            showSecret = torch.cat((secret_i, rev_secret_i, secret_gap_i),0)
        else:
            showSecret = torch.cat((showSecret, secret_i, rev_secret_i, secret_gap_i),0)

    if config.SECRET_CHANS == config.COVER_CHANS:
        showAll = torch.cat((showCover, showSecret),0)
        vutils.save_image(showAll, resultImgName, nrow=3*(config.NUM_COVER+config.NUM_SECRET), padding=1, normalize=True)
    else:
        ContainerImgName = '%s/CoverPics_epoch%03d_batch%04d.png' % (save_path, epoch, i)
        SecretImgName = '%s/SecretPics_epoch%03d_batch%04d.png' % (save_path, epoch, i)
        vutils.save_image(showCover, ContainerImgName, nrow=3*(config.NUM_COVER+config.NUM_SECRET), padding=1, normalize=True)
        # vutils.save_image(showSecret, SecretImgName, nrow=3*(config.NUM_COVER+config.NUM_SECRET), padding=1, normalize=True)


def save_pic_multi(config, cover_img, steg_img, secret_img, rev_img, epoch, i, save_path):

    resultImgName = '%s/ResultPics_epoch%03d_batch%04d.png' % (save_path, epoch, i)

    cover_img = cover_img.cuda()
    steg_img = steg_img.cuda()
    rev_img = rev_img.cuda()
    secret_img = secret_img.cuda()

    cover_gap = steg_img - cover_img
    secret_gap = rev_img - secret_img
    cover_gap = (cover_gap*10 + 0.5).clamp_(0.0, 1.0)
    secret_gap = (secret_gap*10 + 0.5).clamp_(0.0, 1.0)

    coverImgName = 'result_analysis/result/cover/cover_img%04d.png' % (i)
    stegImgName = 'result_analysis/result/steg/1/steg_img%04d.png' % (i)
    res_steg = 'result_analysis/result/res_steg/res_steg_img%04d.png' % (i)
    revImgName = 'result_analysis/result/rev/rev_img%04d.png' % (i)
    secretImgName = 'result_analysis/result/secret/1/secret_img%04d.png' % (i)
    res_rev = 'result_analysis/result/res_rev/res_rev_img%04d.png' % (i)


    secret_i = secret_img[:,0:3,:,:]
    rev_secret_i = rev_img[:,0:3,:,:]
    secret_gap_i = secret_gap[:,0:3,:,:]   

    showCover = torch.cat((cover_img, steg_img, cover_gap),0)
    showSecret = torch.cat((secret_i, rev_secret_i, secret_gap_i),0)
    showAll = torch.cat((showCover, showSecret),0)
    vutils.save_image(showAll, resultImgName, nrow=3*(config.num_cover+config.num_secret), padding=1, normalize=True)


def save_testpic_multi(config, cover_img, steg_img, secret_img, rev_img, epoch, i, save_path):

    resultImgName = '%s/ResultPics_epoch%03d_batch%04d.png' % (save_path, epoch, i)

    cover_img = cover_img.cuda()
    steg_img = steg_img.cuda()
    rev_img = rev_img.cuda()
    secret_img = secret_img.cuda()

    cover_gap = steg_img - cover_img
    secret_gap = rev_img - secret_img
    cover_gap = (cover_gap*10 + 0.5).clamp_(0.0, 1.0)
    secret_gap = (secret_gap*10 + 0.5).clamp_(0.0, 1.0)

    coverImgName = 'result_analysis/result/cover/cover_img%04d.png' % (i)
    stegImgName = 'result_analysis/result/steg/1/steg_img%04d.png' % (i)
    res_steg = 'result_analysis/result/res_steg/res_steg_img%04d.png' % (i)
    revImgName = 'result_analysis/result/rev/rev_img%04d.png' % (i)
    secretImgName = 'result_analysis/result/secret/1/secret_img%04d.png' % (i)
    res_rev = 'result_analysis/result/res_rev/res_rev_img%04d.png' % (i)


    secret_i = secret_img[:,0:3,:,:]
    rev_secret_i = rev_img[:,0:3,:,:]
    secret_gap_i = secret_gap[:,0:3,:,:]

    vutils.save_image(cover_img, coverImgName, padding=1, normalize=True)
    vutils.save_image(steg_img, stegImgName, padding=1, normalize=True)
    vutils.save_image(cover_gap, res_steg, padding=1, normalize=True)
    vutils.save_image(rev_secret_i, revImgName, padding=1, normalize=True)
    vutils.save_image(secret_i, secretImgName, padding=1, normalize=True)
    vutils.save_image(secret_gap_i, res_rev, padding=1, normalize=True)    

    showCover = torch.cat((cover_img, steg_img, cover_gap),0)
    showSecret = torch.cat((secret_i, rev_secret_i, secret_gap_i),0)
    showAll = torch.cat((showCover, showSecret),0)
    vutils.save_image(showAll, resultImgName, nrow=3*(config.num_cover+config.num_secret), padding=1, normalize=True)


# def save_multi_pic(config, cover_img, steg_img, secret_img, rev_img, epoch, i, save_path):
#     cover_img = cover_img.cuda()
#     steg_img = steg_img.cuda()
#     rev_img = rev_img.cuda()
#     secret_img = secret_img.cuda()

#     cover_gap = steg_img - cover_img
#     secret_gap = rev_img - secret_img
#     cover_gap = (cover_gap*10 + 0.5).clamp_(0.0, 1.0)
#     secret_gap = (secret_gap*10 + 0.5).clamp_(0.0, 1.0)


#     showCover = torch.cat((cover_img, steg_img, cover_gap),0)


#     rev_img_cpu = rev_img.cpu()
#     secret_img_cpu = secret_img.cpu()
#     secret_gap_cpu = secret_gap.cpu()
#     SecretCat = torch.cat((secret_img_cpu, rev_img_cpu, secret_gap_cpu), 0)
#     N, C, H, W = SecretCat.shape[0], SecretCat.shape[1], SecretCat.shape[2], SecretCat.shape[3]

#     secret_tif = (secret_img_cpu*65535).to(torch.uint16).numpy()
#     rev_tif = (rev_img_cpu*65535).to(torch.uint16).numpy()
#     secret_gap_tif = (secret_gap_cpu*65535).to(torch.uint16).numpy()
#     tensor_list = [secret_tif, rev_tif, secret_gap_tif]


#     ContainerImgName = '%s/CoverPics_epoch%03d_batch%04d.png' % (save_path, epoch, i)
#     vutils.save_image(showCover, ContainerImgName, nrow=3*(config.NUM_COVER+config.NUM_SECRET), padding=1, normalize=True)

#     SecretImgName = '%s/SecretPics_epoch%03d_batch%04d.tif' % (save_path, epoch, i)
#     RevImgName = '%s/RevPics_epoch%03d_batch%04d.tif' % (save_path, epoch, i)
#     SRgapImgName = '%s/SRgapPics_epoch%03d_batch%04d.tif' % (save_path, epoch, i)


#     for i, tensor in enumerate(tensor_list):
#         # output_tif = f"output_{i}.tif"  # 保存路径
#         output_tif = '%s/SecretPics_epoch%03d_batch%04d.tif' % (save_path, epoch, i)

#         # 写入 TIFF 文件
#         with rasterio.open(
#             output_tif,
#             "w",
#             driver="GTiff",
#             height=H,
#             width=W,
#             count=C,  # 波段数
#             dtype="uint16"
#         ) as dst:
#             for band in range(C):
#                 dst.write(tensor[band], band + 1)  # 写入每个波段
