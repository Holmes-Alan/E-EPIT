import numpy as np
import os
from skimage import metrics
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import logging
from option import args
from einops import rearrange
import xlwt
import torch.nn.functional as F


class ExcelFile():
    def __init__(self):
        self.xlsx_file = xlwt.Workbook()
        self.worksheet = self.xlsx_file.add_sheet(r'sheet1', cell_overwrite_ok=True)
        self.worksheet.write(0, 0, 'Datasets')
        self.worksheet.write(0, 1, 'Scenes')
        self.worksheet.write(0, 2, 'PSNR')
        self.worksheet.write(0, 3, 'SSIM')
        self.worksheet.col(0).width = 256 * 16
        self.worksheet.col(1).width = 256 * 22
        self.worksheet.col(2).width = 256 * 10
        self.worksheet.col(3).width = 256 * 10
        self.sum = 1

    def write_sheet(self, test_name, LF_name, psnr_iter_test, ssim_iter_test):
        ''' Save PSNR & SSIM '''
        for i in range(len(psnr_iter_test)):
            self.add_sheet(test_name, LF_name[i], psnr_iter_test[i], ssim_iter_test[i])

        psnr_epoch_test = float(np.array(psnr_iter_test).mean())
        ssim_epoch_test = float(np.array(ssim_iter_test).mean())
        self.add_sheet(test_name, 'average', psnr_epoch_test, ssim_epoch_test)
        self.sum = self.sum + 1

    def add_sheet(self, test_name, LF_name, psnr_iter_test, ssim_iter_test):
        ''' Save PSNR & SSIM '''
        self.worksheet.write(self.sum, 0, test_name)
        self.worksheet.write(self.sum, 1, LF_name)
        self.worksheet.write(self.sum, 2, '%.6f' % psnr_iter_test)
        self.worksheet.write(self.sum, 3, '%.6f' % ssim_iter_test)
        self.sum = self.sum + 1


def get_logger(log_dir, args):
    '''LOG '''
    logger = logging.getLogger(args.model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model_name))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def create_dir(args):
    log_dir = Path(args.path_log)
    log_dir.mkdir(exist_ok=True)
    if args.task == 'SR':
        task_path = 'SR_' + str(args.angRes_in) + 'x' + str(args.angRes_in) + '_' + str(args.scale_factor) + 'x'
    
    log_dir = log_dir.joinpath(task_path)
    log_dir.mkdir(exist_ok=True)
    log_dir = log_dir.joinpath(args.data_name)
    log_dir.mkdir(exist_ok=True)
    log_dir = log_dir.joinpath(args.model_name)
    log_dir.mkdir(exist_ok=True)

    checkpoints_dir = log_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)

    results_dir = log_dir.joinpath('results/')
    results_dir.mkdir(exist_ok=True)

    return log_dir, checkpoints_dir, results_dir


class Logger():
    def __init__(self, log_dir, args):
        self.logger = get_logger(log_dir, args)

    def log_string(self, str):
        if args.local_rank <= 0:
            self.logger.info(str)
            print(str)


def cal_metrics(args, label, out,):
    if len(label.size()) == 4:
        label = rearrange(label, 'b c (a1 h) (a2 w) -> b c a1 h a2 w', a1=args.angRes_in, a2=args.angRes_in)
        out = rearrange(out, 'b c (a1 h) (a2 w) -> b c a1 h a2 w', a1=args.angRes_in, a2=args.angRes_in)

    if len(label.size()) == 5:
        label = label.permute((0, 1, 3, 2, 4)).unsqueeze(0)
        out = out.permute((0, 1, 3, 2, 4)).unsqueeze(0)

    B, C, U, h, V, w = label.size()
    label_y = label[:, 0, :, :, :, :].data.cpu()
    out_y = out[:, 0, :, :, :, :].data.cpu()

    PSNR = np.zeros(shape=(B, U, V), dtype='float32')
    SSIM = np.zeros(shape=(B, U, V), dtype='float32')
    for b in range(B):
        for u in range(U):
            for v in range(V):
                PSNR[b, u, v] = metrics.peak_signal_noise_ratio(label_y[b, u, :, v, :].numpy(), out_y[b, u, :, v, :].numpy())
                if args.task == 'RE':
                    SSIM[b, u, v] = metrics.structural_similarity(label_y[b, u, :, v, :].numpy(),
                                                                  out_y[b, u, :, v, :].numpy(),
                                                                  gaussian_weights=True,
                                                                  sigma=1.5, use_sample_covariance=False)
                else:
                    a = label_y[b, u, :, v, :].numpy()
                    SSIM[b, u, v] = metrics.structural_similarity(label_y[b, u, :, v, :].numpy(),
                                                                  out_y[b, u, :, v, :].numpy(),
                                                                  data_range=a.max()-a.min())
                pass

    if args.task=='RE':
        for u in range(0, args.angRes_out, (args.angRes_out - 1) // (args.angRes_in - 1)):
            for v in range(0, args.angRes_out, (args.angRes_out - 1) // (args.angRes_in - 1)):
                PSNR[:, u, v] = 0
                SSIM[:, u, v] = 0

    PSNR_mean = PSNR.sum() / np.sum(PSNR > 0)
    SSIM_mean = SSIM.sum() / np.sum(SSIM > 0)

    return PSNR_mean, SSIM_mean


def SAI2MacPI(x, angRes):
    x = rearrange(x, 'b c (u v) h w -> b c (u h) (v w)', u=angRes, v=angRes)
    b, c, hu, wv = x.shape
    h, w = hu // angRes, wv // angRes
    tempU = []
    for i in range(h):
        tempV = []
        for j in range(w):
            tempV.append(x[:, :, i::h, j::w])
        tempU.append(torch.cat(tempV, dim=3))
    out = torch.cat(tempU, dim=2)
    return out

def MacPI2SAI(x, angRes):
	out = []
	for i in range(angRes):
		out_h = []
		for j in range(angRes):
			out_h.append(x[:, :, i::angRes, j::angRes])
		out.append(torch.cat(out_h, 3))
	out = torch.cat(out, 2)
	out = rearrange(out, 'b c (u h) (v w) -> b c (u v) h w', u=angRes, v=angRes)
	return out


def ImageExtend(Im, bdr):
    [_, _, h, w] = Im.size()
    Im_lr = torch.flip(Im, dims=[-1])
    Im_ud = torch.flip(Im, dims=[-2])
    Im_diag = torch.flip(Im, dims=[-1, -2])

    Im_up = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_mid = torch.cat((Im_lr, Im, Im_lr), dim=-1)
    Im_down = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_Ext = torch.cat((Im_up, Im_mid, Im_down), dim=-2)
    Im_out = Im_Ext[:, :, h - bdr[0]: 2 * h + bdr[1], w - bdr[2]: 2 * w + bdr[3]]

    return Im_out


def LFdivide(data, angRes, patch_size, stride):
    data = rearrange(data, '(a1 h) (a2 w) -> (a1 a2) 1 h w', a1=angRes, a2=angRes)
    [_, _, h0, w0] = data.size()

    bdr = (patch_size - stride) // 2
    numU = (h0 + bdr * 2 - 1) // stride
    numV = (w0 + bdr * 2 - 1) // stride
    data_pad = ImageExtend(data, [bdr, bdr+stride-1, bdr, bdr+stride-1])
    # pad = torch.nn.ReflectionPad2d(padding=(bdr, bdr+stride-1, bdr, bdr+stride-1))
    # data_pad = pad(data)
    subLF = F.unfold(data_pad, kernel_size=patch_size, stride=stride)
    subLF = rearrange(subLF, '(a1 a2) (h w) (n1 n2) -> n1 n2 (a1 h) (a2 w)',
                      a1=angRes, a2=angRes, h=patch_size, w=patch_size, n1=numU, n2=numV)

    return subLF


def LFintegrate(subLF, angRes, pz, stride, h, w):
    if subLF.dim() == 4:
        subLF = rearrange(subLF, 'n1 n2 (a1 h) (a2 w) -> n1 n2 a1 a2 h w', a1=angRes, a2=angRes)
        pass
    bdr = (pz - stride) // 2
    outLF = subLF[:, :, :, :, bdr:bdr+stride, bdr:bdr+stride]
    outLF = rearrange(outLF, 'n1 n2 a1 a2 h w -> a1 a2 (n1 h) (n2 w)')
    outLF = outLF[:, :, 0:h, 0:w]

    return outLF


def rgb2ycbcr(x):
    y = np.zeros(x.shape, dtype='double')
    y[:,:,0] =  65.481 * x[:, :, 0] + 128.553 * x[:, :, 1] +  24.966 * x[:, :, 2] +  16.0
    y[:,:,1] = -37.797 * x[:, :, 0] -  74.203 * x[:, :, 1] + 112.000 * x[:, :, 2] + 128.0
    y[:,:,2] = 112.000 * x[:, :, 0] -  93.786 * x[:, :, 1] -  18.214 * x[:, :, 2] + 128.0

    y = y / 255.0
    return y


def ycbcr2rgb(x):
    mat = np.array(
        [[65.481, 128.553, 24.966],
         [-37.797, -74.203, 112.0],
         [112.0, -93.786, -18.214]])
    mat_inv = np.linalg.inv(mat)
    offset = np.matmul(mat_inv, np.array([16, 128, 128]))
    mat_inv = mat_inv * 255

    y = np.zeros(x.shape, dtype='double')
    y[:,:,0] =  mat_inv[0,0] * x[:, :, 0] + mat_inv[0,1] * x[:, :, 1] + mat_inv[0,2] * x[:, :, 2] - offset[0]
    y[:,:,1] =  mat_inv[1,0] * x[:, :, 0] + mat_inv[1,1] * x[:, :, 1] + mat_inv[1,2] * x[:, :, 2] - offset[1]
    y[:,:,2] =  mat_inv[2,0] * x[:, :, 0] + mat_inv[2,1] * x[:, :, 1] + mat_inv[2,2] * x[:, :, 2] - offset[2]
    return y



def split_tensor(tensor, tile_size, stride):
    c = tensor.shape[1]
    mask = torch.ones_like(tensor)
    # use torch.nn.Unfold
    unfold  = torch.nn.Unfold(kernel_size=(tile_size, tile_size), stride=(stride, stride))
    # Apply to mask and original image
    mask_p  = unfold(mask)
    patches = unfold(tensor)
	
    patches = patches.reshape(c, tile_size, tile_size, -1).permute(3, 0, 1, 2)
    patches_base = torch.zeros(patches.size()).to(tensor.device)
	
    tiles = []
    for t in range(patches.size(0)):
         tiles.append(patches[[t], :, :, :])
    return tiles, mask_p, patches_base, (tensor.size(2), tensor.size(3))

def rebuild_tensor(tensor_list, mask_t, base_tensor, t_size, tile_size, stride):
    # base_tensor here is used as a container
    c = tensor_list[0].shape[1]
    for t, tile in enumerate(tensor_list):
         base_tensor[[t], :, :] = tile  
	 
    base_tensor = base_tensor.permute(1, 2, 3, 0).reshape(c*tile_size*tile_size, base_tensor.size(0)).unsqueeze(0)
    fold = torch.nn.Fold(output_size=(t_size[0], t_size[1]), kernel_size=(tile_size, tile_size), stride=(stride, stride))

    output_tensor = fold(base_tensor)/fold(mask_t)
    # output_tensor = fold(base_tensor)
    return output_tensor



def patchify_tensor(features, angRes, patch_size, overlap=10):
    # features = rearrange(features, '(a1 h) (a2 w) -> (a1 a2) 1 h w', a1=angRes, a2=angRes)
    batch_size, channels, height, width = features.size()

    effective_patch_size = patch_size - overlap
    n_patches_height = (height // effective_patch_size)
    n_patches_width = (width // effective_patch_size)

    if n_patches_height * effective_patch_size < height:
        n_patches_height += 1
    if n_patches_width * effective_patch_size < width:
        n_patches_width += 1

    patches = []
    for b in range(batch_size):
        for h in range(n_patches_height):
            for w in range(n_patches_width):
                patch_start_height = min(h * effective_patch_size, height - patch_size)
                patch_start_width = min(w * effective_patch_size, width - patch_size)
                patches.append(features[b:b+1, :,
                               patch_start_height: patch_start_height + patch_size,
                               patch_start_width: patch_start_width + patch_size])
                
    patches = torch.cat(patches, 0)
    patches = rearrange(patches, '(n1 n2) (a1 a2) h w -> n1 n2 (a1 h) (a2 w)',
                      a1=angRes, a2=angRes, h=patch_size, w=patch_size, n1=n_patches_height, n2=n_patches_width)
    # print('inside after===========', patches.shape)
    return patches


def recompose_tensor(patches, full_height, full_width, overlap=10):

    batch_size, channels, patch_size, _ = patches.size()
    effective_patch_size = patch_size - overlap
    n_patches_height = (full_height // effective_patch_size)
    n_patches_width = (full_width // effective_patch_size)

    if n_patches_height * effective_patch_size < full_height:
        n_patches_height += 1
    if n_patches_width * effective_patch_size < full_width:
        n_patches_width += 1

    n_patches = n_patches_height * n_patches_width
    if batch_size % n_patches != 0:
        print("Error: The number of patches provided to the recompose function does not match the number of patches in each image.")
    final_batch_size = batch_size // n_patches

    blending_in = torch.linspace(0.1, 1.0, overlap)
    blending_out = torch.linspace(1.0, 0.1, overlap)
    middle_part = torch.ones(patch_size - 2 * overlap)
    blending_profile = torch.cat([blending_in, middle_part, blending_out], 0)

    horizontal_blending = blending_profile[None].repeat(patch_size, 1)
    vertical_blending = blending_profile[:, None].repeat(1, patch_size)
    blending_patch = horizontal_blending * vertical_blending

    blending_image = torch.zeros(1, channels, full_height, full_width)
    for h in range(n_patches_height):
        for w in range(n_patches_width):
            patch_start_height = min(h * effective_patch_size, full_height - patch_size)
            patch_start_width = min(w * effective_patch_size, full_width - patch_size)
            blending_image[0, :, patch_start_height: patch_start_height + patch_size, patch_start_width: patch_start_width + patch_size] += blending_patch[None]

    recomposed_tensor = torch.zeros(final_batch_size, channels, full_height, full_width)
    if patches.is_cuda:
        blending_patch = blending_patch.cuda()
        blending_image = blending_image.cuda()
        recomposed_tensor = recomposed_tensor.cuda()
    patch_index = 0
    for b in range(final_batch_size):
        for h in range(n_patches_height):
            for w in range(n_patches_width):
                patch_start_height = min(h * effective_patch_size, full_height - patch_size)
                patch_start_width = min(w * effective_patch_size, full_width - patch_size)
                recomposed_tensor[b, :, patch_start_height: patch_start_height + patch_size, patch_start_width: patch_start_width + patch_size] += patches[patch_index] * blending_patch
                patch_index += 1
    recomposed_tensor /= blending_image

    return recomposed_tensor