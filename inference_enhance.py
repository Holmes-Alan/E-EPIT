import importlib
import torch
import torch.backends.cudnn as cudnn
from utils.utils import *
from collections import OrderedDict
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import h5py
from torchvision.transforms import ToTensor
import imageio
from tqdm import tqdm


def MultiTestSetDataLoader(args):
    # get testdataloader of every test dataset
    data_list = None
    if args.data_name in ['ALL', 'RE_Lytro', 'RE_HCI']:
        if args.task == 'SR':
            dataset_dir = args.path_for_test + 'SR_' + str(args.angRes_in) + 'x' + str(args.angRes_in) + '_' + \
                          str(args.scale_factor) + 'x/'
            data_list = os.listdir(dataset_dir)
        elif args.task == 'RE':
            dataset_dir = args.path_for_test + 'RE_' + str(args.angRes_in) + 'x' + str(args.angRes_in) + '_' + \
                          str(args.angRes_out) + 'x' + str(args.angRes_out) + '/' + args.data_name
            data_list = os.listdir(dataset_dir)
    else:
        data_list = [args.data_name]

    test_Loaders = []
    length_of_tests = 0
    for data_name in data_list:
        test_Dataset = TestSetDataLoader(args, data_name, Lr_Info=data_list.index(data_name))
        length_of_tests += len(test_Dataset)

        test_Loaders.append(DataLoader(dataset=test_Dataset, num_workers=args.num_workers, batch_size=1, shuffle=False))

    return data_list, test_Loaders, length_of_tests


class TestSetDataLoader(Dataset):
    def __init__(self, args, data_name = 'ALL', Lr_Info=None):
        super(TestSetDataLoader, self).__init__()
        self.angRes_in = args.angRes_in
        self.angRes_out = args.angRes_out
        if args.task == 'SR':
            self.dataset_dir = args.path_for_test + 'SR_' + str(args.angRes_in) + 'x' + str(args.angRes_in) + '_' + \
                               str(args.scale_factor) + 'x/'
            self.data_list = [data_name]
        elif args.task == 'RE':
            self.dataset_dir = args.path_for_test + 'RE_' + str(args.angRes_in) + 'x' + str(args.angRes_in) + '_' + \
                               str(args.angRes_out) + 'x' + str(args.angRes_out) + '/' + args.data_name + '/'
            self.data_list = [data_name]

        self.file_list = []
        for data_name in self.data_list:
            tmp_list = os.listdir(self.dataset_dir + data_name)
            for index, _ in enumerate(tmp_list):
                tmp_list[index] = data_name + '/' + tmp_list[index]

            self.file_list.extend(tmp_list)

        self.item_num = len(self.file_list)

    def __getitem__(self, index):
        file_name = [self.dataset_dir + self.file_list[index]]
        with h5py.File(file_name[0], 'r') as hf:
            Lr_SAI_y = np.array(hf.get('Lr_SAI_y'))
            Hr_SAI_y = np.array(hf.get('Hr_SAI_y'))
            Sr_SAI_cbcr = np.array(hf.get('Sr_SAI_cbcr'), dtype='single')
            Lr_SAI_y = np.transpose(Lr_SAI_y, (1, 0))
            Hr_SAI_y = np.transpose(Hr_SAI_y, (1, 0))
            Sr_SAI_cbcr  = np.transpose(Sr_SAI_cbcr,  (2, 1, 0))

        Lr_SAI_y = ToTensor()(Lr_SAI_y.copy())
        Hr_SAI_y = ToTensor()(Hr_SAI_y.copy())
        Sr_SAI_cbcr = ToTensor()(Sr_SAI_cbcr.copy())

        Lr_angRes_in = self.angRes_in
        Lr_angRes_out = self.angRes_out
        LF_name = self.file_list[index].split('/')[-1].split('.')[0]

        return Lr_SAI_y, Hr_SAI_y, Sr_SAI_cbcr, [Lr_angRes_in, Lr_angRes_out], LF_name

    def __len__(self):
        return self.item_num


def test_patch(args, Lr_SAI_y, net, device, data_info):
    ''' Crop LFs into Patches '''
    features = rearrange(Lr_SAI_y, '(a1 h) (a2 w) -> 1 (a1 a2) h w', a1=args.angRes_in, a2=args.angRes_in)
    subLFin = patchify_tensor(features, args.angRes_in, args.patch_size_for_test, args.stride_for_test)
    numU, numV, H, W = subLFin.size()
    subLFin = rearrange(subLFin, 'n1 n2 a1h a2w -> (n1 n2) 1 a1h a2w')

    subLFout = torch.zeros(numU * numV, 1, args.angRes_in * args.patch_size_for_test * args.scale_factor,
                        args.angRes_in * args.patch_size_for_test * args.scale_factor)

    ''' SR the Patches '''
    for i in range(0, numU * numV, args.minibatch_for_test):
        tmp = subLFin[i:min(i + args.minibatch_for_test, numU * numV), :, :, :]
        with torch.no_grad():
            net.eval()
            torch.cuda.empty_cache()
            out = net(tmp.to(device), data_info)
            subLFout[i:min(i + args.minibatch_for_test, numU * numV), :, :, :] = out
    hh = args.patch_size_for_test * args.scale_factor
    subLFout = rearrange(subLFout, '(n1 n2) 1 (a1 h) (a2 w) -> (n1 n2) (a1 a2) h w', n1=numU, n2=numV, h=hh, w=hh)
    ''' Restore the Patches to LFs '''
    subLFout = recompose_tensor(subLFout, features.shape[2]*args.scale_factor, features.shape[3]*args.scale_factor, args.stride_for_test*args.scale_factor)
    Sr_SAI_y = rearrange(subLFout, '1 (a1 a2) h w -> 1 1 (a1 h) (a2 w)', a1=args.angRes_out, a2=args.angRes_out)

    return Sr_SAI_y

def main(args):
    ''' Create Dir for Save '''
    # _, _, result_dir = create_dir(args)
    # result_dir = result_dir.joinpath('TEST')
    # result_dir.mkdir(exist_ok=True)
    result_dir = Path(args.path_log)
    result_dir.mkdir(exist_ok=True)

    ''' CPU or Cuda'''
    device = torch.device(args.device)
    # if 'cuda' in args.device:
    #     torch.cuda.set_device(device)


    ''' DATA TEST LOADING '''
    print('\nLoad Test Dataset ...')
    test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader(args)
    print("The number of test data is: %d" % length_of_tests)


    ''' MODEL LOADING '''
    print('\nModel Initial ...')
    MODEL_PATH = 'model.' + args.task + '.' + args.model_name
    MODEL = importlib.import_module(MODEL_PATH)
    net = MODEL.get_model(args)


    ''' Load Pre-Trained PTH '''
    if args.use_pre_ckpt == False:
        net.apply(MODEL.weights_init)
    else:
        ckpt_path = args.path_pre_pth
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        try:
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = 'module.' + k  # add `module.`
                new_state_dict[name] = v
            # load params
            net.load_state_dict(new_state_dict)
            print('Use pretrain model!')
        except:
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                new_state_dict[k] = v
            # load params
            net.load_state_dict(new_state_dict)
            print('Use pretrain model!')
            pass
        pass

    net = net.to(device)
    cudnn.benchmark = True

    ''' Print Parameters '''
    print('PARAMETER ...')
    print(args)

    ''' TEST on every dataset '''
    print('\nStart test...')
    with torch.no_grad():

        for index, test_name in enumerate(test_Names):
            test_loader = test_Loaders[index]

            save_dir = result_dir.joinpath(test_name)
            save_dir.mkdir(exist_ok=True)

            test(test_loader, device, net, save_dir)
    pass

def test(test_loader, device, net, save_dir=None):
    for idx_iter, (Lr_SAI_y, Hr_SAI_y, Sr_SAI_cbcr, data_info, LF_name) in tqdm(enumerate(test_loader), total=len(test_loader), ncols=70):
        [Lr_angRes_in, Lr_angRes_out] = data_info
        data_info[0] = Lr_angRes_in[0].item()
        data_info[1] = Lr_angRes_out[0].item()

        Lr_SAI_y = Lr_SAI_y.squeeze().to(device)  # numU, numV, h*angRes, w*angRes
        Sr_SAI_cbcr = Sr_SAI_cbcr
        Sr_enhance_lr = 0
        Sr_enhance_rl = 0

        for rot in range(4):
            Lr_SAI_y_t = torch.rot90(Lr_SAI_y, rot, [0, 1])
            Sr_SAI_y_t = test_patch(args, Lr_SAI_y_t, net, device, data_info)
            Sr_SAI_y_t = Sr_SAI_y_t.squeeze(0).squeeze(0)
            Sr_SAI_y_t = torch.rot90(Sr_SAI_y_t, -1 * rot, [0, 1])
            Sr_SAI_y_t = Sr_SAI_y_t.unsqueeze(0).unsqueeze(0)
            Sr_enhance_lr += Sr_SAI_y_t
        # Sr_enhance_lr = Sr_enhance_lr / 4.0

        #===== Flip version ============
        Lr_SAI_y = rearrange(Lr_SAI_y, '(a1 h) (a2 w) -> 1 1 (a1 a2) h w', a1=args.angRes_in, a2=args.angRes_in)
        Lr_SAI_y = torch.flip(Lr_SAI_y, dims=(4,))
        Lr_SAI_y = rearrange(Lr_SAI_y, '1 1 (a1 a2) h w -> (a1 h) (a2 w)', a1=args.angRes_in, a2=args.angRes_in)
        for rot in range(4):
            Lr_SAI_y_t = torch.rot90(Lr_SAI_y, rot, [0, 1])
            Sr_SAI_y_t = test_patch(args, Lr_SAI_y_t, net, device, data_info)
            Sr_SAI_y_t = Sr_SAI_y_t.squeeze(0).squeeze(0)
            Sr_SAI_y_t = torch.rot90(Sr_SAI_y_t, -1 * rot, [0, 1])
            Sr_enhance_rl += Sr_SAI_y_t
        # Sr_enhance_rl = Sr_enhance_rl / 4.0
        Sr_enhance_rl = rearrange(Sr_enhance_rl, '(a1 h) (a2 w) -> 1 1 (a1 a2) h w', a1=args.angRes_in, a2=args.angRes_in)
        Sr_enhance_rl = torch.flip(Sr_enhance_rl, dims=(4,))
        Sr_enhance_rl = rearrange(Sr_enhance_rl, '1 1 (a1 a2) h w -> 1 1 (a1 h) (a2 w)', a1=args.angRes_in, a2=args.angRes_in)

        Sr_enhance = (Sr_enhance_lr + Sr_enhance_rl) / 8.0

        ''' Save RGB '''
        if save_dir is not None:
            save_dir_ = save_dir.joinpath(LF_name[0])
            save_dir_.mkdir(exist_ok=True)
            Sr_SAI_ycbcr = torch.cat((Sr_enhance, Sr_SAI_cbcr), dim=1)
            Sr_SAI_rgb = (ycbcr2rgb(Sr_SAI_ycbcr.squeeze().permute(1, 2, 0).numpy()).clip(0,1)*255).astype('uint8')
            Sr_4D_rgb = rearrange(Sr_SAI_rgb, '(a1 h) (a2 w) c -> a1 a2 h w c', a1=args.angRes_out, a2=args.angRes_out)

            # save all views
            for i in range(args.angRes_out):
                for j in range(args.angRes_out):
                    img = Sr_4D_rgb[i, j, :, :, :]
                    path = str(save_dir_) + '/' + 'View' + '_' + str(i) + '_' + str(j) + '.bmp'
                    imageio.imwrite(path, img)
                    pass
                pass
            pass
        pass
    pass


if __name__ == '__main__':
    from option import args
    args.scale_factor = 4
    args.angRes = 5
    args.path_for_test = './data_for_inference/'
    args.stride_for_test = 32
    args.patch_size_for_test = 64
    args.minibatch_for_test = 4


    args.data_name = 'NTIRE_Test_Real'
    args.model_name = 'EPIT_bs'
    # args.path_pre_pth = '/data/LFSR/model/EPIT_5x5_4x_model.pth'
    args.path_pre_pth = 'pth/EPIT_bs_5x5_4x.pth'
    main(args)
