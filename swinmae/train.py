import argparse
import json
import numpy as np
import os
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from utils.utils import HDF5Dataset, id_collate, get_quantiles

import utils.misc as misc
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
import swin_mae
from utils.engine_pretrain import train_one_epoch


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)

    # common parameters
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--save_freq', default=400, type=int)
    parser.add_argument('--checkpoint_encoder', default='', type=str)
    parser.add_argument('--checkpoint_decoder', default='', type=str)
    parser.add_argument('--data_path', default=r'Enki_LLC_valid_noise_preproc.h5', type=str)  # fill in the dataset path here
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    # model parameters
    parser.add_argument('--model', default='enki', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=64, type=int,
                        help='images input size')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # optimizer parameters
    parser.add_argument('--accum_iter', default=1, type=int)
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')

    # other parameters
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.set_defaults(pin_mem=True)

    return parser


def main(args):
    # Print... something (i forgot)
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    
    # Fixed random seeds
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Set up training equipment
    device = torch.device(args.device)
    cudnn.benchmark = True
    
    # Defining data augmentation
    transform_train = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    # Setup dataset
    dataset_train = HDF5Dataset(args.data_path, partition='valid')
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        collate_fn=id_collate,
        drop_last=True,
    )
    
    print("training", len(data_loader_train.dataset) )
    print("Datasets loaded")

    """ default (delete when working)
    # Set dataset
    dataset_train = datasets.ImageFolder(args.data_path, transform=transform_train)
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True
    )
    """

    # Log output
    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter()
    else:
        log_writer = None

    # Set model
    model = swin_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss, mask_ratio=args.mask_ratio)
    model.to(device)
    model_without_ddp = model

    # Set optimizer
    param_groups = [p for p in model_without_ddp.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=5e-2, betas=(0.9, 0.95))  # 原来是5E-2
    loss_scaler = NativeScaler()

    # Create model
    misc.load_model(args=args, model_without_ddp=model_without_ddp)

    # Start the training process
    print(f"Start training for {args.epochs} epochs")
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )

        # Save model and upload to s3 storage every epoch
        # Uploads to s3 storage if rank 0
        if args.output_dir and ((epoch + 1) % args.save_freq == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch + 1)
            
            # create filenames
            local_file = os.path.join(args.output_dir, 'checkpoint-%s.pth' % (epoch + 1))
            s3_file = os.path.join("s3://llc/mae", local_file)
            if local_file[:2] == './':    # remove ./ if hidden output folder
                s3_file = os.path.join('s3://llc/mae', local_file[2:]) 
            # upload to s3
            if args.rank == 0:
                ulmo_io.upload_file_to_s3(local_file, s3_file)
        
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
        
        # upload and update log file per epoch
        log_file = os.path.join(args.output_dir, 'log.txt')
        s3_log_file = os.path.join('s3://llc/mae', log_file)
        if log_file[:2] == './':    # remove ./ if hidden folder
            s3_log_file = os.path.join('s3://llc/mae', log_file[2:]) 
        if args.rank == 0:
            ulmo_io.upload_file_to_s3(log_file, s3_log_file)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    arg = get_args_parser()
    arg = arg.parse_args()
    if arg.output_dir:
        Path(arg.output_dir).mkdir(parents=True, exist_ok=True)
    # I think this works now. Any errors spat out by this is cause something later in the code failed.
    world_size = torch.cuda.device_count()
    main(arg)
