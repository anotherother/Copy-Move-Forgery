import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from DeepLearning.dataloaders.CoMoFodDataloader import CoMoFodDataloader
from DeepLearning.net.architecture.unet.unet import unet
from DeepLearning.net.custom_layers.DiceLoss import DiceLoss
from DeepLearning.utils.learning import train_step

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Copy-Move Forgery Detection")
    parser.add_argument("--dataset_path", default='/media/jacob/DATA_DRIVE1/DATA/IMG_FORGERY',
                        help='Path with data')
    parser.add_argument("--image_size", default=256,
                        help='size of input image')
    parser.add_argument("--n_epoch", default=50, help='Number of epochs for training')
    parser.add_argument("--optimizer", default='Adam', help='Choose optimizer. May be Adam or SGD')
    parser.add_argument("--batch_size", default=4, help='Size of batch for training')
    parser.add_argument('--lr', dest='lr', default=0.001, help='learning rate')
    parser.add_argument('--loss', dest='loss', default="bce", choices=["dice", "bce"],
                      help='Loss functios to use.')

    args = parser.parse_args()

    train_loader = CoMoFodDataloader(datasetPath=args.dataset_path, imgSize=args.image_size)
    dataloader = DataLoader(train_loader, batch_size=args.batch_size)

    use_cuda = torch.cuda.is_available()

    net = unet(channels_in=3, classes_out=1).cuda()

    optimizer = None
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(net.parameters(),
                                     lr=args.lr)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(net.parameters(),
                                    lr=args.lr,
                                    momentum=0.9,
                                    weight_decay=0.0005)
    criterion = None
    if args.loss == "dice":
        criterion = DiceLoss()
    elif args.loss == "bce":
        criterion = nn.BCELoss()

    for epoch_idx in range(args.n_epoch):
        loss_train, time_exe = train_step(model=net, dataloader=dataloader, loss_fn=criterion, optimizer=optimizer)
        print('Loss Train {}'.format(loss_train))