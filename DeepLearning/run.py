import argparse
import torch
from torch.utils.data import DataLoader

from DeepLearning.dataloaders.CoMoFodDataloader import CoMoFodDataloader
from DeepLearning.net.architecture.unet.unet import UNet
from DeepLearning.net.custom_layers.DiceLoss import DiceLoss
from DeepLearning.utils.learning import train_on_epoch

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Copy-Move Forgery Detection")
    parser.add_argument("--dataset_path", default='/media/jacob/DATA_DRIVE1/DATA/IMG_FORGERY',
                        help='Path with data')
    parser.add_argument("--image_size", default=256,
                        help='size of input image')
    parser.add_argument("--n_epoch", default=50, help='Number of epochs for training')
    parser.add_argument("--optimizer", default='Adam', help='Choose optimizer. May be Adam or SGD')
    parser.add_argument("--batch_size", default=32, help='Size of batch for training')
    parser.add_argument('--lr', dest='lr', default=0.001, help='learning rate')
    parser.add_argument('--loss', dest='loss', default="Dice", choices=["Dice", "CE"],
                      help='Loss functios to use.')

    args = parser.parse_args()

    train_loader = CoMoFodDataloader(datasetPath=args.dataset_path, imgSize=args.image_size)
    dataloader = DataLoader(train_loader, batch_size=args.batch_size)

    # Use GPU or not
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Create the model
    net = UNet(n_channels=3, n_classes = 1).to(device)
    net = torch.nn.DataParallel(net, device_ids=list(
        range(torch.cuda.device_count()))).to(device)

    # Definition of the optimizer ADD MORE IF YOU WANT
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(net.parameters(),
                                     lr=args.lr)
    elif args.optimizer == "SGD":
        optimizer = torch.optim.SGD(net.parameters(),
                                    lr=args.lr,
                                    momentum=0.9,
                                    weight_decay=0.0005)

    if args.loss == "Dice":
        criterion = DiceLoss()

    for epoch_idx in range(args.n_epoch):
        loss_train, time_exe = train_on_epoch(model=net, dataloader=dataloader, loss_fn=criterion, optimizer=optimizer, device=device)
        print('Loss Train {}'.format(loss_train))