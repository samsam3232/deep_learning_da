import argparse
import os
import training_setup.one_dataset as one_dataset
import training_setup.if_dann as if_dann

TRAINING_SETUP = {"one_ds" : one_dataset, "if_dann" : if_dann}

def main(args):

    TRAINING_SETUP[args.setup].train(args, ITE=1)
    return 1

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=1.2e-3, type=float, help="Learning rate")
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--start_iter", default=0, type=int)
    parser.add_argument("--end_iter", default=100, type=int)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--valid_freq", default=1, type=int)
    parser.add_argument("--pt", default="lt", type=str, help="lt | reinit")
    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument("--ds", default="OfficeCaltech", type=str, help="CaltechOffice | cifar10 | fashionmnist | cifar100")
    parser.add_argument("--prune_percent", default=10, type=int, help="Pruning percent")
    parser.add_argument("--prune_iterations", default=20, type=int, help="Pruning iterations count")
    parser.add_argument("--iter_to_prune", default=1, type=int, help="Prune every ... iterations")
    parser.add_argument("--lotter", default=False, type=bool)
    parser.add_argument("--source", default="A", type=str)
    parser.add_argument("--target", default="C", type=str)
    parser.add_argument("--setup", default="one_ds", type=str)
    parser.add_argument("--model", default = "vgg", type=str)
    parser.add_argument("--size", default=16, type = int)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--alpha", default=10, type=int)
    parser.add_argument("--freeze_all", action="store_true")
    parser.add_argument("--ganin", action="store_true")
    parser.add_argument("--prune_features", action="store_true")
    parser.add_argument("--prune_classifier", action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    main(args)