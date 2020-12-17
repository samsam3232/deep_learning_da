import torch
import torch.nn as nn
import argparse
import models.vgg_like as vgg_like
import seaborn as sns
import torch.nn.init as init
import numpy as np
import utils
import pickle
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime
import os
import matplotlib.pyplot as plt
import models
import datasets.data_utils as data_utils
import copy

COMBINATIONS = np.array(np.meshgrid(np.arange(start=-1, stop=2), np.arange(start=-1, stop=2))).T.reshape(-1,2)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Plotting Style
sns.set_style('darkgrid')

def main(args, ITE=0):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reinit = True if args.pt == "reinit" else False
    
    date = datetime.now().strftime('%d_%m_%Y_%H:%M:%S').replace(' ' , '_')
    train_source_loader, train_target_loader, val_source_loader = data_utils.get_data(args.ds, args.source, args.target, args.data_dir, args.batch_size, False)

    global model

    model = models.vgg_like.CCN_Model()
    model.to(device)
    model.apply(weight_init)

    initial_state_dict = copy.deepcopy(model.state_dict())
    utils.checkdir(f"{args.data_dir}/saves/vgg_like_1/{date}/{args.ds}/")
    torch.save(model, f"{args.data_dir}/saves/vgg_like_1/{date}/{args.ds}/initial_state_dict_{args.pt}.pth.tar")

    get_masks(model)

    optimizer = optim.Adam(model.parameters(), weight_decay=10**(-3))
    class_loss = nn.NLLLoss()
    disc_loss = nn.NLLLoss()

    for name, param in model.named_parameters():
        print(name, param.size())


    best_accuracy = 0
    ITERATION = args.prune_iterations
    comp = np.zeros(ITERATION,float)
    bestacc = np.zeros(ITERATION,float)
    step = 0
    all_loss = np.zeros(args.end_iter,float)
    all_accuracy = np.zeros(args.end_iter,float)

    if args.lotter:
        model.load_state_dict(torch.load(f"{args.data_dir}/saves/vgg_like_1/{date}/{args.ds}/initial_state_dict_{args.pt}.pth.tar"))

    for _ite in range(args.start_iter, ITERATION):
        if _ite != 0 and (((_ite - args.start_iter) % args.iter_to_prune) == 0) and not args.lotter:
            prune_by_perc(args.prune_percent, activation_source, activation_target)
            if reinit:
                model.apply(weight_init)
                step = 0
                for name, param in model.named_parameters():
                    if 'weight' in name:
                        weight_dev = param.device
                        param.data = torch.from_numpy(param.data.cpu().numpy() * mask[step]).to(weight_dev)
                        step = step + 1
                step = 0
            else:
                original_initialization(mask, initial_state_dict)
            optimizer = optim.Adam(model.parameters(), lr = args.lr, weight_decay=10**(-3))
        else:
            original_initialization(mask, initial_state_dict)
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=10 ** (-3))

        print(f"\n--- Pruning Level [{ITE}:{_ite}/{ITERATION}]: ---")

        # Print the table of Nonzeros in each layer
        comp1 = utils.print_nonzeros(model)
        comp[_ite] = comp1
        pbar = tqdm(range(args.end_iter))

        for iter_ in pbar:

            alpha = ((iter_ ) + (_ite * iter_))
            # Frequency for Testing
            if iter_ % args.valid_freq == 0:
                accuracy = test(model, val_source_loader, class_loss)

                # Save Weights
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    utils.checkdir(f"{args.data_dir}/saves/vgg_like_1/{date}/{args.ds}/")
                    torch.save(model,
                               f"{args.data_dir}/saves/vgg_like_1/{date}/{args.ds}/initial_state_dict_{args.pt}_{_ite}.pth.tar")

            # Training
            loss, activation_source, activation_target = train(model, train_source_loader, train_target_loader, optimizer, class_loss, disc_loss, alpha)
            all_loss[iter_] = loss
            all_accuracy[iter_] = accuracy

            # Frequency for Printing Accuracy and Loss
            if iter_ % args.print_freq == 0:
                pbar.set_description(
                    f'Train Epoch: {iter_}/{args.end_iter} Loss: {loss:.6f} Accuracy: {accuracy:.2f}% Best Accuracy: {best_accuracy:.2f}%')

        bestacc[_ite] = best_accuracy

        # Plotting Loss (Training), Accuracy (Testing), Iteration Curve
        # NOTE Loss is computed for every iteration while Accuracy is computed only for every {args.valid_freq} iterations. Therefore Accuracy saved is constant during the uncomputed iterations.
        # NOTE Normalized the accuracy to [0,100] for ease of plotting.
        plt.plot(np.arange(1, (args.end_iter) + 1),
                 100 * (all_loss - np.min(all_loss)) / np.ptp(all_loss).astype(float), c="blue", label="Loss")
        plt.plot(np.arange(1, (args.end_iter) + 1), all_accuracy, c="red", label="Accuracy")
        plt.title(f"Loss Vs Accuracy Vs Iterations ({args.ds},vgg_like_1)")
        plt.xlabel("Iterations")
        plt.ylabel("Loss and Accuracy")
        plt.legend()
        plt.grid(color="gray")
        utils.checkdir(f"{args.data_dir}/plots/lt/vgg_like_1/{date}/{args.ds}/")
        plt.savefig(
            f"{args.data_dir}/plots/lt/vgg_like_1/{date}/{args.ds}/{args.pt}_LossVsAccuracy_{comp1}_{_ite}.png",
            dpi=1200)
        plt.close()

        # Dump Plot values
        utils.checkdir(f"{args.data_dir}/dumps/lt/vgg_like_1/{date}/{args.ds}/")
        all_loss.dump(f"{args.data_dir}/dumps/lt/vgg_like_1/{date}/{args.ds}/{args.pt}_all_loss_{comp1}_{_ite}.dat")
        all_accuracy.dump(
            f"{args.data_dir}/dumps/lt/vgg_like_1/{date}/{args.ds}/{args.pt}_all_accuracy_{comp1}_{_ite}.dat")

        # Dumping mask
        utils.checkdir(f"{args.data_dir}/dumps/lt/vgg_like_1/{date}/{args.ds}/")
        with open(f"{args.data_dir}/dumps/lt/vgg_like_1/{date}/{args.ds}/{args.pt}_mask_{comp1}_{_ite}.pkl",
                  'wb') as fp:
            pickle.dump(mask, fp)

        # Making variables into 0
        best_accuracy = 0
        all_loss = np.zeros(args.end_iter, float)
        all_accuracy = np.zeros(args.end_iter, float)

    # Dumping Values for Plotting
    utils.checkdir(f"{args.data_dir}/dumps/lt/vgg_like_1/{date}/{args.ds}/")
    comp.dump(f"{args.data_dir}/dumps/lt/vgg_like_1/{date}/{args.ds}/{args.pt}_compression.dat")
    bestacc.dump(f"{args.data_dir}/dumps/lt/vgg_like_1/{date}/{args.ds}/{args.pt}_bestaccuracy.dat")

    # Plotting
    a = np.arange(args.prune_iterations)
    plt.plot(a, bestacc, c="blue", label="Winning tickets")
    plt.title(f"Test Accuracy vs Unpruned Weights Percentage ({args.ds},vgg_like_1)")
    plt.xlabel("Unpruned Weights Percentage")
    plt.ylabel("test accuracy")
    plt.xticks(a, comp, rotation="vertical")
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(color="gray")
    utils.checkdir(f"{args.data_dir}/plots/lt/vgg_like_1/{date}/{args.ds}/")
    plt.savefig(f"{args.data_dir}/plots/lt/vgg_like_1/{date}/{args.ds}/{args.pt}_AccuracyVsWeights.png",
                dpi=1200)
    plt.close()


def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)


def get_distance_mask(activation_source, activation_target, percent):
    act = dict()
    for layer in activation_source:
        act_source = activation_source[layer][1]
        act_target = activation_target[layer][1]
        dist = (torch.sum(act_source, dim=0) / 60) - (torch.sum(act_target, dim = 0) / 60)
        nz = dist.detach().numpy()[np.nonzero(dist.detach().numpy())]
        perc_value = np.percentile(abs(nz), (100-percent))
        dist_mask = np.where((abs(dist) > perc_value), 1, 0)
        act_source_true, act_target_true = (act_source.detach()*dist_mask), (act_target.detach()*dist_mask)
        act[layer] = (act_source_true, act_target_true)
    return act


def get_influence_factor_priv(msk, source, param):

    base_src = (torch.transpose(source[0], 1, 2))
    base_src = (torch.transpose(base_src, 2, 3))[:, :, :, :, None, None]
    base_src = base_src.detach() * torch.ones(list(base_src.shape[:-2]) + list(param.shape)[-2:])
    base_src = base_src[:, :, :, None, :, :, :] * param
    msk = torch.transpose(msk, 1, 2)
    msk = torch.transpose(msk, 2, 3)
    lcs = torch.randperm(50)[:5]
    msk, base_src = msk[lcs], base_src[lcs]
    base_src_copy = base_src
    locs_src = np.where((msk.numpy() != 0))
    locs_src = list(locs_src)
    locs_src[0], locs_src[1], locs_src[2], locs_src[3] = locs_src[0][locs_src[2] != 0], locs_src[1][locs_src[2] != 0], locs_src[2][locs_src[2] != 0], locs_src[3][locs_src[2] != 0]
    locs_src[0], locs_src[1], locs_src[2], locs_src[3] = locs_src[0][locs_src[2] != base_src.shape[2]-1], locs_src[1][locs_src[2] != base_src.shape[2]-1], locs_src[2][locs_src[2] != base_src.shape[2]-1], locs_src[3][locs_src[2] != base_src.shape[2]-1]
    locs_src[0], locs_src[1], locs_src[2], locs_src[3] = locs_src[0][locs_src[1] != 0], locs_src[1][locs_src[1] != 0], locs_src[2][locs_src[1] != 0], locs_src[3][locs_src[1] != 0]
    locs_src[0], locs_src[1], locs_src[2], locs_src[3] = locs_src[0][locs_src[1] != base_src.shape[2]-1], locs_src[1][locs_src[1] != base_src.shape[2]-1], locs_src[2][locs_src[1] != base_src.shape[2]-1], locs_src[3][locs_src[1] != base_src.shape[2]-1]
    for i in COMBINATIONS:
        locs_curr = list(locs_src)
        locs_curr[2], locs_curr[1] = locs_curr[2] - i[0], locs_curr[1] - i[1]
        base_src[locs_curr][:][:, i[0]+1, i[1]+1] = base_src[locs_curr][:][:, i[0]+1, i[1]+1] / msk[locs_curr].view(-1, 1)
    base_src = np.where(base_src.detach() != base_src_copy.detach(), 0, base_src.detach())
    return np.sum(base_src, axis=(0, 1, 2))


def get_influence_factor(msk, source, target, param):

    if_source = get_influence_factor_priv(msk[0], source, param)
    if_target = get_influence_factor_priv(msk[1], target, param)
    return if_source+if_target


def prune_by_perc(percent, activation_source, activation_target):

    global step
    global mask
    global model

    step = 0
    act_mask = get_distance_mask(activation_source, activation_target, percent)
    for name, param in model.named_parameters():

        if ('weight' in name):
            if ('feature' in name) and ('conv' in name):
                tensor = param.data.cpu().numpy()
                influence_factors = get_influence_factor(act_mask[name.split('.')[1]],
                                                         activation_source[name.split('.')[1]],
                                                         activation_target[name.split('.')[1]], param)
                nz = influence_factors[np.nonzero(influence_factors)]
                perc_value = np.percentile(abs(nz), (100-percent))

                weights = param.device
                nm = np.where(abs(influence_factors) > perc_value, 0, mask[step])

                param.data = torch.from_numpy(tensor * nm).to(weights)
            else:
                if 'classifier' in name:
                    percent=20
                tensor = param.data.cpu().numpy()

                nz = tensor[np.nonzero(tensor)]
                perc_value = np.percentile(abs(nz), percent)

                weights = param.device
                nm = np.where(abs(tensor) < perc_value, 0, mask[step])

                param.data = torch.from_numpy(tensor * nm).to(weights)
            mask[step] = nm
            step += 1

    step = 0


def get_masks(model):
    global step
    global mask

    step = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            step += 1
    mask = [None] * step
    step = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            same_size_tens = param.data.cpu().numpy()
            mask[step] = np.ones_like(same_size_tens)
            step += 1
    step = 0


def original_initialization(mask_temp, initial_state_dict):
    global model

    step = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            weight_dev = param.device
            param.data = torch.from_numpy(mask_temp[step] * initial_state_dict[name].cpu().numpy()).to(weight_dev)
            step = step + 1
        if "bias" in name:
            param.data = initial_state_dict[name]
    step = 0

def train(model, source, target, optimizer, class_loss, disc_loss, alpha):

    EPS = 1e-6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    source_iter, target_iter = iter(source), iter(target)
    source_activations, target_activations = dict(), dict()
    a = min(len(source_iter), len(target_iter))
    for i in range(a):
        imgs_source, source_class = source_iter.next()
        imgs_target, target_class = target_iter.next()
        optimizer.zero_grad()
        imgs_source, source_class = imgs_source.to(device), source_class.to(device)
        imgs_target, target_class = imgs_target.to(device), target_class.to(device)
        source_class_pred, source_domain_pred, source_activations_curr = model(imgs_source, alpha)
        loss_cl_source = class_loss(source_class_pred, source_class)
        loss_dis_source = disc_loss(source_domain_pred, torch.zeros((args.batch_size), dtype = torch.long))

        _, target_domain_pred, target_activations_curr = model(imgs_target, alpha)
        loss_dis_target = disc_loss(target_domain_pred, torch.ones((args.batch_size), dtype=torch.long))
        train_loss = loss_cl_source + loss_dis_source + loss_dis_target
        train_loss.backward()

        for key in list(source_activations_curr.keys()):
            if key not in list(source_activations.keys()):
                source_activations[key] = source_activations_curr[key]
                target_activations[key] = target_activations_curr[key]
            else:
                source_activations[key][0] = source_activations_curr[key][0]+ source_activations[key][0]
                target_activations[key][0] = target_activations_curr[key][0]+ target_activations[key][0]
                source_activations[key][1] = source_activations_curr[key][1]+ source_activations[key][1]
                target_activations[key][1] = target_activations_curr[key][1]+ target_activations[key][1]

        for name, p in model.named_parameters():
            if 'weight' in name:
                tensor = p.data.cpu().numpy()
                grad_tensor = p.grad.data.cpu().numpy()
                grad_tensor = np.where(tensor < EPS, 0, grad_tensor)
                p.grad.data = torch.from_numpy(grad_tensor).to(device)
        optimizer.step()

    for key in list(source_activations):
        source_activations[key][0] = source_activations[key][0] / a
        target_activations[key][0] = target_activations[key][0] / a
        source_activations[key][1] = source_activations[key][1] / a
        target_activations[key][1] = target_activations[key][1] / a
    return train_loss.item(), source_activations, target_activations

# Function for Testing
def test(model, test_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, _, _ = model(data, 0.1)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy


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
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    main(args, ITE=1)

print('Finished Training')
