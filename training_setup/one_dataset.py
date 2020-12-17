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

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# Plotting Style
sns.set_style('darkgrid')

MODELS = {"vgg" : models.vgg_like}


def train(args, ITE = 0):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reinit = True if args.pt == "reinit" else False

    date = datetime.now().strftime('%d_%m_%Y_%H:%M:%S').replace(' ', '_')
    train_loader, val_source, val_target = data_utils.get_data(args.ds, args.source, args.target, args.data_dir,
                                                                args.batch_size, "one_ds")

    global model

    model = MODELS[args.model].get_vgg(args.size, args.pretrained)
    model.to(device)

    initial_state_dict = copy.deepcopy(model.state_dict())
    utils.checkdir(f"{args.data_dir}/saves/{args.model}/{date}/{args.ds}/")
    torch.save(model, f"{args.data_dir}/saves/{args.model}/{date}/{args.ds}/initial_state_dict_{args.pt}.pth.tar")

    get_masks(model)

    optimizer = optim.Adam(model.parameters(), weight_decay=10**-3)
    class_loss = nn.NLLLoss()

    best_accuracy = 0
    best_accuracy_source = 0
    best_accuracy_target = 0
    ITERATION = args.prune_iterations
    comp = np.zeros(ITERATION, float)
    bestacc = np.zeros(ITERATION, float)
    bestacc_source = np.zeros(ITERATION, float)
    bestacc_target = np.zeros(ITERATION, float)
    step = 0
    all_loss = np.zeros(args.end_iter, float)
    all_accuracy = np.zeros(args.end_iter, float)
    all_accuracy_source = np.zeros(args.end_iter, float)
    all_accuracy_target = np.zeros(args.end_iter, float)

    for _ite in range(args.start_iter, ITERATION):
        if _ite != 0 and (((_ite - args.start_iter) % args.iter_to_prune) == 0) and not args.lotter:
            prune_by_perc(args.prune_percent)
            if reinit:
                if args.pretrained:
                    model.apply(MODELS[args.model].classifier_init)
                else:
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
            optimizer = optim.Adam(model.parameters(), weight_decay=10 ** (-3))

        print(f"\n--- Pruning Level [{ITE}:{_ite}/{ITERATION}]: ---")

        # Print the table of Nonzeros in each layer
        comp1 = utils.print_nonzeros(model)
        comp[_ite] = comp1
        pbar = tqdm(range(args.end_iter))
        
        for iter_ in pbar:
            # Frequency for Testing
            if iter_ % args.valid_freq == 0:
                accuracy_source, accuracy_target = test(model, val_source, val_target, args.batch_size, class_loss)

                    # Save Weights
                if (0.4 * accuracy_source + 0.6 * accuracy_target) > best_accuracy:
                    best_accuracy = (0.4 * accuracy_source + 0.6 * accuracy_target)
                    utils.checkdir(f"{args.data_dir}/saves/{args.model}/{date}/{args.ds}/")
                    torch.save(model,
                                   f"{args.data_dir}/saves/{args.model}/{date}/{args.ds}/initial_state_dict_{args.pt}_{_ite}.pth.tar")

                if accuracy_source > best_accuracy_source:
                    best_accuracy_source = accuracy_source

                if accuracy_target > best_accuracy_target:
                    best_accuracy_target = accuracy_target

                # Training
            loss = train_iter(model, train_loader, optimizer, class_loss)
            all_loss[iter_] = loss
            all_accuracy[iter_] = (0.4 * accuracy_source + 0.6 * accuracy_target)

                # Frequency for Printing Accuracy and Loss
            if iter_ % args.print_freq == 0:
                pbar.set_description(
                        f'Train Epoch: {iter_}/{args.end_iter} Loss: {loss:.6f} Accuracy: {(0.4 * accuracy_source + 0.6 * accuracy_target):.2f}% Best Accuracy: {best_accuracy:.2f}%'
                        f'Source_accuracy: {accuracy_source:.2f}% Best Accuracy: {best_accuracy_source:.2f}%'
                        f'Source_accuracy: {accuracy_target:.2f}% Best Accuracy: {best_accuracy_target:.2f}%')

        bestacc[_ite] = best_accuracy
        bestacc_source[_ite] = best_accuracy_source
        bestacc_target[_ite] = best_accuracy_target

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
        utils.checkdir(f"{args.data_dir}/plots/lt/{args.model}/{date}/{args.ds}/")
        plt.savefig(
            f"{args.data_dir}/plots/lt/{args.model}/{date}/{args.ds}/{args.pt}_LossVsAccuracy_{comp1}_{_ite}.png",
            dpi=1200)
        plt.close()


        plt.plot(np.arange(1, (args.end_iter) + 1), all_accuracy_source, c="blue", label="Source accuracy")
        plt.plot(np.arange(1, (args.end_iter) + 1), all_accuracy_target, c="red", label="Target_accuracy")
        plt.title(f"Source accuracy Vs Target_accuracy Vs Iterations ({args.ds},vgg_like_1)")
        plt.xlabel("Iterations")
        plt.ylabel("Source accuracy and Target_accuracy")
        plt.legend()
        plt.grid(color="gray")
        utils.checkdir(f"{args.data_dir}/plots/lt/{args.model}/{date}/{args.ds}/")
        plt.savefig(
            f"{args.data_dir}/plots/lt/{args.model}/{date}/{args.ds}/{args.pt}_SourceVSTarget_{comp1}_{_ite}.png",
            dpi=1200)
        plt.close()

        # Dump Plot values
        utils.checkdir(f"{args.data_dir}/dumps/lt/{args.model}/{date}/{args.ds}/")
        all_loss.dump(f"{args.data_dir}/dumps/lt/{args.model}/{date}/{args.ds}/{args.pt}_all_loss_{comp1}_{_ite}.dat")
        all_accuracy.dump(
            f"{args.data_dir}/dumps/lt/{args.model}/{date}/{args.ds}/{args.pt}_all_accuracy_{comp1}_{_ite}.dat")
        all_accuracy_source.dump(
            f"{args.data_dir}/dumps/lt/{args.model}/{date}/{args.ds}/{args.pt}_all_accuracy_source_{comp1}_{_ite}.dat")
        all_accuracy_target.dump(
            f"{args.data_dir}/dumps/lt/{args.model}/{date}/{args.ds}/{args.pt}_all_accuracy_target_{comp1}_{_ite}.dat")

        # Dumping mask
        utils.checkdir(f"{args.data_dir}/dumps/lt/{args.model}/{date}/{args.ds}/")
        with open(f"{args.data_dir}/dumps/lt/{args.model}/{date}/{args.ds}/{args.pt}_mask_{comp1}_{_ite}.pkl",'wb') as fp:
            pickle.dump(mask, fp)

        # Making variables into 0
        best_accuracy = 0
        all_loss = np.zeros(args.end_iter, float)
        all_accuracy = np.zeros(args.end_iter, float)
        all_accuracy_source = np.zeros(args.end_iter, float)
        all_accuracy_target = np.zeros(args.end_iter, float)

        # Dumping Values for Plotting
    utils.checkdir(f"{args.data_dir}/dumps/lt/{args.model}/{date}/{args.ds}/")
    comp.dump(f"{args.data_dir}/dumps/lt/{args.model}/{date}/{args.ds}/{args.pt}_compression.dat")
    bestacc.dump(f"{args.data_dir}/dumps/lt/{args.model}/{date}/{args.ds}/{args.pt}_bestaccuracy.dat")
    bestacc_source.dump(f"{args.data_dir}/dumps/lt/{args.model}/{date}/{args.ds}/{args.pt}_bestaccuracy_source.dat")
    bestacc_target.dump(f"{args.data_dir}/dumps/lt/{args.model}/{date}/{args.ds}/{args.pt}_bestaccuracy_target.dat")

    # Plotting
    a = np.arange(args.prune_iterations)
    plt.plot(a, bestacc, c="blue", label="Winning tickets global accuracy")
    plt.plot(a, bestacc_source, c="red", label="Winning ticket source accuracy")
    plt.plot(a, bestacc_target, c="green", label="Winning ticket target accuracy")
    plt.title(f"Test Accuracies vs Unpruned Weights Percentage ({args.ds},{args.model})")
    plt.xlabel("Unpruned Weights Percentage")
    plt.ylabel("test accuracies")
    plt.xticks(a, comp, rotation="vertical")
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(color="gray")
    utils.checkdir(f"{args.data_dir}/plots/lt/{args.model}/{date}/{args.ds}/")
    plt.savefig(f"{args.data_dir}/plots/lt/{args.model}/{date}/{args.ds}/{args.pt}_AccuraciesVsWeights.png",dpi=1200)
    plt.close()


def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.normal_(m.bias.data)



def prune_by_perc(percent):
    global step
    global mask
    global model

    for name, param in model.classifier.named_parameters():
        if ('weight' in name):
            if ('classifier' in name):
                tensor = param.data.cpu().numpy()
                nz = tensor[np.nonzero(tensor)]
                perc_value = np.percentile(abs(nz), (percent))

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


def train_iter(model, source, optimizer, class_loss):

    EPS = 1e-6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    for img, labels in source:

        optimizer.zero_grad()
        img, labels = img.to(device), labels.to(device)
        preds = model(img)
        loss = class_loss(preds, labels)

        loss.backward()

        for name, p in model.named_parameters():
            if 'weight' in name:
                tensor = p.data.cpu().numpy()
                grad_tensor = p.grad.data.cpu().numpy()
                grad_tensor = np.where(tensor < EPS, 0, grad_tensor)
                p.grad.data = torch.from_numpy(grad_tensor).to(device)
        optimizer.step()

    return loss.item()

# Function for Testing
def test(model, source, target, batch_size, criterion):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_source_loss, test_target_loss, correct_source, correct_target = 0, 0, 0, 0
    with torch.no_grad():
        source_iter, target_iter = iter(source), iter(target)
        for i in range(min(len(source_iter), len(target_iter))):
            imgs_source, source_class = source_iter.next()
            imgs_target, target_class = target_iter.next()
            imgs_source, source_class = imgs_source.to(device), source_class.to(device)
            imgs_target, target_class = imgs_target.to(device), target_class.to(device)
            output_source = model(imgs_source)
            output_target = model(imgs_target)
            test_source_loss += criterion(output_source, source_class).item()
            test_target_loss += criterion(output_target, target_class).item()
            pred_source = output_source.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            pred_target = output_source.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct_source += pred_source.eq(source_class.data.view_as(pred_source)).sum().item()
            correct_target += pred_target.eq(target_class.data.view_as(pred_target)).sum().item()
        test_source_loss /= (min(len(source_iter), len(target_iter)) * batch_size)
        test_target_loss /= (min(len(source_iter), len(target_iter)) * batch_size)
        accuracy_source = 100. * correct_source / (min(len(source_iter), len(target_iter)) * batch_size)
        accuracy_target = 100. * correct_target / (min(len(source_iter), len(target_iter)) * batch_size)
    return accuracy_source, accuracy_target

print('Finished Training')
