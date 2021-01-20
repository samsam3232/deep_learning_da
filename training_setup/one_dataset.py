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
import training_setup.training_utils as training_utils
import pruners.influence_factor as if_pruner
import copy
from collections import defaultdict

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# Plotting Style
sns.set_style('darkgrid')

MODELS = {"vgg" : models.vgg_like.VggGetter}


def train(args, ITE = 0):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reinit = True if args.pt == "reinit" else False

    date = datetime.now().strftime('%d_%m_%Y_%H:%M:%S').replace(' ', '_')
    train_loader, train_target, val_source, val_target = data_utils.get_data(args.ds, args.source, args.target, args.data_dir,
                                                                args.batch_size, "full_opt")

    global vggGetter
    global masks

    vggGetter = MODELS[args.model](size = args.size, pretrained = args.pretrained, freeze_all = args.freeze_all, ganin_da = args.ganin)
    vggGetter.model.to(device)

    initial_state_dict = copy.deepcopy(get_initial_state(vggGetter.model))
    utils.checkdir(f"{args.data_dir}/saves/{args.model}/{date}/{args.ds}/")
    torch.save(vggGetter.model, f"{args.data_dir}/saves/{args.model}/{date}/{args.ds}/initial_state_dict_{args.pt}.pth.tar")

    masks = get_masks(vggGetter.model)

    optimizer = optim.SGD(vggGetter.model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    ITERATION = args.prune_iterations
    end_iter = args.end_iter
    best_accuracy, best_accuracy_source, best_accuracy_target, comp, bestacc,bestacc_source, bestacc_target, all_loss, all_accuracy, all_accuracy_source, all_accuracy_target = training_utils.init_lists(ITERATION, end_iter)

    pruner = if_pruner.IFPruner(masks=masks, device=device, prune_features=args.prune_features, prune_classifier=
                                args.prune_classifier, percent=args.prune_percent)

    for _ite in range(args.start_iter, ITERATION):
        if _ite != 0 and (((_ite - args.start_iter) % args.iter_to_prune) == 0) and not args.lotter:
            vggGetter.model = pruner.IFFeaturesPruner.prune_by_perc(args.prune_percent, activations, vggGetter.model, args.cl)
            if reinit:
                vggGetter.init_weights(args.pretrained, args.freeze_all)
                for m in model.features._modules.keys():
                    if isinstance(m, nn.Sequential):
                        continue
                    module = vggGetter.model.features._modules[m]
                    if "conv" in module._get_name():
                        weight, _ = module.parameters()
                        weight_dev = weight.device
                        weight.data = torch.from_numpy(weight.data.cpu().numpy() * masks["features"][m]).to(weight_dev)
                for m in vggGetter.model.classifier._modules.keys():
                    if isinstance(m, nn.Sequential):
                        continue
                    module = vggGetter.model.classifier._modules[m]
                    if "linear" in module._get_name():
                        weight, _ = module.parameters()
                        weight_dev = weight.device
                        weight.data = torch.from_numpy(weight.data.cpu().numpy() * masks["classifier"][m]).to(weight_dev)
            else:
                original_initialization(masks, initial_state_dict)
            optimizer = optim.Adam(vggGetter.model.parameters(), weight_decay=10 ** (-3))

        print(f"\n--- Pruning Level [{ITE}:{_ite}/{ITERATION}]: ---")

        # Print the table of Nonzeros in each layer
        comp1 = utils.print_nonzeros(vggGetter.model)
        comp[_ite] = comp1
        pbar = tqdm(range(args.end_iter))
        
        for iter_ in pbar:
            # Frequency for Testing
            if iter_ % args.valid_freq == 0:
                accuracy_source, accuracy_target = test(vggGetter, val_source, val_target, args.batch_size, criterion, args.alpha, args.ganin)

                    # Save Weights
                if (0.4 * accuracy_source + 0.6 * accuracy_target) > best_accuracy:
                    best_accuracy = (0.4 * accuracy_source + 0.6 * accuracy_target)
                    utils.checkdir(f"{args.data_dir}/saves/{args.model}/{date}/{args.ds}/")
                    torch.save(vggGetter.model,
                                   f"{args.data_dir}/saves/{args.model}/{date}/{args.ds}/initial_state_dict_{args.pt}_{_ite}.pth.tar")

                if accuracy_source > best_accuracy_source:
                    best_accuracy_source = accuracy_source

                if accuracy_target > best_accuracy_target:
                    best_accuracy_target = accuracy_target

                # Training
            loss, activations_source = train_iter(vggGetter, train_loader, args.ganin, optimizer, criterion, args.alpha)
            activations_target = target_activations(vggGetter, train_target, args.ganin, args.alpha)
            activations = (activations_source, activations_target)
            all_loss[iter_] = loss
            all_accuracy[iter_] = (0.4 * accuracy_source + 0.6 * accuracy_target)
            all_accuracy_source[iter_] = accuracy_source
            all_accuracy_target[iter_] = accuracy_target

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
            pickle.dump(masks, fp)

        # Making variables into 0
        best_accuracy = 0
        best_accuracy_source = 0
        best_accuracy_target = 0
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



def get_masks(model):
    global step
    masks = defaultdict(lambda: defaultdict(lambda : 0))

    for m in model.features._modules.keys():
        if isinstance(m, nn.Sequential):
            continue
        module = model.features._modules[m]
        if "conv" in module._get_name().lower():
            weight, _ = module.parameters()
            masks["features"][m] = weight.data
    for m in model.classifier._modules.keys():
        if isinstance(m, nn.Sequential):
            continue
        module = model.classifier._modules[m]
        if "linear" in module._get_name().lower():
            weight, _ = module.parameters()
            masks["classifier"][m] = weight.data

    return masks

def original_initialization(mask_temp, initial_state_dict):
    global model

    for m in model.features._modules.keys():
        if isinstance(m, nn.Sequential):
            continue
        module = model.features._modules[m]
        if "conv" in module._get_name().lower():
            weight, bias = module.parameters()
            weight_dev = weight.device
            weight.data = torch.from_numpy(initial_state_dict["features"]["weight"][m].cpu().numpy() * mask_temp["features"][m]).to(weight_dev)
            bias.data = initial_state_dict["features"]["bias"][m]
    for m in model.classifier._modules.keys():
        if isinstance(m, nn.Sequential):
            continue
        module = model.classifier._modules[m]
        if "linear" in module._get_name().lower():
            weight, bias = module.parameters()
            weight_dev = weight.device
            weight.data = torch.from_numpy(initial_state_dict["classifier"]["weight"][m].cpu().numpy() * mask_temp["classifier"][m]).to(weight_dev)
            bias.data = initial_state_dict["classifier"]["bias"][m]

def train_iter(model, source, ganin, optimizer, criterion, alpha):

    EPS = 1e-6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    for img, labels in source:

        optimizer.zero_grad()
        img, labels = img.to(device), labels.to(device)
        preds, _, _, activations_source = model(img, alpha, ganin, True, False)
        loss = criterion(preds, labels)

        loss.backward()

        for name, p in model.named_parameters():
            if 'weight' in name:
                tensor = p.data.cpu().numpy()
                grad_tensor = p.grad.data.cpu().numpy()
                grad_tensor = np.where(tensor < EPS, 0, grad_tensor)
                p.grad.data = torch.from_numpy(grad_tensor).to(device)
        optimizer.step()

    return loss.item(), activations_source

def target_activations(model, target, ganin, alpha):

    target_iter = iter(target)
    imgs, _ = target_iter.next()
    imgs.to(model.device)
    _, _, _, activations_target = model(imgs, alpha, ganin, True, False)
    return activations_target

# Function for Testing
def test(model, source, target, batch_size, criterion, alpha, ganin):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.model.eval()
    test_source_loss, test_target_loss, correct_source, correct_target = 0, 0, 0, 0
    with torch.no_grad():
        source_iter, target_iter = iter(source), iter(target)
        for i in range(min(len(source_iter), len(target_iter))):
            imgs_source, source_class = source_iter.next()
            imgs_target, target_class = target_iter.next()
            imgs_source, source_class = imgs_source.to(device), source_class.to(device)
            imgs_target, target_class = imgs_target.to(device), target_class.to(device)
            output_source, _, _, _ = model(imgs_source, alpha, ganin, False, False)
            output_target, _, _, _ = model(imgs_target, alpha, ganin, False, False)
            test_source_loss += criterion(output_source, source_class).item()
            test_target_loss += criterion(output_target, target_class).item()
            pred_source = output_source.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            pred_target = output_target.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct_source += pred_source.eq(source_class.data.view_as(pred_source)).sum().item()
            correct_target += pred_target.eq(target_class.data.view_as(pred_target)).sum().item()
        test_source_loss /= (min(len(source_iter), len(target_iter)) * batch_size)
        test_target_loss /= (min(len(source_iter), len(target_iter)) * batch_size)
        accuracy_source = 100. * correct_source / (min(len(source_iter), len(target_iter)) * batch_size)
        accuracy_target = 100. * correct_target / (min(len(source_iter), len(target_iter)) * batch_size)
    return accuracy_source, accuracy_target

def get_initial_state(model):

    initial_state = {"features":{"weight":dict(),"bias":dict()},"classifier":{"weight":dict(),"bias":dict()}}

    for m in model.features._modules.keys():
        if isinstance(m, nn.Sequential):
            continue
        module = model.features._modules[m]
        if "conv" in module._get_name().lower():
            weight, bias = module.parameters()
            initial_state["features"]["weight"][m] = weight.data
            initial_state["features"]["bias"][m] = bias.data
    for m in model.classifier._modules.keys():
        if isinstance(m, nn.Sequential):
            continue
        module = model.classifier._modules[m]
        if "linear" in module._get_name().lower():
            weight, bias = module.parameters()
            initial_state["classifier"]["weight"][m] = weight.data
            initial_state["classifier"]["bias"][m] = bias.data

    return initial_state

