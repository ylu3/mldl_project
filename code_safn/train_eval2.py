#!/usr/bin/env python3
import numpy as np
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from net import ResBase, ResClassifier, RelativeRotationClassifier
from data_loader import DatasetGeneratorMultimodal, MyTransformer, INPUT_RESOLUTION
from utils import *
from tqdm import tqdm

# Parse arguments
parser = argparse.ArgumentParser()

add_base_args(parser)
parser.add_argument('--weight_ent', default=0.1, type=float, help="Weight for the entropy loss")
args = parser.parse_args()

# Run name
hp_list = [
    'variation0-safn',
    'rgbd-rr-',   # Task
    'resnet18',         # Backbone. For these experiments we only use ResNet18
    'lr='+str(args.lr),            # Learning rate
    'lr_mult='+str(args.lr_mult),       # Learning rate multiplier for the non-pretrained parts of the network,
    'batch_size='+str(args.batch_size),    # Batch size
    'epoches='+str(args.epochs),        # Number of epochs
    'weight_decay='+str(args.weight_decay), # Weight Decay in L2 regulation
    'weight_ent='+str(args.weight_ent),     # Trade-off weight for the entropy regularization loss
    'dataset=ROD-synROD' if not args.smallset else 'dataset=smallset'
]

if args.suffix is not None:
    hp_list.append(args.suffix)

# Map hyperparameters to string
hp_string = '_'.join(map(str, hp_list))
print(f"Run: {hp_string}")

# Initialize checkpoint path and Tensorboard logger
checkpoint_path = os.path.join(args.logdir, hp_string, 'checkpoint.pth')
writer = SummaryWriter(log_dir=os.path.join(args.logdir, hp_string), flush_secs=5)

# Device. If CUDA is not available (!!!) run on CPU
if not torch.cuda.is_available():
    print("WARNING! CUDA not available")
    device = torch.device('cpu')
else:
    device = torch.device(f'cuda:{args.gpu}')
    # Print device name
    print(f"Running on device {torch.cuda.get_device_name(device)}")

# Center crop, no random flip
test_transform = MyTransformer([int((256 - INPUT_RESOLUTION) / 2), int((256 - INPUT_RESOLUTION) / 2)])

"""
    Prepare datasets
"""

data_root_source, data_root_target, split_source_train, split_source_test, split_target = make_paths(args.data_root, args.smallset)

# Source: training set
train_set_source = DatasetGeneratorMultimodal(data_root_source, split_source_train)

# Source: test set
test_set_source = DatasetGeneratorMultimodal(data_root_source, split_source_test, transform=test_transform)

# Target: training set (for entropy)
train_set_target = DatasetGeneratorMultimodal(data_root_target, split_target, ds_name='ROD')

# Target: test set
test_set_target = DatasetGeneratorMultimodal(data_root_target, split_target, ds_name='ROD', transform=test_transform)

# Source: training set (for relative rotation)
rot_set_source = DatasetGeneratorMultimodal(data_root_source, split_source_train)

# Source: test set (for relative rotation)
rot_test_set_source = DatasetGeneratorMultimodal(data_root_source, split_source_test)

# Target: training and test set (for relative rotation)
rot_set_target = DatasetGeneratorMultimodal(data_root_target, split_target, ds_name='ROD')

"""
    Prepare data loaders
"""

# Source training recognition
train_loader_source = DataLoader(train_set_source,
                                 shuffle=True,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers,
                                 drop_last=True)

# Source test recognition
test_loader_source = DataLoader(test_set_source,
                                shuffle=True,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=False)

# Target train
train_loader_target = DataLoader(train_set_target,
                                 shuffle=True,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers,
                                 drop_last=True)

# Target test
test_loader_target = DataLoader(test_set_target,
                                shuffle=True,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=False)

"""
    Set up network & optimizer
"""
# This needs to be changed if a different backbone is used instead of ResNet18
input_dim_F = 512

# RGB feature extractor based on ResNet18
netG_rgb = ResBase()

# Depth feature extractor based on ResNet18
netG_depth = ResBase()

# Main task: classifier
classnumber = 47 if args.smallset is False else 5
netF = ResClassifier(input_dim=input_dim_F * 2, class_num=classnumber, dropout_p=args.dropout_p, extract=True)
netF.apply(weights_init)

# Define a list of the networks. Move everything on the GPU
net_list = [netG_rgb, netG_depth, netF]
net_list = map_to_device(device, net_list)

# Classification loss
ce_loss = nn.CrossEntropyLoss()

# Sanity check
if args.sanitycheck:
    args.weight_decay = 0

# Optimizers
opt_g_rgb = optim.SGD(netG_rgb.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
opt_g_depth = optim.SGD(netG_depth.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
opt_f = optim.SGD(netF.parameters(), lr=args.lr * args.lr_mult, momentum=0.9, weight_decay=args.weight_decay)

optims_list = [opt_g_rgb, opt_g_depth, opt_f]

first_epoch = 1
if args.resume:
    first_epoch = load_checkpoint(checkpoint_path, first_epoch, net_list, optims_list)

for epoch in range(first_epoch, args.epochs + 1):
    print("Epoch {} / {}".format(epoch, args.epochs))
    # ========================= TRAINING =========================

    # Train source (recognition)
    train_loader_source_rec_iter = iter(train_loader_source)

    # Train target (entropy)
    train_target_loader_iter = IteratorWrapper(train_loader_target)

    # Training loop. The tqdm thing is to show progress bar
    with tqdm(total=len(train_loader_source), desc="Training: ") as pb:
        correct = 0.0
        num_predictions = 0.0
        train_loss = 0
        for batch_num, (img_rgb_source, img_depth_source, img_label_source) in enumerate(train_loader_source_rec_iter):
            # The optimization step is performed by OptimizerManager
            with OptimizerManager(optims_list):
                img_rgb_target, img_depth_target, _ = train_target_loader_iter.get_next()

                # Compute source features
                img_rgb_source, img_depth_source, img_label_source = map_to_device(device, (img_rgb_source, img_depth_source, img_label_source))
                img_rgb_target, img_depth_target = map_to_device(device, (img_rgb_target, img_depth_target))

                """
                Compute features for RGB and Depth, concatenate them along the feature dimension
                and then compute the main task logits.
                
                Then compute the classidication loss. 
                """

                feat_rgb_source, _ = netG_rgb(img_rgb_source)
                feat_depth_source, _ = netG_depth(img_depth_source)
                features_source = torch.cat((feat_rgb, feat_depth), 1)
                fc_source, logits_source = netF(features_source)

                feat_rgb_target, _ = netG_rgb(img_rgb_target)
                feat_depth_target, _ = netG_depth(img_depth_target)
                features_target = torch.cat((feat_rgb, feat_depth), 1)
                fc_target, logits_target = netF(features_source)

                # Classification los
                train_loss_cls = ce_loss(logits_source, img_label_source)
                train_loss += train_loss_cls
                correct += (torch.argmax(logits_source, dim=1) == img_label_source).sum().item()
                num_predictions += logits_source.shape[0]

                if args.sanitycheck:
                    print("Sanity check - Training loss (Classification): {} on {} classes.".format(train_loss/(batch_num + 1), classnumber))

                # Entropy loss
                if args.weight_ent > 0.:
                    loss_ent = entropy_loss(logits_target)
                else:
                    loss_ent = 0

                # Backpropagate
                loss = train_loss_cls + args.weight_ent * loss_ent # compute the total loss before backpropagating
                loss.backward()

                del img_rgb, img_depth, img_label_source

                pb.update(1)
        #Output accuracy
        train_acc = correct / num_predictions
        print("Epoch: {} - Training accuracy (Classification): {}".format(epoch, train_acc))

    # ========================= VALIDATION =========================

    # Classification - source
    actual_test_batches = min(len(test_loader_source), args.test_batches or len(test_loader_source))
    with EvaluationManager(net_list), tqdm(total=actual_test_batches, desc="Validation Classification: ") as pb:
        test_source_loader_iter = iter(test_loader_source)
        correct = 0.0
        num_predictions = 0.0
        val_loss_cls = 0.0
        for num_batch, (img_rgb, img_depth, img_label_source) in enumerate(test_source_loader_iter):
            # By default validate only on 100 batches
            if num_batch >= args.test_batches and args.test_batches > 0:
                break
            """
            Move the batch on GPU, compute the features and then the
            main task prediction
            """
            # Compute source features
            img_rgb, img_depth, img_label_source = map_to_device(device, (img_rgb, img_depth, img_label_source))
            feat_rgb, _ = netG_rgb(img_rgb)
            feat_depth, _ = netG_depth(img_depth)
            features_source = torch.cat((feat_rgb, feat_depth), 1)

            # Compute predictions
            preds = netF(features_source)

            val_loss_cls += ce_loss(preds, img_label_source).item()
            correct += (torch.argmax(preds, dim=1) == img_label_source).sum().item()
            num_predictions += preds.shape[0]

            pb.update(1)

        #Output the accuracy
        val_acc_cls = correct / num_predictions
        val_loss_cls_per_batch = val_loss_cls / args.test_batches
        print("Epoch: {} - Validation source accuracy (Classification): {}".format(epoch, val_acc_cls))

    del img_rgb, img_depth, img_label_source, feat_rgb, feat_depth, preds

    #Log accuracy and loss
    writer.add_scalar("Loss/train_cls", train_loss_cls.item(), epoch) # train loss after epoch i
    writer.add_scalar("Loss/val_cls", val_loss_cls_per_batch, epoch) # validation loss per batch after epoch i
    writer.add_scalar("Accuracy/train_cls", train_acc, epoch)
    writer.add_scalar("Accuracy/val_cls", val_acc_cls, epoch)

    # ========================= EVALUAION =========================

    # Classification - target
    with EvaluationManager(net_list), tqdm(total=len(test_loader_target), desc="Evaluation: ") as pb:
        # Test target
        test_loader_target_iter = iter(test_loader_target)
        correct = 0.0
        num_predictions = 0.0
        eval_loss = 0.0
        for num_batch, (img_rgb, img_depth, img_label_target) in enumerate(test_loader_target_iter):
            # Compute source features
            img_rgb, img_depth, img_label_target = map_to_device(device, (img_rgb, img_depth, img_label_target))
            feat_rgb, _ = netG_rgb(img_rgb)
            feat_depth, _ = netG_depth(img_depth)
            features_target = torch.cat((feat_rgb, feat_depth), 1)

            # Compute predictions
            preds = netF(features_target)

            eval_loss += ce_loss(preds, img_label_target).item()
            correct += (torch.argmax(preds, dim=1) == img_label_target).sum().item()
            num_predictions += preds.shape[0]

            pb.update(1)

        #Output accuracy
        eval_acc = correct / num_predictions
        eval_loss_per_batch = eval_loss / len(test_loader_target)
        print("Epoch: {} - Evaluation Target classification accuracy: {}".format(epoch, eval_acc))

    del img_rgb, img_depth, img_label_target, feat_rgb, feat_depth, preds

    # Log loss and accuracy
    writer.add_scalar("Loss/eval_target", eval_loss_per_batch, epoch)
    writer.add_scalar("Accuracy/eval_target", eval_acc, epoch)

    # Save checkpoint
    save_checkpoint(checkpoint_path, epoch, net_list, optims_list)
    print("Checkpoint saved")
