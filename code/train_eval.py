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
parser.add_argument("--weight_rot", default=1.0, type=float, help="Weight for the rotation loss")
parser.add_argument('--weight_ent', default=0.1, type=float, help="Weight for the entropy loss")
args = parser.parse_args()

# Run name
hp_list = [
    'rgbd-rr',          # Task
    'resnet18',         # Backbone. For these experiments we only use ResNet18
    args.epochs,        # Number of epochs
    args.lr,            # Learning rate
    args.lr_mult,       # Learning rate multiplier for the non-pretrained parts of the network,
    args.batch_size,    # Batch size
    args.weight_rot,    # Trade-off weight for the rotation classifier loss
    args.weight_ent     # Trade-off weight for the entropy regularization loss
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
test_transform = MyTransformer([int((256 - INPUT_RESOLUTION) / 2), int((256 - INPUT_RESOLUTION) / 2)], False)

"""
    Prepare datasets
"""

data_root_source, data_root_target, split_source_train, split_source_test, split_target = make_paths(args.data_root, args.samllset)

# Source: training set
train_set_source = DatasetGeneratorMultimodal(data_root_source, split_source_train, do_rot=False)

# Source: test set
test_set_source = DatasetGeneratorMultimodal(data_root_source, split_source_test, do_rot=False,
                                             transform=test_transform)

# Target: training set (for entropy)
train_set_target = DatasetGeneratorMultimodal(data_root_target, split_target, ds_name='ROD',
                                              do_rot=False)

# Target: test set
test_set_target = DatasetGeneratorMultimodal(data_root_target, split_target, ds_name='ROD', do_rot=False,
                                             transform=test_transform)

# Source: training set (for relative rotation)
rot_set_source = DatasetGeneratorMultimodal(data_root_source, split_source_train, do_rot=True)

# Source: test set (for relative rotation)
rot_test_set_source = DatasetGeneratorMultimodal(data_root_source, split_source_test, do_rot=True)

# Target: training and test set (for relative rotation)
rot_set_target = DatasetGeneratorMultimodal(data_root_target, split_target, ds_name='ROD',
                                            do_rot=True)

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

# Source rot
rot_source_loader = DataLoader(rot_set_source,
                               shuffle=True,
                               batch_size=args.batch_size,
                               num_workers=args.num_workers,
                               drop_last=True)

rot_test_source_loader = DataLoader(rot_test_set_source,
                                    shuffle=True,
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers,
                                    drop_last=False)

# Target rot

rot_target_loader = DataLoader(rot_set_target,
                               shuffle=True,
                               batch_size=args.batch_size,
                               num_workers=args.num_workers,
                               drop_last=True)

rot_test_target_loader = DataLoader(rot_set_target,
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
netF = ResClassifier(input_dim=input_dim_F * 2, class_num=47, dropout_p=args.dropout_p)
netF.apply(weights_init)
# Pretext task: relative rotation classifier
netF_rot = RelativeRotationClassifier(input_dim=input_dim_F * 2, class_num=4)
netF_rot.apply(weights_init)

# Define a list of the networks. Move everything on the GPU
net_list = [netG_rgb, netG_depth, netF, netF_rot]
net_list = map_to_device(device, net_list)

# Classification loss
ce_loss = nn.CrossEntropyLoss()

# Optimizers
opt_g_rgb = optim.SGD(netG_rgb.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
opt_g_depth = optim.SGD(netG_depth.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
opt_f = optim.SGD(netF.parameters(), lr=args.lr * args.lr_mult, momentum=0.9, weight_decay=args.weight_decay)
opt_f_rot = optim.SGD(netF_rot.parameters(), lr=args.lr * args.lr_mult, momentum=0.9, weight_decay=args.weight_decay)

optims_list = [opt_g_rgb, opt_g_depth, opt_f, opt_f_rot]

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

    # Source (rotation)
    rot_source_loader_iter = IteratorWrapper(rot_source_loader)
    # Target (rotation)
    rot_target_loader_iter = IteratorWrapper(rot_target_loader)

    # Training loop. The tqdm thing is to show progress bar
    with tqdm(total=len(train_loader_source), desc="Training: ") as pb:
        for batch_num, (img_rgb, img_depth, img_label_source) in enumerate(train_loader_source_rec_iter):
            # The optimization step is performed by OptimizerManager
            with OptimizerManager(optims_list):
                # Compute source features
                img_rgb, img_depth, img_label_source = map_to_device(device, (img_rgb, img_depth, img_label_source))
                """
                Compute features for RGB and Depth, concatenate them along the feature dimension
                and then compute the main task logits.
                
                Then compute the classidication loss. 
                """
                feat_rgb, _ = netG_rgb(img_rgb)
                feat_depth, _ = netG_depth(img_depth)
                features_source = torch.cat((feat_rgb, feat_depth), 1)
                logits = netF(features_source)

                # Classification los
                train_loss_cls = ce_loss(logits, img_label_source)

                # Entropy loss
                if args.weight_ent > 0.:
                    # Load target batch
                    img_rgb, img_depth, _ = train_target_loader_iter.get_next()
                    # Compute target features
                    img_rgb, img_depth = map_to_device(device, (img_rgb, img_depth))
                    feat_rgb, _ = netG_rgb(img_rgb)
                    feat_depth, _ = netG_depth(img_depth)
                    features_target = torch.cat((feat_rgb, feat_depth), 1)
                    logits = netF(features_target)

                    loss_ent = entropy_loss(logits)
                else:
                    loss_ent = 0

                # Backpropagate
                loss = train_loss_cls + args.weight_ent * loss_ent # compute the total loss before backpropagating
                loss.backward()

                del img_rgb, img_depth, img_label_source

                # Relative Rotation
                if args.weight_rot > 0.0:
                    # Load batch: rotation, source
                    img_rgb, img_depth, _, rot_label = rot_source_loader_iter.get_next()
                    """
                    Compute the features (without pooling!), concatenate them and
                    then compute the rotation classification loss
                    """

                    img_rgb, img_depth, rot_label = map_to_device(device, (img_rgb, img_depth, rot_label))

                    # Compute features (without pooling!)
                    _, pooled_rgb = netG_rgb(img_rgb)
                    _, pooled_depth = netG_depth(img_depth)
                    # Prediction
                    logits_rot = netF_rot(torch.cat((pooled_rgb, pooled_depth), 1))

                    # Classification loss for the rleative rotation task
                    loss_train_rot = ce_loss(logits_rot, rot_label)
                    loss = args.weight_rot * loss_train_rot # compute the total loss

                    # Backpropagate
                    loss.backward()
                    loss_train_rot = loss_train_rot.item()

                    del img_rgb, img_depth, rot_label, pooled_rgb, pooled_depth, logits_rot, loss
                    """
                    Same thing, but for target
                    """
                    # Load batch: rotation, target
                    img_rgb, img_depth, _, rot_label = rot_target_loader_iter.get_next()
                    img_rgb, img_depth, rot_label = map_to_device(device, (img_rgb, img_depth, rot_label))

                    # Compute features (without pooling!)
                    _, pooled_rgb = netG_rgb(img_rgb)
                    _, pooled_depth = netG_depth(img_depth)
                    # Prediction
                    logits_rot = netF_rot(torch.cat((pooled_rgb, pooled_depth), 1))

                    # Classification loss for the rleative rotation task
                    loss = args.weight_rot * ce_loss(logits_rot, rot_label)
                    # Backpropagate
                    loss.backward()
                
                    del img_rgb, img_depth, rot_label, pooled_rgb, pooled_depth, logits_rot, loss

                pb.update(1)

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
        print("Epoch: {} - Validation source accuracy (Classification): {}".format(epoch, val_acc))

    del img_rgb, img_depth, img_label_source, feat_rgb, feat_depth, preds

    #Log accuracy and loss
    writer.add_scalar("Loss/train_cls", train_loss_cls.item(), epoch) # train loss after epoch i
    writer.add_scalar("Loss/val_per_batch", val_loss_cls_per_batch, epoch) # validation loss per batch after epoch i
    writer.add_scalar("Accuracy/val_cls", val_acc_cls, epoch)

    # Relative Rotation
    if args.weight_rot > 0.0:

        # Rotation - source
        actual_test_batches = min(len(rot_test_source_loader), args.test_batches or len(rot_test_source_loader))
        with EvaluationManager(net_list), tqdm(total=actual_test_batches, desc="Validation Source Rotaion: ") as pb:
            rot_test_source_loader_iter = iter(rot_test_source_loader)
            correct = 0.0
            num_predictions = 0.0
            val_loss_rot_source = 0.0
            for num_val_batch, (img_rgb, img_depth, _, rot_label) in enumerate(rot_test_source_loader_iter):
                if num_val_batch >= args.test_batches and args.test_batches > 0:
                    break
                img_rgb, img_depth, rot_label = map_to_device(device, (img_rgb, img_depth, rot_label))

                # Compute features (without pooling)
                _, pooled_rgb = netG_rgb(img_rgb)
                _, pooled_depth = netG_depth(img_depth)
                # Compute predictions
                preds = netF_rot(torch.cat((pooled_rgb, pooled_depth), 1))

                val_loss_rot_source += ce_loss(preds, rot_label).item()
                correct += (torch.argmax(preds, dim=1) == rot_label).sum().item()
                num_predictions += preds.shape[0]

                pb.update(1)

            del img_rgb, img_depth, rot_label, preds

            rot_val_acc_source = correct / num_predictions
            val_loss_rot_source_per_batch = val_loss_rot_source / args.test_batches
            print("Epoch: {} - Validation source rotation accuracy: {}".format(epoch, rot_val_acc_source))

        # Rotation - target
        actual_test_batches = min(len(rot_test_target_loader), args.test_batches or len(rot_test_target_loader))
        with EvaluationManager(net_list), tqdm(total=actual_test_batches, desc="Validation Target Rotation: ") as pb:
            rot_test_target_loader_iter = iter(rot_test_target_loader)
            correct = 0.0
            val_loss_rot_target = 0.0
            num_predictions = 0.0
            for num_val_batch, (img_rgb, img_depth, _, rot_label) in enumerate(rot_test_target_loader_iter):
                if num_val_batch >= args.test_batches and args.test_batches > 0:
                    break

                img_rgb, img_depth, rot_label = map_to_device(device, (img_rgb, img_depth, rot_label))

                # Compute features (without pooling)
                _, pooled_rgb = netG_rgb(img_rgb)
                _, pooled_depth = netG_depth(img_depth)
                # Compute predictions
                preds = netF_rot(torch.cat((pooled_rgb, pooled_depth), 1))

                val_loss_rot_target += ce_loss(preds, rot_label).item()
                correct += (torch.argmax(preds, dim=1) == rot_label).sum().item()
                num_predictions += preds.shape[0]

                pb.update(1)

            rot_val_acc_target = correct / num_predictions
            val_loss_rot_target_per_batch = val_loss_rot_target / args.test_batches
            print("Epoch: {} - Validation target rotation accuracy: {}".format(epoch, rot_val_acc_target))

        del img_rgb, img_depth, rot_label, preds

        writer.add_scalar("Loss/rot", loss_train_rot, epoch)
        writer.add_scalar("Loss/rot_val_source_per_batch", val_loss_rot_source_per_batch, epoch)
        writer.add_scalar("Loss/rot_val_target_per_batch", val_loss_rot_target_per_batch, epoch)
        writer.add_scalar("Accuracy/rot_val_source", rot_val_acc_source, epoch)
        writer.add_scalar("Accuracy/rot_val_target", rot_val_acc_target, epoch)

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
        eval_loss_per_batch = eval_loss / args.test_batches
        print("Epoch: {} - Evaluation Target classification accuracy: {}".format(epoch, eval_acc))

    del img_rgb, img_depth, img_label_target, feat_rgb, feat_depth, preds

    # Log loss and accuracy
    writer.add_scalar("Loss/eval_target_per_batch", eval_loss_per_batch, epoch)
    writer.add_scalar("Accuracy/eval_target", eval_acc, epoch)

    # Save checkpoint
    save_checkpoint(checkpoint_path, epoch, net_list, optims_list)
    print("Checkpoint saved")
