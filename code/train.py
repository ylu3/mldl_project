#!/usr/bin/env python3
import numpy as np
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from net import ResBase, ResClassifier, RelativeRotationClassifier
from data_loader import DatasetGeneratorMultimodal, MyTransform, INPUT_RESOLUTION
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
    # Task
    'rgbd-rr',
    # Backbone. For these experiments we only use ResNet18
    'resnet18',
    # Number of epochs
    args.epochs,
    # Learning rate
    args.lr,
    # Learning rate multiplier for the non-pretrained parts of the network
    args.lr_mult,
    # Batch size
    args.batch_size,
    # Trade-off weight for the rotation classifier loss
    args.weight_rot,
    # Trade-off weight for the entropy regularization loss
    args.weight_ent
]
if args.suffix is not None:
    hp_list.append(args.suffix)
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
test_transform = MyTransform([int((256 - INPUT_RESOLUTION) / 2), int((256 - INPUT_RESOLUTION) / 2)], False)

"""
    Prepare datasets
"""

data_root_source, data_root_target, split_source_train, split_source_test, split_target = make_paths(args.data_root)

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
    train_loader_source_rec_iter = train_loader_source
    # Train target (entropy)
    train_target_loader_iter = IteratorWrapper(train_loader_target)

    # Source (rotation)
    rot_source_loader_iter = IteratorWrapper(rot_source_loader)
    # Target (rotation)
    rot_target_loader_iter = IteratorWrapper(rot_target_loader)

    # Training loop. The tqdm thing is to show progress bar
    with tqdm(total=len(train_loader_source), desc="Train  ") as pb:
        for batch_num, (img_rgb, img_depth, img_label_source) in enumerate(train_loader_source_rec_iter):
            # The optimization step is performed by OptimizerManager
            with OptimizerManager(optims_list):
                # Compute source features
                img_rgb, img_depth, img_label_source = map_to_device(device, (img_rgb, img_depth, img_label_source))
                # TODO
                """
                Here you should compute features for RGB and Depth, concatenate them along the feature dimension
                and then compute the main task logits.
                
                Then compute the classidication loss. 
                """

                # Entropy loss
                if args.weight_ent > 0.:
                    # Load target batch
                    img_rgb, img_depth, _ = train_target_loader_iter.get_next()

                    # TODO
                    """
                    Here you should compute target features for RGB and Depth, concatenate them and compute logits.
                    Then you use the logits to compute the entropy loss.
                    """
                else:
                    loss_ent = 0

                # Backpropagate
                loss = ...  # TODO: compute the total loss before backpropagating
                loss.backward()

                del img_rgb, img_depth, img_label_source

                # Relative Rotation
                if args.weight_rot > 0.0:
                    # Load batch: rotation, source
                    img_rgb, img_depth, _, rot_label = rot_source_loader_iter.get_next()

                    # TODO
                    """
                    Here you should compute the features (without pooling!), concatenate them and
                    then compute the rotation classification loss
                    """

                    loss_rot = ...  # TODO
                    loss = ...  # TODO: compute the total loss
                    # Backpropagate
                    loss.backward()

                    loss_rot = loss_rot.item()

                    del img_rgb, img_depth, rot_label, loss

                    # Load batch: rotation, target
                    img_rgb, img_depth, _, rot_label = rot_target_loader_iter.get_next()

                    # TODO
                    """
                    Same thing, but for target
                    """

                    del img_rgb, img_depth, rot_label, loss

                pb.update(1)

    # ========================= VALIDATION =========================

    # Classification - source
    actual_test_batches = min(len(test_loader_source), args.test_batches or len(test_loader_source))
    with EvaluationManager(net_list), tqdm(total=actual_test_batches, desc="TestClS") as pb:
        test_source_loader_iter = iter(test_loader_source)

        for num_batch, (img_rgb, img_depth, img_label_source) in enumerate(test_source_loader_iter):
            # By default validate only on 100 batches
            if num_batch >= args.test_batches and args.test_batches > 0:
                break

            # TODO
            """
            Here you should move the batch on GPU, compute the features and then the
            main task prediction
            """

            pb.update(1)

        # TODO: output the accuracy
        print(f"Epoch: {epoch} - Val SRC CLS accuracy: {...}")

    del img_rgb, img_depth, img_label_source

    # TODO: log accuracy and loss
    writer.add_scalar("Loss/train", ..., epoch)
    writer.add_scalar("Loss/val", ..., epoch)
    writer.add_scalar("Accuracy/val", ..., epoch)

    # Relative Rotation
    if args.weight_rot > 0.0:

        # Rotation - source
        cf_matrix = np.zeros((4, 4))
        actual_test_batches = min(len(rot_test_source_loader), args.test_batches or len(rot_test_source_loader))
        with EvaluationManager(net_list), tqdm(total=actual_test_batches, desc="TestRtS") as pb:
            rot_test_source_loader_iter = iter(rot_test_source_loader)

            for num_val_batch, (img_rgb, img_depth, _, rot_label) in enumerate(rot_test_source_loader_iter):
                if num_val_batch >= args.test_batches and args.test_batches > 0:
                    break

                # TODO: very similar to the previous part

                pb.update(1)

            del img_rgb, img_depth, rot_label

            # TODO
            print(f"Epoch: {epoch} - Val SRC ROT accuracy: {...}")

        # Rotation - target
        actual_test_batches = min(len(rot_test_target_loader), args.test_batches or len(rot_test_target_loader))
        with EvaluationManager(net_list), tqdm(total=actual_test_batches, desc="TestRtT") as pb:
            rot_test_target_loader_iter = iter(rot_test_target_loader)

            for num_val_batch, (img_rgb, img_depth, _, rot_label) in enumerate(rot_test_target_loader_iter):
                if num_val_batch >= args.test_batches and args.test_batches > 0:
                    break

                # TODO: very similar to the previous part

                pb.update(1)

            # TODO
            print(f"Epoch: {epoch} - Val TRG ROT accuracy: {...}")

        del img_rgb, img_depth, rot_label

        # TODO
        writer.add_scalar("Loss/rot", ..., epoch)
        writer.add_scalar("Loss/rot_val", ..., epoch)
        writer.add_scalar("Accuracy/rot_val", ..., epoch)

    # Classification - target
    with EvaluationManager(net_list), tqdm(total=len(test_loader_target), desc="TestClT") as pb:
        # Test target

        for num_batch, (img_rgb, img_depth, img_label_source) in enumerate(test_loader_target):
            # Compute source features
            img_rgb, img_depth, img_label_source = map_to_device(device, (img_rgb, img_depth, img_label_source))

            # TODO: very similar to the previous part

            pb.update(1)

        # TODO: Output accuracy
        print(f"Epoch: {epoch} - Val TRG CLS accuracy: {...}")

    del img_rgb, img_depth, img_label_source

    # TODO: log loss and accuracy
    writer.add_scalar("Loss/train_target", ..., epoch)
    writer.add_scalar("Loss/val_target", ..., epoch)
    writer.add_scalar("Accuracy/val_target", ..., epoch)

    # Save checkpoint
    save_checkpoint(checkpoint_path, epoch, net_list, optims_list)
    print("Checkpoint saved")
