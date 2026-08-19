import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
# from torchflow.data.loaders.nde.image import MNIST
from torchvision.transforms import RandomHorizontalFlip, Pad, RandomAffine, \
    CenterCrop, RandomCrop, Compose, ToPILImage, ToTensor
import math

from .front3d.front3d_fast import Front3DFast
# from .voxelroom.voxelroom_fast import VoxelRoomFast


def add_data_args(parser):
    # Data params
    parser.add_argument('--dataset', type=str, default='front3d', )

    # Train params
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_categories', type=int, default=22)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--pin_memory', type=eval, default=False)
    parser.add_argument('--augmentation', type=str, default=None)
    parser.add_argument('--floor_plan', type=eval, default=False)
    parser.add_argument('--wo_floor', type=eval, default=False)
    parser.add_argument('--data_size', type=int, default=120)
    parser.add_argument('--data_dir', type=str, default='livingroom')
    parser.add_argument('--room_type_condition', type=eval, default=False)
    parser.add_argument('--w_arch', type=eval, default=False)
    parser.add_argument('--specific_room_type', type=str, default=None)
    parser.add_argument('--text_condition', type=eval, default=False)
    parser.add_argument('--mix_condition', type=eval, default=False)
    parser.add_argument('--validation_split', type=float, default=0.0)
    parser.add_argument('--split_seed', type=int, default=0)

def get_data_id(args):
    return '{}'.format(args.dataset)


def get_augmentation(augmentation, dataset, data_shape):
    h, w = data_shape
    if augmentation is None:
        pil_transforms = []
    elif augmentation == 'horizontal_flip':
        pil_transforms = [RandomHorizontalFlip(p=0.5)]
    elif augmentation == 'shift':
        pad_h, pad_w = int(0.07 * h), int(0.07 * w)
        if 'cityscapes' in dataset and 'large' in dataset:
            # Annoying, cityscapes images have a 3-border around every image.
            # This messes up shift augmentation and needs to be dealt with.
            assert h == 128 and w == 256
            print('Special cityscapes transform')
            pad_h, pad_w = int(0.075 * h), int(0.075 * w)
            pil_transforms = [CenterCrop((h - 2, w - 2)),
                              RandomHorizontalFlip(p=0.5),
                              Pad((pad_h, pad_w), padding_mode='edge'),
                              RandomCrop((h - 2, w - 2)),
                              Pad((1, 1), padding_mode='constant', fill=3)]

        else:
            pil_transforms = [RandomHorizontalFlip(p=0.5),
                              Pad((pad_h, pad_w), padding_mode='edge'),
                              RandomCrop((h, w))]
    elif augmentation == 'neta':
        assert h == w
        pil_transforms = [Pad(int(math.ceil(h * 0.04)), padding_mode='edge'),
                          RandomAffine(degrees=0, translate=(0.04, 0.04)),
                          CenterCrop(h)]
    elif augmentation == 'eta':
        assert h == w
        pil_transforms = [RandomHorizontalFlip(),
                          Pad(int(math.ceil(h * 0.04)), padding_mode='edge'),
                          RandomAffine(degrees=0, translate=(0.04, 0.04)),
                          CenterCrop(h)]

    # torchvision.transforms.s
    return pil_transforms


def get_augmentation_3d(augmentation, dataset, data_shape):
    h, w, d = data_shape
    if augmentation is None:
        pil_transforms = []

    # torchvision.transforms.s
    return pil_transforms


def get_data(args):
    if args.dataset == 'front3d':
        data_shape = (1, args.data_size, args.data_size)
        num_classes = args.num_categories
        if args.wo_floor:
            num_classes -= 1
        pil_transforms = get_augmentation(args.augmentation, args.dataset,
                                          (args.data_size, args.data_size))
        pil_transforms = Compose(pil_transforms)
        if not hasattr(args, 'data_dir'):
            # old checkpoint has different args structure
            args.data_dir = args.room_type
            args.specific_room_type = None
        train = Front3DFast(root="datasets", split="unified_w_arch",
                            resolution=(args.data_size, args.data_size),
                            transform=pil_transforms, floor_plan=args.floor_plan, wo_floor=args.wo_floor,
                            room_type_condition=args.room_type_condition, w_arch=args.w_arch, specific_room_type=args.specific_room_type, 
                            text_condition=args.text_condition, mixed_condition=args.mix_condition)
        
        # Architecture-fixed fine-tuning requires a real held-out
        # validation split. Keep the original SemLayoutDiff behavior
        # unchanged unless validation_split > 0.
        if args.validation_split > 0:
            if not 0.0 < args.validation_split < 1.0:
                raise ValueError(
                    f"validation_split must be between 0 and 1, "
                    f"got {args.validation_split}"
                )

            # The current augmentation path transforms img after the
            # Architecture condition has already been created, so the
            # two can become spatially misaligned. Do not allow that
            # for Architecture-fixed training.
            if getattr(args, 'architecture_fixed', False) and args.augmentation is not None:
                raise ValueError(
                    "Architecture-fixed fine-tuning requires "
                    "augmentation=None until synchronized "
                    "img/Architecture augmentation is implemented."
                )

            generator = torch.Generator().manual_seed(args.split_seed)

            if args.room_type_condition:
                # Stratify by room type so train/validation preserve
                # the room-type composition of the unified dataset.
                room_type_to_indices = {}

                for index in range(len(train)):
                    room_type_id = int(
                        torch.unique(train.data[index][0]).item()
                    )
                    room_type_to_indices.setdefault(
                        room_type_id, []
                    ).append(index)

                train_indices = []
                eval_indices = []

                for room_type_id in sorted(room_type_to_indices):
                    indices = room_type_to_indices[room_type_id]

                    permutation = torch.randperm(
                        len(indices),
                        generator=generator
                    ).tolist()

                    shuffled = [indices[i] for i in permutation]

                    num_eval = max(
                        1,
                        int(round(
                            len(shuffled) * args.validation_split
                        ))
                    )

                    # Never consume the entire class for validation.
                    if num_eval >= len(shuffled):
                        num_eval = len(shuffled) - 1

                    eval_indices.extend(shuffled[:num_eval])
                    train_indices.extend(shuffled[num_eval:])
            else:
                permutation = torch.randperm(
                    len(train),
                    generator=generator
                ).tolist()

                num_eval = int(round(
                    len(train) * args.validation_split
                ))

                if num_eval <= 0 or num_eval >= len(train):
                    raise ValueError(
                        "validation_split produced an invalid "
                        f"validation size: {num_eval}"
                    )

                eval_indices = permutation[:num_eval]
                train_indices = permutation[num_eval:]

            train_subset = Subset(train, train_indices)
            eval_subset = Subset(train, eval_indices)

            print(
                f"Dataset split - train: {len(train_subset)}, "
                f"validation: {len(eval_subset)}, "
                f"seed: {args.split_seed}"
            )

            if args.room_type_condition:
                train_weights = [
                    train.weights[i] for i in train_indices
                ]

                sampler = torch.utils.data.WeightedRandomSampler(
                    weights=train_weights,
                    num_samples=len(train_subset),
                    replacement=True
                )

                train_loader = DataLoader(
                    train_subset,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    sampler=sampler,
                    pin_memory=args.pin_memory
                )
            else:
                train_loader = DataLoader(
                    train_subset,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    shuffle=True,
                    pin_memory=args.pin_memory
                )

            eval_loader = DataLoader(
                eval_subset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory
            )

        else:
            # Original SemLayoutDiff behavior.
            if args.room_type_condition:
                sampler = torch.utils.data.WeightedRandomSampler(
                    weights=train.weights,
                    num_samples=len(train),
                    replacement=True
                )

                train_loader = DataLoader(
                    train,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    sampler=sampler,
                    pin_memory=args.pin_memory
                )
            else:
                train_loader = DataLoader(
                    train,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    shuffle=True,
                    pin_memory=args.pin_memory
                )

            eval_loader = DataLoader(
                train,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory
            )

        return train_loader, eval_loader, data_shape, num_classes
