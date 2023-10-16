import argparse
import io
import os
import random
import shutil
import time
from contextlib import contextmanager
from datetime import timedelta
import warnings
import numpy
from enum import Enum

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Elastic ImageNet Training')
parser.add_argument('data', metavar='DIR', nargs='?', default='imagenet',
                    help='path to dataset (default: imagenet)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 32), per worker (GPU)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--dummy', action='store_true', help="use fake data to benchmark")
parser.add_argument("--checkpoint-file", default="/tmp/checkpoint.pth.tar", type=str, help="checkpoint file path, to load and save to")

best_acc1 = 0


def main():
    args = parser.parse_args()
    args.gpu = int(os.environ["LOCAL_RANK"])
    args.world_size = int(os.environ['WORLD_SIZE'])
    args.rank = int(os.environ['RANK'])
    torch.cuda.set_device(args.gpu)
    print(f"=> Set cuda device = {args.gpu}")
    print(f"Rank: {args.rank} WorldSize: {args.world_size}")

    if args.seed is not None:
        os.environ["PYTHONHASHSEED"] = str(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        numpy.random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    main_worker(args.gpu, args)


def main_worker(gpu, args):
    args.gpu = gpu
    dist.init_process_group(
        backend=args.dist_backend, init_method="env://", timeout=timedelta(seconds=10)
    )

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    # Support distributed launch via torchrun
    model.cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    
    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    # Data loading code
    if args.dummy:
        print("=> Dummy data is used!")
        train_dataset = datasets.FakeData(1281167, (3, 224, 224), 1000, transforms.ToTensor())
        val_dataset = datasets.FakeData(50000, (3, 224, 224), 1000, transforms.ToTensor())
    else:
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        val_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)

    g = torch.Generator()
    g.manual_seed(args.seed)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        os.environ['PYTHONHASHSEED'] = str(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        torch.cuda.manual_seed(worker_seed)
        torch.cuda.manual_seed_all(worker_seed)
        numpy.random.seed(worker_seed)
        torch.backends.cudnn.deterministic = True

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, worker_init_fn=seed_worker, generator=g)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    # Resume from checkpoint if available
    state = load_checkpoint(
        args.checkpoint_file, args.gpu, args.arch, model, optimizer, scheduler
    )

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    start_epoch = state.epoch + 1
    print(f"=> start_epoch: {start_epoch}, best_acc1: {state.best_acc1}")

    for epoch in range(start_epoch):
        # Create the randomization of the sampler
        for _ in train_loader:
            break
    
    if os.path.isfile(args.checkpoint_file):
        state.load_rng_state(args.checkpoint_file)

    for epoch in range(start_epoch, args.epochs):
        state.epoch = epoch
        train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)
        
        scheduler.step()
        
        # remember best acc@1 and save checkpoint
        is_best = acc1 > state.best_acc1
        state.best_acc1 = max(acc1, state.best_acc1)

        if gpu == 0:
            save_checkpoint(state, is_best, args.checkpoint_file)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
        images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i + 1)


def validate(val_loader, model, criterion, args):

    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i
                if args.gpu is not None and torch.cuda.is_available():
                    images = images.cuda(args.gpu, non_blocking=True)
                    target = target.cuda(args.gpu, non_blocking=True)

                # compute output
                output = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i + 1)

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader) + (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)

    top1.all_reduce()
    top5.all_reduce()

    if (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
        aux_val_dataset = Subset(val_loader.dataset,
                                 range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        run_validate(aux_val_loader, len(val_loader))

    progress.display_summary()

    return top1.avg


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class State:
    """
    Container for objects that we want to checkpoint. Represents the
    current "state" of the worker. This object is mutable.
    """

    def __init__(self, arch, model, optimizer, scheduler):
        self.epoch = -1
        self.best_acc1 = 0
        self.arch = arch
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = scheduler
        self.snapshot = None

    def capture_snapshot(self):
        """
        Essentially a ``serialize()`` function, returns the state as an
        object compatible with ``torch.save()``. The following should work
        ::

        snapshot = state_0.capture_snapshot()
        state_1.apply_snapshot(snapshot)
        assert state_0 == state_1
        """
        return {
            "epoch": self.epoch,
            "best_acc1": self.best_acc1,
            "arch": self.arch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.lr_scheduler.state_dict(),
            "rng_state": {
                "python": random.getstate(),
                "numpy": numpy.random.get_state(),
                "cpu": torch.random.get_rng_state(),
                "cuda": torch.cuda.random.get_rng_state()
            }
        }

    def apply_snapshot(self, obj, device_id):
        """
        The complimentary function of ``capture_snapshot()``. Applies the
        snapshot object that was returned by ``capture_snapshot()``.
        This function mutates this state object.
        """

        self.epoch = obj["epoch"]
        self.best_acc1 = obj["best_acc1"]
        self.state_dict = obj["state_dict"]
        self.model.load_state_dict(obj["state_dict"])
        self.optimizer.load_state_dict(obj["optimizer"])
        self.lr_scheduler.load_state_dict(obj["scheduler"])

    def save(self, f):
        torch.save(self.capture_snapshot(), f)

    def load(self, f, device_id):
        # Map model to be loaded to specified single gpu.
        snapshot = torch.load(f, map_location=f"cuda:{device_id}")
        self.snapshot = snapshot
        self.apply_snapshot(snapshot, device_id)

    def load_rng_state(self, f):
        # Rng state needs to be loaded onto the cpu, so we cant reuse the snapshot from before
        snapshot = torch.load(f)
        random.setstate(snapshot["rng_state"]["python"])
        numpy.random.set_state(snapshot["rng_state"]["numpy"])
        torch.random.set_rng_state(snapshot["rng_state"]["cpu"])
        torch.cuda.random.set_rng_state(snapshot["rng_state"]["cuda"])
        

def load_checkpoint(
    checkpoint_file: str,
    device_id: int,
    arch: str,
    model: torch.nn.parallel.DistributedDataParallel,
    optimizer,  # SGD
    scheduler,
) -> State:
    """
    Loads a local checkpoint (if any). Otherwise, checks to see if any of
    the neighbors have a non-zero state. If so, restore the state
    from the rank that has the most up-to-date checkpoint.

    .. note:: when your job has access to a globally visible persistent storage
              (e.g. nfs mount, S3) you can simply have all workers load
              from the most recent checkpoint from such storage. Since this
              example is expected to run on vanilla hosts (with no shared
              storage) the checkpoints are written to local disk, hence
              we have the extra logic to broadcast the checkpoint from a
              surviving node.
    """

    state = State(arch, model, optimizer, scheduler)

    if os.path.isfile(checkpoint_file):
        print(f"=> loading checkpoint file: {checkpoint_file}")
        state.load(checkpoint_file, device_id)
        print(f"=> loaded checkpoint file: {checkpoint_file}")
        print(f"=> epoch: {state.epoch}")
        return state

    # logic below is unnecessary when the checkpoint is visible on all nodes!
    # create a temporary cpu pg to broadcast most up-to-date checkpoint
    with tmp_process_group(backend="gloo") as pg:
        rank = dist.get_rank(group=pg)

        # get rank that has the largest state.epoch
        epochs = torch.zeros(dist.get_world_size(), dtype=torch.int32)
        epochs[rank] = state.epoch
        dist.all_reduce(epochs, op=dist.ReduceOp.SUM, group=pg)
        t_max_epoch, t_max_rank = torch.max(epochs, dim=0)
        max_epoch = t_max_epoch.item()
        max_rank = t_max_rank.item()

        # max_epoch == -1 means no one has checkpointed return base state
        if max_epoch == -1:
            print(f"=> no workers have checkpoints, starting from epoch 0")
            return state

        # broadcast the state from max_rank (which has the most up-to-date state)
        # pickle the snapshot, convert it into a byte-blob tensor
        # then broadcast it, unpickle it and apply the snapshot
        print(f"=> using checkpoint from rank: {max_rank}, max_epoch: {max_epoch}")

        with io.BytesIO() as f:
            torch.save(state.capture_snapshot(), f)
            raw_blob = numpy.frombuffer(f.getvalue(), dtype=numpy.uint8)

        blob_len = torch.tensor(len(raw_blob))
        dist.broadcast(blob_len, src=max_rank, group=pg)
        print(f"=> checkpoint broadcast size is: {blob_len}")

        if rank != max_rank:
            # pyre-fixme[6]: For 1st param expected `Union[List[int], Size,
            #  typing.Tuple[int, ...]]` but got `Union[bool, float, int]`.
            blob = torch.zeros(blob_len.item(), dtype=torch.uint8)
        else:
            blob = torch.as_tensor(raw_blob, dtype=torch.uint8)

        dist.broadcast(blob, src=max_rank, group=pg)
        print(f"=> done broadcasting checkpoint")

        if rank != max_rank:
            with io.BytesIO(blob.numpy()) as f:
                snapshot = torch.load(f)
            state.apply_snapshot(snapshot, device_id)

        # wait till everyone has loaded the checkpoint
        dist.barrier(group=pg)

    print(f"=> done restoring from previous checkpoint")
    return state


@contextmanager
def tmp_process_group(backend):
    cpu_pg = dist.new_group(backend=backend)
    try:
        yield cpu_pg
    finally:
        dist.destroy_process_group(cpu_pg)


def save_checkpoint(state: State, is_best: bool, filename: str):
    checkpoint_dir = os.path.dirname(filename)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # save to tmp, then commit by moving the file in case the job
    # gets interrupted while writing the checkpoint
    tmp_filename = filename + ".tmp"
    torch.save(state.capture_snapshot(), tmp_filename)
    os.rename(tmp_filename, filename)
    print(f"=> saved checkpoint for epoch {state.epoch} at {filename}")
    if is_best:
        best = os.path.join(checkpoint_dir, "model_best.pth.tar")
        print(f"=> best model found at epoch {state.epoch} saving to {best}")
        shutil.copyfile(filename, best)


if __name__ == '__main__':
    main()
