import torch, time, argparse
from torch import nn, optim
from models import get_backbone
from dataset import get_cifar10_loaders
from utils import train_one_epoch, validate, save_checkpoint

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_backbone(args.model, pretrained=args.pretrained).to(device)

    if args.mode == 'finetune':
        # freeze backbone conv layers except last layer optionally
        for name, p in model.named_parameters():
            if 'fc' not in name and 'classifier' not in name:
                p.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    else:
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    train_loader, unlabeled_loader, val_loader = get_cifar10_loaders(labeled_fraction=args.labeled_fraction, batch_size=args.batch_size)

    for epoch in range(args.epochs):
        t0 = time.time()
        train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, args)
        val_acc = validate(model, val_loader, criterion, device)
        print(f'Epoch {epoch} val_acc {val_acc:.4f} time {(time.time()-t0):.1f}s')
        save_checkpoint(model, optimizer, epoch, 'checkpoints/')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['baseline','finetune','pseudo'], default='baseline')
    parser.add_argument('--model', default='resnet18')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--optimizer', choices=['adamw','sgd'], default='sgd')
    parser.add_argument('--labeled_fraction', type=float, default=1.0)
    args = parser.parse_args()
    main(args)
