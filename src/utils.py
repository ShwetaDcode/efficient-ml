import torch, os
from tqdm import tqdm
import torch.nn.functional as F

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, args):
    model.train()
    pbar = tqdm(loader)
    for xb, yb in pbar:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        loss = criterion(out, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_description(f'Epoch {epoch} loss {loss.item():.4f}')

def validate(model, loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            pred = out.argmax(dim=1)
            correct += (pred==yb).sum().item()
            total += yb.size(0)
    return correct/total

def save_checkpoint(model, optimizer, epoch, folder='checkpoints'):
    os.makedirs(folder, exist_ok=True)
    torch.save({'epoch':epoch, 'model_state':model.state_dict(), 'opt':optimizer.state_dict()}, os.path.join(folder, f'ckpt_ep{epoch}.pt'))
