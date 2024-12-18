import torch
from tqdm.autonotebook import tqdm
from .utils import AverageMeter, get_train_transforms, get_valid_transforms, collate_fn
import sys
import argparse
import pandas as pd
from .dataset import CropDataset
from .model import DETRModel
from torch.utils.data import DataLoader
from detr.models.matcher import HungarianMatcher
from detr.models.detr import SetCriterion
from .utils import convert_to_detr 
from sklearn.model_selection import StratifiedKFold

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--backbone", default="resnet50", type=str,  choices=['resnet50', 'resnet101', 'resnet152'])
parser.add_argument("--img_size", default=1024, type=int)
parser.add_argument("--batch_size", default=8, type=int)
parser.add_argument("--null_class_coef", default=0.5, type=float)
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--train_loc", default='/content/Train.csv', type=str)
parser.add_argument("--folds", default=5, type=int)
parser.add_argument("--init_lr", default=2e-5, type=float)
parser.add_argument("--seed", default=42, type=int)
args = parser.parse_args()
print(args)


def eval_fn(data_loader, model,criterion, device):
    model.eval()
    criterion.eval()
    summary_loss = AverageMeter()

    with torch.no_grad():

        tk0 = tqdm(data_loader, total=len(data_loader))
        for step, (images, targets, image_ids) in enumerate(tk0):

            images = list(image.to(device) for image in images)
            targets = [{k: torch.tensor(v).to(device) for k, v in t.items()} for t in targets]

            output = model(images)

            loss_dict = criterion(output, targets)
            weight_dict = criterion.weight_dict

            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            summary_loss.update(losses.item(),args.batch_size)
            tk0.set_postfix(loss=summary_loss.avg)

    return summary_loss


def train_fn(data_loader,model,criterion,optimizer,device,scheduler,epoch):
    model.train()
    criterion.train()

    summary_loss = AverageMeter()

    tk0 = tqdm(data_loader, total=len(data_loader))

    for step, (images, targets, image_ids) in enumerate(tk0):

        images = list(image.to(device) for image in images)
        targets = [{k: torch.tensor(v).to(device) for k, v in t.items()} for t in targets]


        output = model(images)

        loss_dict = criterion(output, targets)
        weight_dict = criterion.weight_dict

        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        optimizer.zero_grad()

        losses.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        summary_loss.update(losses.item(),args.batch_size)
        tk0.set_postfix(loss=summary_loss.avg)

    return summary_loss



def run(fold, train, df_folds):

    df_train = df_folds[df_folds['fold'] != fold]
    df_valid = df_folds[df_folds['fold'] == fold]

    class_map = [{x: y} for x, y in zip(range(train['class'].nunique), train['class'].unique().tolist())]

    train_dataset = CropDataset(
    image_ids=df_train.index.values,
    dataframe=train,
    class_map=class_map,
    transforms=get_train_transforms()
    )

    valid_dataset = CropDataset(
    image_ids=df_valid.index.values,
    dataframe=train,
    class_map=class_map,
    transforms=get_valid_transforms()
    )

    train_data_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn
    )

    valid_data_loader = DataLoader(
    valid_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn
    )

    num_classes = train['class'].nunique() + 1
    num_queries = 100
    matcher = HungarianMatcher()
    weight_dict = weight_dict = {'loss_ce': 1, 'loss_bbox': 1 , 'loss_giou': 1}
    losses = ['labels', 'boxes', 'cardinality']

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = DETRModel(num_classes=num_classes,num_queries=num_queries)
    model = model.to(device)
    criterion = SetCriterion(num_classes-1, matcher, weight_dict, eos_coef = args.nul_class_coef, losses=losses)
    criterion = criterion.to(device)


    optimizer = torch.optim.AdamW(model.parameters(), lr=args.init_lr)

    best_loss = 10**5
    for epoch in range(args.epochs):
        train_loss = train_fn(train_data_loader, model,criterion, optimizer,device,scheduler=None,epoch=epoch)
        valid_loss = eval_fn(valid_data_loader, model,criterion, device)

        print('|EPOCH {}| TRAIN_LOSS {}| VALID_LOSS {}|'.format(epoch+1,train_loss.avg,valid_loss.avg))

        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            print('Best model found for Fold {} in Epoch {}........Saving Model'.format(fold,epoch+1))
            torch.save(model.state_dict(), f'detr_best_{fold}.pth')



if __name__ == '__main__':
    DATA_DIR = args.train_loc

    train = pd.read_csv(DATA_DIR)
    train_cleaned = convert_to_detr(train)

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    df_folds = train_cleaned[['Image_ID']].copy()
    df_folds.loc[:, 'bbox_count'] = 1
    df_folds = df_folds.groupby('Image_ID').count()
    df_folds.loc[:, 'stratify_group'] = df_folds['bbox_count'].apply(lambda x: f'_{x // 15}').values.astype(str)
    df_folds.loc[:, 'fold'] = 0

    for fold_number, (train_index, val_index) in enumerate(skf.split(X=df_folds.index, y=df_folds['stratify_group'])):
        df_folds.loc[df_folds.iloc[val_index].index, 'fold'] = fold_number
    
    run(fold=0, train=train_cleaned, df_folds=df_folds)
