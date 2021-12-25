from __future__ import print_function, division
import argparse
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from opt.dataloader_fusion import face_Dataset, my_transforms
from arch.xcep_fusion import xception
from tqdm import tqdm
from utils.metrics_intra import get_metrics
import random
import pandas as pd

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device: ', device)
criterion = torch.nn.BCEWithLogitsLoss()

def parse_args():
    parser = argparse.ArgumentParser(description='cq')
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--warm_start_epoch', default=0, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--batch_size_train', default=32, type=int)
    parser.add_argument('--batch_size_val', default=64, type=int)
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--save_model', default=3000, type=int)
    parser.add_argument('--disp_step', default=500, type=int)
    parser.add_argument('--weight_decay', default=0.00001, type=float)
    parser.add_argument('--loss_weight1', default=1.0, type=float)
    parser.add_argument('--loss_weight2', default=0.1, type=float)
    parser.add_argument('--save_root', default='./Training_results/Xception_fusion/', type=str)
    parser.add_argument('--train_csv', default='./misc_intra/Train/train.csv', type=str)
    parser.add_argument('--val_csv', default='./misc_intra/Val100/val.csv', type=str)
    parser.add_argument('--root_path', default='', type=str)
    parser.add_argument('--model_name', default="xcep_fusion/", type=str)
    return parser.parse_args()

def fix_seed(seed):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def validation(model, dataloader, num_imgs, thre):
    print('Validating...')
    model.eval()
    batch_val_losses = []
    SCORE = np.zeros(num_imgs)
    LABEL = np.zeros(num_imgs)
    for num, data in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            faces = data['faces'].to(device)
            label_cls = data['labels'].to(device)
            x,_,_,_,_,_,_ = model(faces)
            val_loss = criterion(torch.squeeze(x), label_cls)
            pred_score = torch.squeeze(torch.sigmoid(x), 1)
            batch_val_losses.append(val_loss.item())
            SCORE[num * args.batch_size_val:(num + 1) * args.batch_size_val] = pred_score.cpu().numpy()
            LABEL[num * args.batch_size_val:(num + 1) * args.batch_size_val] = label_cls.cpu().numpy()
    avg_val_loss = round(sum(batch_val_losses) / (len(batch_val_losses)), 5)
    pred = SCORE[:, np.newaxis]
    y_true = LABEL[:, np.newaxis]
    AUC, ACC = get_metrics(pred, y_true, thre)
    return avg_val_loss, AUC, ACC


def train(args, model):
    avg_train_loss_list = np.array([])
    training_dataset = face_Dataset(csv_file=args.train_csv, transform=my_transforms(299, RandomHorizontalFlip=True))
    training_dataloader = DataLoader(training_dataset, batch_size=args.batch_size_train, shuffle=True, num_workers=args.num_workers)

    val_dataset = face_Dataset(csv_file=args.val_csv, transform=my_transforms(299, RandomHorizontalFlip=False))
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size_val, shuffle=True, num_workers=args.num_workers)

    Num_val_imgs = len(pd.read_csv(args.val_csv,header=None))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # result folder
    res_folder_name = args.save_root + args.model_name
    if not os.path.exists(res_folder_name):
        os.makedirs(res_folder_name)
        os.mkdir(res_folder_name + '/ckpt/')
    else:
        print("WARNING: RESULT PATH ALREADY EXISTED -> " + res_folder_name)
    print('find models here: ', res_folder_name)
    writer = SummaryWriter(res_folder_name)
    f1 = open(res_folder_name + "/training_log.csv", 'a+')
    Best_AUC = 0
    # training
    steps_per_epoch = len(training_dataloader)
    for epoch in range(args.warm_start_epoch, args.epochs):
        batch_train_losses = []
        step_loss = np.zeros(steps_per_epoch, dtype=np.float)
        for step, data in enumerate(tqdm(training_dataloader)):
            model.train()
            optimizer.zero_grad()
            frames = data['faces'].to(device)
            masks1 = data['masks1'].to(device)
            masks2 = data['masks2'].to(device)
            masks3 = data['masks3'].to(device)
            noises1 = data['noises1'].to(device)
            noises2 = data['noises2'].to(device)
            noises3 = data['noises3'].to(device)
            labels = data['labels'].to(device)
            pred, msk1, msk2, msk3, nis1, nis2, nis3 = model(frames)
            predicted_label = torch.squeeze(pred, 1)
            loss_cls = criterion(predicted_label, labels)
            loss_seg = criterion(msk1, masks1) + criterion(msk2, masks2) + criterion(msk3, masks3)
            loss_nis = torch.mean(torch.abs(noises1-nis1)) + torch.mean(torch.abs(noises2-nis2)) + torch.mean(torch.abs(noises3-nis3))

            loss = loss_cls + args.loss_weight1*loss_seg + args.loss_weight2*loss_nis
            step_loss[step] = loss
            batch_train_losses.append(loss.item())
            loss.backward()
            optimizer.step()
            Global_step = epoch * steps_per_epoch + (step + 1)

            if Global_step % args.disp_step == 0:
                avg_loss = np.mean(step_loss[(step + 1) - args.disp_step: (step + 1)])
                now_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                step_log_msg = '[%s] Epoch: %d/%d | Global_step: %d |average loss: %f' % (
                now_time, epoch + 1, args.epochs, Global_step, avg_loss)
                writer.add_scalar('Loss/train', avg_loss, Global_step)
                print('\n', step_log_msg)

            if Global_step % args.save_model == 0:
                now_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                avg_train_loss = np.mean(step_loss[(step + 1) - args.disp_step: (step + 1)])
                avg_train_loss_list = np.append(avg_train_loss_list, avg_train_loss)
                log_msg = '[%s] Epoch: %d/%d | 1/10 average epoch loss: %f' % (now_time, epoch + 1, args.epochs, avg_train_loss)
                print('\n', log_msg)
                f1.write(log_msg)
                f1.write('\n')

                # validation
                val_loss, AUC, ACC = validation(model, val_dataloader, Num_val_imgs, 0.5)
                val_msg = '[%s] Epoch: %d/%d | Global_step: %d | average validation loss: %f | ACC: %f| AUC: %f' % (now_time, epoch + 1, args.epochs, Global_step, val_loss, ACC, AUC)
                print('\n', val_msg)
                f1.write(val_msg)
                f1.write('\n')

                # save model
                if AUC > Best_AUC:
                    Best_AUC =  AUC
                    torch.save(model.state_dict(), res_folder_name + '/ckpt/' + 'epoch-%d_step-%d.pth' % (epoch + args.warm_start_epoch, Global_step))
                    np.save(res_folder_name + '/avg_train_loss_list.np', avg_train_loss_list)
                    cur_learning_rate = [param_group['lr'] for param_group in optimizer.param_groups]
                    print('Saved model. lr %f' % cur_learning_rate[0])
                    f1.write('Saved model. lr %f' % cur_learning_rate[0])
                    f1.write('\n')
    f1.close()


def main(args):
    model = xception(pretrained=True)
    model = model.to(device)
    model = torch.nn.DataParallel(model, device_ids=[0, 1])
    print(model)
    print("number of model parameters:", sum([np.prod(p.size()) for p in model.parameters()]))
    train(args, model)

if __name__ == '__main__':
    args = parse_args()
    if args.random_seed is not None:
        fix_seed(args.random_seed)
    print(args)
    main(args)
