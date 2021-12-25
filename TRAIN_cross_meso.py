from __future__ import print_function, division
import argparse
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from opt.dataloader import face_Dataset, my_transforms
from arch.Meso import Meso4
from tqdm import tqdm
import random
import pandas as pd
from utils.metrics_cross import get_metrics

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
    parser.add_argument('--save_root', default='./Training_results/Cross_df/', type=str)
    parser.add_argument('--root_path', default='', type=str)
    parser.add_argument('--model_name', default="Meso4/", type=str)
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


def val(model, dataloader, num_imgs):
    print('Validating...')
    model.eval()
    batch_val_losses = []
    SCORE = np.zeros(num_imgs)
    LABEL = np.zeros(num_imgs)
    for num, data in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            faces = data['faces'].to(device)
            label_cls = data['labels'].to(device)
            x = model(faces)
            pred_score = torch.squeeze(torch.sigmoid(x), 1)
            val_loss = criterion(torch.squeeze(x, 1), label_cls)
            batch_val_losses.append(val_loss.item())
            SCORE[num * args.batch_size_val:(num + 1) * args.batch_size_val] = pred_score.cpu().numpy()
            LABEL[num * args.batch_size_val:(num + 1) * args.batch_size_val] = label_cls.cpu().numpy()

    avg_val_loss = round(sum(batch_val_losses) / (len(batch_val_losses)), 5)
    pred = SCORE[:, np.newaxis]
    y_true = LABEL[:, np.newaxis]

    return avg_val_loss, pred, y_true


def combine_csv(file_array, result_name):
    combine_cmd = "cat "
    for i in range(len(file_array)):
        combine_cmd += (file_array[i] + " ")
    combine_cmd += " > " + result_name
    os.system(combine_cmd)


def train(args, model):
    avg_train_loss_list = np.array([])

    train_csv_file_df_c23 = './misc_cross/Train/c23_df_train.csv'
    train_csv_file_or_c23 = './misc_cross/Train/c23_or_train.csv'
    train_csv_file_df_c40 = './misc_cross/Train/c40_df_train.csv'
    train_csv_file_or_c40 = './misc_cross/Train/c40_or_train.csv'

    val_csv_file_df_c23 = './misc_cross/Val100/c23_df_val.csv'
    val_csv_file_or_c23 = './misc_cross/Val100/c23_or_val.csv'
    val_csv_file_df_c40 = './misc_cross/Val100/c40_df_val.csv'
    val_csv_file_or_c40 = './misc_cross/Val100/c40_or_val.csv'

    training_csv_files = [train_csv_file_df_c23, train_csv_file_or_c23, train_csv_file_df_c40, train_csv_file_or_c40]
    val_csv_files = [val_csv_file_df_c23, val_csv_file_or_c23, val_csv_file_df_c40, val_csv_file_or_c40]

    tem_train_file_name = 'training_meso_cross_train'
    tem_val_file_name = 'training_meso_cross_val'
    combine_csv(training_csv_files, tem_train_file_name)
    combine_csv(val_csv_files, tem_val_file_name)
    tem_train_file_path = args.root_path + tem_train_file_name
    tem_val_file_path = args.root_path + tem_val_file_name

    training_dataset = face_Dataset(csv_file=tem_train_file_path,
                                    transform=my_transforms(256, RandomHorizontalFlip=False))
    training_dataloader = DataLoader(training_dataset, batch_size=args.batch_size_train, shuffle=True,
                                     num_workers=args.num_workers)

    val_dataset = face_Dataset(csv_file=tem_val_file_path, transform=my_transforms(256, RandomHorizontalFlip=False))
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size_val, shuffle=True, num_workers=args.num_workers)

    Num_val_imgs_1 = len(pd.read_csv(val_csv_file_df_c23, header=None))
    Num_val_imgs_2 = len(pd.read_csv(val_csv_file_or_c23, header=None))
    Num_val_imgs_3 = len(pd.read_csv(val_csv_file_df_c40, header=None))
    Num_val_imgs_4 = len(pd.read_csv(val_csv_file_or_c40, header=None))
    Num_val_all = Num_val_imgs_1 + Num_val_imgs_2 + Num_val_imgs_3 + Num_val_imgs_4
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

    # training
    steps_per_epoch = len(training_dataloader)
    Learning_Rate = args.lr
    Best_AUC = 0.0
    for epoch in range(args.warm_start_epoch, args.epochs):
        batch_train_losses = []
        step_loss = np.zeros(steps_per_epoch, dtype=np.float)
        for step, data in enumerate(tqdm(training_dataloader)):
            model.train()
            optimizer.zero_grad()
            frames = data['faces'].to(device)
            labels = data['labels'].to(device)
            predicted_label = torch.squeeze(model(frames), 1)
            loss = criterion(predicted_label, labels)
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
                log_msg = '[%s] Epoch: %d/%d | 1/10 average epoch loss: %f' % (
                    now_time, epoch + 1, args.epochs, avg_train_loss)
                print('\n', log_msg)
                f1.write(log_msg)
                f1.write('\n')

                # validation
                val_loss, pred, y = val(model, val_dataloader, Num_val_all)
                threshold = 0.5
                AUC, ACC, FPR, FNR, EER, AP = get_metrics(pred, y, threshold)

                val_msg = '[%s] Epoch: %d/%d | Global_step: %d | average df validation loss: %f | AUC: %f | ACC: %f| FPR: %f| FTR: %f| EER: %f| AP: %f' % (
                    now_time, epoch + 1, args.epochs, Global_step, val_loss, AUC, ACC, FPR, FNR, EER, AP)
                print('\n', val_msg)
                f1.write(val_msg)
                f1.write('\n')

                # save model
                if AUC > Best_AUC and Global_step > 50000:
                    Best_AUC = AUC
                    torch.save(model.state_dict(), res_folder_name + '/ckpt/' + 'best.pth')
                    np.save(res_folder_name + '/avg_train_loss_list.np', avg_train_loss_list)
                    cur_learning_rate = [param_group['lr'] for param_group in optimizer.param_groups]
                    print('Saved model. lr %f' % cur_learning_rate[0])
                    f1.write('Saved model. lr %f' % cur_learning_rate[0])
                    f1.write('\n')
    f1.close()


def main(args):
    model = Meso4()
    model = model.to(device)
    print(model)
    print("number of model parameters:", sum([np.prod(p.size()) for p in model.parameters()]))
    train(args, model)


if __name__ == '__main__':
    args = parse_args()
    if args.random_seed is not None:
        fix_seed(args.random_seed)
    print(args)
    main(args)
