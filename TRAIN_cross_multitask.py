from __future__ import print_function, division
import argparse
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from opt.dataloader_multitask_cross import face_Dataset, my_transforms
from arch.multitask import Encoder, Decoder, ActivationLoss, ReconstructionLoss, SegmentationLoss
from tqdm import tqdm
import pandas as pd
import random
from sklearn import metrics
from utils.metrics_cross import get_metrics

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device: ', device)
criterion = torch.nn.BCEWithLogitsLoss()


def fix_seed(seed):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description='cq')
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--lr_decay_rate', default=1.0, type=float)
    parser.add_argument('--weight', default=1.0, type=float)
    parser.add_argument('--lr_decay_step', default=30000, type=str)
    parser.add_argument('--warm_start_epoch', default=0, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--face_size', default=256, type=int)
    parser.add_argument('--mask_size', default=256, type=int)
    parser.add_argument('--batch_size_train', default=32, type=int)
    parser.add_argument('--batch_size_val', default=64, type=int)
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--save_model', default=3000, type=int)
    parser.add_argument('--disp_step', default=500, type=int)
    parser.add_argument('--weight_decay', default=0.00001, type=float)
    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--save_root', default='./Training_results/Cross_df/', type=str)
    parser.add_argument('--root_path', default='', type=str)
    parser.add_argument('--model_name', default="Multitask/", type=str)
    parser.add_argument('--temp_name', default="training_multitask_cross", type=str)
    return parser.parse_args()


def val(ENCODER, DECODER, dataloader, num_imgs):
    print('Validating...')
    ENCODER.eval()
    DECODER.eval()
    tol_label = np.array([], dtype=np.float)
    tol_pred = np.array([], dtype=np.float)
    tol_pred_prob = np.array([], dtype=np.float)

    for num, data in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            faces = data['faces'].to(device)
            label_cls = data['labels'].to(device)

            latent = ENCODER(faces).reshape(-1, 2, 64, 16, 16)
            zero_abs = torch.abs(latent[:, 0]).view(latent.shape[0], -1)
            zero = zero_abs.mean(dim=1)

            one_abs = torch.abs(latent[:, 1]).view(latent.shape[0], -1)
            one = one_abs.mean(dim=1)

            y = torch.eye(2)
            y = y.to(device)

            y = y.index_select(dim=0, index=label_cls.data.long())
            latent = (latent * y[:, :, None, None, None]).reshape(-1, 128, 16, 16)
            seg, rect = DECODER(latent)
            output_pred = np.zeros((faces.shape[0]), dtype=np.float)
            for i in range(faces.shape[0]):
                if one[i] >= zero[i]:
                    output_pred[i] = 1.0
                else:
                    output_pred[i] = 0.0

            tol_label = np.concatenate((tol_label, label_cls.cpu().numpy()))
            tol_pred = np.concatenate((tol_pred, output_pred))

            pred_prob = torch.softmax(torch.cat((zero.reshape(zero.shape[0], 1), one.reshape(one.shape[0], 1)), dim=1),
                                      dim=1)
            tol_pred_prob = np.concatenate((tol_pred_prob, pred_prob[:, 1].data.cpu().numpy()))

    acc_test = metrics.accuracy_score(tol_label, tol_pred)
    val_loss = 0

    return val_loss, tol_pred_prob, tol_label, acc_test


def combine_csv(file_array, result_name):
    combine_cmd = "cat "
    for i in range(len(file_array)):
        combine_cmd += (file_array[i] + " ")
    combine_cmd += " > " + result_name
    os.system(combine_cmd)


def train(args, ENCODER, DECODER):
    ENCODER.to(device)
    DECODER.to(device)
    act_loss_fn = ActivationLoss()
    rect_loss_fn = ReconstructionLoss()
    seg_loss_fn = SegmentationLoss()

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

    tem_train_file_name = 'training_xcep_cross_train'
    tem_val_file_name = 'training_xcep_cross_val'
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

    optimizer_encoder = torch.optim.Adam(ENCODER.parameters(), lr=args.lr, betas=(0.9, 0.999),
                                         weight_decay=args.weight_decay)
    optimizer_decoder = torch.optim.Adam(DECODER.parameters(), lr=args.lr, betas=(0.9, 0.999),
                                         weight_decay=args.weight_decay)

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
    Best_AUC = 0.0
    for epoch in range(args.warm_start_epoch, args.epochs):

        step_loss = np.zeros(steps_per_epoch, dtype=np.float)
        step_cls_loss = np.zeros(steps_per_epoch, dtype=np.float)
        step_msk_loss = np.zeros(steps_per_epoch, dtype=np.float)
        step_rect_loss = np.zeros(steps_per_epoch, dtype=np.float)

        for step, data in enumerate(tqdm(training_dataloader)):
            ENCODER.train()
            DECODER.train()
            optimizer_encoder.zero_grad()
            optimizer_decoder.zero_grad()
            faces = data['faces'].to(device)
            masks = data['masks'].to(device)
            labels = data['labels'].to(device)

            latent = ENCODER(faces).reshape(-1, 2, 64, 16, 16)

            zero_abs = torch.abs(latent[:, 0]).view(latent.shape[0], -1)
            zero = zero_abs.mean(dim=1)

            one_abs = torch.abs(latent[:, 1]).view(latent.shape[0], -1)
            one = one_abs.mean(dim=1)

            loss_act = act_loss_fn(zero, one, labels)
            loss_act_data = loss_act.item()

            y = torch.eye(2)
            y = y.to(device)

            y = y.index_select(dim=0, index=labels.data.long())
            latent = (latent * y[:, :, None, None, None]).reshape(-1, 128, 16, 16)

            seg, rect = DECODER(latent)

            loss_seg = criterion(seg, masks)
            loss_seg_data = loss_seg.item()

            loss_rect = rect_loss_fn(rect, faces)
            loss_rect_data = loss_rect.item()

            loss_total = loss_act + loss_seg + loss_rect
            loss_total.backward()

            step_loss[step] = loss_total
            step_cls_loss[step] = loss_act
            step_msk_loss[step] = loss_seg
            step_rect_loss[step] = loss_rect

            optimizer_decoder.step()
            optimizer_encoder.step()

            Global_step = epoch * steps_per_epoch + (step + 1)

            if Global_step % args.disp_step == 0:
                avg_loss = np.mean(step_loss[(step + 1) - args.disp_step: (step + 1)])
                avg_cls_loss = np.mean(step_cls_loss[(step + 1) - args.disp_step: (step + 1)])
                avg_msk_loss = np.mean(step_msk_loss[(step + 1) - args.disp_step: (step + 1)])
                avg_rect_loss = np.mean(step_rect_loss[(step + 1) - args.disp_step: (step + 1)])
                now_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                step_log_msg = '[%s] Epoch: %d/%d | Global_step: %d |average loss: %f |average class loss: %f |average mask loss: %f |average rect loss: %f' % (
                    now_time, epoch + 1, args.epochs, Global_step, avg_loss, avg_cls_loss, avg_msk_loss, avg_rect_loss)
                writer.add_scalar('Loss/train', avg_loss, Global_step)
                print('\n', step_log_msg)

            if Global_step % args.save_model == 0 or Global_step % steps_per_epoch == 0:
                now_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                avg_train_loss = np.mean(step_loss[(step + 1) - args.disp_step: (step + 1)])
                avg_train_loss_list = np.append(avg_train_loss_list, avg_train_loss)
                log_msg = '[%s] Epoch: %d/%d | 1/10 average epoch loss: %f' % (
                now_time, epoch + 1, args.epochs, avg_train_loss)
                print('\n', log_msg)
                f1.write(log_msg)
                f1.write('\n')

                # validation
                val_loss, pred, y, ACC = val(ENCODER, DECODER, val_dataloader, Num_val_all)
                threshold = 0.5
                AUC, ACC, FPR, FNR, EER, AP = get_metrics(pred, y, threshold)
                val_msg = '[%s] Epoch: %d/%d | Global_step: %d | average df validation loss: %f | AUC: %f | ACC: %f| FPR: %f| FNR: %f| EER: %f| AP: %f' % (
                    now_time, epoch + 1, args.epochs, Global_step, val_loss, AUC, ACC, FPR, FNR, EER, AP)
                print('\n', val_msg)
                f1.write(val_msg)
                f1.write('\n')

                # save model
                if AUC > Best_AUC and Global_step > 50000:
                    Best_AUC = AUC
                    torch.save(ENCODER.state_dict(), res_folder_name + '/ckpt/' + 'encoder_best.pth')
                    torch.save(DECODER.state_dict(), res_folder_name + '/ckpt/' + 'decoder_best.pth')
                    np.save(res_folder_name + '/avg_train_loss_list.np', avg_train_loss_list)
                    f1.write('Saved model.')
                    f1.write('\n')

    f1.close()


def main(args):
    encoder = Encoder(3)
    decoder = Decoder(3)
    print("number of encoder parameters:", sum([np.prod(p.size()) for p in encoder.parameters()]))
    print("number of decoder parameters:", sum([np.prod(p.size()) for p in decoder.parameters()]))
    train(args, encoder, decoder)


if __name__ == '__main__':
    args = parse_args()
    print(args)
    if args.random_seed is not None:
        fix_seed(args.random_seed)
    main(args)
