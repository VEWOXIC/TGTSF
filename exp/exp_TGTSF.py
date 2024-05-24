from data_provider.data_factory import TGTSF_data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, PatchTST, TGTSF_PatchTST, TTSF
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric
from utils.augmentations import augmentation

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

from thop import profile

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'PatchTST': PatchTST,
            'TGTSF': TGTSF_PatchTST,
            'TTSF': TTSF,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        if self.args.news_pre_embed + self.args.des_pre_embed == 2:
            self.text_encoder=None
        else: # if any one need text_encoder
            from sentence_transformers import SentenceTransformer
            self.text_encoder = SentenceTransformer(self.args.text_encoder).to(self.device)
            print(self.args.text_encoder)
            print('text_encoder loaded')

            
        return model

    def _get_data(self, flag):
        data_set, data_loader = TGTSF_data_provider(self.args, flag, text_encoder=self.text_encoder)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        total_time = 0
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_news, batch_des) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                # batch_x_mark = batch_x_mark.float().to(self.device)
                # batch_y_mark = batch_y_mark.float().to(self.device)
                batch_news = batch_news.float().to(self.device)
                batch_des = batch_des.float().to(self.device)

                if 'TSF' in self.args.model:
                    time_now = time.time()
                    # compatability of PatchTST
                    # batch_news = batch_news.unfold(dimension=1, size=self.args.seq_len, step=self.args.stride)
                    # _, patch_num, patch_size, news_num, __ = batch_news.shape
                    # batch_news = batch_news.reshape(_, patch_num, patch_size*news_num, __)
                    # batch_des = batch_des.unfold(dimension=1, size=self.args.seq_len, step=self.args.stride)
                    # batch_des = batch_des[:, :, -1, :, :] # only take the last day of the patch (can change)
                    ##################################
                    outputs = self.model(batch_x, batch_news, batch_des)

                    total_time += time.time() - time_now

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        print('vali time: ', total_time/(i+1))
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        # x, y, news, des = next(iter(test_loader))
        # x=x.float().to(self.device)
        # y=y.float().to(self.device)
        # news=news.float().to(self.device)
        # des=des.float().to(self.device)
        # time_now = time.time()
        # macs, params = profile(self.model, inputs=(x, news, des,))
        # total_time = time.time() - time_now
        # print('FLOPs: ', macs)
        # print('params: ', params)
        # print('Total time: ', total_time)


        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        if self.args.finetune:
            print('>>>>>>FINETUNE loading model<<<<<<')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting[3:], 'checkpoint.pth')))
            # freeze the parameters of PatchTST
            self.model.encoder.load_state_dict(torch.load(os.path.join('./checkpoints/' + 'investing_DLinear_60_7_DLinear_custom_ftM_sl60_ll0_pl7_dm768_nh16_el3_dl1_df256_fc1_ebtimeF_dtTrue_Exp_0', 'checkpoint.pth'), map_location=self.device, strict=False))
            for name, param in self.model.encoder.named_parameters():
                param.requires_grad = False

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
            
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_news, batch_des) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_news = batch_news.float().to(self.device)
                batch_des = batch_des.float().to(self.device)


                if self.args.in_batch_augmentation:
                    aug = augmentation('batch')
                    methods = {'f_mask':aug.freq_mask, 'f_mix': aug.freq_mix, 'noise':aug.noise,'noise_input':aug.noise_input,'vFlip':aug.vFlip,'hFlip':aug.hFlip,'time_comb':aug.time_combination,'upsample':aug.linear_upsampling}
                    for step in range(self.args.aug_data_size):
                        xy = methods[self.args.aug_method](batch_x, batch_y[:, -self.args.pred_len:, :], rate=self.args.aug_rate, dim=1)
                        batch_x2, batch_y2 = xy[:, :self.args.seq_len, :], xy[:, -self.args.label_len-self.args.pred_len:, :]
                        if 'noise' not in self.args.aug_method:
                            batch_x = torch.cat([batch_x,batch_x2],dim=0)
                            batch_y = torch.cat([batch_y,batch_y2],dim=0)
                            batch_news = torch.cat([batch_news,batch_news],dim=0)
                            batch_des = torch.cat([batch_des,batch_des],dim=0)
                        else:
                            print('noise')
                            batch_x = batch_x2
                            batch_y = batch_y2

                # print(batch_x.shape, batch_y.shape, batch_news.shape, batch_des.shape)

                if 'TSF' in self.args.model:
                    outputs = self.model(batch_x, batch_news, batch_des)
                    
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                    
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        print(f"Test for {setting}")
        test_data, test_loader = self._get_data(flag='test')
        
        x, y, news, des = next(iter(test_loader))
        x=x.float().to(self.device)
        y=y.float().to(self.device)
        news=news.float().to(self.device)
        des=des.float().to(self.device)
        time_now = time.time()
        macs, params = profile(self.model, inputs=(x, news, des,))
        total_time = time.time() - time_now
        print('FLOPs: ', macs)
        print('params: ', params)
        print('Total time: ', total_time)

        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_news, batch_des) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_news = batch_news.float().to(self.device)
                batch_des = batch_des.float().to(self.device)

                # # decoder input
                # dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                # dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                    # if 'Linear' in self.args.model or 'TST' in self.args.model:
                    #         outputs = self.model(batch_x)
                    # else:
                    #     if self.args.output_attention:
                    #         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    #     else:
                    #         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                if 'TSF' or 'TGTSF' in self.args.model:
                        outputs = self.model(batch_x, batch_news, batch_des)
                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)


                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                batch_x = batch_x.detach().cpu().numpy()

                b, l, c = outputs.shape
                outputs = outputs.reshape(-1, c)
                batch_y = batch_y.reshape(-1, c)
                outputs = test_data.scaler.inverse_transform(outputs)
                batch_y = test_data.scaler.inverse_transform(batch_y)
                outputs = outputs.reshape(b, l, c)
                batch_y = batch_y.reshape(b, l, c)

                b, l, c = batch_x.shape
                batch_x = batch_x.reshape(-1, c)
                batch_x = test_data.scaler.inverse_transform(batch_x)
                batch_x = batch_x.reshape(b, l, c)

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x)
                if i % 20 == 0:
                    input = batch_x
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, rse:{}, rmse:{}, mape:{}, mspe:{}, corr:{}'.format(mse, mae, rse, rmse, mape, mspe, corr))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
