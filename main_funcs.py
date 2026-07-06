import os
import torch
from tqdm import tqdm
import logging
from torch.nn import functional as F
from torch.utils import tensorboard
import numpy as np
from torch.amp import autocast
from torch.cuda.amp import GradScaler
import cv2
from PIL import Image
import gc
from utils import metric, split_sample_path
import pandas as pd


class TrainingModel:

    def __init__(self, fold_id, net, cont_loss, clas_loss, loader_train, loader_val, loader_test, config, optimizer, scheduler=None):

        self.logger = logging.getLogger("MultiModaCL")
        self.cont_loss = cont_loss
        self.clas_loss = clas_loss
        self.model = net
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = GradScaler(enabled=config.cuda)
        self.loader = loader_train
        self.loader_val = loader_val
        self.loader_test = loader_test
        self.fold_id = fold_id
        self.device = torch.device("cuda" if config.cuda else "cpu")
        if config.cuda and not torch.cuda.is_available():
            raise ValueError("No GPU found: set cuda=False parameter.")
        self.config = config
        if hasattr(config, 'lam_cls'):
            self.lam_cls = config.lam_cls
        self.metrics = {}

        if hasattr(config, 'pretrained_path') and config.pretrained_path is not None:
            self.load_model(config.pretrained_path + '_fold_' + str(fold_id) + '.pth')

        self.model.to(self.device)
        self.cont_loss.to(self.device)
        self.clas_loss.to(self.device)

    def pretraining(self):
        print("Pretraining started")
        print(self.cont_loss)
        print(self.optimizer)
        best_val_loss = 1000
        sum_writer = tensorboard.SummaryWriter(os.path.join(self.config.output_dir, str(self.fold_id))) # type: ignore

        for epoch in range(self.config.nb_epochs):
            
            self.adjust_learning_rate(self.optimizer, epoch, self.config.lr, self.config.nb_epochs, self.config.warm_epochs, self.config.power_lr)
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch}, Learning Rate: {current_lr}")
            
            ## Training step
            self.model.train()
            nb_batch = len(self.loader)
            training_loss = 0
            pbar = tqdm(total=nb_batch, desc="Training", mininterval=20)

            for batch in self.loader:
                pbar.update()
                inputs_1 = batch['image_1'].to(self.device)
                inputs_2 = batch['image_2'].to(self.device)
                labels = batch['label'].to(self.device)

                if self.config.use_var:
                    var = batch['var'].to(self.device)
                else:
                    var = None
                self.optimizer.zero_grad()

                with autocast(device_type='cuda' if self.config.cuda else 'cpu'):
                    feat_i, z_i = self.model(inputs_1, var)
                    feat_j, z_j = self.model(inputs_2, var)
                    loss_i, loss_j, logits, target = self.cont_loss(z_i, z_j, labels)
                    batch_loss = (loss_i + loss_j) / 2
                
                self.scaler.scale(batch_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.scheduler is not None:
                    self.scheduler.step()
                training_loss += float(batch_loss.item()) / nb_batch
            pbar.close()

            ## Validation step
            nb_batch = len(self.loader_val)
            pbar = tqdm(total=nb_batch, desc="Validation", mininterval=20)
            val_loss = 0
            with torch.no_grad():
                self.model.eval()
                for batch in self.loader_val:
                    pbar.update()
                    inputs_1 = batch['image_1'].to(self.device)
                    inputs_2 = batch['image_2'].to(self.device)
                    labels = batch['label'].to(self.device)
                    if self.config.use_var:
                        var = batch['var'].to(self.device)
                    else:
                        var = None
                        
                    feat_i, z_i = self.model(inputs_1, var)
                    feat_j, z_j = self.model(inputs_2, var)
                    loss_i, loss_j, logits, target = self.cont_loss(z_i, z_j, labels)
                    batch_loss = (loss_i + loss_j) / 2
                    val_loss += float(batch_loss.item()) / nb_batch
            pbar.close()

            print("\nEpoch [{}/{}] Training loss = {:.4f}\t Validation loss = {:.4f}\t".format(
                epoch, self.config.nb_epochs, training_loss, val_loss), flush=True)
            sum_writer.add_scalar('train/loss', training_loss, epoch)
            sum_writer.add_scalar('valid/loss', val_loss, epoch)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print("Best model!")
                torch.save({
                    "epoch": epoch,
                    "model": self.model.state_dict(),
                    # "optimizer": self.optimizer.state_dict()
                    },
                    os.path.join(self.config.output_dir, "MultiModaCL_Pretrained_Best_fold_{fold}.pth".format(fold=self.fold_id)))


    def fine_tuning(self):
        print("Finetuning started")
        print(self.cont_loss)
        print(self.clas_loss)
        print("Lam_cls: {}".format(self.lam_cls))
        print(self.optimizer)
        best_val_loss, best_bacc, best_f1 = 1000, 0, 0
        sum_writer = tensorboard.SummaryWriter(os.path.join(self.config.output_dir, "fold_" + str(self.fold_id))) # type: ignore
        best_acc, best_es_acc, best_es_auc, best_es_class_attr_scaled_f1 = 0, 0, 0, 0

        for epoch in range(self.config.nb_epochs):
            
            self.adjust_learning_rate(self.optimizer, epoch, self.config.lr, self.config.nb_epochs, self.config.warm_epochs, self.config.power_lr)
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch}, Learning Rate: {current_lr}")

            ## Training step
            self.model.train()
            nb_batch = len(self.loader)
            train_cont_losses, train_clas_losses, training_loss = 0, 0, 0
            pbar = tqdm(total=nb_batch, desc="Training", mininterval=20)
            for batch in self.loader:
                pbar.update()
                inputs_1 = batch['image_1'].to(self.device)
                inputs_2 = batch['image_2'].to(self.device)
                labels = batch['label'].to(self.device)
                sens_attr = batch['sens_attr'] if batch['sens_attr'][0] != -1 else None
                if self.config.use_var:
                    var = batch['var'].to(self.device)
                else:
                    var = None
                self.optimizer.zero_grad()

                with autocast(device_type='cuda' if self.config.cuda else 'cpu'):
                    feat_i, z_i, y_i = self.model(inputs_1, var)
                    feat_j, z_j, y_j = self.model(inputs_2, var)
                    cont_loss_i, cont_loss_j, logits, target = self.cont_loss(z_i, z_j, labels)
                    batch_cont_loss = (cont_loss_i + cont_loss_j) / 2
                    clas_loss_i = self.clas_loss(y_i, labels)
                    clas_loss_j = self.clas_loss(y_j, labels)
                    batch_clas_loss = (clas_loss_i + clas_loss_j) / 2
                    batch_loss = (1 - self.lam_cls) * batch_cont_loss + self.lam_cls * batch_clas_loss
                
                self.scaler.scale(batch_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.scheduler is not None:
                    self.scheduler.step()
                train_cont_losses += float(batch_cont_loss.item()) / nb_batch
                train_clas_losses += float(batch_clas_loss.item()) / nb_batch
                training_loss += float(batch_loss.item()) / nb_batch
            pbar.close()

            ## Validation step
            nb_batch = len(self.loader_val)
            pbar = tqdm(total=nb_batch, desc="Validation", mininterval=20)
            val_cont_loss, val_clas_loss, val_loss, correct = 0, 0, 0, 0
            prob_list, label_list, sens_attr_list = [], [], []

            with torch.no_grad():
                self.model.eval()
                for batch in self.loader_val:
                    pbar.update()
                    inputs_1 = batch['image_1'].to(self.device)
                    inputs_2 = batch['image_2'].to(self.device)
                    labels = batch['label'].to(self.device)
                    sens_attr = batch['sens_attr'] if batch['sens_attr'][0] != -1 else None
                    if self.config.use_var:
                        var = batch['var'].to(self.device)
                    else:
                        var = None
                        
                    feat_i, z_i, y_i = self.model(inputs_1, var)
                    feat_j, z_j, y_j = self.model(inputs_2, var)
                    cont_loss_i, cont_loss_j, logits, target = self.cont_loss(z_i, z_j, labels)
                    batch_cont_loss = (cont_loss_i + cont_loss_j) / 2
                    clas_loss_i = self.clas_loss(y_i, labels)
                    clas_loss_j = self.clas_loss(y_j, labels)
                    batch_clas_loss = (clas_loss_i + clas_loss_j) / 2
                    batch_loss = (1 - self.lam_cls) * batch_cont_loss + self.lam_cls * batch_clas_loss
                    
                    val_cont_loss += float(batch_cont_loss.item()) / nb_batch
                    val_clas_loss += float(batch_clas_loss.item()) / nb_batch
                    val_loss += float(batch_loss.item()) / nb_batch
                    prob_1 = F.softmax(y_i, dim=1)
                    prob_2 = F.softmax(y_j, dim=1)
                    prob_list.append(((prob_1 + prob_2) / 2).cpu().numpy())
                    label_list.append(labels.cpu().numpy())
                    if sens_attr is not None:
                        sens_attr_list.append(sens_attr.cpu().numpy())
                    pred_i = y_i.argmax(dim=1, keepdim=True)
                    pred_j = y_j.argmax(dim=1, keepdim=True)
                    correct += (pred_i.eq(labels.view_as(pred_i)).sum().item() + pred_j.eq(labels.view_as(pred_j)).sum().item()) / 2
            
            accuracy = correct / len(self.loader_val.dataset)
            return_dict = metric(np.concatenate(prob_list, axis=0), np.concatenate(label_list, axis=0), np.concatenate(sens_attr_list, axis=0) if len(sens_attr_list) > 0 else None, equity_alpha=self.config.equity_alpha)
            bacc = return_dict['overall_bacc']
            f1 = return_dict['overall_classification_report']['macro avg']['f1-score']
            if len(sens_attr_list) > 0:
                ES_acc = return_dict['overall_ES_acc']
                ES_auc = return_dict['overall_ES_AUC']
                ES_class_attr_scaled_f1 = return_dict['overall_ES_Class_Attr_Scaled_F1']
            else:
                ES_acc = None
                ES_auc = None
                ES_class_attr_scaled_f1 = None
            pbar.close()
            
            sum_writer.add_scalar('train/cont_loss', train_cont_losses, epoch)
            sum_writer.add_scalar('train/clas_loss', train_clas_losses, epoch)
            sum_writer.add_scalar('train/loss', training_loss, epoch)
            sum_writer.add_scalar('valid/cont_loss', val_cont_loss, epoch)
            sum_writer.add_scalar('valid/clas_loss', val_clas_loss, epoch)
            sum_writer.add_scalar('valid/loss', val_loss, epoch)
            sum_writer.add_scalar('valid/acc', accuracy, epoch)
            sum_writer.add_scalar('valid/bacc', bacc, epoch)
            sum_writer.add_scalar('valid/f1', f1, epoch)
            if len(sens_attr_list) > 0:
                sum_writer.add_scalar('valid/ES_acc', ES_acc, epoch)
                sum_writer.add_scalar('valid/ES_auc', ES_auc, epoch)
                sum_writer.add_scalar('valid/ES_class_attr_scaled_f1', ES_class_attr_scaled_f1, epoch)

            print("\nEpoch [{}/{}] Train: cont loss = {:.4f}, clas loss = {:.4f}, loss = {:.4f}, Val: cont loss {:.4f}, clas loss {:.4f}, loss = {:.4f}, accuracy = {:.4f}, bacc = {:.4f}, f1 = {:.4f}".format(
                epoch, self.config.nb_epochs, train_cont_losses, train_clas_losses,
                training_loss, val_cont_loss, val_clas_loss, val_loss, accuracy, bacc, f1), flush=True)
            if len(sens_attr_list) > 0:
                print("Val: ES_acc = {:.4f}, ES_auc = {:.4f}, ES_class_attr_scaled_f1 = {:.4f}".format(ES_acc, ES_auc, ES_class_attr_scaled_f1), flush=True)
            
            if ES_class_attr_scaled_f1 is not None and self.config.metric == "class_attr_scaled_f1" and ES_class_attr_scaled_f1 > best_es_class_attr_scaled_f1:
                best_es_class_attr_scaled_f1 = ES_class_attr_scaled_f1
                print("Best model!")
                torch.save({
                    "epoch": epoch,
                    "model": self.model.state_dict(),
                    # "optimizer": self.optimizer.state_dict()
                    },
                    os.path.join(self.config.output_dir, "MultiModaCL_Finetuned_Best_fold_{fold}.pth".format(fold=self.fold_id)))
            elif self.config.metric == "f1" and f1 > best_f1:
                best_f1 = f1
                print("Best model!")
                torch.save({
                    "epoch": epoch,
                    "model": self.model.state_dict(),
                    # "optimizer": self.optimizer.state_dict()
                    },
                    os.path.join(self.config.output_dir, "MultiModaCL_Finetuned_Best_fold_{fold}.pth".format(fold=self.fold_id)))
            elif self.config.metric == "bacc" and bacc > best_bacc:
                best_bacc = bacc
                print("Best model!")
                torch.save({
                    "epoch": epoch,
                    "model": self.model.state_dict(),
                    # "optimizer": self.optimizer.state_dict()
                    },
                    os.path.join(self.config.output_dir, "MultiModaCL_Finetuned_Best_fold_{fold}.pth".format(fold=self.fold_id)))
            elif self.config.metric == "acc" and accuracy > best_acc:
                best_acc = accuracy
                print("Best model!")
                torch.save({
                    "epoch": epoch,
                    "model": self.model.state_dict(),
                    # "optimizer": self.optimizer.state_dict()
                    },
                    os.path.join(self.config.output_dir, "MultiModaCL_Finetuned_Best_fold_{fold}.pth".format(fold=self.fold_id)))
    
    
    def test(self):
        
        # if hasattr(self.config, 'pretrained_path') and self.config.pretrained_path is not None:
        self.load_model(os.path.join(self.config.output_dir, "MultiModaCL_Finetuned_Best_fold_{fold}.pth".format(fold=self.fold_id)))
        nb_batch = len(self.loader_test)
        pbar = tqdm(total=nb_batch, desc="Test", mininterval=20)
        predict_dict, truth_dict, count_dict, ensemble_dict, ensemble_prob_dict, sens_attr_dict = {}, {}, {}, {}, {}, {}
        case_correct, img_correct = 0, 0
        name_list, predict_list, label_list, prob_list, sens_attr_list = [], [], [], [], []

        with torch.no_grad():
            self.model.eval()
            for batch in self.loader_test:
                pbar.update()
                inputs = batch['image'].to(self.device, non_blocking=True)
                labels = batch['label'].to(self.device, non_blocking=True)
                name = batch['name'][0]
                sens_attr = batch['sens_attr'] if batch['sens_attr'][0] != -1 else None
                if self.config.use_var:
                    var = batch['var'].to(self.device, non_blocking=True)
                else:
                    var = None

                patient = split_sample_path(name)[0]
                if patient not in predict_dict:
                    predict_dict[patient] = []
                if patient not in truth_dict:
                    truth_dict[patient] = labels.cpu().numpy()[0]
                if patient not in count_dict:
                    count_dict[patient] = 0
                if patient not in sens_attr_dict and sens_attr is not None and sens_attr.cpu().numpy()[0] != -1:
                    sens_attr_dict[patient] = sens_attr.cpu().numpy()[0]
                count_dict[patient] += 1

                feat, z, y = self.model(inputs, var)
                pred = y.argmax(dim=1, keepdim=True)
                prob = F.softmax(y, dim=1)
                img_correct += pred.eq(labels.view_as(pred)).sum().item()
                name_list.append(name)
                if sens_attr is not None and sens_attr.cpu().numpy()[0] != -1:
                    sens_attr_list.append(sens_attr.cpu().numpy()[0])
                predict_list.append(torch.flatten(pred).cpu().numpy()[0])
                label_list.append(torch.flatten(labels).cpu().numpy()[0])
                prob_np = prob.squeeze(0).cpu().numpy()
                prob_list.append(prob_np)

                predict_dict[patient].append(prob_np)

        for patient in predict_dict.keys():
            ensemble_prob = np.mean(np.stack(predict_dict[patient], axis=0), axis=0)
            ensemble = np.argmax(ensemble_prob, axis=0)
            ensemble_prob_dict[patient] = ensemble_prob
            ensemble_dict[patient] = int(ensemble)
            if ensemble == truth_dict[patient]:
                case_correct += 1

        case_acc = case_correct / len(predict_dict)
        img_acc = img_correct / len(self.loader_test.dataset)
        case_result_dict = metric(list(ensemble_prob_dict.values()), list(truth_dict.values()), list(sens_attr_dict.values()) if len(sens_attr_dict) > 0 else None, group_wise=True, equity_alpha=self.config.equity_alpha)
        case_f1 = case_result_dict['overall_classification_report']['macro avg']['f1-score']
        case_auc = case_result_dict['overall_auc']
        case_bacc = case_result_dict['overall_bacc']
        case_con_matrix = case_result_dict['overall_confusion_matrix']
        case_class_report = case_result_dict['overall_classification_report']
        img_result_dict = metric(prob_list, label_list, sens_attr_list if len(sens_attr_list) > 0 else None, group_wise=True, equity_alpha=self.config.equity_alpha) 
        img_f1 = img_result_dict['overall_classification_report']['macro avg']['f1-score']
        img_auc = img_result_dict['overall_auc']
        img_bacc = img_result_dict['overall_bacc']
        img_con_matrix = img_result_dict['overall_confusion_matrix']
        img_class_report = img_result_dict['overall_classification_report']
        pbar.close()

        csv_dict = {
            'case_0_f1': case_class_report['0']['f1-score'], # type: ignore
            'case_1_f1': case_class_report['1']['f1-score'], # type: ignore
            'case_f1': case_f1,
            'case_acc': case_acc,
            'case_bacc': case_bacc,
            'case_auc': case_auc,
            'img_0_f1': img_class_report['0']['f1-score'], # type: ignore
            'img_1_f1': img_class_report['1']['f1-score'], # type: ignore
            'img_f1': img_f1,
            'img_acc': img_acc,
            'img_bacc': img_bacc,
            'img_auc': img_auc,
        }

        if len(sens_attr_dict) > 0:
            csv_dict['case_ES_acc'] = case_result_dict['overall_ES_acc']
            csv_dict['case_ES_AUC'] = case_result_dict['overall_ES_AUC']
            csv_dict['case_ES_Class_Attr_Scaled_F1'] = case_result_dict['overall_ES_Class_Attr_Scaled_F1']
            csv_dict['case_EO'] = case_result_dict['overall_EO']
            csv_dict['case_DP'] = case_result_dict['overall_DP']
            csv_dict['case_attr_0_fungal_f1'] = case_result_dict['attr_0_classification_report']['0']['f1-score']
            csv_dict['case_attr_0_bacterial_f1'] = case_result_dict['attr_0_classification_report']['1']['f1-score']
            csv_dict['case_attr_0_f1'] = case_result_dict['attr_0_classification_report']['macro avg']['f1-score']
            csv_dict['case_attr_0_AUC'] = case_result_dict['attr_0_AUC']
            csv_dict['case_attr_0_acc'] = case_result_dict['attr_0_classification_report']['accuracy']
            csv_dict['case_attr_1_fungal_f1'] = case_result_dict['attr_1_classification_report']['0']['f1-score']
            csv_dict['case_attr_1_bacterial_f1'] = case_result_dict['attr_1_classification_report']['1']['f1-score']
            csv_dict['case_attr_1_f1'] = case_result_dict['attr_1_classification_report']['macro avg']['f1-score']
            csv_dict['case_attr_1_AUC'] = case_result_dict['attr_1_AUC']
            csv_dict['case_attr_1_acc'] = case_result_dict['attr_1_classification_report']['accuracy']
            csv_dict['img_ES_acc'] = img_result_dict['overall_ES_acc']
            csv_dict['img_ES_AUC'] = img_result_dict['overall_ES_AUC']
            csv_dict['img_ES_Class_Attr_Scaled_F1'] = img_result_dict['overall_ES_Class_Attr_Scaled_F1']
            csv_dict['img_EO'] = img_result_dict['overall_EO']
            csv_dict['img_DP'] = img_result_dict['overall_DP']
            csv_dict['img_attr_0_fungal_f1'] = img_result_dict['attr_0_classification_report']['0']['f1-score']
            csv_dict['img_attr_0_bacterial_f1'] = img_result_dict['attr_0_classification_report']['1']['f1-score']
            csv_dict['img_attr_0_f1'] = img_result_dict['attr_0_classification_report']['macro avg']['f1-score']
            csv_dict['img_attr_0_AUC'] = img_result_dict['attr_0_AUC']
            csv_dict['img_attr_0_acc'] = img_result_dict['attr_0_classification_report']['accuracy']
            csv_dict['img_attr_1_fungal_f1'] = img_result_dict['attr_1_classification_report']['0']['f1-score']
            csv_dict['img_attr_1_bacterial_f1'] = img_result_dict['attr_1_classification_report']['1']['f1-score']
            csv_dict['img_attr_1_f1'] = img_result_dict['attr_1_classification_report']['macro avg']['f1-score']
            csv_dict['img_attr_1_AUC'] = img_result_dict['attr_1_AUC']
            csv_dict['img_attr_1_acc'] = img_result_dict['attr_1_classification_report']['accuracy']

        fold_col = f'fold_{self.fold_id}'
        csv_save_path = os.path.join(self.config.output_dir, 'test_results.csv' if self.config.site is None else f'test_results_{self.config.site}.csv')
        df_to_save = pd.DataFrame(csv_dict, index=[0]).T # type: ignore
        df_to_save.columns = [fold_col]
        if not os.path.exists(csv_save_path):
            df_to_save.to_csv(csv_save_path, mode='w', header=True)
        else:
            existing_df = pd.read_csv(csv_save_path, index_col=0)
            existing_df[fold_col] = df_to_save[fold_col]
            existing_df.to_csv(csv_save_path, mode='w', header=True)
        print("Test results saved to {}".format(csv_save_path))

        out_file = open(os.path.join(self.config.out_file_path + '_fold_{}'.format(self.fold_id) + '.txt'), 'w', encoding='utf-8')
        result = "Case Result: Accuracy = {:.4f}, F1-score = {:.4f}, BACC = {:.4f}, AUC = {:.4f}\n".format(case_acc, case_f1, case_bacc, case_auc)
        print('\n' + result)
        out_file.write(result)
        case_con_matrix = case_con_matrix.tolist()
        print('Confusion matrix: ')
        out_file.write('Confusion matrix: ' + '\n')
        for l in case_con_matrix:
            print(l)
            out_file.write(str(l) + '\n')
        print('Classification Report: ')
        out_file.write('Classification Report: ' + '\n')
        print(case_class_report)
        out_file.write(str(case_class_report)) # type: ignore
        if len(sens_attr_dict) > 0:
            es_result = "Equalized Odds (EO) = {:.4f}, Demographic Parity (DP) = {:.4f}\n".format(case_result_dict['overall_EO'], case_result_dict['overall_DP'])
            print(es_result)
            out_file.write(es_result)
            eq_acc_auc = "Equalized Accuracy = {:.4f}, Equalized AUC = {:.4f}, Equalized Class-Attr Scaled F1 = {:.4f}\n".format(
                case_result_dict['overall_ES_acc'], case_result_dict['overall_ES_AUC'], case_result_dict['overall_ES_Class_Attr_Scaled_F1'])
            print(eq_acc_auc)
            out_file.write(eq_acc_auc)
        title = '\nName\tpredict\ttruth\tattr\n'
        print(title)
        out_file.write(title)
        for patient in predict_dict.keys():
            content = str(patient) + '\t' + str(ensemble_dict[patient]) + '\t' + str(truth_dict[patient])
            if len(sens_attr_dict) > 0:
                content += '\t' + str(sens_attr_dict[patient])
            else:
                content += '\t-1'
            print(content)
            out_file.write(content + '\n')

        result = "Image Result: Accuracy = {:.4f}, F1-score = {:.4f}, BACC = {:.4f}, AUC = {:.4f}\n".format(img_acc, img_f1, img_bacc, img_auc)
        print('\n' + result)
        out_file.write(result)
        img_con_matrix = img_con_matrix.tolist()
        print('Confusion matrix: ')
        out_file.write('Confusion matrix: ' + '\n')
        for l in img_con_matrix:
            print(l)
            out_file.write(str(l) + '\n')
        print('Classification Report: ')
        out_file.write('Classification Report: ' + '\n')
        print(img_class_report)
        out_file.write(str(img_class_report)) # type: ignore
        if len(sens_attr_dict) > 0:
            es_result = "Equalized Odds (EO) = {:.4f}, Demographic Parity (DP) = {:.4f}\n".format(img_result_dict['overall_EO'], img_result_dict['overall_DP'])
            print(es_result)
            out_file.write(es_result)
            eq_acc_auc = "Equalized Accuracy = {:.4f}, Equalized AUC = {:.4f}, Equalized Class-Attr Scaled F1 = {:.4f}\n".format(
                img_result_dict['overall_ES_acc'], img_result_dict['overall_ES_AUC'], img_result_dict['overall_ES_Class_Attr_Scaled_F1'])
            print(eq_acc_auc)
            out_file.write(eq_acc_auc)
        title = '\nName\tprobability\tpredict\ttruth\tattr\n'
        print(title)
        out_file.write(title)
        for i in range(len(name_list)):
            content = str(name_list[i]) + '\t' + str(prob_list[i]) + '\t' + str(predict_list[i]) + '\t' + str(label_list[i])
            if len(sens_attr_list) > 0:
                content += '\t' + str(sens_attr_list[i])
            else:
                content += '\t-1'
            print(content)
            out_file.write(content + '\n')
        out_file.close()
    
    def heatmap(self):
        
        # if hasattr(self.config, 'pretrained_path') and self.config.pretrained_path is not None:
        self.load_model(os.path.join(self.config.output_dir, "MultiModaCL_Finetuned_Best_fold_{fold}.pth".format(fold=self.fold_id)))
        nb_batch = len(self.loader_test)
        pbar = tqdm(total=nb_batch, desc="Test", mininterval=20)
        # predict_dict, truth_dict, count_dict, ensemble_dict = {}, {}, {}, {}
        # case_correct, img_correct = 0, 0
        # name_list, predict_list, label_list = [], [], []
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        self.model.train()
        for batch in self.loader_test:
            pbar.update()
            inputs = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            name = batch['name'][0]
            if self.config.use_var:
                var = batch['var'].to(self.device)
            else:
                var = None

            name_parts = split_sample_path(name)
            patient = name_parts[0]
                # if patient not in predict_dict:
                #     predict_dict[patient] = 0
                # if patient not in truth_dict:
                #     truth_dict[patient] = labels.cpu().numpy()[0]
                # if patient not in count_dict:
                #     count_dict[patient] = 0
                # count_dict[patient] += 1
                
            feature_maps = []
            def hook_fn(module, input, output):
                    feature_maps.append(output)
            hook = self.model.block4[-1].mlp.dwconv.dwconv.register_forward_hook(hook_fn)
            
            
            feat, z, y = self.model(inputs, var)
            pred = y.argmax(dim=1, keepdim=True)
            feature_maps[0].retain_grad()
            self.model.zero_grad()
            y[0, pred].backward()
            grads = feature_maps[0].grad
            features = feature_maps[0].cpu()
            weights = torch.mean(grads, dim=(2, 3)).cpu()
            cam = torch.zeros(features.shape[2:], dtype=torch.float32)
            for i, w in enumerate(weights[0]):
                cam += w * features[0, i, :, :]
            cam = F.relu(cam)
            cam -= cam.min()
            cam /= cam.max()
            cam = cv2.resize(cam.detach().numpy(), (inputs.shape[2], inputs.shape[3]))
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET) # type: ignore
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                
            img = inputs[0].detach().cpu().numpy().transpose(1, 2, 0)
            ori_img = (img * std + mean) * 255.0
            ori_img = np.clip(ori_img, 0, 255).astype(np.uint8)
            superimposed_img = heatmap * 0.3 + np.array(ori_img)
            superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

            os.makedirs(os.path.join(self.config.output_dir, 'gradcam', patient), exist_ok=True)
            save_path = os.path.join(self.config.output_dir, 'gradcam', patient, name_parts[-1][:-4] + '.png')
            superimposed_img_pil = Image.fromarray(superimposed_img)
            superimposed_img_pil.save(save_path)
            hook.remove()
            gc.collect()



    def load_model(self, path):
        checkpoint = None
        try:
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        except BaseException as e:
            raise ValueError('Impossible to load the checkpoint: %s' % str(e))
        if checkpoint is not None:
            try:
                if hasattr(checkpoint, "state_dict"):
                    unexpected = self.model.load_state_dict(checkpoint.state_dict(), strict=False)
                    print('Model loading info: {}'.format(unexpected))
                elif isinstance(checkpoint, dict):
                    if "model" in checkpoint:
                        unexpected = self.model.load_state_dict(checkpoint["model"], strict=False)
                        print('Model loading info: {}'.format(unexpected))
                else:
                    unexpected = self.model.load_state_dict(checkpoint)
                    print('Model loading info: {}'.format(unexpected))
            except BaseException as e:
                raise ValueError('Error while loading the model\'s weights: %s' % str(e))


    def adjust_learning_rate(self, optimizer, epoch, base_lr, nb_epochs, warm_epochs, power_lr):
        if epoch < warm_epochs:
            lr = epoch / warm_epochs * base_lr
        else:
            lr = base_lr * ((1 - (epoch - warm_epochs) / nb_epochs) ** (power_lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


