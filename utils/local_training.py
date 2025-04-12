import logging
import numpy as np

import torch
import torch.optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from utils.losses import LA_R , LA_E
from sklearn.metrics import f1_score,accuracy_score ,recall_score

from utils.loss_dp import DPLoss



def globaltest(net, test_dataset, args):
    net.eval()
    
    X_test = test_dataset['x']
    y_test = test_dataset['y']
    test_data = [(x, y) for x, y in zip(X_test, y_test)]
    total_acc = []
    total_score=[]
    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    pred = np.array([])
    with torch.no_grad():
        for images, labels in test_loader:

            images = images.float().to(args.device)
            labels = labels.long().cuda().to(args.device)
            outputs = net(images)
            predictions, features = outputs

            accuracy = accuracy_score(labels.cpu().numpy(), predictions.detach().argmax(dim=1).cpu().numpy())
            pred = np.concatenate([pred, predictions.detach().argmax(dim=1).cpu().numpy()], axis=0)
            
            recall = recall_score(labels.cpu().numpy(), predictions.detach().argmax(dim=1).cpu().numpy(), average='macro')
            score = 2 * (accuracy * recall) / (accuracy + recall)
            
            total_acc.append(accuracy)
            total_score.append(score)
    total_acc = torch.tensor(total_acc).mean()
    total_score = torch.tensor(total_score).mean()

            
    return total_acc,total_score


class DatasetSplit(Dataset):
    def __init__(self, dataset,dataset_aug, idxs):
        self.dataset = dataset
        self.dataset_aug = dataset_aug
        
        
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image = self.dataset['x'][self.idxs[item]]
        label = self.dataset['y'][self.idxs[item]]
        
        image_aug = self.dataset_aug['x'][self.idxs[item]]
        label_aug = self.dataset_aug['y'][self.idxs[item]]
        
        
        return self.idxs[item], image, label ,image_aug ,label_aug

    def get_num_of_each_class(self, args):
        class_sum = np.array([0] * args.n_classes)
        for idx in self.idxs:
            label = self.dataset['y'][idx]
            class_sum[int(label)] += 1
        return class_sum.tolist()



class LocalUpdate(object):
    def __init__(self, args, id, dataset,dataset_aug, idxs):
        self.args = args
        self.id = id
        self.idxs = idxs
        self.local_dataset = DatasetSplit(dataset,dataset_aug, idxs)
     #   self.class_num_list = self.local_dataset.get_num_of_each_class(self.args)
        # logging.info(
        #     f'client{id} each class num: {self.class_num_list}, total: {len(self.local_dataset)}')
        self.ldr_train = DataLoader(
            self.local_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        self.epoch = 0
        self.iter_num = 0
        self.lr = self.args.base_lr


    def train_out_index(self,pre_teacher):
        pre_teacher.eval()
        label_r = np.array([])
        label_e = np.array([])
        
        with torch.no_grad():
            for (_, images, labels ,image_aug,label_aug) in self.ldr_train:

                images, labels = images.float().to(self.args.device), labels.long().cuda().to(self.args.device)
                
                image_aug= image_aug.float().to(self.args.device)
                
                teacher_output,f = pre_teacher(images)
                soft_label = torch.softmax(teacher_output/1, dim=1)
                max_probs, targets_u = torch.max(soft_label, dim=1)
                targets_u=targets_u.long().to('cuda')
                mask = targets_u == labels
                indices1 = mask.nonzero().squeeze()
                labels_2 = targets_u[indices1].cpu().numpy()
                label_r = np.append(label_r,labels_2)
        label_r = np.bincount(label_r.astype(int), minlength=5).tolist()
        return label_r
    

    def train_FedDSHAR(self, student_net, teacher_net,pre_teacher,class_list_fiter_r, writer, weight_kd):
        
        n_feature=725
        dp='dis'
        label_e = np.array([])
        student_net.train()
        teacher_net.eval()
        pre_teacher.eval()
        self.optimizer =torch.optim.Adam(student_net.parameters(), lr=self.args.lr_fed,betas=(0.9, 0.999), weight_decay=5e-4)

        epoch_loss = []
        criterion_1 = LA_R(cls_num_list=[elem * 2 for elem in class_list_fiter_r])
        
        with torch.no_grad():
            for (_, images, labels ,image_aug,label_aug) in self.ldr_train:

                images, labels = images.float().to(self.args.device), labels.long().cuda().to(self.args.device)
                
                image_aug= image_aug.float().to(self.args.device)
                
                teacher_output,f = teacher_net(image_aug)
                soft_label = torch.softmax(teacher_output/1, dim=1)
                max_probs, targets_u = torch.max(soft_label, dim=1)
                targets_u=targets_u.long().to('cuda')                
                mask = targets_u != labels
                indices2 = mask.nonzero().squeeze()
                labels_3 = targets_u[indices2].cpu().numpy()
                label_e = np.append(label_e,labels_3)
        label_e = np.bincount(label_e.astype(int), minlength=5).tolist()
        criterion_2 = LA_E(cls_num_list=label_e)

        for epoch in range(self.args.local_ep):
            batch_loss = []
            for (img_idx, images, labels,image_aug,label_aug ) in self.ldr_train:
                #images, labels = images.to(self.args.device), labels.to(self.args.device)
                images, labels = images.float().to(self.args.device), labels.long().cuda().to(self.args.device)
                
                image_aug= image_aug.float().to(self.args.device)
                
                logits1,f1 = student_net(images)
                
                logits2,f2 = student_net(image_aug)
                
                
                
                loss_dp = torch.zeros(1).cuda()
                dp_layer = DPLoss(loss_type=dp, input_dim=n_feature)
                loss_dp = dp_layer.compute(f1, f2)                
                
                
                with torch.no_grad():
                    teacher_output,f = pre_teacher(images)
                    soft_label = torch.softmax(teacher_output/1, dim=1)
                    max_probs, targets_u = torch.max(soft_label, dim=1)
                    targets_u=targets_u.long().to('cuda')
                


                mask = targets_u == labels
                indices1 = mask.nonzero().squeeze()

                
               
                mask = targets_u != labels
                indices2 = mask.nonzero().squeeze()

                with torch.no_grad():
                    teacher_output,f = teacher_net(image_aug)
                    soft_label = torch.softmax(teacher_output/0.8, dim=1)
                
                # 计算自定义损失 Lx2
                Lx1 = criterion_1(logits1, logits2, labels, soft_label, weight_kd, indices1, indices2)
                Lx2 = criterion_2(logits1, logits2, labels, soft_label, weight_kd, indices1, indices2)

                # 检查 Lx2 中有效（finite）的数值，并对其取平均
                valid_mask2 = torch.isfinite(Lx2)
                if valid_mask2.sum() > 0:
                    Lx2_avg = Lx2[valid_mask2].mean()
                else:
                    Lx2_avg = torch.tensor(0.0, device=Lx2.device)

                valid_mask1 = torch.isfinite(Lx1)
                if valid_mask1.sum() > 0:
                    Lx1_avg = Lx1[valid_mask1].mean()
                else:
                    Lx1_avg = torch.tensor(0.0, device=Lx1.device)

                loss = Lx2_avg +Lx1_avg +1 * loss_dp

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_loss.append(loss.item())
                writer.add_scalar(f'client{self.id}/loss_train', loss.item(), self.iter_num)
                self.iter_num += 1
                writer.add_scalar(
                    f'client{self.id}/loss_train', loss.item(), self.iter_num)
                self.iter_num += 1
            self.epoch = self.epoch + 1

            epoch_loss.append(np.array(batch_loss).mean())

        return student_net.state_dict(), np.array(epoch_loss).mean()

    
  