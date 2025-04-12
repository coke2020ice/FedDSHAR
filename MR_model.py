"""
basetrain_teacher.py
--------------------
该模块负责MR_model的训练。函数 train_teacher(args) 加载训练和测试数据，
根据参数设置训练 HARCNN 模型，并将模型保存为 args.teacher_save_path。
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model.build_model import HARCNN
from utils.loss_dp import DPLoss
# 定义数据集加载类（与原来 basetrain.py 中类似）
class LoadDataset(Dataset):
    def __init__(self, dataset1,dataset2):
        # 假设数据集字典中 'x' 包含原始数据与增强数据拼接，前半部分为原始数据
        total = dataset1['x'].shape[0]
        half = total // 2
        self.x_data = dataset1['x'][:half]
        self.x_data_aug = dataset2['x'][half:]
        self.y_data = dataset1['y'][:half]
        self.length = half

    def __getitem__(self, index):
        return self.x_data[index], self.x_data_aug[index], self.y_data[index]

    def __len__(self):
        return self.length

# 定义测试函数，用于评估教师模型
def test_model(test_loader, model, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.float().to(device)
            labels = labels.long().to(device)
            outputs, _ = model(data)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    from sklearn.metrics import accuracy_score
    return accuracy_score(all_labels, all_preds)

def MR_model(args):
    device = args.device
    # 加载训练数据
    data_dir = os.path.join('dataset', 'harbox')
    
    train_file = os.path.join(data_dir, 'basedata.npz')


    train_data = np.load(train_file)
    X_train, y_train = train_data['x'], train_data['y']
    X_train = torch.Tensor(X_train).type(torch.float32)
    y_train = torch.Tensor(y_train).type(torch.int64)


    train_data={}
    train_data['x']=  X_train
    train_data['y'] = y_train
    #train_data = [(x, y) for x, y in zip(X_train, y_train)
    
    train_file_aug = os.path.join(data_dir, 'DA_RandSampling_'+ 'basedata.npz')
    
    train_data_aug  = np.load(train_file_aug)
    X_train_aug, y_train_aug = train_data_aug['x'], train_data_aug['y']
    X_train_aug = torch.Tensor(X_train_aug).type(torch.float32)
    y_train_aug = torch.Tensor(y_train_aug).type(torch.int64)
    
    
    train_data_aug={}
    train_data_aug['x']=  X_train_aug
    train_data_aug['y'] = y_train_aug
    


    
    train_dataset = LoadDataset(train_data,train_data_aug)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,shuffle=True, num_workers=0)
    # 加载测试数据
    
    
    train_file = os.path.join(data_dir, 'test.npz')
    test_data_np = np.load(train_file)
    X_test = torch.Tensor(test_data_np['x']).float()
    y_test = torch.Tensor(test_data_np['y']).long()
    test_dataset = [(x, y) for x, y in zip(X_test, y_test)]
    test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset),
                             shuffle=False, drop_last=True, num_workers=0)

    # 初始化模型
    data_name = "harbox"  # 可根据需要扩展到参数
    model = HARCNN(data_name).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.teacher_lr)
    
    use_data_aug = False  # 教师训练中是否使用数据增强，默认关闭
    dp='dis'
    n_feature=725
    print("开始教师模型训练...")
    for epoch in range(args.teacher_epochs):
        model.train()
        for batch_idx, (data, data_aug, labels) in enumerate(train_loader):
            data, labels = data.float().to(device), labels.long().to(device)
            data_aug = data_aug.float().to(device)
            optimizer.zero_grad()
            output1 = model(data)
            output2 = model(data_aug)

            pre1, feature1 = output1#['output'], output1['proto']
            pre2, feature2 = output2#['output'], output2['proto']

            raw_feature = feature1
            aug_feature = feature2


            loss_dp = torch.zeros(1).cuda()
            dp_layer = DPLoss(loss_type=dp, input_dim=n_feature)
            loss_dp = dp_layer.compute(raw_feature, aug_feature)

            # 直接在GPU上合并预测结果和标签
            pre_all = torch.cat([pre1, pre2], dim=0)
            labels_all = torch.cat([labels, labels], dim=0)

            loss = criterion(pre_all, labels_all) + 0.1*loss_dp
            loss.backward()
            optimizer.step()
           # torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
        acc = test_model(test_loader, model, device)
        print(f"MR_model Epoch [{epoch+1}/{args.teacher_epochs}] - 测试准确率: {acc:.4f}")
    # 保存教师模型
    torch.save(model.state_dict(), args.teacher_save_path)
    print(f"MR_model已保存至 {args.teacher_save_path}")
