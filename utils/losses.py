import torch
import torch.nn as nn
import torch.nn.functional as F





class LA_R(nn.Module):
    def __init__(self, cls_num_list, tau=1):
        super(LA_R, self).__init__()
        cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list = tau * torch.log(cls_p_list)
        self.m_list = m_list.view(1, -1)

    def forward(self, x1,x2 ,target, soft_target, w_kd ,indices1,indices2):
        # 检查 x1 的第二维是否和 self.m_list 的第二维一致
        
        # expected_dim = self.m_list.shape[1]
        # if x1.shape[1] < expected_dim:
        #     diff = expected_dim - x1.shape[1]
        #     # 在 x1 的后面补零，使其尺寸达到 expected_dim
        #     x1 = F.pad(x1, (0, diff), mode='constant', value=0)
        #     x2 = F.pad(x2, (0, diff), mode='constant', value=0)
        
        
        x_m1 = x1 + self.m_list
        x_m2 = x2 + self.m_list
        log_pred1 = torch.log_softmax(x_m1, dim=-1)
        log_pred1 = torch.where(torch.isinf(log_pred1), torch.full_like(log_pred1, 0), log_pred1)

        log_pred2 = torch.log_softmax(x_m2, dim=-1)
        log_pred2 = torch.where(torch.isinf(log_pred2), torch.full_like(log_pred2, 0), log_pred2)
        

        kl = F.kl_div(log_pred1[indices2], soft_target[indices2], reduction='batchmean')
        return   F.nll_loss(log_pred1[indices1], target[indices1])  +  F.nll_loss(log_pred2[indices1], target[indices1])  #+  w_kd *  kl 






class LA_E(nn.Module):
    def __init__(self, cls_num_list, tau=1):
        super(LA_E, self).__init__()
        cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list = tau * torch.log(cls_p_list)
        self.m_list = m_list.view(1, -1)

    def forward(self, x1,x2 ,target, soft_target, w_kd ,indices1,indices2):
        x_m1 = x1 + self.m_list
        log_pred1 = torch.log_softmax(x_m1, dim=-1)
        log_pred1 = torch.where(torch.isinf(log_pred1), torch.full_like(log_pred1, 0), log_pred1)

        kl = F.kl_div(log_pred1[indices2], soft_target[indices2], reduction='batchmean')
        return  w_kd *  kl 
        #return   F.nll_loss(log_pred1[indices1], target[indices1])  +  F.nll_loss(log_pred2[indices1], target[indices1]) + w_kd *  kl 
