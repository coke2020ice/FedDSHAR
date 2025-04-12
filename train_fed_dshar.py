"""
train_fed_dshar.py
------------------

"""

import os
import copy
import numpy as np
import torch
from model.build_model import HARCNN  
from utils.sampling import iid_sampling, non_iid_dirichlet_sampling
from utils.utils import set_output_files, get_current_consistency_weight
import random

from utils.local_training import LocalUpdate, globaltest
from utils.FedAvg import FedAvg

def get_dataset(args):
    data_dir = os.path.join('dataset', 'harbox')
    
    train_file = os.path.join(data_dir, 'train.npz')
    train_data = np.load(train_file)
    train_data = {key: train_data[key] for key in train_data.files}

    train_file_aug = os.path.join(data_dir, 'DA_TimeWarp_'+'train.npz')
    train_data_aug = np.load(train_file_aug)
    train_data_aug = {key: train_data_aug[key] for key in train_data_aug.files}

    test_file = os.path.join(data_dir, 'test.npz')
    test_data = np.load(test_file)
    test_data = {key: test_data[key] for key in test_data.files}

    n_train = len(train_data["x"])
    n_clients = args.n_clients
    seed = args.seed
    
    
    y_train = train_data["y"]
    n_clients=30
    seed=0
    if args.iid:
        dict_users = iid_sampling(n_train, args.n_clients, args.seed)
    else:

        dict_users = non_iid_dirichlet_sampling(
            y_train=y_train,
            num_classes=args.num_classes,
            p=args.non_iid_prob,
            num_users=args.n_clients,
            seed=args.seed,
            alpha_dirichlet=args.alpha_dirichlet
        )
    return train_data, train_data_aug, test_data, dict_users


# --------------------- Add Noise ---------------------------
def noisify_label(true_label, num_classes=5, noise_type="pairflip"):
    #return (true_label - 1) % num_classes
    label_lst = list(range(num_classes))
    label_lst.remove(true_label)
    return random.sample(label_lst, k=1)[0]
    




def add_noise(train_data, dict_users, p=0.5,
              high_noise_lower=0.5, high_noise_upper=1.0,
              low_noise_lower=0.0, low_noise_upper=0.5):

    y_train_noisy = copy.deepcopy(train_data)
    noise_rate1 = []


    num_classes = len(set(train_data))  
    num_low_noise_classes = int((1 - p) * num_classes)
    num_high_noise_classes = num_classes - num_low_noise_classes

    all_classes = list(range(num_classes))
    low_noise_classes = random.sample(all_classes, num_low_noise_classes)
    high_noise_classes = [cls for cls in all_classes if cls not in low_noise_classes]

    for user, indices in dict_users.items():
        for idx in indices:
            true_label = train_data[idx]
            if true_label in high_noise_classes:
                noise_rate = np.random.uniform(high_noise_lower, high_noise_upper)
            else:
                noise_rate = np.random.uniform(low_noise_lower, low_noise_upper)

            if random.random() < noise_rate:
                y_train_noisy[idx] = noisify_label(true_label, num_classes)
        noise_rate1.append(noise_rate)
    print('mean noise rate:', sum(noise_rate1) / len(noise_rate1))
    return y_train_noisy



def train_fed_dshar(args):
    device = args.device
    data_name = "harbox"
    netglob = HARCNN(data_name).to(device)
    
# ------------------------------ Data Loading ------------------------------
# Assume the data is located at args.output_dir/dataset/harbox/; modify as needed.
    dataset_train,train_data_aug, dataset_test, dict_users = get_dataset(args)
    y_train = dataset_train['y']
    y_train_noisy = add_noise(
        y_train, dict_users,
        p=args.p,
        high_noise_lower=args.high_noise_lower,
        high_noise_upper=args.high_noise_upper,
        low_noise_lower=args.low_noise_lower,
        low_noise_upper=args.low_noise_upper
    )
    
    print('y_train_noisy',np.sum(y_train_noisy == y_train) / len(y_train))
    dataset_train['y'] = y_train_noisy
    train_data_aug['y'] = y_train_noisy
# ------------------------------ Initialize Federated Training Modules ------------------------------

# Create a simple IID client partition, with each client receiving an equal share of the training samples.
    
    pre_teacher = HARCNN(data_name).to(args.device)
    pre_teacher.load_state_dict(torch.load('./MR_model.pth'))
    
    n_train = len(dataset_train["x"])
    n_clients = args.n_clients

    user_id = list(range(args.n_clients))
    
    trainer_locals = []
    for i in range(n_clients):
        trainer = LocalUpdate(args, i, copy.deepcopy(dataset_train), copy.deepcopy(train_data_aug), dict_users[i])
        trainer_locals.append(trainer)

    fiter_index_r={}
    for idx in user_id:
        local = trainer_locals[idx]
        label_r  = local.train_out_index(pre_teacher=copy.deepcopy(pre_teacher).to(args.device))
        fiter_index_r[idx] =label_r


# ------------------------------ Federated Training Rounds ------------------------------

    writer, models_dir = set_output_files(args)
    for rnd in range(args.fed_epochs):
        weight_kd = get_current_consistency_weight(rnd, args.begin, args.end) * args.a
        writer.add_scalar('train/w_kd', weight_kd, rnd)
        w_locals = []
        for i in range(n_clients):
            local = trainer_locals[i]
            
            # Use the current loop index i for selecting the filter list
            class_list_fiter_r = fiter_index_r[i] 


            w_local, loss_local = local.train_FedDSHAR(
                student_net=copy.deepcopy(netglob).to(device),
                teacher_net=copy.deepcopy(netglob).to(device),
                pre_teacher=copy.deepcopy(pre_teacher).to(device),
                class_list_fiter_r=class_list_fiter_r,
                writer=writer, 
                weight_kd=1.0   
            )
            w_locals.append(copy.deepcopy(w_local))
        
        # Aggregate the local models to update the global model
        num_samples_list = [len(dict_users[i]) for i in range(n_clients)]
        w_glob = FedAvg(w_locals, num_samples_list)
        netglob.load_state_dict(copy.deepcopy(w_glob))
        
        # Evaluate the global model
        acc, score = globaltest(copy.deepcopy(netglob).to(device), dataset_test, args)
        print(f"FedDSHAR Round {rnd} - Accuracy: {acc:.4f}, Score: {score:.4f}")

  # Save the global model
    global_model_path = os.path.join(args.output_dir, "global_model.pth")
    torch.save(netglob.state_dict(), global_model_path)
    print(f"Federated global model saved to {global_model_path}")

