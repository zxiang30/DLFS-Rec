import datetime
import math
import os
import pickle as pkl
import random

import numpy as np
import torch
from scipy.sparse import csr_matrix
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from datasets import DLFSRecDataset


def get_metric(pred_list, topk=10):
    NDCG = 0.0
    HIT = 0.0
    MRR = 0.0
    # [batch] the answer's rank
    for rank in pred_list:
        MRR += 1.0 / (rank + 1.0)
        if rank < topk:
            NDCG += 1.0 / np.log2(rank + 2.0)
            HIT += 1.0
    return HIT / len(pred_list), NDCG / len(pred_list), MRR / len(pred_list)


# Calculates the ideal discounted cumulative gain at k
def idcg_k(k):
    res = sum([1.0 / math.log(i + 2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res


def ndcg_k(actual, predicted, topk):
    res = 0
    for user_id in range(len(actual)):
        k = min(topk, len(actual[user_id]))
        idcg = idcg_k(k)
        dcg_k = sum([int(predicted[user_id][j] in
                         set(actual[user_id])) / math.log(j + 2, 2) for j in range(topk)])
        res += dcg_k / idcg
    return res / float(len(actual))


def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    return sum_recall / true_users


def cal_mrr(actual, predicted):
    sum_mrr = 0.
    true_users = 0
    num_users = len(predicted)
    for i in range(num_users):
        r = []
        act_set = set(actual[i])
        pred_list = predicted[i]
        for item in pred_list:
            if item in act_set:
                r.append(1)
            else:
                r.append(0)
        r = np.array(r)
        if np.sum(r) > 0:
            sum_mrr += np.reciprocal(np.where(r == 1)[0] + 1, dtype=np.float)[0]
            true_users += 1
    return sum_mrr / len(predicted)


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'{path} created')


def get_local_time():
    r"""Get current time

    Returns:
        str: current time
    """
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y_%H-%M-%S')

    return cur


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, checkpoint_path, patience=10, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def compare(self, score):
        for i in range(len(score)):
            # 有一个指标增加了就认为是还在涨
            if score[i] > self.best_score[i] + self.delta:
                return False
        return True

    def __call__(self, score, model):
        # score HIT@10 NDCG@10

        if self.best_score is None:
            self.best_score = score
            self.score_min = np.array([0] * len(score))
            self.save_checkpoint(score, model)
        elif self.compare(score):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            # ({self.score_min:.6f} --> {score:.6f}) # 这里如果是一个值的话输出才不会有问题
            print(f'Validation score increased.  Saving model ...')
        torch.save(model.state_dict(), self.checkpoint_path)
        self.score_min = score


def generate_rating_matrix_valid(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        row.append(user_id)
        col.append(0)
        data.append(1)
        for item in item_list[:-2]:
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix


def generate_rating_matrix_test(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []

    for user_id, item_list in enumerate(user_seq):
        row.append(user_id)
        col.append(0)
        data.append(1)
        for item in item_list[:-1]:
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix


def get_rating_matrix(seq_dic):
    valid_rating_matrix = generate_rating_matrix_valid(seq_dic['user_seq'], seq_dic['n_users'], seq_dic['n_items'])
    test_rating_matrix = generate_rating_matrix_test(seq_dic['user_seq'], seq_dic['n_users'], seq_dic['n_items'])
    return valid_rating_matrix, test_rating_matrix


def get_user_seqs_and_max_item(data_file):
    lines = open(data_file).readlines()
    user_seq = []
    item_set = set()
    for line in lines:
        user, items = line.strip().split(' ', 1)
        items = items.split()
        items = [int(item) for item in items]
        user_seq.append(items)
        item_set = item_set | set(items)
    max_item = max(item_set)
    return user_seq, max_item


def get_user_seqs_and_sample(data_file, sample_file):
    lines = open(data_file).readlines()
    user_seq = []
    item_set = set()
    for line in lines:
        user, items = line.strip().split(' ', 1)
        items = items.split(' ')
        items = [int(item) for item in items]
        user_seq.append(items)
        item_set = item_set | set(items)
    max_item = max(item_set)
    num_users = len(lines)
    lines = open(sample_file).readlines()
    sample_seq = []
    for line in lines:
        user, items = line.strip().split(' ', 1)
        items = items.split(' ')
        items = [int(item) for item in items]
        sample_seq.append(items)

    assert len(user_seq) == len(sample_seq)

    return user_seq, max_item, num_users, sample_seq


def get_user_sample(sample_file):
    lines = open(sample_file).readlines()
    sample_seq = []
    for line in lines:
        user, items = line.strip().split(' ', 1)
        items = items.split(' ')
        items = [int(item) for item in items]
        sample_seq.append(items)
    return sample_seq


def get_data_dic(args):
    dat = pkl.load(open(f'{args.data_dir}{args.data_name}_all_multi_word.dat', 'rb'))
    data = {}

    user_reviews = dat['user_seq_token']
    data['user_seq_wt'] = []
    data['user_seq'] = []
    for u in user_reviews:
        data['user_seq_wt'].append(user_reviews[u])
        items = [item for item, time in user_reviews[u]]
        data['user_seq'].append(items)

    data['user_seq_wt_dic'] = user_reviews
    data['items_feat'] = dat['items_feat']
    data['n_items'] = len(dat['item2id'])
    data['n_users'] = len(dat['user2id']) - 1
    data['n_categories'] = len(dat['category2id'])
    data['n_brands'] = len(dat['brand2id'])
    data['feature_size'] = 6 + 1 + data['n_categories'] + data['n_brands'] - 2
    data['sample_seq'] = get_user_sample(args.data_dir + args.data_name + '_sample.txt')
    return data


def get_dataloader(args, seq_dic):
    train_dataset = DLFSRecDataset(args, seq_dic['user_seq_wt'], data_type='train')
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    eval_dataset = DLFSRecDataset(args, seq_dic['user_seq_wt'], test_neg_items=seq_dic['sample_seq'], data_type='valid')
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    test_dataset = DLFSRecDataset(args, seq_dic['user_seq_wt'], test_neg_items=seq_dic['sample_seq'], data_type='test')
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size)

    return train_dataloader, eval_dataloader, test_dataloader


def get_feats_vec(feats, args):
    feats = torch.tensor(feats)
    feat_category = torch.zeros(feats.size(0), args['n_categories'])
    category_vec = feat_category.scatter_(index=feats[:, 1:-1].long(), value=1, dim=-1)
    feat_brand = torch.zeros(feats.size(0), args['n_brands'])
    brand_vec = feat_brand.scatter_(index=feats[:, -1:].long(), value=1, dim=-1)
    vec = torch.cat((feats[:, :1], category_vec[:, 1:], brand_vec[:, 1:]), dim=1)
    return vec
