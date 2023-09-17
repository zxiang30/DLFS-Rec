import numpy as np
import torch
import tqdm
from torch.optim import Adam

from utils import get_metric, recall_at_k, ndcg_k


def wasserstein_distance(mean1, cov1, mean2, cov2):
    ret = torch.sum((mean1 - mean2) * (mean1 - mean2), -1)
    cov1_sqrt = torch.sqrt(torch.clamp(cov1, min=1e-24))
    cov2_sqrt = torch.sqrt(torch.clamp(cov2, min=1e-24))
    ret = ret + torch.sum((cov1_sqrt - cov2_sqrt) * (cov1_sqrt - cov2_sqrt), -1)

    return ret


def wasserstein_distance_matmul(mean1, cov1, mean2, cov2):
    mean1 = mean1.unsqueeze(dim=1)
    cov1 = cov1.unsqueeze(dim=1)
    mean1_2 = torch.sum(mean1 ** 2, -1, keepdim=True)
    mean2_2 = torch.sum(mean2 ** 2, -1, keepdim=True)
    ret = -2 * torch.matmul(mean1, mean2.transpose(1, 2)) + mean1_2 + mean2_2.transpose(1, 2)

    cov1_2 = torch.sum(cov1, -1, keepdim=True)
    cov2_2 = torch.sum(cov2, -1, keepdim=True)

    cov_ret = -2 * torch.matmul(torch.sqrt(torch.clamp(cov1, min=1e-24)),
                                torch.sqrt(torch.clamp(cov2, min=1e-24)).transpose(1, 2)) + cov1_2 + cov2_2.transpose(
        1, 2)

    return ret + cov_ret


class Trainer:
    def __init__(self, model, train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):

        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        self.model = model
        print(self.model)
        if self.cuda_condition:
            self.model.cuda()

        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        # self.data_name = self.args.data_name
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(self.model.parameters(), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader)

    def valid(self, epoch, full_sort=False):
        if full_sort:
            self.args.train_matrix = self.args.valid_rating_matrix
        return self.iteration(epoch, self.eval_dataloader, full_sort, train=False)

    def test(self, epoch, full_sort=False):
        if full_sort:
            self.args.train_matrix = self.args.test_rating_matrix
        return self.iteration(epoch, self.test_dataloader, full_sort, train=False)

    def iteration(self, epoch, dataloader, full_sort=False, train=True):
        raise NotImplementedError

    def get_sample_scores(self, epoch, pred_list):
        pred_list = (-pred_list).argsort().argsort()[:, 0]
        HIT_1, NDCG_1, MRR = get_metric(pred_list, 1)
        HIT_5, NDCG_5, MRR = get_metric(pred_list, 5)
        HIT_10, NDCG_10, MRR = get_metric(pred_list, 10)
        post_fix = {
            "Epoch": epoch,
            "HIT@1": '{:.4f}'.format(HIT_1), "NDCG@1": '{:.4f}'.format(NDCG_1),
            "HIT@5": '{:.4f}'.format(HIT_5), "NDCG@5": '{:.4f}'.format(NDCG_5),
            "HIT@10": '{:.4f}'.format(HIT_10), "NDCG@10": '{:.4f}'.format(NDCG_10),
            "MRR": '{:.4f}'.format(MRR),
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR], str(post_fix)

    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [5, 10, 15, 20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "HIT@5": '{:.4f}'.format(recall[0]), "NDCG@5": '{:.4f}'.format(ndcg[0]),
            "HIT@10": '{:.4f}'.format(recall[1]), "NDCG@10": '{:.4f}'.format(ndcg[1]),
            "HIT@20": '{:.4f}'.format(recall[3]), "NDCG@20": '{:.4f}'.format(ndcg[3])
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[3], ndcg[3]], str(post_fix)

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        original_state_dict = self.model.state_dict()
        print(original_state_dict.keys())
        new_dict = torch.load(file_name)
        print(new_dict.keys())
        for key in new_dict:
            original_state_dict[key] = new_dict[key]
        self.model.load_state_dict(original_state_dict)

    def cross_entropy(self, seq_out, pos_ids, neg_ids):
        # [batch seq_len hidden_size]
        pos_emb = self.model.item_embeddings(pos_ids)
        neg_emb = self.model.item_embeddings(neg_ids)

        seq_emb = seq_out[:, -1, :]  # [batch*seq_len hidden_size]
        pos_logits = torch.sum(pos_emb * seq_emb, -1)  # [batch*seq_len]
        neg_logits = torch.sum(neg_emb * seq_emb, -1)
        loss = torch.mean(
            - torch.log(torch.sigmoid(pos_logits) + 1e-24) -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24)
        )

        return loss

    def predict_sample(self, seq_out, test_neg_sample):
        # [batch 100 hidden_size]
        test_item_emb = self.model.item_embeddings(test_neg_sample)
        # [batch hidden_size]
        test_logits = torch.bmm(test_item_emb, seq_out.unsqueeze(-1)).squeeze(-1)  # [B 100]
        return test_logits

    def predict_full(self, seq_out):
        # [item_num hidden_size]
        test_item_emb = self.model.item_embeddings.weight
        # [batch hidden_size ]
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred


class DLFSRecTrainer(Trainer):

    def __init__(self, model,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):
        super(DLFSRecTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader, args
        )

    def get_sample_scores(self, epoch, pred_list):
        pred_list = pred_list.argsort().argsort()[:, 0]
        HIT_1, NDCG_1, MRR = get_metric(pred_list, 1)
        HIT_5, NDCG_5, MRR = get_metric(pred_list, 5)
        HIT_10, NDCG_10, MRR = get_metric(pred_list, 10)
        post_fix = {
            "Epoch": epoch,
            "HIT@1": '{:.4f}'.format(HIT_1), "NDCG@1": '{:.4f}'.format(NDCG_1),
            "HIT@5": '{:.4f}'.format(HIT_5), "NDCG@5": '{:.4f}'.format(NDCG_5),
            "HIT@10": '{:.4f}'.format(HIT_10), "NDCG@10": '{:.4f}'.format(NDCG_10),
            "MRR": '{:.4f}'.format(MRR),
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR], str(post_fix)

    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [5, 10, 15, 20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "HIT@5": '{:.4f}'.format(recall[0]), "NDCG@5": '{:.4f}'.format(ndcg[0]),
            "HIT@10": '{:.4f}'.format(recall[1]), "NDCG@10": '{:.4f}'.format(ndcg[1]),
            "HIT@20": '{:.4f}'.format(recall[3]), "NDCG@20": '{:.4f}'.format(ndcg[3])
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[3], ndcg[3]], str(post_fix)

    def bpr_optimization(self, seq_mean_out, seq_cov_out, pos_ids, pos_cxt, neg_ids, neg_cxt):

        pos_mean_emb = self.model.item_mean_embeddings(pos_ids)
        pos_cov_emb = self.model.item_cov_embeddings(pos_ids)
        neg_mean_emb = self.model.item_mean_embeddings(neg_ids)
        neg_cov_emb = self.model.item_cov_embeddings(neg_ids)

        pos_attrs = self.args.items_feature[pos_ids]
        neg_attrs = self.args.items_feature[neg_ids]

        if self.args.side_info_fused:

            pos_mean_side_dense = self.model.side_mean_dense(torch.cat((pos_cxt, pos_attrs), dim=1))
            pos_cov_side_dense = self.model.side_cov_dense(torch.cat((pos_cxt, pos_attrs), dim=1))
            neg_mean_side_dense = self.model.side_mean_dense(torch.cat((neg_cxt, neg_attrs), dim=1))
            neg_cov_side_dense = self.model.side_cov_dense(torch.cat((neg_cxt, neg_attrs), dim=1))

            if self.args.fusion_type == 'concat':
                pos_mean_emb = self.model.mean_fusion_layer(torch.cat((pos_mean_emb, pos_mean_side_dense), dim=1))
                pos_cov_emb = self.model.cov_fusion_layer(torch.cat((pos_cov_emb, pos_cov_side_dense), dim=1))
                neg_mean_emb = self.model.mean_fusion_layer(torch.cat((neg_mean_emb, neg_mean_side_dense), dim=1))
                neg_cov_emb = self.model.cov_fusion_layer(torch.cat((neg_cov_emb, neg_cov_side_dense), dim=1))

            elif self.args.fusion_type == 'gate':
                pos_mean_concat = torch.cat(
                    [pos_mean_emb.unsqueeze(-2), pos_mean_side_dense.unsqueeze(-2)], dim=-2)
                pos_mean_emb, _ = self.model.mean_fusion_layer(pos_mean_concat)
                pos_cov_concat = torch.cat(
                    [pos_cov_emb.unsqueeze(-2), pos_cov_side_dense.unsqueeze(-2)], dim=-2)
                pos_cov_emb, _ = self.model.cov_fusion_layer(pos_cov_concat)

                neg_mean_concat = torch.cat(
                    [neg_mean_emb.unsqueeze(-2), neg_mean_side_dense.unsqueeze(-2)], dim=-2)
                neg_mean_emb, _ = self.model.mean_fusion_layer(neg_mean_concat)
                neg_cov_concat = torch.cat(
                    [neg_cov_emb.unsqueeze(-2), neg_cov_side_dense.unsqueeze(-2)], dim=-2)
                neg_cov_emb, _ = self.model.cov_fusion_layer(neg_cov_concat)

            else:
                pos_mean_emb = pos_mean_emb + pos_mean_side_dense
                pos_cov_emb = pos_cov_emb + pos_cov_side_dense
                neg_mean_emb = neg_mean_emb + neg_mean_side_dense
                neg_cov_emb = neg_cov_emb + neg_cov_side_dense

        pos_cov_emb = self.model.elu(pos_cov_emb) + 1
        neg_cov_emb = self.model.elu(neg_cov_emb) + 1

        seq_mean_emb = seq_mean_out[:, -1, :]
        seq_cov_emb = seq_cov_out[:, -1, :]

        pos_logits = wasserstein_distance(seq_mean_emb, seq_cov_emb, pos_mean_emb, pos_cov_emb)
        neg_logits = wasserstein_distance(seq_mean_emb, seq_cov_emb, neg_mean_emb, neg_cov_emb)
        pos_vs_neg = wasserstein_distance(pos_mean_emb, pos_cov_emb, neg_mean_emb, neg_cov_emb)

        istarget = (pos_ids > 0).view(pos_ids.size(0)).float()
        loss = torch.sum(-torch.log(torch.sigmoid(neg_logits - pos_logits) + 1e-24) * istarget) / torch.sum(istarget)

        pvn_loss = self.args.pvn_weight * torch.sum(torch.clamp(pos_logits - pos_vs_neg, 0) * istarget) / torch.sum(
            istarget)
        auc = torch.sum(((torch.sign(neg_logits - pos_logits) + 1) / 2) * istarget) / torch.sum(istarget)

        return loss, auc, pvn_loss

    def dist_predict(self, seq_mean_out, seq_cov_out, test_ids, test_cxt):

        test_mean_emb = self.model.item_mean_embeddings(test_ids)
        test_cov_emb = self.model.item_cov_embeddings(test_ids)

        test_attrs = self.args.items_feature[test_ids]
        if self.args.side_info_fused:

            test_mean_side_dense = self.model.side_mean_dense(torch.cat((test_cxt, test_attrs), dim=2))
            test_cov_side_dense = self.model.side_cov_dense(torch.cat((test_cxt, test_attrs), dim=2))

            if self.args.fusion_type == 'concat':
                test_mean_emb = self.model.mean_fusion_layer(torch.cat((test_mean_emb, test_mean_side_dense), dim=2))
                test_cov_emb = self.model.cov_fusion_layer(torch.cat((test_cov_emb, test_cov_side_dense), dim=2))
            elif self.args.fusion_type == 'gate':
                test_mean_concat = torch.cat(
                    [test_mean_emb.unsqueeze(-2), test_mean_side_dense.unsqueeze(-2)], dim=-2)
                test_mean_emb, _ = self.model.mean_fusion_layer(test_mean_concat)
                test_cov_concat = torch.cat(
                    [test_cov_emb.unsqueeze(-2), test_cov_side_dense.unsqueeze(-2)], dim=-2)
                test_cov_emb, _ = self.model.cov_fusion_layer(test_cov_concat)
            else:
                test_mean_emb = test_mean_emb + test_mean_side_dense
                test_cov_emb = test_cov_emb + test_cov_side_dense

        test_item_mean_emb = test_mean_emb
        test_item_cov_emb = self.model.elu(test_cov_emb) + 1
        return wasserstein_distance_matmul(seq_mean_out, seq_cov_out, test_item_mean_emb, test_item_cov_emb)

    def iteration(self, epoch, dataloader, full_sort=False, train=True):

        str_code = "train" if train else "test"
        rec_data_iter = tqdm.tqdm(enumerate(dataloader),
                                  desc="Recommendation EP_%s:%d" % (str_code, epoch),
                                  total=len(dataloader),
                                  bar_format="{l_bar}{r_bar}")
        if train:
            self.model.train()

            rec_avg_loss = 0.0
            rec_avg_auc = 0.0
            rec_avg_pvn_loss = 0.0

            train_num = 0.0
            for i, batch in rec_data_iter:
                batch = tuple(items[k].to(self.device) for items in batch for k in items)
                _, inputs_id, inputs_cxt, answers_id, answers_cxt, negs_id, negs_cxt = batch
                sequence_mean_output, sequence_cov_output = self.model(inputs_id, inputs_cxt)
                loss, batch_auc, pvn_loss = self.bpr_optimization(sequence_mean_output, sequence_cov_output, answers_id,
                                                                  answers_cxt, negs_id,
                                                                  negs_cxt)
                loss += pvn_loss
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                rec_avg_loss += loss.item() * len(answers_id)
                rec_cur_loss = loss.item()
                rec_avg_auc += batch_auc.item() * len(answers_id)
                rec_avg_pvn_loss += pvn_loss.item() * len(answers_id)
                train_num += len(answers_id)

            post_fix = {
                "epoch": epoch,
                "rec_loss": '{:.4f}'.format(rec_cur_loss),
                "rec_avg_loss": '{:.4f}'.format(rec_avg_loss / train_num),
                "rec_avg_pvn_loss": '{:.4f}'.format(rec_avg_pvn_loss / train_num),
                "rec_avg_auc": '{:.4f}'.format(rec_avg_auc / train_num)
            }

            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))

            with open(self.args.log_file, 'a') as f:
                f.write(str(post_fix) + '\n')

        else:
            self.model.eval()
            if full_sort:
                with torch.no_grad():
                    for i, batch in rec_data_iter:
                        batch = tuple(items[k].to(self.device) for items in batch for k in items)
                        users_id, inputs_id, inputs_cxt, pos_id, pos_cxt, _, _, _, _ = batch

                        pos_id = pos_id.unsqueeze(-1)
                        test_items = torch.arange(self.args.item_size).view(-1, self.args.item_size).expand(
                            inputs_id.size(0), self.args.item_size).cuda()
                        test_cxt = pos_cxt.unsqueeze(1).expand(inputs_id.size(0), self.args.item_size,
                                                                   pos_cxt.size(-1))

                        recommend_mean_output, recommend_cov_output = self.model(inputs_id, inputs_cxt)
                        recommend_mean_output = recommend_mean_output[:, -1, :]
                        recommend_cov_output = recommend_cov_output[:, -1, :]

                        rating_pred = self.dist_predict(recommend_mean_output, recommend_cov_output, test_items,
                                                        test_cxt)
                        rating_pred = rating_pred.squeeze(dim=1).cpu().data.numpy().copy()

                        batch_user_index = users_id.cpu().numpy()
                        rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 1e+24
                        # get the first 40 items
                        ind = np.argpartition(rating_pred, 40)[:, :40]
                        # get the first 40 scores
                        arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                        # get the first 40 items index in order
                        arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::]
                        # get the first 40 items in order
                        batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                        if i == 0:
                            pred_list = batch_pred_list
                            answer_list = pos_id.cpu().data.numpy()
                        else:
                            pred_list = np.append(pred_list, batch_pred_list, axis=0)
                            answer_list = np.append(answer_list, pos_id.cpu().data.numpy(), axis=0)

                    return self.get_full_sort_score(epoch, answer_list, pred_list)
            else:
                pred_list = None
                with torch.no_grad():
                    for i, batch in rec_data_iter:
                        batch = tuple(items[k].to(self.device) for items in batch for k in items)
                        _, inputs_id, inputs_cxt, pos_id, pos_cxt, _, _, negs_id, negs_cxt = batch

                        test_items = torch.cat((pos_id.unsqueeze(-1), negs_id), 1)
                        test_cxt = torch.cat((pos_cxt.unsqueeze(1), negs_cxt), 1)

                        recommend_mean_output, recommend_cov_output = self.model(inputs_id, inputs_cxt)
                        recommend_mean_output = recommend_mean_output[:, -1, :]
                        recommend_cov_output = recommend_cov_output[:, -1, :]

                        rating_pred = self.dist_predict(recommend_mean_output, recommend_cov_output, test_items,
                                                        test_cxt)

                        test_logits = rating_pred.squeeze(dim=1).cpu().data.numpy().copy()

                        if i == 0:
                            pred_list = test_logits
                        else:
                            pred_list = np.append(pred_list, test_logits, axis=0)
                return self.get_sample_scores(epoch, pred_list)
