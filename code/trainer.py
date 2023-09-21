from torch.utils.data import DataLoader
from ent_init_model import EntInit
from rgcn_model import RGCN
from kge_model import KGEModel
import torch
import torch.nn.functional as F
import torch.nn as nn
from collections import defaultdict as ddict
import numpy as np
from torch.autograd import Variable
from VAE import VAE, GVAE, another_GVAE
import os
import pdb

class Trainer(nn.Module):
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args
        # models
        self.ent_init = EntInit(args).to(args.gpu)
        self.rgcn = RGCN(args).to(args.gpu)
        self.kge_model = KGEModel(args).to(args.gpu)
        self.vae = another_GVAE(args, args.ent_dim, args.vae_hidden_dims).to(args.gpu)
        self.multi_type_loss = MultiLossLayer(6).to(args.gpu)
        # self.multi_kg_loss = MultiLossLayer(args.metatrain_bs).to(args.gpu)

        self.loss_fct = torch.nn.CrossEntropyLoss()

    def get_loss_cle(self, tri, neg_tail_ent, neg_head_ent, neg_rel, ent_emb):
        # pdb.set_trace()
        neg_tail_score = self.kge_model((tri, neg_tail_ent), ent_emb, mode='tail-batch')
        neg_head_score = self.kge_model((tri, neg_head_ent), ent_emb, mode='head-batch')
        neg_rel_score = self.kge_model((tri, neg_rel), ent_emb, mode='rel-batch')
        pos_score = self.kge_model(tri, ent_emb)

        temp = 0.05
        score = torch.cat([pos_score/temp, neg_tail_score/temp, neg_head_score/temp, neg_rel_score/temp], dim=1)
        labels = torch.zeros(score.size(0)).long().cuda()
        loss_cl = self.loss_fct(score, labels)
        return loss_cl

    def get_loss_clm(self, sup_tri, ent_emb, ent_emb_vae, sup_tri_self_last_list, ent_emb_last_list):
        # pdb.set_trace()
        temp = 0.05
        neg_tail_score_list = []
        for lastall in zip(sup_tri_self_last_list, ent_emb_last_list):
            sup_neg_tail = torch.from_numpy(np.array([np.random.choice(np.arange(len(lastall[0])), self.args.metatrain_num_neg) for _ in sup_tri])).to(self.args.gpu)
            neg_tail_score = self.kge_model((sup_tri, sup_neg_tail), ent_emb, lastall[1], mode='tail-batch')
            neg_tail_score_list.append(neg_tail_score / temp)

        pos_score = self.kge_model(sup_tri, ent_emb, ent_emb_vae) / temp

        # pdb.set_trace()
        all_score=[]
        all_score.append(pos_score)
        all_score.extend(neg_tail_score_list)
        score = torch.cat(all_score, dim=1)
        labels = torch.zeros(score.size(0)).long().cuda()
        loss_cl = self.loss_fct(score, labels)
        return loss_cl

    def get_loss_task(self, tri, neg_tail_ent, neg_head_ent, ent_emb):
        # pdb.set_trace()
        neg_tail_score = self.kge_model((tri, neg_tail_ent), ent_emb, mode='tail-batch')
        neg_head_score = self.kge_model((tri, neg_head_ent), ent_emb, mode='head-batch')
        pos_score = self.kge_model(tri, ent_emb)

        neg_score = torch.cat([neg_tail_score, neg_head_score])
        neg_score = (F.softmax(neg_score * self.args.adv_temp, dim=1).detach()
                     * F.logsigmoid(-neg_score)).sum(dim=1)
        pos_score = F.logsigmoid(pos_score).squeeze(dim=1)
        positive_sample_loss = - pos_score.mean()
        negative_sample_loss = - neg_score.mean()

        # pdb.set_trace()
        loss_link = (positive_sample_loss + negative_sample_loss) / 2

        return loss_link


    def get_ent_emb(self, sup_g_bidir):
        # pdb.set_trace()
        self.ent_init(sup_g_bidir)
        # ent_emb = self.rgcn(sup_g_bidir)
        ent_emb, ent_emb_vae, mu, log_var = self.vae(sup_g_bidir)
        loss_vae = self.vae.vae_loss_function(ent_emb_vae, ent_emb, mu, log_var)

        return ent_emb, ent_emb_vae, loss_vae

    def evaluate(self, ent_emb, eval_dataloader, num_cand='all'):
        results = ddict(float)
        count = 0

        eval_dataloader.dataset.num_cand = num_cand

        if num_cand == 'all':
            for batch in eval_dataloader:
                pos_triple, tail_label, head_label = [b.to(self.args.gpu) for b in batch]
                head_idx, rel_idx, tail_idx = pos_triple[:, 0], pos_triple[:, 1], pos_triple[:, 2]

                # tail prediction
                pred = self.kge_model((pos_triple, None), ent_emb, mode='tail-batch')

                b_range = torch.arange(pred.size()[0], device=self.args.gpu)
                target_pred = pred[b_range, tail_idx]
                pred = torch.where(tail_label.bool(), -torch.ones_like(pred) * 10000000, pred)
                pred[b_range, tail_idx] = target_pred

                tail_ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True),
                                               dim=1, descending=False)[b_range, tail_idx]

                # head prediction
                pred = self.kge_model((pos_triple, None), ent_emb, mode='head-batch')

                b_range = torch.arange(pred.size()[0], device=self.args.gpu)
                target_pred = pred[b_range, head_idx]
                pred = torch.where(head_label.bool(), -torch.ones_like(pred) * 10000000, pred)
                pred[b_range, head_idx] = target_pred

                head_ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True),
                                               dim=1, descending=False)[b_range, head_idx]

                ranks = torch.cat([tail_ranks, head_ranks])
                ranks = ranks.float()
                count += torch.numel(ranks)
                results['mr'] += torch.sum(ranks).item()
                results['mrr'] += torch.sum(1.0 / ranks).item()

                for k in [1, 5, 10]:
                    results['hits@{}'.format(k)] += torch.numel(ranks[ranks <= k])

            for k, v in results.items():
                results[k] = v / count

        else:
            for i in range(self.args.num_sample_cand):
                for batch in eval_dataloader:
                    pos_triple, tail_cand, head_cand = [b.to(self.args.gpu) for b in batch]

                    b_range = torch.arange(pos_triple.size()[0], device=self.args.gpu)
                    target_idx = torch.zeros(pos_triple.size()[0], device=self.args.gpu, dtype=torch.int64)
                    # tail prediction
                    pred = self.kge_model((pos_triple, tail_cand), ent_emb, mode='tail-batch')
                    tail_ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True),
                                                   dim=1, descending=False)[b_range, target_idx]
                    # head prediction
                    pred = self.kge_model((pos_triple, head_cand), ent_emb, mode='head-batch')
                    head_ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True),
                                                   dim=1, descending=False)[b_range, target_idx]

                    ranks = torch.cat([tail_ranks, head_ranks])
                    ranks = ranks.float()
                    count += torch.numel(ranks)
                    results['mr'] += torch.sum(ranks).item()
                    results['mrr'] += torch.sum(1.0 / ranks).item()

                    for k in [1, 5, 10]:
                        results['hits@{}'.format(k)] += torch.numel(ranks[ranks <= k])

            for k, v in results.items():
                results[k] = v / count

        return results



    def clone(self):
        clone = Trainer(self.args)
        clone.load_state_dict(self.state_dict())
        if self.is_cuda():
            clone.cuda()
        return clone

    def point_grad_to(self, target):
        '''
        Set .grad attribute of each parameter to be proportional
        to the difference between self and target
        '''
        for p, target_p in zip(self.parameters(), target.parameters()):
            if p.grad is None:
                if self.is_cuda():
                    p.grad = Variable(torch.zeros(p.size())).cuda()
                else:
                    p.grad = Variable(torch.zeros(p.size()))
            p.grad.data.zero_()  # not sure this is required
            p.grad.data.add_(p.data - target_p.data)

    def is_cuda(self):
        return next(self.parameters()).is_cuda

class MultiLossLayer(nn.Module):
    """
        计算自适应损失权重
        implementation of "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"
    """

    def __init__(self, num_loss):
        """
        Args:
            num_loss (int): number of multi-task loss
        """
        super(MultiLossLayer, self).__init__()
        # sigmas^2 (num_loss,)
        # uniform init
        # 从均匀分布U(a, b)中生成值，填充输入的张量或变量，其中a为均匀分布中的下界，b为均匀分布中的上界
        self.sigmas_sq = nn.Parameter(nn.init.uniform_(torch.empty(num_loss), a=0.2, b=1.0), requires_grad=True)

    def get_loss(self, loss_set):
        """
        Args:
            loss_set (Tensor): multi-task loss (num_loss,)
        """
        # 1/2σ^2
        # (num_loss,)
        # self.sigmas_sq -> tensor([0.9004, 0.4505]) -> tensor([0.6517, 0.8004]) -> tensor([0.7673, 0.6247])
        # 出现左右两个数随着迭代次数的增加，相对大小交替变换
        # pdb.set_trace()
        factor = torch.div(1.0, torch.mul(2.0, self.sigmas_sq))
        # loss part (num_loss,)
        loss_part = torch.sum(torch.mul(factor, loss_set))
        # regular part 正则项，防止某个σ过大而引起训练严重失衡。
        # regular_part = torch.sum(torch.log(self.sigmas_sq))

        loss = loss_part

        return loss












