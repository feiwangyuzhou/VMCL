from utils import get_g_bidir
from datasets import TrainSubgraphDataset, ValidSubgraphDataset
from torch.utils.data import DataLoader
import torch
from torch import nn
from torch import optim
from trainer import Trainer
import dgl
from collections import defaultdict as ddict
from torch.utils.tensorboard import SummaryWriter
import json
from utils import Log
import os
from utils import get_indtest_test_dataset_and_train_g
from datasets import KGEEvalDataset
import pdb


class MetaTrainer(nn.Module):
    def __init__(self, args, generator):
        super(MetaTrainer, self).__init__()
        # pdb.set_trace()
        # dataloader
        self.args = args

        train_subgraph_dataset = TrainSubgraphDataset(args)
        valid_subgraph_dataset = ValidSubgraphDataset(args)
        self.train_subgraph_dataloader = DataLoader(train_subgraph_dataset, batch_size=args.metatrain_bs,
                                                    shuffle=True, collate_fn=TrainSubgraphDataset.collate_fn,
                                                    generator=generator)
        self.valid_subgraph_dataloader = DataLoader(valid_subgraph_dataset, batch_size=args.metatrain_bs,
                                                    shuffle=False, collate_fn=ValidSubgraphDataset.collate_fn,
                                                    generator=generator)

        indtest_test_dataset, indtest_train_g = get_indtest_test_dataset_and_train_g(args)
        self.indtest_train_g = indtest_train_g.to(args.gpu)
        self.indtest_test_dataloader = DataLoader(indtest_test_dataset, batch_size=args.indtest_eval_bs,
                                                  shuffle=False, collate_fn=KGEEvalDataset.collate_fn,
                                                  generator=generator)

        # writer and logger
        self.name = args.name
        self.writer = SummaryWriter(os.path.join(args.tb_log_dir, self.name))
        self.logger = Log(args.log_dir, self.name).get_logger()
        self.logger.info(json.dumps(vars(args)))
        # state dir
        self.state_path = os.path.join(args.state_dir, self.name)
        if not os.path.exists(self.state_path):
            os.makedirs(self.state_path)

        self.net = Trainer(args)
        # optim
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.args.metatrain_lr)
        # self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.args.metatrain_lr)
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.9, patience=10)

    def load_pretrain(self):
        state = torch.load(self.args.pretrain_state, map_location=self.args.gpu)
        self.net.load_state_dict(state)
        # self.ent_init.load_state_dict(state['ent_init'])
        # self.rgcn.load_state_dict(state['rgcn'])
        # self.kge_model.load_state_dict(state['kge_model'])

    def get_optimizer(self, net, state=None):
        optimizer = optim.Adam(net.parameters(), lr=self.args.metatrain_lr)
        if state is not None:
            optimizer.load_state_dict(state)
        return optimizer

    def get_metaKGs(self, batch, sup_g_list):
        sup_tri_self_last_list = []
        ent_emb_last_list = []
        # pdb.set_trace()
        for i in range(len(batch)):
            ent_emb_last = sup_g_list[len(batch)-i-1].ndata['h']
            ent_emb_last_vae = sup_g_list[len(batch)-i-1].ndata['g']
            ent_emb_last_list.insert(0, ent_emb_last_vae)
            ent_emb_last_list.insert(0, ent_emb_last)
            _, _, _, _, sup_tri_self_last = [d.to(self.args.gpu) for d in batch[len(batch)-i-1][:5]]
            sup_tri_self_last_list.insert(0, sup_tri_self_last)
            sup_tri_self_last_list.insert(0, sup_tri_self_last)
            if len(sup_tri_self_last_list) >= self.args.num_sample * 2:
                break
        # pdb.set_trace()
        return sup_tri_self_last_list, ent_emb_last_list

    def train(self):
        step = 0
        best_step = 0
        best_eval_rst = {'mrr': 0, 'hits@1': 0, 'hits@5': 0, 'hits@10': 0}
        bad_count = 0
        self.logger.info('start meta-training')
        # pdb.set_trace()
        for e in range(self.args.metatrain_num_epoch):
            for batch in self.train_subgraph_dataloader:
                self.net.train()

                # Clone model
                # old_model_vars = self.net.state_dict()
                # old_opt_vars = self.optimizer.state_dict()
                # net_clone = self.net.clone()
                # optimizer_clone = self.get_optimizer(net_clone, self.optimizer.state_dict())

                batch_loss = []
                batch_sup_g = dgl.batch([get_g_bidir(d[0], self.args) for d in batch]).to(self.args.gpu)
                _, _, loss_vae = self.net.get_ent_emb(batch_sup_g)
                sup_g_list = dgl.unbatch(batch_sup_g)

                # pdb.set_trace()
                sup_tri_self_last_list, ent_emb_last_list = self.get_metaKGs(batch, sup_g_list)

                for batch_i, data in enumerate(batch):
                    # pdb.set_trace()
                    akg_loss = 0
                    ent_emb = sup_g_list[batch_i].ndata['h']
                    ent_emb_vae = sup_g_list[batch_i].ndata['g']

                    # within
                    # pdb.set_trace()
                    que_tri, que_neg_tail_ent, que_neg_head_ent, que_neg_rel = [d.to(self.args.gpu) for d in data[5:]]
                    sup_tri, sup_neg_tail_ent, sup_neg_head_ent, sup_neg_rel, sup_tri_self = [d.to(self.args.gpu) for d in data[:5]]
                    loss_cle_que = self.net.get_loss_cle(que_tri, que_neg_tail_ent, que_neg_head_ent, que_neg_rel, ent_emb+ent_emb_vae)
                    loss_cle_sup = self.net.get_loss_cle(sup_tri, sup_neg_tail_ent, sup_neg_head_ent, sup_neg_rel, ent_emb+ent_emb_vae)
                    akg_loss = akg_loss + loss_cle_que * 0.001 + loss_cle_sup * 0.001
                    # akg_loss = akg_loss + loss_cle_que + loss_cle_sup

                    loss_task_que = self.net.get_loss_task(que_tri, que_neg_tail_ent, que_neg_head_ent, ent_emb + ent_emb_vae)
                    loss_task_sup = self.net.get_loss_task(sup_tri, sup_neg_tail_ent, sup_neg_head_ent, ent_emb + ent_emb_vae)
                    akg_loss = akg_loss + loss_task_que + loss_task_sup

                    # between meta-KGs
                    loss_clm = self.net.get_loss_clm(sup_tri_self, ent_emb, ent_emb_vae, sup_tri_self_last_list, ent_emb_last_list)
                    akg_loss = akg_loss + loss_clm * 0.001
                    if len(sup_tri_self_last_list) >= self.args.num_sample*2:
                        # pdb.set_trace()
                        sup_tri_self_last_list=sup_tri_self_last_list[2:]
                        ent_emb_last_list=ent_emb_last_list[2:]
                    sup_tri_self_last_list.append(sup_tri_self)
                    sup_tri_self_last_list.append(sup_tri_self)
                    ent_emb_last_list.append(ent_emb)
                    ent_emb_last_list.append(ent_emb_vae)

                    batch_loss.append(akg_loss)

                # pdb.set_trace()
                # batch_loss /= len(batch)
                self.optimizer.zero_grad()
                # final_loss = self.net.multi_kg_loss.get_loss(torch.stack(batch_loss))
                final_loss = sum(batch_loss) / len(batch) + loss_vae * 0.001
                final_loss.backward()
                self.optimizer.step()

                # self.optimizer.zero_grad()
                # self.net.point_grad_to(net_clone)
                # self.optimizer.step()
                # self.net.load_state_dict(old_model_vars)
                # self.optimizer.load_state_dict(old_opt_vars)

                step += 1
                self.logger.info('step: {} | loss: {:.4f}'.format(step, final_loss.item()))
                self.write_training_loss(final_loss.item(), step)

                if step % self.args.metatrain_check_per_step == 0:
                    self.net.eval()
                    eval_res = self.evaluate_valid_subgraphs()
                    self.write_evaluation_result(eval_res, step)

                    if eval_res['mrr'] > best_eval_rst['mrr']:
                        best_eval_rst = eval_res
                        best_step = step
                        self.logger.info('best model | mrr {:.4f}'.format(best_eval_rst['mrr']))
                        self.save_checkpoint(step)
                        bad_count = 0
                    else:
                        bad_count += 1
                        self.logger.info('best model is at step {0}, mrr {1:.4f}, bad count {2}'.format(
                            best_step, best_eval_rst['mrr'], bad_count))

        self.logger.info('finish meta-training')
        self.logger.info('save best model')
        self.save_model(best_step)

        self.logger.info('best validation | mrr: {:.4f}, hits@1: {:.4f}, hits@5: {:.4f}, hits@10: {:.4f}'.format(
            best_eval_rst['mrr'], best_eval_rst['hits@1'],
            best_eval_rst['hits@5'], best_eval_rst['hits@10']))

        self.before_test_load()
        self.evaluate_indtest_test_triples(num_cand=50)

    # def evaluate_valid_subgraphs(self):
    #     all_results = ddict(int)
    #     for batch in self.valid_subgraph_dataloader:
    #         for batch_i, data in enumerate(batch):
    #             net_clone = self.net.clone()
    #             optimizer_clone = self.get_optimizer(net_clone, self.optimizer.state_dict())
    #             batch_sup_g = get_g_bidir(data[0], self.args).to(self.args.gpu)
    #             ent_emb = net_clone.get_ent_emb(batch_sup_g)
    #             sup_tri, sup_neg_tail_ent, sup_neg_head_ent, sup_neg_rel, sup_tri_self = [d.to(self.args.gpu) for d in data[2:]]
    #             loss_sup = net_clone.get_loss(sup_tri, sup_neg_tail_ent, sup_neg_head_ent, sup_neg_rel, ent_emb)
    #
    #             optimizer_clone.zero_grad()
    #             loss_sup.backward()
    #             optimizer_clone.step()
    #
    #             batch_sup_g = get_g_bidir(data[0], self.args).to(self.args.gpu)
    #             ent_emb = net_clone.get_ent_emb(batch_sup_g)
    #             que_dataloader = data[1]
    #             results = self.net.evaluate(ent_emb, que_dataloader)
    #
    #             for k, v in results.items():
    #                 all_results[k] += v
    #
    #     for k, v in all_results.items():
    #         all_results[k] = v / self.args.num_valid_subgraph
    #
    #     self.logger.info('valid on valid subgraphs')
    #     self.logger.info('mrr: {:.4f}, hits@1: {:.4f}, hits@5: {:.4f}, hits@10: {:.4f}'.format(
    #         all_results['mrr'], all_results['hits@1'],
    #         all_results['hits@5'], all_results['hits@10']))
    #
    #     return all_results

    def evaluate_valid_subgraphs(self):
        all_results = ddict(int)
        for batch in self.valid_subgraph_dataloader:
            batch_sup_g = dgl.batch([get_g_bidir(d[0], self.args) for d in batch]).to(self.args.gpu)
            self.net.get_ent_emb(batch_sup_g)
            sup_g_list = dgl.unbatch(batch_sup_g)

            for batch_i, data in enumerate(batch):
                que_dataloader = data[1]
                ent_emb = sup_g_list[batch_i].ndata['h']
                ent_emb_vae = sup_g_list[batch_i].ndata['g']

                # ent_emb_vae, mu, log_var = self.net.vae(ent_emb)

                results = self.net.evaluate(ent_emb+ent_emb_vae, que_dataloader)

                for k, v in results.items():
                    all_results[k] += v

        for k, v in all_results.items():
            all_results[k] = v / self.args.num_valid_subgraph

        self.logger.info('valid on valid subgraphs')
        self.logger.info('mrr: {:.4f}, hits@1: {:.4f}, hits@5: {:.4f}, hits@10: {:.4f}'.format(
            all_results['mrr'], all_results['hits@1'],
            all_results['hits@5'], all_results['hits@10']))

        return all_results

    def write_training_loss(self, loss, step):
        self.writer.add_scalar("training/loss", loss, step)

    def write_evaluation_result(self, results, e):
        self.writer.add_scalar("evaluation/mrr", results['mrr'], e)
        self.writer.add_scalar("evaluation/hits10", results['hits@10'], e)
        self.writer.add_scalar("evaluation/hits5", results['hits@5'], e)
        self.writer.add_scalar("evaluation/hits1", results['hits@1'], e)

    def save_checkpoint(self, step):
        # state = {'ent_init': self.ent_init.state_dict(),
        #          'rgcn': self.rgcn.state_dict(),
        #          'kge_model': self.kge_model.state_dict()}
        # delete previous checkpoint
        for filename in os.listdir(self.state_path):
            if self.name in filename.split('.') and os.path.isfile(os.path.join(self.state_path, filename)):
                os.remove(os.path.join(self.state_path, filename))
        # save checkpoint
        torch.save(self.net.state_dict(), os.path.join(self.args.state_dir, self.name,
                                       self.name + '.' + str(step) + '.ckpt'))

    def save_model(self, best_step):
        os.rename(os.path.join(self.state_path, self.name + '.' + str(best_step) + '.ckpt'),
                  os.path.join(self.state_path, self.name + '.best'))

    def before_test_load(self):
        state = torch.load(os.path.join(self.state_path, self.name + '.best'), map_location=self.args.gpu)
        self.net.load_state_dict(state)
        # self.ent_init.load_state_dict(state['ent_init'])
        # self.rgcn.load_state_dict(state['rgcn'])
        # self.kge_model.load_state_dict(state['kge_model'])

    def evaluate_indtest_test_triples(self, num_cand='all'):
        """do evaluation on test triples of ind-test-graph"""
        ent_emb, ent_emb_vae, _ = self.net.get_ent_emb(self.indtest_train_g)

        # ent_emb_vae, mu, log_var = self.net.vae(ent_emb)

        results = self.net.evaluate(ent_emb+ent_emb_vae, self.indtest_test_dataloader, num_cand=num_cand)

        self.logger.info(f'test on ind-test-graph, sample {num_cand}')
        self.logger.info('mrr: {:.4f}, hits@1: {:.4f}, hits@5: {:.4f}, hits@10: {:.4f}'.format(
            results['mrr'], results['hits@1'],
            results['hits@5'], results['hits@10']))

        return results
