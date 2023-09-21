import torch
from torch import optim
from torch import nn
import numpy as np
from utils import get_posttrain_train_valid_dataset, get_indtest_test_dataset_and_train_g
from torch.utils.data import DataLoader
from datasets import KGETrainDataset, KGEEvalDataset
from trainer import Trainer
from torch.utils.tensorboard import SummaryWriter
from utils import Log
import os
import json
import pdb


class PostTrainer(nn.Module):
    def __init__(self, args):
        super(PostTrainer, self).__init__()
        self.args = args
        # dataloader
        train_dataset, valid_dataset = get_posttrain_train_valid_dataset(args)
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.args.posttrain_bs,
                                      collate_fn=KGETrainDataset.collate_fn)
        self.valid_dataloader = DataLoader(valid_dataset, batch_size=args.indtest_eval_bs,
                                      collate_fn=KGEEvalDataset.collate_fn)

        indtest_test_dataset, indtest_train_g = get_indtest_test_dataset_and_train_g(args)
        self.indtest_train_g = indtest_train_g.to(args.gpu)
        self.indtest_test_dataloader = DataLoader(indtest_test_dataset, batch_size=args.indtest_eval_bs,
                                                  shuffle=False, collate_fn=KGEEvalDataset.collate_fn)

        # writer and logger
        self.name = args.name
        self.writer = SummaryWriter(os.path.join(args.tb_log_dir, self.name))
        self.logger_final = Log(args.log_dir, "final").get_logger()
        self.logger = Log(args.log_dir, self.name).get_logger()
        self.logger.info(json.dumps(vars(args)))
        # state dir
        self.state_path = os.path.join(args.state_dir, self.name)
        if not os.path.exists(self.state_path):
            os.makedirs(self.state_path)

        self.net = Trainer(args)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.args.posttrain_lr)
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.9, patience=10)

        self.load_metatrain()

    def load_metatrain(self):
        state = torch.load(self.args.metatrain_state, map_location=self.args.gpu)
        self.net.load_state_dict(state)
        # self.ent_init.load_state_dict(state['ent_init'])
        # self.rgcn.load_state_dict(state['rgcn'])
        # self.kge_model.load_state_dict(state['kge_model'])

    # def get_ent_emb(self, sup_g_bidir):
    #     self.net.ent_init(sup_g_bidir)
    #     ent_emb = self.net.rgcn(sup_g_bidir)
    #
    #     return ent_emb

    def train(self):
        self.logger.info('start fine-tuning')

        # print epoch test rst
        # self.evaluate_indtest_test_triples(num_cand=50)

        # self.net.eval()
        eval_res = self.evaluate_indtest_valid_triples()
        self.write_evaluation_result(eval_res, 0)

        best_step = 0
        best_eval_rst = eval_res
        self.save_checkpoint(0)
        bad_count = 0

        for i in range(1, self.args.posttrain_num_epoch + 1):
            losses = []
            # pdb.set_trace()
            self.net.train()
            for batch in self.train_dataloader:

                pos_triple, neg_tail_ent, neg_head_ent, neg_rel = [b.to(self.args.gpu) for b in batch]
                ent_emb, ent_emb_vae, loss_vae = self.net.get_ent_emb(self.indtest_train_g)
                # pdb.set_trace()
                # ent_emb = self.get_ent_emb(self.indtest_train_g)
                # ent_emb_vae, mu, log_var = self.net.vae(ent_emb)
                # loss_vae = self.net.vae.vae_loss_function(ent_emb_vae, ent_emb, mu, log_var)

                loss = self.net.get_loss_task(pos_triple, neg_tail_ent, neg_head_ent, ent_emb+ent_emb_vae)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                losses.append(loss.item())

            self.logger.info('epoch: {} | loss: {:.4f}'.format(i, np.mean(losses)))

            # if i % self.args.posttrain_check_per_epoch == 0:
            #     self.evaluate_indtest_test_triples(num_cand=50, epoch=i)
            if i % self.args.posttrain_check_per_epoch == 0:
                # self.net.eval()
                eval_res = self.evaluate_indtest_valid_triples()
                self.write_evaluation_result(eval_res, i)

                if eval_res['mrr'] > best_eval_rst['mrr']:
                    bad_count = 0
                # elif (eval_res['hits@1'] == best_eval_rst['hits@1']) and (eval_res['hits@10'] > best_eval_rst['hits@10']):
                #     bad_count = 0
                else:
                    bad_count += 1

                if bad_count == 0:
                    best_eval_rst = eval_res
                    best_step = i
                    self.logger.info('best model | mrr {:.4f}'.format(best_eval_rst['mrr']))
                    self.save_checkpoint(i)
                else:
                    self.logger.info('best model is at step {0}, mrr {1:.4f}, bad count {2}'.format(
                        best_step, best_eval_rst['mrr'], bad_count))

                # self.scheduler.step(best_step)

        self.logger.info('finish meta-training')
        self.logger.info('save best model')
        self.save_model(best_step)

        self.logger.info('best validation | mrr: {:.4f}, hits@1: {:.4f}, hits@5: {:.4f}, hits@10: {:.4f}'.format(
            best_eval_rst['mrr'], best_eval_rst['hits@1'],
            best_eval_rst['hits@5'], best_eval_rst['hits@10']))

        self.before_test_load()

        self.evaluate_indtest_test_triples(num_cand=50, epoch=best_step)

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



    def evaluate_indtest_valid_triples(self, num_cand='all'):
        ent_emb, ent_emb_vae, loss_vae = self.net.get_ent_emb(self.indtest_train_g)

        # ent_emb_vae, mu, log_var = self.net.vae(ent_emb)

        results = self.net.evaluate(ent_emb+ent_emb_vae, self.valid_dataloader, num_cand)

        self.logger.info('valid on ind-test-graph')
        self.logger.info('mrr: {:.4f}, hits@1: {:.4f}, hits@5: {:.4f}, hits@10: {:.4f}'.format(
            results['mrr'], results['hits@1'],
            results['hits@5'], results['hits@10']))

        return results

    def evaluate_indtest_test_triples(self, num_cand='all', epoch=None):
        """do evaluation on test triples of ind-test-graph"""
        ent_emb, ent_emb_vae, loss_vae = self.net.get_ent_emb(self.indtest_train_g)

        # ent_emb_vae, mu, log_var = self.net.vae(ent_emb)

        results = self.net.evaluate(ent_emb+ent_emb_vae, self.indtest_test_dataloader, num_cand=num_cand)

        # self.logger.info(f'test on ind-test-graph, sample {num_cand}')
        self.logger.info('epoch: {:}, mrr: {:.4f}, hits@1: {:.4f}, hits@5: {:.4f}, hits@10: {:.4f}'.format(epoch,
            results['mrr'], results['hits@1'],
            results['hits@5'], results['hits@10']))

        self.logger_final.info('name: {:}, {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(self.name,
            results['mrr'], results['hits@1'],
            results['hits@5'], results['hits@10']))

        return results
