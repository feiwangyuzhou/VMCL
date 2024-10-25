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


class TestTrainer(nn.Module):
    def __init__(self, args):
        super(TestTrainer, self).__init__()
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

        # self.load_metatrain()

    def load_metatrain(self):
        state = torch.load(self.args.metatrain_state, map_location=self.args.gpu)
        self.net.load_state_dict(state)
        # self.ent_init.load_state_dict(state['ent_init'])
        # self.rgcn.load_state_dict(state['rgcn'])
        # self.kge_model.load_state_dict(state['kge_model'])

    def get_ent_emb(self, sup_g_bidir):
        self.net.ent_init(sup_g_bidir)
        ent_emb = self.net.rgcn(sup_g_bidir)

        return ent_emb

    def train(self):
        self.logger.info('prediction')

        self.before_test_load()

        self.evaluate_indtest_test_triples(num_cand=50)

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
        ent_emb = self.net.get_ent_emb(self.indtest_train_g)

        ent_emb_vae, mu, log_var = self.net.vae(ent_emb)

        results = self.net.evaluate(ent_emb+ent_emb_vae, self.valid_dataloader, num_cand)

        self.logger.info('valid on ind-test-graph')
        self.logger.info('mrr: {:.4f}, hits@1: {:.4f}, hits@5: {:.4f}, hits@10: {:.4f}'.format(
            results['mrr'], results['hits@1'],
            results['hits@5'], results['hits@10']))

        return results

    def evaluate_indtest_test_triples(self, num_cand='all', epoch=None):
        """do evaluation on test triples of ind-test-graph"""
        ent_emb = self.net.get_ent_emb(self.indtest_train_g)

        ent_emb_vae, mu, log_var = self.net.vae(ent_emb)

        results = self.net.evaluate(ent_emb+ent_emb_vae, self.indtest_test_dataloader, num_cand=num_cand)

        # self.logger.info(f'test on ind-test-graph, sample {num_cand}')
        self.logger.info('epoch: {:}, mrr: {:.4f}, hits@1: {:.4f}, hits@5: {:.4f}, hits@10: {:.4f}'.format(epoch,
            results['mrr'], results['hits@1'],
            results['hits@5'], results['hits@10']))

        self.logger_final.info('name: {:}, {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(self.name,
            results['mrr'], results['hits@1'],
            results['hits@5'], results['hits@10']))

        return results
