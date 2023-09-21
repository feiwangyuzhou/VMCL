import pickle
import torch
import numpy as np
from collections import defaultdict as ddict
import lmdb
from tqdm import tqdm
import random
from utils import serialize, get_g, get_hr2t_rt2h_sup_que
import dgl
import pdb

def gen_subgraph_datasets(args):
    # pdb.set_trace()
    print(f'-----There is no sub-graphs for {args.data_name}, so start generating sub-graphs before meta-training!-----')
    data = pickle.load(open(args.data_path, 'rb'))
    train_g = get_g(data['train_graph']['train'] + data['train_graph']['valid']
                    + data['train_graph']['test'])
    # pdb.set_trace()
    nrel = data['train_graph']['nrel']

    BYTES_PER_DATUM = get_average_subgraph_size(args, args.num_sample_for_estimate_size, train_g) * 2
    map_size = (args.num_train_subgraph + args.num_valid_subgraph) * BYTES_PER_DATUM
    env = lmdb.open(args.db_path, map_size=map_size, max_dbs=2)
    train_subgraphs_db = env.open_db("train_subgraphs".encode())
    valid_subgraphs_db = env.open_db("valid_subgraphs".encode())

    for idx in tqdm(range(args.num_train_subgraph)):
        str_id = '{:08}'.format(idx).encode('ascii')
        datum = sample_one_subgraph(args, train_g, nrel, Flag=False)
        with env.begin(write=True, db=train_subgraphs_db) as txn:
            txn.put(str_id, serialize(datum))

    for idx in tqdm(range(args.num_valid_subgraph)):
        str_id = '{:08}'.format(idx).encode('ascii')
        datum = sample_one_subgraph(args, train_g, nrel)
        with env.begin(write=True, db=valid_subgraphs_db) as txn:
            txn.put(str_id, serialize(datum))


def sample_one_subgraph(args, bg_train_g, nrel=None, Flag=True):
    # get graph with bi-direction
    bg_train_g_undir = dgl.graph((torch.cat([bg_train_g.edges()[0], bg_train_g.edges()[1]]),
                                  torch.cat([bg_train_g.edges()[1], bg_train_g.edges()[0]])))

    sel_nodes = None
    # induce sub-graph by sampled nodes
    while True:
        while True:
            # pdb.set_trace()
            if sel_nodes is None:
                sel_nodes = []
            cand_nodes = np.arange(bg_train_g.num_nodes())
            for i in range(args.rw_0):
                rw, _ = dgl.sampling.random_walk(bg_train_g_undir,
                                                 np.random.choice(cand_nodes, 1, replace=False).repeat(args.rw_1),
                                                 length=args.rw_2)
                sel_nodes.extend(np.unique(rw.reshape(-1)))
                sel_nodes = list(np.unique(sel_nodes)) if -1 not in sel_nodes else list(np.unique(sel_nodes))[1:]
                cand_nodes = sel_nodes
            sub_g = dgl.node_subgraph(bg_train_g, sel_nodes)

            if sub_g.num_nodes() >= 50:
                # pdb.set_trace()
                break
            # else:
            #     pdb.set_trace()

        sub_tri = torch.stack([sub_g.edges()[0],
                               sub_g.edata['rel'],
                               sub_g.edges()[1]])
        sub_tri = sub_tri.T.tolist()
        random.shuffle(sub_tri)

        ent_freq = ddict(int)
        rel_freq = ddict(int)
        triples_reidx = []
        ent_reidx = dict()
        entidx = 0
        sup_tri_self = []
        for tri in sub_tri:
            h, r, t = tri
            if h not in ent_reidx.keys():
                ent_reidx[h] = entidx
                entidx += 1
            if t not in ent_reidx.keys():
                ent_reidx[t] = entidx
                entidx += 1
            ent_freq[ent_reidx[h]] += 1
            ent_freq[ent_reidx[t]] += 1
            rel_freq[r] += 1
            triples_reidx.append([ent_reidx[h], r, ent_reidx[t]])

            if r==0:
                # pdb.set_trace()
                sup_tri_self.append([ent_reidx[h], r, ent_reidx[t]])
                if h!=t:
                    print("error")
                    pdb.set_trace()

        # if Flag is True: # for valid
        # randomly get query triples
        que_tris = []
        sup_tris = []
        for idx, tri in enumerate(triples_reidx):
            h, r, t = tri
            if ent_freq[h] > 2 and ent_freq[t] > 2 and rel_freq[r] > 2:
                que_tris.append(tri)
                ent_freq[h] -= 1
                ent_freq[t] -= 1
                rel_freq[r] -= 1
            else:
                sup_tris.append(tri)

            if len(que_tris) >= int(len(triples_reidx)*0.1):
                break

        sup_tris.extend(triples_reidx[idx+1:])

        if len(que_tris) >= int(len(triples_reidx)*0.05):
            break
        # else: # for train, do not need query
        #     que_tris = []
        #     sup_tris = []
        #     sup_tris.extend(triples_reidx)
        #     break

    # hr2t, rt2h
    hr2t, rt2h, ht2r = get_hr2t_rt2h_sup_que(sup_tris, que_tris)
    # pdb.set_trace()

    return sup_tris, sup_tri_self, que_tris, hr2t, rt2h, ht2r, nrel


def get_average_subgraph_size(args, sample_size, bg_train_g):
    total_size = 0
    for i in range(sample_size):
        datum = sample_one_subgraph(args, bg_train_g)
        total_size += len(serialize(datum))
    return total_size / sample_size