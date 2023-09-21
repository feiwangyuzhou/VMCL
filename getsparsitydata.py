import pickle
import numpy as np
import pdb

def reidx(tri):
    tri_reidx = []
    ent_reidx = dict()
    reidx_ent = dict()
    entidx = 0
    # ent_reidx['superent'] = entidx
    # entidx += 1
    # pdb.set_trace()
    rel_reidx = dict()
    reidx_rel = dict()
    relidx = 0
    # rel_reidx['self'] = relidx
    # relidx += 1
    # rel_reidx['superrel'] = relidx
    # relidx += 1
    degree_rel = {}
    for h, r, t in tri:
        if h not in ent_reidx.keys():
            ent_reidx[h] = entidx
            reidx_ent[entidx] = h
            entidx += 1
        if t not in ent_reidx.keys():
            ent_reidx[t] = entidx
            reidx_ent[entidx] = t
            entidx += 1
        if r not in rel_reidx.keys():
            rel_reidx[r] = relidx
            reidx_rel[relidx] = r
            degree_rel[relidx] = 1
            relidx += 1
        else:
            degree_rel[rel_reidx[r]] += 1
        tri_reidx.append((ent_reidx[h], rel_reidx[r], ent_reidx[t]))
    # pdb.set_trace()
    # for h in ent_reidx.keys():
    #     tri_reidx.append([ent_reidx[h], rel_reidx['self'], ent_reidx[h]])

    return tri_reidx, dict(rel_reidx), dict(ent_reidx), degree_rel, reidx_ent, reidx_rel


def reidx_withr(tri, rel_reidx):
    tri_reidx = []
    ent_reidx = dict()
    entidx = 0
    for h, r, t in tri:
        if h not in ent_reidx.keys():
            ent_reidx[h] = entidx
            entidx += 1
        if t not in ent_reidx.keys():
            ent_reidx[t] = entidx
            entidx += 1
        tri_reidx.append([ent_reidx[h], rel_reidx[r], ent_reidx[t]])
    # pdb.set_trace()
    # for h in ent_reidx.keys():
    #     tri_reidx.append([ent_reidx[h], rel_reidx['self'], ent_reidx[h]])

    return tri_reidx, dict(ent_reidx)


def reidx_withr_ande(tri, rel_reidx, ent_reidx):
    tri_reidx = []
    for h, r, t in tri:
        tri_reidx.append([ent_reidx[h], rel_reidx[r], ent_reidx[t]])
    return tri_reidx


def sourceKG(data_name):
    # pdb.set_trace()
    train_tri = []
    file = open('./data/{}/train.txt'.format(data_name))
    train_tri.extend([l.strip().split() for l in file.readlines()])
    file.close()

    train_tri, fix_rel_reidx, ent_reidx, degree_rel, reidx_ent, reidx_rel = reidx(train_tri)
    np.random.shuffle(train_tri)
    num_all = len(train_tri)
    delete_20 = []
    delete_40 = []
    delete_60 = []
    delete_80 = []
    # pdb.set_trace()
    for tri in train_tri:
        if degree_rel[tri[1]] > 1:
            if len(delete_20) < num_all * 0.2:
                delete_20.append((tri[0], tri[1], tri[2]))
            if len(delete_40) < num_all * 0.4:
                delete_40.append((tri[0], tri[1], tri[2]))
            if len(delete_60) < num_all * 0.6:
                delete_60.append((tri[0], tri[1], tri[2]))
            if len(delete_80) < num_all * 0.8:
                delete_80.append((tri[0], tri[1], tri[2]))
            else:
                break
            degree_rel[tri[1]] -=1

    # pdb.set_trace()
    train_tri_20 = set(train_tri)-set(delete_80)
    train_tri_40 = set(train_tri)-set(delete_60)
    train_tri_60 = set(train_tri)-set(delete_40)
    train_tri_80 = set(train_tri)-set(delete_20)


    triples_path_w = './data/{}/train_20.txt'.format(data_name)
    with open(triples_path_w, "w") as f_w:
        for tri in train_tri_20:
            f_w.write(str(reidx_ent[tri[0]]) + " " + str(reidx_rel[tri[1]]) + " " + str(reidx_ent[tri[2]]) + "\n")
    triples_path_w = './data/{}/train_40.txt'.format(data_name)
    with open(triples_path_w, "w") as f_w:
        for tri in train_tri_40:
            f_w.write(str(reidx_ent[tri[0]]) + " " + str(reidx_rel[tri[1]]) + " " + str(reidx_ent[tri[2]]) + "\n")
    triples_path_w = './data/{}/train_60.txt'.format(data_name)
    with open(triples_path_w, "w") as f_w:
        for tri in train_tri_60:
            f_w.write(str(reidx_ent[tri[0]]) + " " + str(reidx_rel[tri[1]]) + " " + str(reidx_ent[tri[2]]) + "\n")
    triples_path_w = './data/{}/train_80.txt'.format(data_name)
    with open(triples_path_w, "w") as f_w:
        for tri in train_tri_80:
            f_w.write(str(reidx_ent[tri[0]]) + " " + str(reidx_rel[tri[1]]) + " " + str(reidx_ent[tri[2]]) + "\n")







def targetKG(data_name):

    file = open('./data/{}_ind/train.txt'.format(data_name))
    ind_train_tri = ([l.strip().split() for l in file.readlines()])
    file.close()

    np.random.shuffle(ind_train_tri)
    num_all = len(ind_train_tri)
    # pdb.set_trace()
    train_tri_20 = ind_train_tri[int(num_all*0.8):]
    train_tri_40 = ind_train_tri[int(num_all*0.6):]
    train_tri_60 = ind_train_tri[int(num_all*0.4):]
    train_tri_80 = ind_train_tri[int(num_all*0.2):]

    triples_path_w = './data/{}_ind/train_20.txt'.format(data_name)
    with open(triples_path_w, "w") as f_w:
        for tri in train_tri_20:
            f_w.write(tri[0] + " " + tri[1] + " " + tri[2] + "\n")
    triples_path_w = './data/{}_ind/train_40.txt'.format(data_name)
    with open(triples_path_w, "w") as f_w:
        for tri in train_tri_40:
            f_w.write(tri[0] + " " + tri[1] + " " + tri[2] + "\n")
    triples_path_w = './data/{}_ind/train_60.txt'.format(data_name)
    with open(triples_path_w, "w") as f_w:
        for tri in train_tri_60:
            f_w.write(tri[0] + " " + tri[1] + " " + tri[2] + "\n")
    triples_path_w = './data/{}_ind/train_80.txt'.format(data_name)
    with open(triples_path_w, "w") as f_w:
        for tri in train_tri_80:
            f_w.write(tri[0] + " " + tri[1] + " " + tri[2] + "\n")


if __name__ == '__main__':
    targetKG('nell_v1')
    targetKG('nell_v2')
    targetKG('nell_v3')
    targetKG('nell_v4')
