import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import math
import random
import torch
import numpy as np
from tqdm import tqdm
from time import time
from prettytable import PrettyTable
from utils.parser import parse_args
from utils.data_loader import load_data
from modules.HAKG import Recommender
from utils.evaluate import test
from utils.helper import early_stopping
import torch.nn as nn

n_users = 0
n_items = 0
n_entities = 0
n_nodes = 0
n_relations = 0


def get_feed_data(train_entity_pairs, train_user_set, args, n_items):
    """生成训练数据的负采样"""
    def negative_sampling(user_item, train_user_set):
        neg_items = list()
        for user, _ in user_item.cpu().numpy():
            user = int(user)
            each_negs = list()
            neg_item = np.random.randint(low=0, high=n_items, size=args.num_neg_sample)
            if len(set(neg_item) & set(train_user_set[user]))==0:
                each_negs += list(neg_item)
            else:
                neg_item = list(set(neg_item) - set(train_user_set[user]))
                each_negs += neg_item
                while len(each_negs)<args.num_neg_sample:
                    n1 = np.random.randint(low=0, high=n_items, size=1)[0]
                    if n1 not in train_user_set[user]:
                        each_negs += [n1]
            neg_items.append(each_negs)

        return neg_items

    feed_dict = {}
    entity_pairs = train_entity_pairs
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['pos_items'] = entity_pairs[:, 1]
    feed_dict['neg_items'] = torch.LongTensor(negative_sampling(entity_pairs,train_user_set))
    return feed_dict


def train_model(args, seed=2020, verbose=True):
    """
    训练模型的核心函数
    
    Args:
        args: 参数对象
        seed: 随机种子
        verbose: 是否打印详细信息
    
    Returns:
        best_result: 最佳结果字典，包含各指标
    """
    # fix the random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:"+str(args.gpu_id)) if args.cuda else torch.device("cpu")
    if verbose:
        print('device:', device)

    # build dataset
    train_cf, test_cf, user_dict, n_params, graph, mat_list, adj_train = load_data(args)
    adj_mat_list, mean_mat_list = mat_list

    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_entities = n_params['n_entities']
    n_relations = n_params['n_relations']
    n_nodes = n_params['n_nodes']

    # cf data
    train_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf], np.int32))
    test_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in test_cf], np.int32))

    # define model
    model = Recommender(n_params, args, graph, mean_mat_list, adj_train, user_dict['train_item_set']).to(device)
    
    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    cur_best = 0
    best_result = None
    stopping_step = 0
    should_stop = False
    
    # 早停参数
    early_stop_enabled = getattr(args, 'early_stop', True)
    patience = getattr(args, 'patience', 10)

    if verbose:
        print("start training ...")
        if early_stop_enabled:
            print(f"Early stopping enabled: metric=recall@20, patience={patience}")
    
    iter_num = math.ceil(len(train_cf_pairs) / args.batch_size)
    
    for epoch in range(args.epoch):
        torch.cuda.empty_cache()
        if epoch % 20 == 1 or epoch == 0:
            # shuffle training data
            index = np.arange(len(train_cf))
            np.random.shuffle(index)
            train_cf_pairs = train_cf_pairs[index]
            if verbose:
                print("start prepare feed data...")
            all_feed_data = get_feed_data(train_cf_pairs, user_dict['train_user_set'], args, n_items)

        # training
        model.train()
        loss, s, cor_loss = 0, 0, 0
        train_s_t = time()
        
        iterator = tqdm(range(iter_num)) if verbose else range(iter_num)
        for i in iterator:
            batch = dict()
            batch['users'] = all_feed_data['users'][i*args.batch_size:(i+1)*args.batch_size].to(device)
            batch['pos_items'] = all_feed_data['pos_items'][i*args.batch_size:(i+1)*args.batch_size].to(device)
            batch['neg_items'] = all_feed_data['neg_items'][i*args.batch_size:(i+1)*args.batch_size,:].to(device)

            # epoch-based 触发
            batch['do_cl2'] = (i == 0)

            batch_loss = model(batch)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss.item()
            s += args.batch_size
        train_e_t = time()

        if epoch % 5 == 0 or epoch == 1:
            # testing
            model.eval()
            test_s_t = time()
            with torch.no_grad():
                ret = test(model, user_dict, n_params)
            test_e_t = time()

            if verbose:
                train_res = PrettyTable()
                train_res.field_names = ["Epoch", "training time", "tesing time", "Loss", "recall", "ndcg", "precision", "hit_ratio"]
                train_res.add_row(
                    [epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss, ret['recall'], ret['ndcg'], ret['precision'], ret['hit_ratio']]
                )
                print(train_res)
                f = open('./result/{}.txt'.format(args.dataset), 'a+')
                f.write(str(train_res) + '\n')
                f.close()

            # 更新最佳结果（以 recall@20 为早停计数标准）
            if ret['recall'][1] > cur_best:
                cur_best = ret['recall'][1]
                stopping_step = 0  # 重置早停计数器
                best_result = {
                    'epoch': epoch,
                    'recall': ret['recall'],
                    'ndcg': ret['ndcg'],
                    'precision': ret['precision'],
                    'hit_ratio': ret['hit_ratio'],
                    'loss': loss
                }
                
                # save weight
                if args.save:
                    torch.save(model.state_dict(), args.out_dir + 'model_' + args.dataset + '.ckpt')
            else:
                # 没有提升，增加早停计数器
                stopping_step += 1
                if verbose and early_stop_enabled:
                    print(f"  No improvement for {stopping_step}/{patience} evaluation rounds")
            
            # 检查是否触发早停
            if early_stop_enabled and stopping_step >= patience:
                if verbose:
                    print(f"\nEarly stopping triggered at epoch {epoch}!")
                    print(f"Best Recall@20: {cur_best:.4f} at epoch {best_result['epoch']}")
                should_stop = True
                break

        else:
            if verbose:
                print('using time %.4f, training loss at epoch %d: %.4f' % (train_e_t - train_s_t, epoch, loss))
        
        if should_stop:
            break

    return best_result


def print_args(args):
    """训练前打印当前参数信息"""
    print("\n" + "="*60)
    print("Training Arguments")
    print("="*60)
    # 只打印与训练/数据/模型相关的常用参数，避免过长
    keys = ['dataset', 'data_path', 'epoch', 'batch_size', 'lr', 'dim', 'l2',
            'mae_msize', 'angle_loss_w', 'mae_loss_w', 'context_hops', 'margin',
            'num_neg_sample', 'loss_f', 'node_dropout_rate', 'mess_dropout_rate',
            'early_stop', 'patience', 'out_dir', 'save', 'cuda', 'gpu_id']
    for k in keys:
        if hasattr(args, k):
            print(f"  {k}: {getattr(args, k)}")
    print("="*60 + "\n")


if __name__ == '__main__':
    """read args"""
    args = parse_args()
    
    # 训练前打印参数信息
    print_args(args)
    
    # 确保输出目录存在
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs('./result/', exist_ok=True)
    
    # 训练模型
    best_result = train_model(args, seed=2020, verbose=True)
    
    if best_result:
        print("\n" + "="*50)
        print("Best Result:")
        print(f"  Epoch: {best_result['epoch']}")
        print(f"  NDCG: {best_result['ndcg']}")
        print(f"  Recall: {best_result['recall']}")
        print(f"  Precision: {best_result['precision']}")
        print(f"  Hit Ratio: {best_result['hit_ratio']}")
        print("="*50)
