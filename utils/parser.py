import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="HAKG")

    # ===== dataset ===== #
    parser.add_argument("--dataset", nargs="?", default="mooccube_rel8", help="Choose a dataset:[mooper_fil25_entity4,mooccube-KGAN,yelp2018,mooccube_rel4,mooccube_18_1_1_to_4_1,]")
    parser.add_argument(
        "--data_path", nargs="?", default="data/", help="Input data path."
    )

    # ===== train ===== #
    parser.add_argument('--epoch', type=int, default=1000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=4096, help='batch size')
    parser.add_argument('--test_batch_size', type=int, default=2048, help='test batch size')
    parser.add_argument('--dim', type=int, default=64, help='embedding size')
    parser.add_argument('--l2', type=float, default=1e-5, help='l2 regularization weight')
    parser.add_argument('--angle_loss_w', type=float, default=0.005, help='angle loss weight')
    parser.add_argument('--mae_loss_w', type=float, default=0.1, help='mae loss weight')
    parser.add_argument('--mae_msize', type=int, default=64, help='knowledge-aware MAE top-k mask size')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument("--inverse_r", type=bool, default=True, help="consider inverse relation or not")
    parser.add_argument("--node_dropout", type=bool, default=True, help="consider node dropout or not")
    parser.add_argument("--node_dropout_rate", type=float, default=0.5, help="ratio of node dropout")
    parser.add_argument("--mess_dropout", type=bool, default=True, help="consider message dropout or not")
    parser.add_argument("--mess_dropout_rate", type=float, default=0.1, help="ratio of node dropout")
    parser.add_argument("--batch_test_flag", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--channel", type=int, default=2, help="hidden channels for model")
    parser.add_argument("--cuda", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
    parser.add_argument('--Ks', nargs='?', default='[10, 20]', help='Output sizes of every layer')
    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')
    
    parser.add_argument('--lambda1', default=0.2, type=float, help='weight of cl loss')
    parser.add_argument('--temp', default=0.2, type=float, help='temperature in cl loss')
    parser.add_argument('--q', default=5, type=int, help='rank')
    parser.add_argument('--emb', default='16_64_0.1', type=str, help='rank')
    

    # ===== relation context ===== #
    parser.add_argument('--context_hops', type=int, default=3, help='number of context hops')

    parser.add_argument('--num_neg_sample', type=int, default=50, help='the number of negative sample')
    parser.add_argument('--margin', type=float, default=0.9, help='the margin of contrastive_loss')
    parser.add_argument('--loss_f', nargs="?", default="contrastive_loss",
                        help="Choose a loss function:[inner_bpr, dis_bpr, contrastive_loss]")

    # ===== save model ===== #
    parser.add_argument("--save", type=bool, default=True, help="save model or not")
    parser.add_argument("--out_dir", type=str, default="./model_para/", help="output directory for model")
    parser.add_argument("--ckpt", type=str, default=None, help="path to checkpoint (for case study / eval)")
    parser.add_argument("--user_id", type=int, default=None, help="user id for case study")
    parser.add_argument("--course_for_triplet", type=int, default=None, help="item id for triplet score analysis in case study")

    # ===== early stopping ===== #
    parser.add_argument('--early_stop', action='store_true', help='enable early stopping')
    parser.add_argument('--patience', type=int, default=20, 
                        help='number of evaluation rounds without improvement before stopping (each round = 2 epochs)')

    # ===== grid search ===== #
    parser.add_argument('--grid_search', action='store_true', help='enable grid search mode')
    parser.add_argument('--batch_size_list', type=str, default='4096', 
                        help='comma-separated list of batch sizes for grid search')
    parser.add_argument('--lr_list', type=str, default='0.0001', 
                        help='comma-separated list of learning rates for grid search')
    parser.add_argument('--angle_loss_w_list', type=str, default='0.005', 
                        help='comma-separated list of angle loss weights for grid search')
    parser.add_argument('--mae_loss_w_list', type=str, default='0.1', 
                        help='comma-separated list of mae loss weights for grid search')
    parser.add_argument('--mae_msize_list', type=str, default='256', 
                        help='comma-separated list of mae mask sizes for grid search (e.g. 64,128,256,512,1024)')
    parser.add_argument('--margin_list', type=str, default='0.7,0.9,1.2', 
                        help='comma-separated list of margins for grid search')
    parser.add_argument('--context_hops_list', type=str, default='3,4', 
                        help='comma-separated list of context hops for grid search')
    parser.add_argument('--grid_search_metric', type=str, default='recall', 
                        choices=['ndcg', 'recall', 'precision', 'hit_ratio'],
                        help='metric to optimize in grid search')
    parser.add_argument('--grid_search_output', type=str, default='./grid_search_results/', 
                        help='output directory for grid search results')

    return parser.parse_args()


def parse_grid_search_params(args):
    """解析网格搜索参数，返回参数网格字典"""
    grid_params = {}
    
    # 解析各参数列表
    grid_params['batch_size'] = [int(x) for x in args.batch_size_list.split(',')]
    grid_params['lr'] = [float(x) for x in args.lr_list.split(',')]
    grid_params['angle_loss_w'] = [float(x) for x in args.angle_loss_w_list.split(',')]
    grid_params['mae_loss_w'] = [float(x) for x in args.mae_loss_w_list.split(',')]
    grid_params['mae_msize'] = [int(x) for x in args.mae_msize_list.split(',')]
    grid_params['margin'] = [float(x) for x in args.margin_list.split(',')]
    grid_params['context_hops'] = [int(x) for x in args.context_hops_list.split(',')]
    
    return grid_params
