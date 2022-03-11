# Default Hyperparameters
data_names = ["50salads", "HAPT", "GTEA", "mHealth"]
init_ratio_dict = {"50salads": 0.00016, "HAPT": 0.00006, "GTEA": 0.0006, "mHealth": 0.0003, "Breakfast":0.0006}  # set as the number of segments in each dataset
input_length_dict = {"TCN":{"50salads": 512, "HAPT": 512, "GTEA": 128, "SAMSUNG": 128, "HASC_BDD": 512, "Sleep": 1024, "ECG": 512, "mHealth": 256, "Breakfast":512},
                     "MSTCN":{"50salads": 2048, "HAPT": 1024, "GTEA": 1024, "mHealth": 1024, "Breakfast":2048},
                     "SSTCN":{"50salads": 2048, "HAPT": 1024, "GTEA": 1024, "mHealth": 1024, "Breakfast":2048}}  #
num_query_ratio = {"50salads": 0.00075, "HAPT": 0.00025, "GTEA": 0.0066, "SAMSUNG": 0.0005, "HASC_BDD": 0.0005, "Sleep": 0.005, "ECG": 0.0001, "mHealth": 0.00025, "Breakfast":0.0005}  #
max_num_prop_dict = {"50salads": 289 // 2, "HAPT": 716 // 2, "GTEA": 35 // 2, "SAMSUNG": 10 // 2, "HASC_BDD": 370 // 2, "Sleep": 1432 // 2, "ECG": 78 // 2, "mHealth": 2933 // 2, "Breakfast": 354 // 2}  # (mean of segment length)/2
data_epoch_dict = {"50salads":50, "HAPT": 50, "GTEA": 50, "mHealth": 50, "Breakfast":50}
data_batch_dict = {"TCN":{"50salads": 32, "HAPT": 32, "GTEA": 32, "SAMSUNG": 32, "HASC_BDD": 32, "Sleep": 32, "ECG": 32, "mHealth": 32,  "Breakfast": 32},
                   "MSTCN":{"50salads": 1, "HAPT": 8, "GTEA": 1, "mHealth": 2,  "Breakfast": 1},
                   "SSTCN":{"50salads": 1, "HAPT": 8, "GTEA": 1, "mHealth": 2,  "Breakfast": 1}} # file-level training
# uncertainty_dict = {"50salads": "margin", "HAPT": "margin", "GTEA": "margin", "mHealth": "margin", "Breakfast":"margin"}
lr_dict = {"TCN": {"50salads": 0.001, "HAPT": 0.0005, "GTEA": 0.001, "SAMSUNG": 0.001, "HASC_BDD": 0.001,
                   "Sleep": 0.001, "ECG": 0.001, "mHealth": 0.0005, "Breakfast": 0.001},
           "MSTCN": {"50salads": 0.0005, "HAPT": 0.0005, "GTEA": 0.0005, "mHealth": 0.0005, "Breakfast":0.0005},
           "SSTCN": {"50salads": 0.001, "HAPT": 0.001, "GTEA": 0.001, "mHealth": 0.001, "Breakfast":0.001}}
background_class_dict = {"50salads":[17,18], "HAPT": [], "GTEA": [10], "mHealth": [], }
import argparse

parser = argparse.ArgumentParser(description='parameters for TSAL')
parser.add_argument('--dataset', type=str, default='HAPT', help='dataset name')
parser.add_argument('--model', type=str, default='MSTCN', help='model name')
parser.add_argument('--gpu', type=str, default="0", help='gpu number')
parser.add_argument('--seed', type=int, default=0, help='experiment seed')
parser.add_argument('--lp', type=str, default="platprob", help='label propagation method')
parser.add_argument('--al', type=str, default="random", help='active learning method')
parser.add_argument('--eta', type=float, default=0.8, help='prob/repr hyperparameter')
parser.add_argument('--tau', type=int, default=10, help='plateau propagation hyperparameter')
parser.add_argument('--num_query', type=int, default=15, help='number of query')
parser.add_argument('--no_plat_reg', type=int, default=0, help='whether or not delete regularization on plateau')
parser.add_argument('--temp', type=float, default=2, help='temperature scaling factor')
# parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
# parser.add_argument('--length', type=int, default=512, help='input length')


args = parser.parse_args()

NAME = args.dataset
MODEL = args.model
GPU = args.gpu
SEED = args.seed
LP = args.lp
AL = args.al
ETA = args.eta
TAU = args.tau
NUMQUERY = args.num_query
NOPLATREG = args.no_plat_reg
TEMP = args.temp
if LP == "prob":
    TAU = 3
    print("TAU = 3 due to LP == prob")
if NAME == "GTEA":
    TAU = 5
    print("TAU = 5 due to NAME == GTEA")

# LR = args.lr
# LENGTH = args.length
import warnings
warnings.filterwarnings('ignore')
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
np.set_printoptions(precision=4)

from TSAL import TSAL
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass
tf.random.set_seed(SEED)
lp_methods_dict = {"tenth":int(np.maximum(int(0.1 * max_num_prop_dict[NAME]), 1)), "true":"true",
                   "full":int(max_num_prop_dict[NAME]), "zero":0, "cls":"cls", "platprob":"platprob",
                   "platrepr":"platrepr","prob":"prob", "repr":"repr", "abs":"abs"}
def main():

    if not (NAME in data_names):
        print("Wrong data name, select from the following data list: ", data_names)

    lp_methods = ["zero","tenth","full","cls","true","platprob","platrepr","prob","repr","abs"]
    al_list = ["random","margin","entropy","conf","utility","badge", "llal", "core"]
    if not (LP in lp_methods):
        print("Wrong LP name, ", LP,": select from the following LP list: ", lp_methods)
    if not (AL in al_list):
        print("Wrong AL name, ", AL,": select from the following al list: ", al_list)


    lp_method = lp_methods_dict[LP]
    tsal = TSAL(data_name=NAME, model_name=MODEL,input_length=input_length_dict[MODEL][NAME], init_ratio=init_ratio_dict[NAME], seed=SEED,
                total_num_query_step=NUMQUERY, num_epoch=data_epoch_dict[NAME], batch_size=data_batch_dict[MODEL][NAME],
                max_num_prop=max_num_prop_dict[NAME], tau=TAU, lr=lr_dict[MODEL][NAME], bg_class=background_class_dict[NAME],
                al_name=AL, is_label_propagation=lp_method, no_plat_reg=NOPLATREG, temp=TEMP)
    result = tsal.doAL(num_query_ratio=num_query_ratio[NAME], eta=ETA)  #
    if NOPLATREG==1:
        np.save(os.path.join(os.getcwd(), "metadata", f"{NAME}_{MODEL}_no{LP}_{AL}_{ETA}_{TAU}_{TEMP}_{SEED}.npy"), result)
    else:
        np.save(os.path.join(os.getcwd(), "metadata", f"{NAME}_{MODEL}_{LP}_{AL}_{ETA}_{TAU}_{TEMP}_{SEED}.npy"), result)


if __name__ == "__main__":
    print(f"DATA: {NAME}\nMODEL: {MODEL}\nLR: {lr_dict[MODEL][NAME]}\nBATCHSIZE: {data_batch_dict[MODEL][NAME]}\n"
          f"NUMEPOCH: {data_epoch_dict[NAME]}\nLP: {LP}\nAL: {AL}\nETA: {ETA}\nTAU: {TAU}\nSEED: {SEED}")
    main()

