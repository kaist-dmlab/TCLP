import os
from tqdm import tqdm
import sys
import numpy as np
import preprocessing
import tensorflow as tf
import model
import data_loader
import pickle
import matplotlib.pyplot as plt
from eval import *

class ModelManager():
    def __init__(self, model_name, input_length, num_class, dim, lr, seed, K=5, is_LLAL=False):
        self.model_name = model_name
        self.input_length = input_length
        self.num_class = num_class
        self.dim = dim
        self.seed = seed
        self.K = K
        self.model_init = 0
        self.is_LLAL = is_LLAL

        if self.model_name == "TCN":
            self.model = model.TCN(self.num_class, self.dim, lr)
        elif self.model_name == "LongTCN":
            self.model = model.TCN(self.num_class, self.dim, lr=lr, num_block=1, num_dilation=15)
        elif self.model_name == "RNN":
            self.model = model.RNN(self.num_class, self.dim, lr=lr)
        elif self.model_name == "MSTCN":
            self.model = model.MSTCN(self.num_class, self.dim, lr=lr, is_LLAL=is_LLAL)
        elif self.model_name == "SSTCN":
            self.model = model.MSTCN(self.num_class, self.dim, lr=lr, num_stage=1, num_dilation=5, is_LLAL=is_LLAL)
        else:
            print("wrong model name in ModelManager")



        if is_LLAL:
            self.model.input_LLAL(np.zeros((1, self.input_length, self.dim)))
            self.model.call_LLAL(np.zeros((1, self.input_length, self.model.num_stage*self.model.num_filters)))
        else:
            self.model(np.zeros((1, self.input_length, self.dim)))
        self.weights_init = self.model.get_weights()

    def train_test_generator(self, X, y, y_seg, mask, file_boundaries):
        assert(self.seed<=self.K-1)
        self.test_data_start = len(X) // self.K * self.seed
        if self.seed == self.K-1:
            self.test_data_end = len(X)
        else:
            self.test_data_end = len(X) // self.K * (self.seed + 1)

        self.X_long_train = np.concatenate([X[:self.test_data_start],X[self.test_data_end:]])
        self.y_long_train = np.concatenate([y[:self.test_data_start],y[self.test_data_end:]])
        self.y_seg_long_train = np.concatenate([y_seg[:self.test_data_start],y_seg[self.test_data_end:]])
        self.mask_long_train = np.concatenate([mask[:self.test_data_start],mask[self.test_data_end:]])
        self.file_boundaries_train = np.concatenate([file_boundaries[:self.test_data_start],file_boundaries[self.test_data_end:]])

        self.X_long_test = X[self.test_data_start:self.test_data_end]
        self.y_long_test = y[self.test_data_start:self.test_data_end]
        self.y_seg_long_test = y_seg[self.test_data_start:self.test_data_end]
        self.mask_long_test = mask[self.test_data_start:self.test_data_end]  # TODO: labeled_or_not index check needed
        self.file_boundaries_test = file_boundaries[self.test_data_start:self.test_data_end]

        print(self.X_long_train.shape, self.y_long_train.shape, self.y_seg_long_train.shape, self.mask_long_train.shape,
              self.file_boundaries_train.shape)

        return self.X_long_train, self.y_long_train, self.y_seg_long_train, self.mask_long_train, self.file_boundaries_train, self.X_long_test, self.y_long_test, self.y_seg_long_test, self.mask_long_test, self.file_boundaries_test


    def load_train_data(self, X, y, y_seg, mask, file_boundaries, fully_supervised=False):
        self.dataloader = data_loader.DataLoader(self. input_length, X, y, y_seg, mask, file_boundaries, self.seed, fully_supervised)

    def get_unlabeled_ECE(self):
        output_final = self.model.predict(X_long=self.X_long_train,file_boundaries=self.file_boundaries_train)

        ECE = 0
        pred = np.argmax(output_final, axis=1)
        conf = np.max(output_final, axis=1)
        prev_i = 0
        for i in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
            if np.sum((conf>prev_i) & (conf<i)) == 0:
                prev_i = i
                continue
            conf_bin = np.mean(conf[(conf>prev_i) & (conf<i)])
            acc_bin = np.sum(pred[(conf>prev_i) & (conf<i)]==self.y_long_train[(conf>prev_i) & (conf<i)])/np.sum((conf>prev_i) & (conf<i))
            ECE += np.sum((conf>prev_i) & (conf<i))/len(conf)*np.abs(acc_bin-conf_bin)
            prev_i = i
        return ECE

    def test_model(self, bg_class = []):
        file_boundary_ind = np.where(self.file_boundaries_test==1)[0].tolist()
        start = 0  # test_data_start_ind
        if len(file_boundary_ind) > 0:
            if not len(self.file_boundaries_test)-1 in file_boundary_ind:
                file_boundary_ind.append(len(self.file_boundaries_test)-1)
            for i in file_boundary_ind:
                output_final_file = self.model(self.X_long_test[np.newaxis, start:i+1]).numpy()[0]
                if start == 0:
                    output_final = output_final_file
                else:
                    output_final = np.concatenate([output_final, output_final_file], axis=0)
                start = i+1
        else:
            output_final = self.model(self.X_long_test[np.newaxis, :, :]).numpy()
        output_final = output_final.reshape((-1, self.num_class))
        y_test_flatten = tf.reshape(self.y_long_test, [-1]).numpy()
        if len(output_final)!=len(y_test_flatten):
            print(self.y_long_test.shape, self.X_long_test.shape, len(self.file_boundaries_test), file_boundary_ind, len(output_final), len(y_test_flatten))
            print("shapes are different when testing")


        return get_all_metrics(np.argmax(output_final,axis=1),y_test_flatten,bg_class=bg_class)

    def train_model_new(self, epoch, batch_size, is_test=False):
        self.model.set_weights(self.weights_init)
        prob, indice_list = self.dataloader.window_scoring(slide_size=self.input_length)
        num_steps_per_epoch = len(prob) // batch_size # epoch is set by # of proped+queried labels
        train_loss = []
        test_acc=[]
        # for i in tqdm(range(epoch), leave=True, desc="train_model"):
        least_train_loss = np.inf
        for i in tqdm(range(epoch), leave=True, desc="train_model"):
            train_loss_per_epoch = 0
            idxs_prob = np.random.choice(len(prob), batch_size*num_steps_per_epoch, p=prob)
            idxs = indice_list[idxs_prob].tolist()
            dataset = self.dataloader.dataset_generator(idxs, batch_size)
            for X_windowed, y_windowed, y_seg_windowed, mask_windowed in dataset: # should be substituted by [for x,y,mask in dataset: ...]
                if self.is_LLAL and i<=37:
                    loss_final = self.model.train_step_llal_full(X_windowed, y_windowed, mask_windowed)
                elif self.is_LLAL and i>37:
                    loss_final = self.model.train_step_llal_part(X_windowed, y_windowed, mask_windowed)
                else:
                    loss_final  = self.model.train_step(X_windowed, y_windowed, mask_windowed)
                train_loss_per_epoch += loss_final.numpy()
            if is_test:
                metric = self.test_model()
                print(i, train_loss_per_epoch, metric)
                test_acc.append(metric)

            if least_train_loss > train_loss_per_epoch:
                weights_least_loss = self.model.get_weights()
                least_train_loss = train_loss_per_epoch
            train_loss.append(train_loss_per_epoch)
        self.model.set_weights(weights_least_loss)
        return train_loss, test_acc

    def train_model(self, epoch, batch_size, is_test=False):
        self.model.set_weights(self.weights_init)
        prob, indice_list = self.dataloader.window_scoring(slide_size=self.input_length)
        num_steps_per_epoch = len(prob) // batch_size # epoch is set by # of proped+queried labels
        train_loss = []
        test_acc=[]
        # for i in tqdm(range(epoch), leave=True, desc="train_model"):
        least_train_loss = np.inf
        for i in tqdm(range(epoch), leave=True, desc="train_model"):
            train_loss_per_epoch = 0
            for j in range(num_steps_per_epoch):
                indice = np.random.choice(len(prob), batch_size, p=prob)  # oversampling
                X_windowed, y_windowed, y_seg_windowed, mask_windowed = self.dataloader.batch_generator(indice_list[indice].tolist())
                if self.is_LLAL and i<=37:
                    loss_final = self.model.train_step_llal_full(X_windowed, y_windowed, mask_windowed)
                elif self.is_LLAL and i>37:
                    loss_final = self.model.train_step_llal_part(X_windowed, y_windowed, mask_windowed)
                else:
                    loss_final  = self.model.train_step(X_windowed, y_windowed, mask_windowed)
                train_loss_per_epoch += loss_final.numpy()
            if is_test:
                metric = self.test_model()
                print(i, train_loss_per_epoch, metric)
                test_acc.append(metric)

            if least_train_loss > train_loss_per_epoch:
                weights_least_loss = self.model.get_weights()
                least_train_loss = train_loss_per_epoch
            train_loss.append(train_loss_per_epoch)
        self.model.set_weights(weights_least_loss)
        return train_loss, test_acc


if __name__ == "__main__":
    # python3 model_manager.py gpu data_name model_name input_length learning_rate
    with tf.device("/GPU:"+sys.argv[1]):
        data = preprocessing.Preprocessing(sys.argv[2], 0.1)
        X_long, y_long, y_seg_long, file_boundaries = data.generate_long_time_series()
        MM = ModelManager(model_name=sys.argv[3], input_length=int(sys.argv[4]), num_class=len(np.unique(y_long)),
                          dim=X_long.shape[1], lr=float(sys.argv[5]))
        MM.load_train_data(X=X_long, y=y_long, y_seg=y_seg_long, mask=np.ones(len(y_long)), file_boundaries=file_boundaries)
        train_loss, test_acc = MM.train_model(1000, 32, y_long)

