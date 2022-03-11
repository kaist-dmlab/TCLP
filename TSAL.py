import os

from preprocessing import Preprocessing
from plateau import Plateaus
from model_manager import ModelManager
import numpy as np
from scipy.ndimage import gaussian_filter1d
from operator import itemgetter
import time
from sklearn.metrics import pairwise_distances
import pdb
from scipy import stats, signal
import matplotlib.pyplot as plt
from eval import f_score

class TSAL:
    def __init__(self, data_name, model_name, input_length, init_ratio, total_num_query_step, num_epoch, batch_size,
                 max_num_prop, lr, receptive_len=32, boundary_threshold=0.5, tau=15, seed=0, bg_class=[],
                 al_name="utility", is_label_propagation="platprob", no_plat_reg=0, temp=2):
        self.start_time = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
        self.prop_debug_dir = os.path.join("figures","propagation_results",self.start_time)
        self.seed = seed
        data = Preprocessing(data_name, 0.1)
        self.X, self.y_true, self.y_seg_true, self.file_boundaries = data.generate_long_time_series()
        self.num_class = len(np.unique(self.y_true))

        if al_name == "llal":
            is_LLAL=True
        else:
            is_LLAL=False

        self.model_manager = ModelManager(model_name=model_name, input_length=input_length, num_class=self.num_class,
                                          dim=self.X.shape[1], lr=lr, seed=self.seed, is_LLAL=is_LLAL)
        self.X_train, self.y_true_train, self.y_seg_true_train, _, self.file_boundaries_train, _0 ,_1, _2 ,_3, _4 = \
            self.model_manager.train_test_generator(self.X, self.y_true, self.y_seg_true, np.array([]), self.file_boundaries)

        self.num_epoch = num_epoch
        self.batch_size = batch_size

        self.input_length = input_length  # unit of label propagation and model input

        self.total_length = len(self.X)

        self.init_ratio = init_ratio
        self.total_num_query_step = total_num_query_step
        self.num_avail = len(self.X_train)  # only use front part of given long time series, unit is query_window_size

        self.receptive_len = receptive_len
        self.boundary_threshold = boundary_threshold

        self.bg_class=bg_class
        self.al_name = al_name
        self.is_label_propagation = is_label_propagation


        self.max_num_prop = max_num_prop
        self.labeled_or_not = np.zeros(self.num_avail)
        self.tau = tau # minimum length for predicted plateaus
        self.no_plat_reg = no_plat_reg
        self.temp=temp
        self.label_first_data()

    def label_first_data(self):
        # Label all class with equal number
        num_first_data = np.maximum(int(self.num_avail * self.init_ratio),1)
        num_init_label_per_class = int(num_first_data/self.num_class)
        if num_init_label_per_class < 1:
            num_init_label_per_class = 1
        self.init_ind_selected = []
        np.random.seed(self.seed)
        for i in range(self.num_class):
            init_label = np.random.choice(np.where(self.y_true_train==i)[0], size=num_init_label_per_class, replace=False)
            self.init_ind_selected += init_label.tolist()
        self.init_ind_selected = np.array(self.init_ind_selected)
        np.random.seed()
        self.labeled_or_not[self.init_ind_selected] = 1
        print(f"Initial labeling done: {np.sum(self.labeled_or_not)}, num_init_label_per_class:{num_init_label_per_class}")
        self.labeled_or_not_init = np.copy(self.labeled_or_not)

        # segmenter initialization
        self.Plateau = Plateaus(self.num_class, self.num_avail, tau=self.tau, no_plat_reg = self.no_plat_reg)


    def timestamp_uncertainty(self, X):
        def calculate_margin(x):
            indice = np.argsort(x)[-2:][::-1]
            margin_ts = x[indice[0]] - x[indice[1]]
            return margin_ts

        def entropy_1d(y):
            return -np.dot(y, np.log(y))

        if self.al_name == "conf":
            return -np.apply_along_axis(np.max, len(X.shape) - 1, X)
        elif self.al_name == "entropy":
            return np.apply_along_axis(entropy_1d, len(X.shape) - 1, X)
        else:
            return -np.apply_along_axis(calculate_margin, len(X.shape) - 1, X)

    def query_scoring(self, uncertainty):
        # output: score_list, indices_list
        labeled_or_not = np.copy(self.labeled_or_not)
        indices_list = np.where(labeled_or_not == 0)[0]
        indices_list_lb = np.where(labeled_or_not == 1)[0]
        if self.al_name == "margin" or self.al_name == "conf" or self.al_name == "entropy":
            score_list = uncertainty[indices_list]
        elif self.al_name == "random":
            score_list = np.random.rand(len(indices_list))
        elif self.al_name == "utility":
            labeled_or_not_proped = np.copy(self.labeled_or_not_propagated)
            indices_list = np.where(labeled_or_not_proped == 0)[0]
            score_list = np.random.rand(len(indices_list))

        elif self.al_name == "badge":
            X = self.model_manager.model.get_gradient(self.X_train,
                                                      self.y_pred_class, self.file_boundaries_train)
            K = self.num_queried_timestamp_per_al_step
            ind = np.argmax([np.linalg.norm(s, 2) for s in X])
            mu = [X[ind]]
            indsAll = [ind]
            centInds = [0.] * len(X)
            cent = 0
            while len(mu) < K:
                if len(mu) == 1:
                    D2 = pairwise_distances(X, mu).ravel().astype(float)
                else:
                    newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
                    for i in range(len(X)):
                        if D2[i] >  newD[i]:
                            centInds[i] = cent
                            D2[i] = newD[i]
                if sum(D2) == 0.0: pdb.set_trace()
                D2 = D2.ravel().astype(float)
                Ddist = (D2 ** 2)/ sum(D2 ** 2)
                customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
                ind = customDist.rvs(size=1)[0]
                while ind in indsAll: ind = customDist.rvs(size=1)[0]
                mu.append(X[ind])
                indsAll.append(ind)
                cent += 1
                score_list = np.zeros(self.num_queried_timestamp_per_al_step)
                indices_list = np.array(indsAll)

        elif self.al_name == "core":
            embedding = self.model_manager.model.predict_penultimate(X_long=self.X_train,
                                                                     file_boundaries=self.file_boundaries_train)
            X = embedding[indices_list, :]
            X_set = embedding[indices_list_lb, :]
            n = self.num_queried_timestamp_per_al_step
            m = np.shape(X)[0]
            if np.shape(X_set)[0] == 0:
                min_dist = np.tile(float("inf"), m)
            else:
                dist_ctr = pairwise_distances(X, X_set)
                min_dist = np.amin(dist_ctr, axis=1)
            idxs = []
            for i in range(n):
                idx = min_dist.argmax()
                idxs.append(idx)
                dist_new_ctr = pairwise_distances(X, X[[idx], :])
                for j in range(m):
                    min_dist[j] = min(min_dist[j], dist_new_ctr[j, 0])
            indices_list = np.array(idxs)
            score_list = np.zeros(len(idxs))
        else:
            raise ValueError("Not proper scoring name")
        return score_list, indices_list

    def acquisition(self):
        self.y_pred = self.model_manager.model.predict(X_long=self.X_train,
                                                       file_boundaries=self.file_boundaries_train)
        self.y_pred_class = np.argmax(self.y_pred, axis=1)
        unlabeled_ind = np.where(self.labeled_or_not == 0)[0]
        labeled_ind = np.where(self.labeled_or_not == 1)[0]

        uncertainty = self.timestamp_uncertainty(self.y_pred)
        score_list, indices_list = self.query_scoring(uncertainty)

        if self.is_semi_supervised:
            score_list = score_list * -1  # we need to find most certain labels

        # top-k Labeling by oracle
        selected_qwin = np.argsort(score_list)[-self.num_queried_timestamp_per_al_step:][::-1].tolist()
        if len(indices_list[selected_qwin]) == self.num_queried_timestamp_per_al_step:
            self.labeled_or_not[indices_list[selected_qwin]] = 1  # oracle-label is done
            self.queried_indices = indices_list[selected_qwin]
        else:
            query_indices = np.random.choice(np.arange(self.num_avail),size=self.num_queried_timestamp_per_al_step).tolist()
            self.labeled_or_not[query_indices] = 1  # oracle-label is done
            self.queried_indices = query_indices
            print(indices_list[selected_qwin])
            print("less number acquired through AL")

        if self.is_label_propagation:
            self.label_propagation()  # propagate values
            self.labeled_or_not_propagated_before_prop = np.array([])
            prop_indices = np.where(self.labeled_or_not_propagated == 1)[0].astype(np.int64)

    def model_fitting(self):
        if not self.is_label_propagation:  # if label propagation is not allowed, use original data
            self.y = np.copy(self.y_true_train)
            self.y_seg = np.copy(self.y_seg_true_train)
            self.labeled_or_not_propagated = np.copy(self.labeled_or_not)
        self.model_manager.load_train_data(self.X_train, self.y, self.y_seg, self.labeled_or_not_propagated, self.file_boundaries_train)
        self.model_manager.train_model(self.num_epoch,self.batch_size, is_test=False)

    def doAL(self, num_query_ratio=0.005, is_semi_supervised=False,  eta=0.8):
        self.labeled_or_not = np.copy(self.labeled_or_not_init)
        self.query_step = 0
        self.labeled_or_not_propagated_before_prop = np.array([])
        self.num_queried_timestamp_per_al_step = int(num_query_ratio * self.total_length)

        self.is_semi_supervised = is_semi_supervised
        self.segmenter_acc = 1 # init segmenter_acc
        self.eta = eta

        print(self.al_name, "AL with", self.is_label_propagation, "Label Propagation")
        num_total_query = 0
        test_acc = []
        num_labeled = []
        num_labeled_propagated = []
        prop_accuracy = []
        prop_mean_iou = []
        boundary_accuracy = []
        plateau_log = []

        self.labeled_or_not_propagated = np.copy(self.labeled_or_not_init)
        self.y = np.copy(self.y_true_train)  # queried label, initial label, and propagated labels
        self.y_seg = np.copy(self.y_seg_true_train)

        # Initialize Segmenter with Plateaus
        self.model_fitting()
        print("classifier initialized")
        self.y_pred = self.model_manager.model.predict(X_long=self.X_train, file_boundaries=self.file_boundaries_train)
        if self.is_label_propagation=="platprob":
            self.Plateau.find_and_fit(self.y_pred)
            self.Plateau.add_plateaus(zip(self.init_ind_selected, self.y_true_train[self.init_ind_selected])) # add initial points
            self.Plateau.update_queried_plateaus()
            self.Plateau.merge_and_split()
        self.plateau_log_per_step = []
        print("propagator initialized")
        for query_step in range(self.total_num_query_step):
            num_total_query += self.num_queried_timestamp_per_al_step
            self.acquisition()
            self.model_fitting()
            print(str(query_step) + "/" + str(self.total_num_query_step), end=' ')
            print(f"{np.sum(self.labeled_or_not):.0f}", end=' ')
            num_labeled.append(np.sum(self.labeled_or_not))
            print(f"{np.sum(self.labeled_or_not_propagated):.0f}", end=' ')
            num_labeled_propagated.append(np.sum(self.labeled_or_not_propagated))
            print(f"{np.sum(self.labeled_or_not_propagated) / np.sum(self.labeled_or_not):.1f}", end=' ')
            test_acc.append(self.model_manager.test_model(bg_class=self.bg_class))
            for i in test_acc[-1]:
                print(f"{i:.3f}", end=" ")
            prop_indices = np.where(self.labeled_or_not_propagated == 1)[0].astype(np.int64)
            prop_acc_one = np.sum(self.y[prop_indices] == self.y_true_train[prop_indices]) / np.sum(
                self.labeled_or_not_propagated)
            prop_accuracy.append(prop_acc_one)
            print(f"{prop_acc_one:.3f}", end=" ")
            y_prop = np.zeros_like(self.y)
            y_prop[:]=-1
            y_prop[prop_indices] = self.y[prop_indices]
            _,_,_,mean_iou = f_score(y_prop,self.y_true_train,[.5], self.bg_class)
            prop_mean_iou.append(mean_iou)
            print(f"{mean_iou:.3f}", end=" ")
            print(f"ECE: {self.model_manager.get_unlabeled_ECE()*100:.3f}", end=" ")
            boundary_accuracy.append(self.segmenter_acc)
            print(f"{self.segmenter_acc:.2f}", end=" ")
            if len(self.plateau_log_per_step) > 0:
                plateau_log.append(self.plateau_log_per_step)
                print(self.plateau_log_per_step)
            else:
                print()
            self.query_step += 1
        if len(plateau_log) > 0:
            return [num_labeled, num_labeled_propagated, test_acc, prop_accuracy, prop_mean_iou, boundary_accuracy, plateau_log]
        else:
            return [num_labeled, num_labeled_propagated, test_acc, prop_accuracy, prop_mean_iou, boundary_accuracy]

    def label_propagation(self):
        y_ref = np.copy(self.y_true_train)
        y_seg_ref = np.copy(self.y_seg_true)

        self.labeled_or_not_propagated = np.copy(self.labeled_or_not)
        self.y = np.copy(self.y_true_train)
        self.y_seg = np.copy(self.y_seg_true)
        labeled_index = np.where(self.labeled_or_not_propagated == 1)[0]
        boundary_index = np.where(self.y_seg_true_train > self.boundary_threshold)[0]  # true boundary indices
        num_maximum_prop = self.max_num_prop
        self.segmenter_acc = np.sum(self.y_seg_true[boundary_index.tolist()] == 1) / len(boundary_index)

        if self.is_label_propagation == "true":
            for index in labeled_index:
                left_indices = np.concatenate(
                    [boundary_index[boundary_index < index], labeled_index[labeled_index < index]], axis=0)
                if np.sum(left_indices < index) > 0:  # boundary has to exist before index
                    nearest_lhs_ind = np.max(left_indices)  # nearest left boundary index
                    if index - nearest_lhs_ind > num_maximum_prop:
                        self.labeled_or_not_propagated[index - num_maximum_prop:index] = 1
                        self.y[index - num_maximum_prop:index] = y_ref[index]
                        self.y_seg[index - num_maximum_prop:index] = y_seg_ref[index]
                        if np.sum(self.y_true_train[index - num_maximum_prop:index]!= y_ref[index])>0:
                            print("propagation is wrong [index - num_maximum_prop:index]")
                            print(self.y_true_train[index - num_maximum_prop:index])
                            print(self.y_seg_true[index - num_maximum_prop:index])
                            print(self.y[index - num_maximum_prop:index])
                    else:
                        self.labeled_or_not_propagated[nearest_lhs_ind:index] = 1
                        self.y[nearest_lhs_ind + 1:index] = y_ref[index]
                        self.y_seg[nearest_lhs_ind + 1:index] = y_seg_ref[index]
                        if np.sum(self.y_true_train[nearest_lhs_ind + 1:index]!= y_ref[index])>0:
                            print("propagation is wrong [nearest_lhs_ind + 1:index]")
                            print(self.y_true_train[nearest_lhs_ind + 1:index])
                            print(self.y_seg_true[nearest_lhs_ind + 1:index])
                            print(self.y[nearest_lhs_ind + 1:index])
                right_indices = np.concatenate(
                    [boundary_index[boundary_index > index], labeled_index[labeled_index > index]], axis=0)

                if np.sum(right_indices > index) > 0:  # boundary has to exist after index
                    if index in boundary_index:
                        continue
                    nearest_rhs_ind = np.min(right_indices)  # nearest right boundary index
                    if nearest_rhs_ind - index > num_maximum_prop:
                        self.labeled_or_not_propagated[index:index + num_maximum_prop] = 1
                        self.y[index:index + num_maximum_prop] = y_ref[index]
                        self.y_seg[index:index + num_maximum_prop] = y_seg_ref[index]  # only non-boundary label propagation
                        if np.sum(self.y_true_train[index:index + num_maximum_prop]!= y_ref[index])>0:
                            print("propagation is wrong [index:index + num_maximum_prop]")
                            print(self.y_true_train[index:index + num_maximum_prop])
                            print(self.y_seg_true[index:index + num_maximum_prop])
                            print(self.y[index:index + num_maximum_prop])
                    else:
                        self.labeled_or_not_propagated[index:nearest_rhs_ind] = 1
                        self.y[index:nearest_rhs_ind] = y_ref[index]
                        self.y_seg[index:nearest_rhs_ind] = y_seg_ref[index]  # only non-boundary label propagation
                        if np.sum(self.y_true_train[index:nearest_rhs_ind]!= y_ref[index])>0:
                            print("propagation is wrong [index:nearest_rhs_ind]")
                            print(self.y_true_train[index:nearest_rhs_ind])
                            print(self.y_seg_true[index:nearest_rhs_ind])
                            print(self.y[index:nearest_rhs_ind])

            self.labeled_or_not_propagated_before_prop = np.copy(self.labeled_or_not_propagated)

        elif type(self.is_label_propagation)==int:
            if self.is_label_propagation==0:
                pass
            boundary_index = []
            for index in labeled_index:
                self.y[index - self.is_label_propagation:index + self.is_label_propagation] = y_ref[index]
                self.y_seg[index - self.is_label_propagation:index + self.is_label_propagation] = y_seg_ref[index]
                self.labeled_or_not_propagated[index - self.is_label_propagation:index + self.is_label_propagation] = 1
                boundary_index += [index - self.is_label_propagation,index + self.is_label_propagation]
            self.segmenter_acc = np.sum(self.y_seg_true[boundary_index] == 1) / len(boundary_index)

        # TCLP
        elif self.is_label_propagation == "platprob":
            self.Plateau.add_plateaus(zip(self.queried_indices, self.y_true_train[self.queried_indices]))
            # Temperature scaling
            T = self.temp
            z = self.model_manager.model.predict_logit(X_long=self.X_train, file_boundaries=self.file_boundaries_train)
            z = z.transpose()
            z = z / T
            max_z = np.max(z,axis=0)
            exp_z = np.exp(z-max_z)
            sum_exp_z = np.sum(exp_z,axis=0)
            y = exp_z / sum_exp_z

            self.Plateau.find_and_fit(y)
            num_trained, num_query_seg_before, num_pred_seg = self.Plateau.update_queried_plateaus()
            num_merge, num_split, num_seg_after_query = self.Plateau.merge_and_split()
            self.plateau_log_per_step = [num_trained, num_query_seg_before, num_pred_seg, num_merge, num_split, num_seg_after_query]
            boundary_index = []
            for pl in self.Plateau.segmenter:
                start, end = pl.propagation_timestamp(self.eta)
                pl_queried_ind = pl.queried_ts_list[0]
                self.labeled_or_not_propagated[start:end] = 1
                self.y[start:end] = y_ref[pl_queried_ind]
                boundary_index += [start, end]
            self.segmenter_acc = np.sum(self.y_seg_true[boundary_index] == 1) / len(boundary_index)

        # ESP
        elif self.is_label_propagation == "repr":
            self.x_penul = self.model_manager.model.predict_penultimate(X_long=self.X_train,
                                                                           file_boundaries=self.file_boundaries_train)
            sim_long = np.zeros(self.num_avail)
            for i in range(self.num_avail-1):
                sim = np.dot(self.x_penul[i],self.x_penul[i+1])/(np.sqrt(np.dot(self.x_penul[i],self.x_penul[i])))*\
                      (np.sqrt(np.dot(self.x_penul[i+1],self.x_penul[i+1])))
                sim_long[i+1] = sim

            sim_mv_avg = np.convolve(sim_long, np.ones(self.tau),"same")/self.tau
            sim_diff = np.abs(sim_mv_avg-sim_long)
            boundary_index, _ = signal.find_peaks(sim_diff)

            num_maximum_prop = self.tau # tau represents maximum prop for repr method
            for index in labeled_index:
                left_indices = np.concatenate(
                    [boundary_index[boundary_index < index], labeled_index[labeled_index < index]], axis=0)
                if np.sum(left_indices < index) > 0:  # boundary has to exist before index
                    nearest_lhs_ind = np.max(left_indices)  # nearest left boundary index
                    if index - nearest_lhs_ind > num_maximum_prop:
                        self.labeled_or_not_propagated[index - num_maximum_prop:index] = 1
                        self.y[index - num_maximum_prop:index] = y_ref[index]
                        self.y_seg[index - num_maximum_prop:index] = y_seg_ref[index]
                    else:
                        self.labeled_or_not_propagated[nearest_lhs_ind:index] = 1
                        self.y[nearest_lhs_ind + 1:index] = y_ref[index]
                        self.y_seg[nearest_lhs_ind + 1:index] = y_seg_ref[index]
                right_indices = np.concatenate(
                    [boundary_index[boundary_index > index], labeled_index[labeled_index > index]], axis=0)

                if np.sum(right_indices > index) > 0:  # boundary has to exist after index
                    if index in boundary_index:
                        continue
                    nearest_rhs_ind = np.min(right_indices)  # nearest right boundary index
                    if nearest_rhs_ind - index > num_maximum_prop:
                        self.labeled_or_not_propagated[index:index + num_maximum_prop] = 1
                        self.y[index:index + num_maximum_prop] = y_ref[index]
                        self.y_seg[index:index + num_maximum_prop] = y_seg_ref[index]  # only non-boundary label propagation
                    else:
                        self.labeled_or_not_propagated[index:nearest_rhs_ind] = 1
                        self.y[index:nearest_rhs_ind] = y_ref[index]
                        self.y_seg[index:nearest_rhs_ind] = y_seg_ref[index]  # only non-boundary label propagation
            self.labeled_or_not_propagated_before_prop = np.copy(self.labeled_or_not_propagated)

        # PTP
        elif self.is_label_propagation == "prob": 
            boundary_index=[]
            for index in labeled_index:
                prop_right_done = False
                prop_left_done = False
                label = self.y_pred_class[index]
                sim_at_label_right = self.y_pred[index][label]
                sim_at_label_left = self.y_pred[index][label]
                for i in range(self.tau): # expand radius = self.tau
                    if not prop_right_done and index+i<len(self.y_pred):
                        if self.y_pred[index+i][label] > sim_at_label_right*self.eta and self.y_pred_class[index+i]==label: # eta = 0.8
                            sim_at_label_right*=self.eta
                            pass
                        else:
                            prop_right_ind = index+i
                            prop_right_done = True
                    if not prop_left_done:
                        if self.y_pred[index-i][label] > sim_at_label_left*self.eta and self.y_pred_class[index-i]==label:
                            sim_at_label_left*=self.eta
                            pass
                        else:
                            prop_left_ind = index-i
                            prop_left_done = True
                if not prop_right_done:
                    prop_right_ind = index + self.tau
                if not prop_left_done:
                    prop_left_ind = index - self.tau
                boundary_index += [prop_right_ind,prop_left_ind]
                self.labeled_or_not_propagated[prop_left_ind:prop_right_ind] = 1
                self.y[prop_left_ind:prop_right_ind] = y_ref[index]
            self.segmenter_acc = np.sum(self.y_seg_true[boundary_index] == 1) / len(boundary_index)

        elif self.is_label_propagation == "platrepr":
            self.x_penul = self.model_manager.model.predict_penultimate(X_long=self.X_train,
                                                                        file_boundaries=self.file_boundaries_train)
            sim_long = np.zeros((self.num_avail,1))
            for i in range(self.num_avail-1):
                sim = np.dot(self.x_penul[i],self.x_penul[i+1])/(np.sqrt(np.dot(self.x_penul[i],self.x_penul[i])))* \
                      (np.sqrt(np.dot(self.x_penul[i+1],self.x_penul[i+1])))
                sim_long[i+1] = sim
            self.Plateau.add_plateaus(zip(self.queried_indices, self.y_true_train[self.queried_indices]))
            self.Plateau.find_and_fit(sim_long.transpose())
            num_trained, num_query_seg_before, num_pred_seg = self.Plateau.update_queried_plateaus()
            num_merge, num_split, num_seg_after_query = self.Plateau.merge_and_split()
            self.plateau_log_per_step = [num_trained, num_query_seg_before, num_pred_seg, num_merge, num_split, num_seg_after_query]

            boundary_index = []
            for pl in self.Plateau.segmenter:
                start, end = pl.propagation_timestamp(self.eta)
                pl_queried_ind = pl.queried_ts_list[0]
                self.labeled_or_not_propagated[start:end] = 1
                self.y[start:end] = y_ref[pl_queried_ind]
                boundary_index += [start, end]
            self.segmenter_acc = np.sum(self.y_seg_true[boundary_index] == 1) / len(boundary_index)

        else:
            print("LP not specified - No propagation applied")

        self.labeled_or_not_propagated_before_prop = np.copy(self.labeled_or_not_propagated)


if __name__ == "__main__":

    data_names = ["50salads", "HAPT", "GTEA", "Sleep", "SAMSUNG", "HASC_BDD"]

    init_ratio_dict = {"50salads": 0.0001, "HAPT": 0.0001, "GTEA": 0.001, "SAMSUNG": 0.0001, "HASC_BDD": 0.0001,
                       "Sleep": 0.01}  # set as the number of segments in each dataset
    input_length_dict = {"50salads": 256, "HAPT": 512, "GTEA": 128, "SAMSUNG": 128, "HASC_BDD": 512, "Sleep": 1024}  #
    end_ratio_dict = {"50salads": 0.004, "HAPT": 0.005, "GTEA": 0.04, "SAMSUNG": 0.003, "HASC_BDD": 0.004,
                      "Sleep": 0.03}  #
    num_query_ratio = {"50salads": 0.00005, "HAPT": 0.00005, "GTEA": 0.00066, "SAMSUNG": 0.00005, "HASC_BDD": 0.00005,
                       "Sleep": 0.0005}  #
    max_num_prop_dict = {"50salads": 575 // 2, "HAPT": 716 // 2, "GTEA": 35 // 2, "SAMSUNG": 10 // 2,
                         "HASC_BDD": 371 // 2, "Sleep": 1432 // 2}  # (mean of segment length)/2
    data_epoch_dict = {"50salads": 35, "HAPT": 30, "GTEA": 30, "SAMSUNG": 30, "HASC_BDD": 30, "Sleep": 60}
    data_batch_dict = {"50salads": 32, "HAPT": 32, "GTEA": 32, "SAMSUNG": 32, "HASC_BDD": 32, "Sleep": 32}
    tau_dict = {"50salads": 15, "HAPT": 30, "GTEA": 5, "SAMSUNG": 1, "HASC_BDD": 30, "Sleep": 30}
    lr_dict = {"50salads": 0.001, "HAPT": 0.0001, "GTEA": 0.001, "SAMSUNG": 0.001, "HASC_BDD": 0.001, "Sleep": 0.001}
    data_epoch_dict = {"50salads": 35, "HAPT": 500, "GTEA": 30, "SAMSUNG": 30, "HASC_BDD": 30, "Sleep": 500}
    al_method = ["random","uncertainty","seg"]
    al = "random"
    lp_method = ["zero", "true", "seg"]
    lp = "seg"
    seed = 0


    for name in ["HAPT"]:
        tsal = TSAL(data_name=name, input_length=input_length_dict[name], init_ratio=init_ratio_dict[name],
                    seed=int(seed), end_ratio=end_ratio_dict[name], num_epoch=data_epoch_dict[name],
                    batch_size=data_batch_dict[name], max_num_prop=max_num_prop_dict[name], lr=lr_dict[name],
                    tau=tau_dict[name])

        result = tsal.doAL(num_query_ratio=num_query_ratio[name], al_name=al,
                       boundary_threshold=0.5, is_label_propagation=lp)









