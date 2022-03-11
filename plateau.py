from scipy.optimize import curve_fit
import numpy as np
from multiprocessing import Pool
import time
import matplotlib.pyplot as plt
import matplotlib.colors as pltc
from tqdm import tqdm

def plateau_function(x,c,w,s):
    return 1 / ((np.exp(s * ((x - c) - w)) + 1) * (np.exp(s * (-(x - c) - w)) + 1))


class Plateaus:
    def __init__(self, num_class, num_timestamp, tau=15, no_plat_reg=0):
        self.segmenter = [] # representing segments in long time series
        self.queried_plateaus = [] # Plateau instances constructed from queried timestamps
        self.pred_plateaus = []
        self.tau = tau # length threshold for segment cands
        self.num_class = num_class
        self.num_timestamp = num_timestamp
        self.class_lengths_list = []
        self.plateau_id = 0
        self.num_cpu = 3
        self.no_plat_reg = no_plat_reg

    def add_plateaus(self, queried_timestamps_labels):

        self.queried_plateaus = []
        total_average = np.mean([i for j in self.class_lengths_list for i in j])
        average_length = {}
        class_no_data = []
        for i in range(self.num_class):
            if len(self.class_lengths_list[i]) > 0:
                average_length[i] = np.mean(self.class_lengths_list[i])
            else:
                class_no_data.append(i)
        # if there is no data for a class, set the length as the total average length
        for i in class_no_data:
            average_length[i] = total_average


        for ts, l in queried_timestamps_labels:
            if average_length[l] < 0:
                print("negative average length")

            self.queried_plateaus.append(Plateau(x_len=self.tau, c=ts, w=self.tau/2, s=0.5, c_=5, l=l,
                                                 queried_ts_list=[ts], id=self.plateau_id))

            self.plateau_id+=1
        if len(self.segmenter)==0:
            for pl in self.queried_plateaus:
                self.segmenter.append(pl.copy())

    def generate_probs_mask(self):
        probs = np.zeros((self.num_class, self.num_timestamp))
        mask = np.zeros((self.num_class, self.num_timestamp))
        sorted_queried_plts = sorted(self.segmenter)
        ind = 0
        for pl in sorted_queried_plts:
            pl_prob = pl.generate_probs()
            if len(pl_prob) % 2 == 0:
                # TODO: deal with a timestamp allocated to multiple same class plateaus
                # using dictionary for each timestamp?
                probs[pl.class_label, pl.c - len(pl_prob) / 2:pl.c + len(pl_prob) / 2] = pl_prob
                mask[pl.class_label, pl.c - len(pl_prob) / 2:pl.c + len(pl_prob) / 2] = ind
            else:
                probs[pl.class_label, pl.c - len(pl_prob) / 2:pl.c + len(pl_prob) / 2 + 1] = pl_prob
                mask[pl.class_label, pl.c - len(pl_prob) / 2:pl.c + len(pl_prob) / 2 + 1] = ind
            ind += 1
        return probs, mask


    def fit(self, cls_output_and_predicted_plateaus):
        cls_output = cls_output_and_predicted_plateaus[0]
        class_segment_cand = cls_output_and_predicted_plateaus[1]
        min_w = 5
        min_s = 0.05
        l, start, end = class_segment_cand
        ydata = cls_output[l, start:end]  # index info disappears
        xdata = np.arange(len(ydata))
        bounds_lower = [0, 0, 0]  # c, w, s lower bound
        bounds_upper = [len(xdata), len(xdata) / 2, 1]  # c, w, s upper bound
        initial_param = [len(xdata) / 2, len(xdata) / 2, 0.5]
        try:  # if optimal parameters not found, do not use it
            fit_params, _ = curve_fit(plateau_function, xdata, ydata, bounds=(bounds_lower, bounds_upper),
                                      p0=initial_param, maxfev=5000)
            c_fit = fit_params[0]
            w_fit = fit_params[1]
            s_fit = fit_params[2]
            if (w_fit < min_w) or (s_fit < min_s):
                return 0
            pred_plateau = Plateau(x_len=len(ydata), c=int(c_fit + start), w=w_fit, s=s_fit, c_=c_fit, l=l,
                                   queried_ts_list=[-1], id=-1)  # -1 means predicted plateau
            pred_plateau.score = np.sum(ydata[pred_plateau.generate_probs() > 0.5])
        except Exception:
            pass
        return pred_plateau

    def find_and_fit(self, cls_output):
        self.pred_plateaus = []

        ############################ Single Version ######################
        centers = []
        for pl in self.segmenter:
            centers.append(int(pl.c))
            # centers += pl.queried_ts_list
        pl_center_binary_array = np.zeros(cls_output.shape[1])
        pl_center_binary_array[centers] = 1

        min_w = 5
        min_s = 0.05

        class_segment_cands = []
        self.class_lengths_list = []
        for i in range(self.num_class):
            class_segment_cands.append([])
            self.class_lengths_list.append([self.tau])

        for conf_threshold in [0.3, 0.5, 0.7, 0.9]:
            cls, ts = np.where(cls_output > conf_threshold)
            if len(cls)==0:
                return

            prev_c = cls[0]
            prev_t = ts[0]
            consecutive_ts = [prev_t]
            for c, t in zip(cls[1:], ts[1:]):
                if (c == prev_c) & (t == prev_t+1):
                    consecutive_ts.append(t)
                else:
                    if len(consecutive_ts) > self.tau and np.sum(pl_center_binary_array[consecutive_ts[0]:consecutive_ts[-1]+1])>0: # very few update happens.
                    # if len(consecutive_ts) > self.tau:
                        class_segment_cands[c].append((consecutive_ts[0], consecutive_ts[-1]+1))  # save only start:end index
                        self.class_lengths_list[c].append(len(consecutive_ts))
                    consecutive_ts = []
                prev_c = c
                prev_t = t

        # fit to the predicted segments
        for i in range(self.num_class):
            for start, end in tqdm(class_segment_cands[i], leave=False, desc=f"find_and_fit_predicted_plateaus({i}/{self.num_class})"):
                ydata = cls_output[i, start:end] # index info disappears
                xdata = np.arange(len(ydata))
                bounds_lower = [0,0,0] # c, w, s lower bound
                bounds_upper = [len(xdata), len(xdata)/2,1] # c, w, s upper bound
                initial_param = [len(xdata)/2,len(xdata)/2,0.5]
                try: # if optimal parameters not found, do not use it
                    fit_params, _ = curve_fit(plateau_function, xdata, ydata, bounds=(bounds_lower, bounds_upper),
                                          p0=initial_param, maxfev=5000)
                    c_fit = fit_params[0]
                    w_fit = fit_params[1]
                    s_fit = fit_params[2]
                    if (w_fit < min_w) or (s_fit < min_s):
                        continue
                    pred_plateau = Plateau(x_len=len(ydata), c=int(c_fit + start), w=w_fit, s=s_fit, c_=c_fit, l=i,
                                           queried_ts_list=[-1], id=-1)  # -1 means predicted plateau

                    pred_plateau.score = np.sum(ydata[pred_plateau.generate_probs() > 0.5])/len(pred_plateau.x)
                    self.pred_plateaus.append(pred_plateau)
                except Exception:
                    pass


    def search_pred_plateau(self, pred_plateaus_and_queried_plateau):
        pred_plateaus = pred_plateaus_and_queried_plateau[0]
        queried_plateau = pred_plateaus_and_queried_plateau[1]
        lc, lw, ls = pred_plateaus_and_queried_plateau[2], pred_plateaus_and_queried_plateau[3], pred_plateaus_and_queried_plateau[4]

        max_score = 0
        for pred_plateau in pred_plateaus:
            pred_c, pred_w, pred_l = pred_plateau.c, pred_plateau.w, pred_plateau.class_label
            # if queried_plateau center is in the predicted plateau, update!
            if (queried_plateau.class_label == pred_l) and (pred_c - pred_w < queried_plateau.c) \
                and (queried_plateau.c < pred_c + pred_w):
                if max_score < pred_plateau.score:
                    target_plateau = pred_plateau
                    max_score = pred_plateau.score
        if max_score == 0:
            return 0  # no update
        else:
            queried_plateau.c = queried_plateau.c - lc * (queried_plateau.c - target_plateau.c)
            queried_plateau.w = queried_plateau.w - lw * (queried_plateau.w - target_plateau.w)
            queried_plateau.s = queried_plateau.s - ls * (queried_plateau.s - target_plateau.s)
            queried_plateau.x = np.arange(int(queried_plateau.w * 2))
            return 1


    def update_queried_plateaus(self, lc=0.5, lw=0.5, ls=0.25):
        """
        :param lc, lw, ls: learning rate for c, w, s
        :return: void, update queried plateaus
        """

        #### Parallel Version ####
        # cls_output_and_predicted_plateaus = []
        # for queried_plateau in self.segmenter:
        #     cls_output_and_predicted_plateaus.append((self.pred_plateaus, queried_plateau, lc, lw, ls))
        #
        # with Pool(self.num_cpu) as p:
        #     update_bool = p.map(self.search_pred_plateau, cls_output_and_predicted_plateaus)
        # num_trained = np.sum(update_bool)

        #### Single Version ####
        num_trained = 0
        num_max_prop = 0
        processed_plateau_id = []
        for queried_plateau in tqdm(self.segmenter, leave=False, desc="update_queried_plateaus"):
            if queried_plateau.id in processed_plateau_id:
                print()
                print("{queried_plateau.id} already processed!")
                continue
            else:
                processed_plateau_id.append(queried_plateau.id)

            max_score = 0
            for pred_plateau in self.pred_plateaus:
                pred_c, pred_w, pred_l = pred_plateau.c, pred_plateau.w, pred_plateau.class_label
                # if queried_plateau center is in the predicted plateau, update!
                if (pred_c-pred_w < queried_plateau.c) and (queried_plateau.c < pred_c+pred_w) and \
                        (queried_plateau.class_label == pred_l):
                    if max_score < pred_plateau.score:
                        target_plateau = pred_plateau
                        max_score = pred_plateau.score
            if max_score == 0:
                continue # no update
            else:
                queried_plateau_c = queried_plateau.c - lc * (queried_plateau.c - target_plateau.c)
                queried_plateau_w = queried_plateau.w - lw * (queried_plateau.w - target_plateau.w)
                queried_plateau_s = queried_plateau.s - ls * (queried_plateau.s - target_plateau.s)
                queried_plateau_x_len = int(queried_plateau_w * 2)

                if queried_plateau_x_len % 2 == 0:
                    start = int(queried_plateau_c - queried_plateau_x_len / 2)
                    end = int(queried_plateau_c + queried_plateau_x_len / 2)
                else:
                    start = int(queried_plateau_c - queried_plateau_x_len / 2)
                    end = int(queried_plateau_c + queried_plateau_x_len / 2 + 1)
                is_valid_update = True
                for labeled_ts in queried_plateau.queried_ts_list:
                    if not labeled_ts in list(range(start,end)): # if update makes plateau not contain labeled timestamp, update is cancelled
                        is_valid_update = False
                        break
                if queried_plateau_w < 0:
                    is_valid_update = False
                # Supress too much propagation
                if (queried_plateau_w > queried_plateau.w*2) and (self.no_plat_reg==0):
                    queried_plateau_w = queried_plateau.w*2
                    queried_plateau_x_len = int(queried_plateau_w*2)
                    num_max_prop += 1
                if is_valid_update:
                    num_trained += 1
                    queried_plateau.c = queried_plateau_c
                    queried_plateau.w = queried_plateau_w
                    queried_plateau.s = queried_plateau_s
                    queried_plateau.x = np.arange(queried_plateau_x_len)
                else:
                    continue
        return num_trained, len(self.segmenter), len(self.pred_plateaus)

    def check_duplicate_propagation(self):
        # timestamps such that plateau value > 0.5 and same class are merged into a new plateau
        # timestamps such that plateau value > 0.5 and different class are splitted/changed
        timestamp_plateaus_list = []
        for i in range(self.num_timestamp):
            timestamp_plateaus_list.append([])  # timestamp i has plateau indices list
        sorted_queried_plts = sorted(self.segmenter)

        ind = 0
        for pl in sorted_queried_plts:
            start, end = pl.start_end_timestamp()
            for ts in range(start, end):
                timestamp_plateaus_list[ts].append(ind) # ragged list constructed
            ind += 1

        for i in range(self.num_timestamp):
            if len(timestamp_plateaus_list[i])>1:
                print(f"timestamp {i} is allocated to multiple plateaus:{timestamp_plateaus_list[i]},"
                      f"{[(sorted_queried_plts[j].start_end_timestamp(), sorted_queried_plts[j].class_label, sorted_queried_plts[j].queried_ts_list) for j in timestamp_plateaus_list[i]]}")


    def split(self, pl1, pl2):
        # Need to be more concise!
        start1, end1 = pl1.start_end_timestamp()
        start2, end2 = pl2.start_end_timestamp()

        if pl1.c < pl2.c:
            pl_left = pl1
            pl_right = pl2
        else:
            pl_left = pl2
            pl_right = pl1

        split_timestamp = (np.max(pl_left.queried_ts_list) + np.min(pl_right.queried_ts_list)) / 2

        pl_left.c = (split_timestamp + start1) / 2
        pl_left.w = (split_timestamp - start1) / 2
        pl_left.x = np.arange(int(pl_left.w * 2))
        pl_left.c_ = pl_left.w

        pl_right.c = (split_timestamp + end2) / 2
        pl_right.w = (-split_timestamp + end2) / 2
        pl_right.x = np.arange(int(pl_right.w * 2))
        pl_right.c_ = pl_right.w



    def merge(self, pl1, pl2):
        start1, end1 = pl1.start_end_timestamp()
        start2, end2 = pl2.start_end_timestamp()

        if start1<start2 and end2<end1:
            new_plateau = pl1.copy()
            new_plateau.id = self.plateau_id
            self.plateau_id += 1
            return [new_plateau]
        elif start2<start1 and end1<end2:
            new_plateau = pl2.copy()
            new_plateau.id = self.plateau_id
            self.plateau_id += 1
            return [new_plateau]
        x_len = np.maximum(end2 - start1, end1 - start2)
        start = np.minimum(start1, start2)
        new_plateau = Plateau(x_len=x_len, c=start + int(x_len / 2), w=int(x_len / 2), s=(pl1.s + pl2.s) / 2,
                              l=pl1.class_label, c_=x_len / 2, queried_ts_list=pl1.queried_ts_list + pl2.queried_ts_list,
                              id=self.plateau_id)
        self.plateau_id += 1
        return [new_plateau]

    def merge_and_split(self):
        num_merge = 0
        num_split = 0

        prev_plateaus = self.segmenter + self.queried_plateaus
        prev_plateaus = np.unique(prev_plateaus).tolist()
        prev_plateaus.sort(reverse=True)
        updated_plateaus = []
        # print(f"start with {len(self.segmenter), len(self.queried_plateaus), len(prev_plateaus)} prev_plateaus")
        while len(prev_plateaus) > 0:
            pl1 = prev_plateaus.pop() # erase pl1 in prev_plateaus
            start1, end1 = pl1.start_end_timestamp()
            neighbors = []
            for pl2 in prev_plateaus + updated_plateaus: # comparison to all plateaus that exit
                if pl1 == pl2:
                    print("same plateau was in current plateaus")
                    continue
                start2, end2 = pl2.start_end_timestamp()
                if (start2 < start1 < end2 < end1) or (start1 < start2 < end1 < end2)\
                    or (start2 < start1 < end1 < end2) or (start1 < start2 < end2 < end1):
                    neighbors.append(pl2)
            if len(neighbors) == 0:
                updated_plateaus.append(pl1) # add plateaus for next update
                continue
            # merge first and then split
            # find a neighbor for merge
            is_merged = False
            for pl2 in neighbors:
                if pl1.class_label == pl2.class_label:
                    is_merged = True
                    for pl in [pl1, pl2]:
                        if pl in updated_plateaus:
                            updated_plateaus.remove(pl) # remove previous plateaus after merge
                    new_plateau = self.merge(pl1, pl2)
                    if pl2 in prev_plateaus: prev_plateaus.remove(pl2)
                    prev_plateaus += new_plateau
                    # updated_plateaus += new_plateau  # TODO:  updated plateaus > 1000???
                    num_merge += 1
                    break
            if is_merged: continue # break while loop for neighbor
            # start split when no merge occurs
            updated_plateaus += [pl1]
            if pl1 in prev_plateaus: prev_plateaus.remove(pl1)
            for pl2 in neighbors:
                self.split(pl1, pl2)
                num_split += 1
            updated_plateaus += [pl1]
        self.segmenter = np.unique(updated_plateaus).tolist()
        # print(f"merge split num_seg {num_merge}, {num_split}, {len(self.segmenter)}", end=" ")
        return num_merge, num_split, len(self.segmenter)

    def plot_segmenter(self, y_true, indice_start=0, indice_end=20000):
        fig = plt.figure(figsize=(20,4))
        ax = fig.add_subplot(111)
        y_true = y_true.flatten()
        all_colors = [k for k, v in pltc.cnames.items()]

        duration = []
        ind = 0
        for label_ts_prev, label_ts in zip(y_true[indice_start:indice_end-1], y_true[indice_start + 1:indice_end]):
            ind += 1
            if label_ts_prev != label_ts:
                duration.append([int(y_true[ind - 1]), ind])
        if duration[-1][-1] < indice_end:
            duration.append([int(y_true[indice_end]), indice_end-indice_start])

        axv_ind = 0
        for label, ind in duration:
            if axv_ind == 0:
                plt.axvspan(0, ind, color=all_colors[label], alpha=0.5)
                plt.text(ind / 2, 0.5, str(label), ha='center')
            else:
                plt.axvspan(prev_ind, ind, color=all_colors[label], alpha=0.5)
                plt.text((ind + prev_ind) / 2, 0.5, str(label), ha='center')
            axv_ind += 1
            prev_ind = ind


        probs = np.zeros(self.num_timestamp)
        probs[:] = -1 # means no label
        sorted_queried_plts = sorted(self.segmenter)

        for pl in sorted_queried_plts:
            pl_prob = pl.generate_probs()
            if len(pl_prob) % 2 == 0:
                probs[int(pl.c - len(pl_prob) / 2):int(pl.c + len(pl_prob) / 2)] = pl.class_label
            else:
                probs[int(pl.c - len(pl_prob) / 2):int(pl.c + len(pl_prob) / 2 + 1)] = pl.class_label
            # probs[int(pl.c - len(pl_prob) / 2):int(pl.c - len(pl_prob) / 2)+len(pl_prob)] = pl_prob
        plt.plot(probs[indice_start:indice_end], color="black")
        ax.set_title('Propagated/True class label')
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Class label")

        return plt


class Plateau:
    def __init__(self, x_len, c, w, s, l, c_, queried_ts_list, id):
        """
        Plateau class implementation for representing segments in long time series
        :param x_len: length of prob generated by this plateau
        :param c: center timestamp
        :param w: width
        :param s: slop
        :param l: class label
        :param c_: center for generating probs, can be float!
        """
        self.c = c
        self.c_ = c_
        self.w = w
        self.s = s
        self.score = 0
        self.class_label = l
        self.x = np.arange(x_len)
        self.queried_ts_list=queried_ts_list
        self.id = id

    def __gt__(self, plateau):
        return self.c > plateau.c

    def function(self, x):
        return 1 / ((np.exp(self.s * ((x - self.c_) - self.w)) + 1) * (np.exp(self.s * (-(x - self.c_) - self.w)) + 1))

    def generate_probs(self):
        # only output prob > 0.5
        return self.function(self.x)

    def generate_probs_n(self, n):
        # generate probs for n timestamps
        m = int(n/2)
        x = np.arange(n)
        x = x-(m-int(len(self.x)/2)) # move center c_
        return self.function(x)

    def start_end_timestamp(self):
        # convert c-w, c+w into real timestamps
        if len(self.x) % 2 == 0:
            start = int(self.c) - int(len(self.x)/2)
            end = int(self.c) + int(len(self.x)/2)
        else:
            start = int(self.c) - int(len(self.x)/2)
            end = int(self.c) + int(len(self.x)/2) + 1

        return start, end

    def propagation_timestamp(self, threshold=0.5):
        start, end = self.start_end_timestamp()
        x = np.arange(start, end)
        prop_x = x[self.generate_probs() > threshold]
        if len(prop_x) < 1:
            return 0, 0
        return prop_x[0], prop_x[-1]

    def __str__(self):
        return f"start_end:{self.start_end_timestamp()}, qts:{self.queried_ts_list}, x_len:{len(self.x)}" 
    def __eq__(self, other):
        if isinstance(other, Plateau):
            return self.id == other.id
        return False

    def copy(self):
        return Plateau(len(self.x), self.c, self.w, self.s, self.class_label, self.c_, self.queried_ts_list, self.id)



if __name__ == "__main__":
    start_t = time.time()
    num_class = 2
    num_timestamp = 1000
    conf = np.random.uniform(0,1,(num_class,num_timestamp))
    query_size = 10
    queried_timestamp = np.random.choice(np.arange(num_timestamp), query_size, replace=False)
    label_timestamp = np.random.choice(np.arange(num_class), query_size, replace=True)
    queried_timestamp_labels = zip(queried_timestamp, label_timestamp)


    seg = Plateaus(num_class, num_timestamp, tau=10)
    print(seg.num_cpu)
    seg.find_and_fit(conf)
    print("fitting done",len(seg.pred_plateaus))
    seg.add_plateaus(queried_timestamp_labels)
    print("plateaus added")
    print(seg.update_queried_plateaus())
    print("plateaus updated")



    seg.merge_and_split()
    print("merge_and_split")
    seg.check_duplicate_propagation()
    print(time.time()-start_t)
    y_true = np.random.choice(np.arange(num_class), replace=True, size=num_timestamp)
    seg.plot_segmenter(y_true)