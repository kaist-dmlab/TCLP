from tqdm import tqdm
import numpy as np
import tensorflow as tf



class DataLoader():
    def __init__(self, input_length, X, y, y_seg, mask, file_boundaries, seed, fully_supervised=False):
        self.input_length = input_length

        self.X_long = X # timestamp, dim
        self.y_long = y[:, np.newaxis] # timestamp, 1
        self.y_seg_long = y_seg[:, np.newaxis] # timestamp, 1
        self.mask_long = mask
        self.file_boundaries = file_boundaries
        self.num_class = len(np.unique(y))

        num_ts_per_class = [] # make maximum num_ts_per_class same as minimum num_ts_per_class by multiply the factor
        for i in range(self.num_class):
            num = np.sum(y[mask==1]==i)
            if num > 0:
                num_ts_per_class.append(num)
            else:
                num_ts_per_class.append(1)

        lr_class = np.min(num_ts_per_class)/np.array(num_ts_per_class)
        self.lr_mask = np.copy(mask)
        if fully_supervised == True:
            pass
        else:
            for i in range(self.num_class):
                self.lr_mask[(mask==1) & (y==i)] = lr_class[i]


    def dataset_generator(self, idxs, batch_size):
        '''
        Prefetch next batch before training with gpu is done for efficiency

        :param idxs: sampled indices for an epoch, the order should be kept same.
        :param batch_size: batch size
        :return: tf.dataset for training loop in model_manager.py
        '''

        tensor_data = self.batch_generator(idxs)
        dataset = tf.data.Dataset.from_tensor_slices(tensor_data).cache()
        return dataset.batch(batch_size).prefetch(1)


    def batch_generator(self, indices):
        '''

        :param indices: timestamps in long time series
        :return: generate windows for a tensor batch
        '''
        windowed_X = []
        windowed_y_seg = []
        windowed_y = []
        windowed_mask = [] # class-balanced learning rate for timestamp where label propagated, otw 0
        window_size = self.input_length
        for i in indices:
            windowed_X.append(self.X_long[i:i + window_size])
            windowed_y.append(self.y_long[i:i + window_size])
            windowed_y_seg.append(self.y_seg_long[i:i + window_size])
            windowed_mask.append(self.lr_mask[i:i + window_size])
        return tf.stack(windowed_X), tf.stack(windowed_y), tf.stack(windowed_y_seg), tf.stack(windowed_mask)

    def window_scoring(self, slide_size):
        '''

        :param slide_size: slide size for making a window in a batch
        :return: score oversampling for each class in a window
        '''
        y_long = self.y_long
        mask_long = tf.cast(self.mask_long, dtype=tf.bool).numpy()
        y = tf.cast(self.y_long, dtype=tf.float32).numpy()
        int_class, counts = np.unique(y_long[mask_long], return_counts=True)
        class_scoring_dict = {}
        for i in range(len(int_class)):
            class_scoring_dict[int_class[i]] = 1 / counts[i]  # low score for high frequency class
        for i in range(len(int_class)):
            y[y_long == i] = class_scoring_dict[int_class[i]]
        y[np.invert(mask_long)] = 0
        window_sample_prob = []
        indice_list = []
        window_size = self.input_length
        num_iter = (len(self.X_long) - window_size) // slide_size + 1
        for i in tqdm(range(num_iter), leave=False, desc="window_scoring"):
            label = y[i * slide_size:i * slide_size + window_size]
            mask = mask_long[i * slide_size:i * slide_size + window_size]
            file_boundary = self.file_boundaries[i * slide_size:i * slide_size + window_size]
            num_label = np.sum(mask)
            # if (num_label > 0) & (np.sum(file_boundary) == 0):
            if num_label > 0:
                score = np.sum(label[mask]) / num_label
                window_sample_prob.append(score)
                indice_list.append(i * slide_size) # save window indice where label exist
        window_sample_prob = np.array(window_sample_prob)
        window_sample_prob = window_sample_prob / np.sum(window_sample_prob)
        return window_sample_prob, np.array(indice_list)

