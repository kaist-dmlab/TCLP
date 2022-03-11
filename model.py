import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv1D, BatchNormalization, ReLU, Dropout, Softmax, InputLayer, LSTM, TimeDistributed
from tensorflow.keras import Model
from tensorflow.keras.losses import Loss
import numpy as np
from tqdm import tqdm



class TCN(Model):
    def __init__(self, num_class, dim, lr=0.001, num_block=3, kernel_size=3, num_filters=32, num_dilation=5, dropout_rate=0.2, *args, **kwargs):
        super(TCN, self).__init__(*args, **kwargs)
        self.num_class = num_class
        self.res_cnn = [] # layers for residual connection
        self.tcn_blocks = []
        for j in range(num_block):
            tcn = []
            if j < num_block - 1:
                self.res_cnn.append(Conv1D(filters=num_filters, kernel_size=1, strides=1, padding='same'))
            if j == 0:
                tcn.append(InputLayer(input_shape=(None, dim), batch_size=None))  # no batch axis for input_shape
            for i in range(num_dilation):
                tcn.append(Conv1D(filters=num_filters, kernel_size=kernel_size, strides=1, padding='same', dilation_rate = [2 ** i]))
                tcn.append(BatchNormalization())
                tcn.append(ReLU())
                tcn.append(Dropout(rate=dropout_rate))
            if j == num_block - 1:
                # tcn.append(Dense(num_batch00, activation="relu"))
                tcn.append(Dense(self.num_class, activation="softmax"))
            self.tcn_blocks.append(tcn)
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    def call(self, x, training = False):
        for i in range(len(self.tcn_blocks)):
            inputs = x
            for layer in self.tcn_blocks[i]:
                x = layer(x, training=training)
            if i < len(self.tcn_blocks)-1:
                inputs = self.res_cnn[i](inputs, training=training)
                x = x + inputs
            else:
                pass
        return x

    @tf.function
    def train_step(self, x, y, lossMask):
        with tf.GradientTape() as tape:
            output = self.call(x, training=True)
            # print(output)
            loss = self.loss(y, output)
            lossMask = tf.cast(lossMask, dtype=tf.float32)
            loss = tf.math.multiply(loss, lossMask) / tf.math.reduce_sum(lossMask)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return tf.math.reduce_sum(loss)

    def predict(self, X_long, file_boundaries=[]):
        file_boundary_ind = np.where(file_boundaries==1)[0].tolist()
        start = 0  # test_data_start_ind
        if len(file_boundary_ind) > 0:
            if not len(file_boundaries)-1 in file_boundary_ind:
                file_boundary_ind.append(len(file_boundaries)-1)
            for i in file_boundary_ind:
                output_final_file = self.call(X_long[np.newaxis, start:i+1]).numpy()[0]
                if start == 0:
                    output_final = output_final_file
                else:
                    output_final = np.concatenate([output_final, output_final_file], axis=0)
                start = i+1
        else:
            output_final = self.call(X_long[np.newaxis, :, :]).numpy()
        output_final = output_final.reshape((-1, self.num_class))
        return output_final


    def call_penultimate(self, x, training=False):
        for i in range(len(self.tcn_blocks)):
            inputs = x
            for j in range(len(self.tcn_blocks[i])):
                if i == len(self.tcn_blocks)-1 and j==len(self.tcn_blocks[i])-1:
                    continue
                else:
                    x = self.tcn_blocks[i][j](x, training=training)
            if i < len(self.tcn_blocks)-1:
                inputs = self.res_cnn[i](inputs, training=training)
                x = x + inputs
            else:
                pass
        return x

    def predict_penultimate(self, X_long, file_boundaries=[]):
        file_boundary_ind = np.where(file_boundaries == 1)[0].tolist()
        if len(file_boundary_ind)==0:
            output_final = self.call_penultimate(X_long[np.newaxis, :]).numpy()[0]
        else:
            if not (len(X_long) in file_boundary_ind):
                file_boundary_ind.append(len(X_long))
            # print(file_boundary_ind)
            start = 0
            for i in file_boundary_ind:
                output_final_file = self.call_penultimate(X_long[np.newaxis, start:i]).numpy()[0]
                if start == 0:
                    output_final = output_final_file
                else:
                    output_final = np.concatenate([output_final, output_final_file], axis=0)
                start = i
        return output_final

    def call_gradient(self, x, y, training=False):
        '''

        :param x: array-like instance with shape (batch, timestamp, dim)
        :param y: sparse label vector (batch, timestamp, dim=1)
        :return: gradient vector of each timestamp (timestamp, penultimate_dim * num_class)
        '''
        for i in range(len(self.tcn_blocks)):
            inputs = x
            for j in range(len(self.tcn_blocks[i])):
                if i == len(self.tcn_blocks)-1 and j==len(self.tcn_blocks[i])-1:
                    penultimate_output = x
                x = self.tcn_blocks[i][j](x, training=training)
            if i < len(self.tcn_blocks)-1:
                inputs = self.res_cnn[i](inputs, training=training)
                x = x + inputs
            else:
                pass

        cout = tf.reshape(tf.squeeze(x),[x.shape[1],self.num_class,1]) # remove batch axis -> timestamp, num_class
        out = tf.reshape(tf.squeeze(penultimate_output), [x.shape[1],1,penultimate_output.shape[2]]) # timestamp, num_channel
        y = tf.reshape(tf.squeeze(tf.one_hot(y, depth=self.num_class)),[x.shape[1],self.num_class,1])# timestamp, num_class
        dy_dz = cout - y
        gradient = tf.reshape(tf.matmul(dy_dz,out), [x.shape[1],-1])
        return gradient

    def get_gradient(self, X_long, y_long, file_boundaries):
        file_boundary_ind = np.where(file_boundaries == 1)[0].tolist()
        # file_boundary_ind = file_boundary_ind[file_boundary_ind<=len(X_long)//2].tolist()
        if len(file_boundary_ind)==0:
            output_final = self.call_gradient(X_long[np.newaxis, :], y_long[np.newaxis, :]).numpy()
        else:
            if not (len(X_long) in file_boundary_ind):
                file_boundary_ind.append(len(X_long))
            start = 0
            for i in file_boundary_ind:
                output_final_file = self.call_gradient(X_long[np.newaxis, start:i], y_long[np.newaxis, start:i]).numpy()
                if start == 0:
                    output_final = output_final_file
                else:
                    output_final = np.concatenate([output_final, output_final_file], axis=0)
                start = i
        return output_final


class TMSE_loss(Loss):
    def call(self, y_true, y_pred, max_value=4, reduction="mean"):

        y_pred = tf.clip_by_value(y_pred, clip_value_min=1e-8, clip_value_max=1)
        delta_tc_square = tf.keras.metrics.mean_squared_error(tf.math.log(y_pred[:,1:,:]),tf.stop_gradient(tf.math.log(y_pred[:,:-1,:])))
        delta_tc_tilda = tf.clip_by_value(delta_tc_square, clip_value_min=0, clip_value_max=max_value**2)
        if reduction == "mean":
            return tf.math.reduce_mean(delta_tc_tilda)
        elif reduction == "none":
            return delta_tc_tilda
        else:
            raise NotImplementedError

class LLAL_loss(Loss):
    def call(self, y_true, y_pred, margin=1.0):
        target_loss = tf.stop_gradient(y_true)
        idx_half = len(y_pred)//2
        diff_pred = (y_pred-tf.reverse(y_pred,axis=[0]))[:idx_half]
        diff_target = (target_loss-tf.reverse(target_loss,axis=[0]))[:idx_half]

        one = 2*tf.math.sign(tf.clip_by_value(diff_target, clip_value_min=0, clip_value_max=tf.float32.max)) - 1
        loss = tf.math.reduce_sum(tf.clip_by_value(margin-one*diff_pred, clip_value_min=0, clip_value_max=tf.float32.max))
        return loss

class MSTCN(Model):
    def __init__(self, num_class, dim, lr=0.005, num_stage=4, kernel_size=3, num_filters=64, num_dilation=10, dropout_rate=0.5, is_LLAL=False, *args, **kwargs):
        super(MSTCN, self).__init__(*args, **kwargs)
        self.num_class = num_class
        self.num_dilation = num_dilation
        self.num_stage = num_stage
        self.num_filters = num_filters
        self.tcn_stage = []

        # self.mask = Masking(mask_value=0.0)
        for j in range(num_stage):
            tcn = []

            tcn.append(Conv1D(filters=num_filters, kernel_size=1, strides=1, padding='same'))
            for i in range(num_dilation):
                dilated_conv = []
                dilated_conv.append(Conv1D(filters=num_filters, kernel_size=kernel_size, strides=1, padding='same', dilation_rate = [2 ** i]))
                dilated_conv.append(BatchNormalization())
                dilated_conv.append(ReLU())
                dilated_conv.append(Conv1D(filters=num_filters, kernel_size=1, strides=1, padding='same'))
                dilated_conv.append(Dropout(rate=dropout_rate))
                tcn.append(dilated_conv)
            tcn.append(Conv1D(filters=num_class, kernel_size=1, strides=1, padding='same', activation=None))
            tcn.append(Softmax())
            self.tcn_stage.append(tcn)
        self.cls_loss = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        self.seg_loss = TMSE_loss()
        self.seg_loss_no_reduction = TMSE_loss(reduction="none")
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        self.is_LLAL = is_LLAL
        if self.is_LLAL:
            # https://github.com/Mephisto405/Learning-Loss-for-Active-Learning/blob/master/models/lossnet.py
            self.fc_list = []
            for j in range(num_stage-1):
                self.fc_list.append(Dense(units = 128, activation="relu"))
            self.fc_list.append(Dense(units=1, activation="linear"))
        self.llal_loss = LLAL_loss()

    def call(self, x, training = False):
        for i in range(len(self.tcn_stage)):
            for j in range(self.num_dilation + 3):
                if 0 < j and j < self.num_dilation + 1:
                    input = x
                    for k in range(5):
                        x = self.tcn_stage[i][j][k](x, training=training)
                    x = input + x
                else:
                    x = self.tcn_stage[i][j](x, training=training)
        return x

    def call_training(self, x, training = False):
        outputs = []
        for i in range(len(self.tcn_stage)): # loop for
            for j in range(self.num_dilation+3):
                if 0 < j and j < self.num_dilation + 1:
                    input = x
                    for k in range(5):
                        x = self.tcn_stage[i][j][k](x, training=training)
                    x = input + x
                else:
                    x = self.tcn_stage[i][j](x, training=training)
            outputs.append(x)
        return outputs

    @tf.function
    def train_step_llal_full(self, x, y, lossMask, lambd=0.15):
        with tf.GradientTape() as tape:
            outputs = self.call_training(x, training=True)
            lossMask = tf.cast(lossMask, dtype=tf.float32)
            loss = tf.math.divide(tf.math.reduce_sum(tf.math.multiply(self.cls_loss(y, outputs[0]),lossMask)),tf.math.reduce_sum(tf.cast(lossMask!=0,tf.float32))) + lambd*self.seg_loss([],outputs[0])
            for i in range(len(self.tcn_stage)-1):
                loss += tf.math.divide(tf.math.reduce_sum(tf.math.multiply(self.cls_loss(y, outputs[i+1]),lossMask)), tf.math.reduce_sum(tf.cast(lossMask!=0,tf.float32))) + lambd*self.seg_loss([],outputs[i+1])
            y_hat = tf.math.argmax(outputs[len(self.tcn_stage)-1],axis=2)
            target_loss = tf.math.multiply(self.cls_loss(y_hat, outputs[len(self.tcn_stage)-1]),lossMask) + lambd*self.seg_loss_no_reduction([],outputs[len(self.tcn_stage)-1]) # after some epochs, do not propagate gradients to MSTCN
            input_LLAL = self.input_LLAL(x, training=True) # x.shape = batch, timestamp, dim / input_LLAL.shape = batch, timestamp, num_stage*num_filter
            output_LLAL = self.call_LLAL(input_LLAL, training=True) # shape = timestamp, 1
            # print(loss,tf.math.divide(tf.math.reduce_sum(tf.math.multiply(self.llal_loss(target_loss, output_LLAL),tf.cast(lossMask!=0,tf.float32))), tf.math.reduce_sum(tf.cast(lossMask!=0,tf.float32))))
            loss += 0.000001*tf.math.reduce_mean(self.llal_loss(target_loss, output_LLAL)) # currently, lambda=1, but can be changed TODO: change the lambda for learning loss module
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss

    @tf.function
    def train_step_llal_part(self, x, y, lossMask, lambd=0.15):
        with tf.GradientTape() as tape:
            outputs = self.call_training(x, training=True)
            lossMask = tf.cast(lossMask, dtype=tf.float32)
            loss = tf.math.divide(tf.math.reduce_sum(tf.math.multiply(self.cls_loss(y, outputs[0]),lossMask)),tf.math.reduce_sum(tf.cast(lossMask!=0,tf.float32))) + lambd*self.seg_loss([],outputs[0])
            for i in range(len(self.tcn_stage)-1):
                loss += tf.math.divide(tf.math.reduce_sum(tf.math.multiply(self.cls_loss(y, outputs[i+1]),lossMask)), tf.math.reduce_sum(tf.cast(lossMask!=0,tf.float32))) + lambd*self.seg_loss([],outputs[i+1])
            y_hat = tf.math.argmax(outputs[len(self.tcn_stage)-1],axis=2)
            target_loss = tf.math.multiply(self.cls_loss(y_hat, outputs[len(self.tcn_stage)-1]),lossMask) + lambd*self.seg_loss_no_reduction([],outputs[len(self.tcn_stage)-1]) # after some epochs, do not propagate gradients to MSTCN
            input_LLAL = self.input_LLAL(x, training=False) # x.shape = batch, timestamp, dim / input_LLAL.shape = batch, timestamp, num_stage*num_filter
            output_LLAL = self.call_LLAL(input_LLAL, training=True) # shape = timestamp, 1
            loss += 0.000001*tf.math.reduce_mean(self.llal_loss(target_loss, output_LLAL)) # currently, lambda=1, but can be changed
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss

    @tf.function
    def train_step(self, x, y, lossMask, lambd=0.15):
        with tf.GradientTape() as tape:
            outputs = self.call_training(x, training=True)
            lossMask = tf.cast(lossMask, dtype=tf.float32)
            loss = tf.math.divide(tf.math.reduce_sum(tf.math.multiply(self.cls_loss(y, outputs[0]),lossMask)),tf.math.reduce_sum(tf.cast(lossMask!=0,tf.float32))) + lambd*self.seg_loss([],outputs[0])
            for i in range(len(self.tcn_stage)-1):
                loss += tf.math.divide(tf.math.reduce_sum(tf.math.multiply(self.cls_loss(y, outputs[i+1]),lossMask)), tf.math.reduce_sum(tf.cast(lossMask!=0,tf.float32))) + lambd*self.seg_loss([],outputs[i+1])
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss

    def predict(self, X_long, file_boundaries=[]):
        file_boundary_ind = np.where(file_boundaries==1)[0].tolist()
        start = 0  # test_data_start_ind
        if len(file_boundary_ind) > 0:
            if not len(file_boundaries)-1 in file_boundary_ind:
                file_boundary_ind.append(len(file_boundaries)-1)
            for i in file_boundary_ind:
                output_final_file = self.call(X_long[np.newaxis, start:i+1]).numpy()[0]
                if start == 0:
                    output_final = output_final_file
                else:
                    output_final = np.concatenate([output_final, output_final_file], axis=0)
                start = i+1
        else:
            output_final = self.call(X_long[np.newaxis, :, :]).numpy()
        output_final = output_final.reshape((-1, self.num_class))

        return output_final

    def call_logit(self, x, training=False):
        for i in range(len(self.tcn_stage)): # loop for
            for j in range(self.num_dilation+3):
                if 0 < j and j < self.num_dilation + 1:
                    input = x
                    for k in range(5):
                        x = self.tcn_stage[i][j][k](x, training=training)
                    x = input + x
                else:
                    if i==len(self.tcn_stage)-1 and j == self.num_dilation+2:
                        break
                    x = self.tcn_stage[i][j](x, training=training)
        return x
    def predict_logit(self, X_long, file_boundaries=[]):
        file_boundary_ind = np.where(file_boundaries == 1)[0].tolist()
        if len(file_boundary_ind)==0:
            output_final = self.call_logit(X_long[np.newaxis, :]).numpy()[0]
        else:
            if not (len(X_long) in file_boundary_ind):
                file_boundary_ind.append(len(X_long))
            # print(file_boundary_ind)
            start = 0
            for i in file_boundary_ind:
                output_final_file = self.call_logit(X_long[np.newaxis, start:i]).numpy()[0]
                if start == 0:
                    output_final = output_final_file
                else:
                    output_final = np.concatenate([output_final, output_final_file], axis=0)
                start = i
        return output_final

    def call_penultimate(self, x, training=False):
        for i in range(len(self.tcn_stage)): # loop for
            for j in range(self.num_dilation+3):
                if 0 < j and j < self.num_dilation + 1:
                    input = x
                    for k in range(5):
                        x = self.tcn_stage[i][j][k](x, training=training)
                    x = input + x
                else:
                    if i==len(self.tcn_stage)-1 and j == self.num_dilation+1:
                        break
                    x = self.tcn_stage[i][j](x, training=training)
        return x

    def predict_penultimate(self, X_long, file_boundaries=[]):
        file_boundary_ind = np.where(file_boundaries == 1)[0].tolist()
        if len(file_boundary_ind)==0:
            output_final = self.call_penultimate(X_long[np.newaxis, :]).numpy()[0]
        else:
            if not (len(X_long) in file_boundary_ind):
                file_boundary_ind.append(len(X_long))
            # print(file_boundary_ind)
            start = 0
            for i in file_boundary_ind:
                output_final_file = self.call_penultimate(X_long[np.newaxis, start:i]).numpy()[0]
                if start == 0:
                    output_final = output_final_file
                else:
                    output_final = np.concatenate([output_final, output_final_file], axis=0)
                start = i
        return output_final

    def call_gradient(self, x, y, training=False):
        '''

        :param x: array-like instance with shape (batch, timestamp, dim)
        :param y: sparse label vector (batch, timestamp, dim=1)
        :return: gradient vector of each timestamp (timestamp, penultimate_dim * num_class)
        '''
        for i in range(len(self.tcn_stage)): # loop for
            for j in range(self.num_dilation+3):
                if 0 < j and j < self.num_dilation + 1:
                    input = x
                    for k in range(5):
                        x = self.tcn_stage[i][j][k](x, training=training)
                    x = input + x
                else:
                    if i==len(self.tcn_stage)-1 and j == self.num_dilation+1:
                        penultimate_output = x
                    x = self.tcn_stage[i][j](x, training=training)

        cout = tf.reshape(tf.squeeze(x),[x.shape[1],self.num_class,1]) # remove batch axis -> timestamp, num_class
        out = tf.reshape(tf.squeeze(penultimate_output), [x.shape[1],1,penultimate_output.shape[2]]) # timestamp, 1, num_channel
        y = tf.reshape(tf.squeeze(tf.one_hot(y, depth=self.num_class)),[x.shape[1],self.num_class,1])# timestamp, num_class, 1
        dy_dz = cout - y

        cout_np = tf.clip_by_value(cout, clip_value_min=1e-8, clip_value_max=1).numpy().reshape(x.shape[1],self.num_class)
        delta_raw = (tf.math.log(cout_np[1:,:])-tf.math.log(cout_np[:-1,:])).numpy() # timestamp-1, num_class
        delta = tf.math.abs(delta_raw).numpy()
        delta_tilda = tf.clip_by_value(delta, clip_value_min=0, clip_value_max=4).numpy()

        d_delta_dw = 1/cout_np[1:,:]
        d_delta_dw[delta_raw<0] = d_delta_dw[delta_raw<0]*-1
        d_delta_dw[delta>4] = 0
        d_delta_dw = 2 * d_delta_dw * delta_tilda / ((x.shape[1]-1)*self.num_class)
        d_delta_dw_complete = np.zeros((x.shape[1],self.num_class))
        d_delta_dw_complete[1:,:] = d_delta_dw

        d_delta_dw_complete = tf.cast(tf.reshape(tf.constant(d_delta_dw_complete), [x.shape[1],self.num_class,1]),tf.dtypes.float32)
        gradient = tf.reshape(tf.matmul(dy_dz,out) + 0.15*tf.matmul(d_delta_dw_complete,out), [x.shape[1],-1])
        return gradient

    def get_gradient(self, X_long, y_long, file_boundaries):
        file_boundary_ind = np.where(file_boundaries == 1)[0].tolist()
        # file_boundary_ind = file_boundary_ind[file_boundary_ind<=len(X_long)//2].tolist()
        if len(file_boundary_ind)==0:
            output_final = self.call_gradient(X_long[np.newaxis, :], y_long[np.newaxis, :]).numpy()
        else:
            if not (len(X_long) in file_boundary_ind):
                file_boundary_ind.append(len(X_long))
            start = 0
            for i in file_boundary_ind:
                output_final_file = self.call_gradient(X_long[np.newaxis, start:i], y_long[np.newaxis, start:i]).numpy()
                if start == 0:
                    output_final = output_final_file
                else:
                    output_final = np.concatenate([output_final, output_final_file], axis=0)
                start = i
        return output_final

    def call_LLAL(self, x, training=False):
        '''

        :param x: penultimate output from all stages (batch, timestamp, num_stage*num_filter)
        :return: predicted loss whose shape is batch, timestamp,1
        '''
        interm_outputs = []
        for j in range(self.num_stage-1):
            interm_outputs.append(self.fc_list[j](x[:,:,j*self.num_filters:(j+1)*self.num_filters], training=training))
        concat_output = tf.concat(interm_outputs, axis=2)
        pred_loss = tf.squeeze(self.fc_list[-1](concat_output))
        return pred_loss

    def input_LLAL(self,x, training=False):
        '''

        :param x: time series (batch, timestamp, dim)
        :return: input for loss prediction module (batch, timestamp, num_stage*num_filter)
        '''
        output = []
        for i in range(len(self.tcn_stage)):
            for j in range(self.num_dilation+3):
                if 0 < j and j < self.num_dilation + 1:
                    input = x
                    for k in range(5):
                        x = self.tcn_stage[i][j][k](x, training=training)
                    x = input + x
                else:
                    if i < len(self.tcn_stage)-1 and j == self.num_dilation+1: # get all penultimate output from stages except final stage
                        output.append(x) # x.shape = timestamp, num_filters
                    x = self.tcn_stage[i][j](x, training=training)
        return tf.concat(output, axis=2) # output.shape = timestamp, num_stage*num_filters

    def predict_loss(self, X_long, file_boundaries):
        file_boundary_ind = np.where(file_boundaries == 1)[0].tolist()
        # file_boundary_ind = file_boundary_ind[file_boundary_ind<=len(X_long)//2].tolist()
        if len(file_boundary_ind)==0:
            output_final = self.input_LLAL(X_long[np.newaxis, :])
            output_final = self.call_LLAL(output_final).numpy()
        else:
            if not (len(X_long) in file_boundary_ind):
                file_boundary_ind.append(len(X_long))
            start = 0
            for i in file_boundary_ind:
                output_final_file = self.input_LLAL(X_long[np.newaxis, start:i])
                output_final_file = self.call_LLAL(output_final_file).numpy()
                if start == 0:
                    output_final = output_final_file
                else:
                    output_final = np.concatenate([output_final, output_final_file], axis=0)
                start = i
        return output_final

    def call_loss(self, x, y, lambd=0.15):
        '''

        :param x: times series data (batch, timestamp, dim)
        :param y: confident class label for each timestamp (batch, timestamp)
        :return: target loss made from confident class(batch, timestamp)
        '''
        output = self.call(x, training=False)
        target_loss = self.cls_loss(y, output) + lambd*self.seg_loss_no_reduction([],output)
        target_loss = tf.squeeze(target_loss)
        return target_loss


    def get_target_loss(self, X_long, y_long, file_boundaries):
        # y_long should be argmax(y_pred)
        file_boundary_ind = np.where(file_boundaries == 1)[0].tolist()
        if len(file_boundary_ind)==0:
            output_final = self.call_loss(X_long[np.newaxis, :], y_long[np.newaxis, :]).numpy()
        else:
            if not (len(X_long) in file_boundary_ind):
                file_boundary_ind.append(len(X_long))
            start = 0
            for i in file_boundary_ind:
                output_final_file = self.call_loss(X_long[np.newaxis, start:i], y_long[np.newaxis, start:i]).numpy()
                if start == 0:
                    output_final = output_final_file
                else:
                    output_final = np.concatenate([output_final, output_final_file], axis=0)
                start = i
        return output_final




class RNN(Model):
    def __init__(self, num_class, dim, lr=0.001, num_layer=3, num_hidden=100, *args, **kwargs):
        super(RNN, self).__init__(*args, **kwargs)
        self.num_class = num_class
        rnn_layers  = []
        rnn_layers.append(InputLayer(input_shape=(None, dim), batch_size=None))
        for i in range(num_layer):
            rnn_layers.append(LSTM(num_hidden, return_sequences=True))
        rnn_layers.append(TimeDistributed(Dense(num_class, activation="softmax")))
        self.model = Sequential(rnn_layers)

        self.loss = tf.keras.losses.SparseCategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    def call(self, x, training=False):
        return self.model(x)

    @tf.function
    def train_step(self, x, y, lossMask):
        with tf.GradientTape() as tape:
            output = self.call(x, training=True)
            # print(output)
            loss = self.loss(y, output)
            lossMask = tf.cast(lossMask, dtype=tf.float32)
            loss = tf.math.multiply(loss, lossMask) / tf.math.reduce_sum(lossMask)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return tf.math.reduce_sum(loss)

    def predict(self, X_long, file_boundaries=[]):
        file_boundary_ind = np.where(file_boundaries==1)[0].tolist()
        start = 0  # test_data_start_ind
        if len(file_boundary_ind) > 0:
            if not len(file_boundaries)-1 in file_boundary_ind:
                file_boundary_ind.append(len(file_boundaries)-1)
            for i in file_boundary_ind:
                output_final_file = self.call(X_long[np.newaxis, start:i+1]).numpy()[0]
                if start == 0:
                    output_final = output_final_file
                else:
                    output_final = np.concatenate([output_final, output_final_file], axis=0)
                start = i+1
        else:
            output_final = self.call(X_long[np.newaxis, :, :]).numpy()
        output_final = output_final.reshape((-1, self.num_class))
        return output_final


if __name__=="__main__":
    input_length = 256
    num_batch = 32
    dim = 20
    num_class = 10
    total_timestamp = 200000
    file_boundaries = np.zeros(total_timestamp)
    file_boundaries[50000]=1
    file_boundaries[70000]=1
    file_boundaries[120000]=1
    tcn = MSTCN(num_class,dim,is_LLAL=True) # num_class, dim
    print("call", tcn(np.random.rand(num_batch,input_length,dim)).shape)  # batch, timestamp, dim
    # need to call the model at least one time to initialize parameters of the model
    print("call_penul", tcn.call_penultimate(np.random.rand(num_batch,input_length,dim)).shape)  # batch, timestamp, dim
    # print("call_training", tcn.call_training(np.random.rand(num_batch,input_length,dim)))  # batch, timestamp, dim

    print("pred_penul", tcn.predict_penultimate(np.zeros((total_timestamp,dim)),file_boundaries).shape)
    for i in range(1):
        print(tcn.train_step(np.random.rand(num_batch,input_length,dim), np.zeros((num_batch,input_length)), np.ones((num_batch,input_length)),curr_epoch=40,file_boundaries=file_boundaries)) # if current epoch is over 80% of total epochs
        print("call_after_training_one_step", np.sum(tf.math.is_nan(tcn(np.random.rand(num_batch,input_length,dim).astype(np.float32))).numpy()))

    g=tcn.call_gradient(np.random.rand(1,total_timestamp,dim), np.random.randint(num_class,size=total_timestamp)[np.newaxis,:])
    G = tcn.get_gradient(np.random.rand(total_timestamp,dim), np.random.randint(num_class,size=total_timestamp), file_boundaries)
    print(g.shape)
    print(G.shape)
    print(tcn.predict_loss(np.zeros((total_timestamp,dim)),file_boundaries).shape)
    print(tcn.get_target_loss(np.random.rand(total_timestamp,dim), np.random.randint(num_class,size=total_timestamp), file_boundaries).shape)

    tcn.summary()
