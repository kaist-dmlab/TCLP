import os
import sys
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass
tf.random.set_seed(int(sys.argv[6]))

from model_manager import ModelManager
import numpy as np
import preprocessing
import matplotlib.pyplot as plt

# python3 model_manager.py gpu data_name model_name input_length learning_rate seed epoch

data = preprocessing.Preprocessing(sys.argv[2], 0.1)
X_long, y_long, y_seg_long, file_boundaries = data.generate_long_time_series()
MM = ModelManager(model_name=sys.argv[3], input_length=int(sys.argv[4]), num_class=len(np.unique(y_long)),
                  dim=X_long.shape[1], lr=float(sys.argv[5]), seed=int(sys.argv[6]))
X_train, y_train, y_seg_train, _, file_boundaries_train, X_test, y_test, _2 ,_3, file_boundaries_test = MM.train_test_generator(X=X_long, y=y_long, y_seg=y_seg_long, mask=[], file_boundaries=file_boundaries)
mask = np.zeros(len(X_train))
mask[:]=1

MM.load_train_data(X=X_train, y=y_train, y_seg=y_seg_train, mask=mask, file_boundaries=file_boundaries_train, fully_supervised=True)
train_loss, test_acc = MM.train_model(int(sys.argv[7]), int(sys.argv[8]), True)
metrics = np.array(test_acc)
np.savetxt("./metadata/"+sys.argv[2]+"_"+sys.argv[3]+"_"+sys.argv[4]+"_"+sys.argv[5]+"_"+sys.argv[6]+"_"+sys.argv[8].__str__().replace(",","_")+".txt", metrics)
print(metrics)
for i in range(metrics.shape[1]-1):
    plt.plot(metrics[:,i], label=f"class {i}")
plt.plot(metrics[:,-1], label="total_acc", c="black")
plt.legend(loc="lower right")
plt.title(sys.argv[2]+"_"+sys.argv[3]+"_"+sys.argv[4]+"_"+sys.argv[5]+"_"+sys.argv[6]+"_"+sys.argv[8])
plt.savefig("./metadata/"+sys.argv[2]+"_"+sys.argv[3]+"_"+sys.argv[4]+"_"+sys.argv[5]+"_"+sys.argv[6]+"_"+sys.argv[8].__str__().replace(",","_")+".png")

print(np.max(metrics[:,-1]))

import tensorflow_probability as tfp
logits = MM.model.predict(X_test)
print(tfp.stats.expected_calibration_error(num_bins=10,logits=logits,labels_true=y_test))
print(MM.get_unlabeled_ECE())