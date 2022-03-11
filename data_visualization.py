import numpy as np
import pandas as pd
from preprocessing import Preprocessing
from matplotlib import pyplot as plt
import matplotlib.colors as pltc
import os
from datetime import datetime


class DataVisualization():
    def __init__(self, data_name):
        self.data_name = data_name
        dataloader = Preprocessing(data_name, 0.1)
        self.X, self.y, self.y_seg, self.file_boundaries = dataloader.generate_long_time_series()
        self.all_colors = [k for k,v in pltc.cnames.items()]

    def compute_segment_location(self, indice_start, indice_end):
        duration = []
        ind = 0
        for label_ts_prev, label_ts in zip(self.y[indice_start:indice_end - 1], self.y[indice_start + 1:indice_end]):
            ind += 1
            if label_ts_prev != label_ts:
                duration.append([int(self.y[ind - 1]), ind])
        if duration[-1][-1] < indice_end:
            duration.append([int(self.y[-1]), indice_end-indice_start])
        return duration

    def graph(self, start, length):

        indice_start = start
        indice_end = start+length

        duration = self.compute_segment_location(indice_start, indice_end)

        plt.figure(figsize=(15, 2))
        if (self.X.shape[1] < 100):
            plt.plot(self.X[indice_start:indice_end])

        axv_ind = 0
        for label, ind in duration:
            if axv_ind == 0:
                plt.axvspan(0, ind, color=self.all_colors[label], alpha=0.5)
                plt.text(ind / 2, 0.5, str(label), ha='center')
            else:
                plt.axvspan(prev_ind, ind, color=self.all_colors[label], alpha=0.5)
                plt.text((ind + prev_ind) / 2, 0.5, str(label), ha='center')
            axv_ind += 1
            prev_ind = ind

        plt.plot(self.y_seg[indice_start:indice_end])

        plt.xlabel(str((1/sampling_rate_dict[self.data_name]))+" sec")
        plt.ylabel("Raw value")
        plt.tight_layout()
        plt.show()

        # plt.savefig("./figures/raw_data_" + self.data_name + "_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".png", dpi=300, bbox_inches='tight')
        plt.savefig("./figures/raw_data_" + self.data_name + ".png", dpi=300, bbox_inches='tight')

    def segment_analysis(self):
        duration = self.compute_segment_location(0, len(self.X))
        num_class = len(np.unique(self.y))
        segment_len_class = []
        for i in range(num_class):
            segment_len_class.append([])
        prev_ind = 0
        for label, ind in duration:
            segment_len_class[label].append(ind - prev_ind)
            prev_ind = ind
        fig, axs = plt.subplots(num_class, 1, figsize=(4,num_class*1.5))
        mean = []
        std = []
        num_seg = []
        for i in range(num_class):
            print("Average Segment Length of Class " + str(i) + ":", np.mean(segment_len_class[i]))
            mean.append(np.mean(segment_len_class[i]))
            print("STD of Segment Length of Class " + str(i) + ":", np.std(segment_len_class[i]))
            std.append(np.std(segment_len_class[i]))
            print("Number of Segment of Class " + str(i) + ":", len(segment_len_class[i]))
            num_seg.append(len(segment_len_class[i]))
            axs[i].hist(segment_len_class[i])
            axs[i].set_title(self.data_name+"-"+str(i))
        segment_len_all = np.concatenate(segment_len_class)
        print("Average segment length: ", np.mean(segment_len_all))
        mean.append(np.mean(segment_len_all))
        print("STD segment length: ", np.std(segment_len_all))
        std.append(np.std(segment_len_all))
        print("Number of segments: ", len(segment_len_all))
        num_seg.append(len(segment_len_all))
        segment_info_df = pd.DataFrame([mean,std,num_seg])
        segment_info_df = segment_info_df.transpose()
        segment_info_df.columns=["mean","std","#segments"]
        segment_info_df.to_excel(os.path.join(os.getcwd(),"figures",self.data_name+".xlsx"))
        plt.tight_layout()
        fig.show()
        fig.savefig("./figures/segment_hist_" + self.data_name + ".png", dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    sampling_rate_dict = {"50salads": 30, "GTEA": 15, "SAMSUNG": 1, "Sleep": 100, "HAPT": 50, "Breakfast":15,
                          "HASC_BDD": 125, "PAMAP2": 100, "en-disease": 1, "ECG":500, "mHealth":50}

    for name in ["50salads","GTEA","HAPT","mHealth"]:
    # for name in ["mHealth"]:
        data = Preprocessing(name,boundary_ratio=0.1)
        X_long, y_long, y_seg_long, file_boundaries = data.generate_long_time_series()
        print(type(X_long),type(y_long),type(y_seg_long),type(file_boundaries))
        print("########### Data Specification ###########")
        print("Number of timestamp and data dimension:", X_long.shape)
        print("Number of class:", len(np.unique(y_long)))
        print("Class label:", np.unique(y_long))
        for i in range(len(np.unique(y_long))):
            print("Number of timestamp for class " + str(i) + ":", len(np.where(y_long == i)[0]))
        print("Number of boundary class:", len(np.unique(y_seg_long)))
        print("Boundary Class label:", np.unique(y_seg_long))
        for i in range(len(np.unique(y_seg_long))):
            print("Number of timestamp for Boundary Class " + str(i) + ":",
                  len(np.where(y_seg_long == i)[0]))
        print(f"file boundaries {np.where(file_boundaries == 1)[0]}")
        prev = 0
        file_length = []
        for i in np.where(file_boundaries == 1)[0]:
            file_length.append(i-prev)
            prev = i
        print(f"mean file length: {np.mean(file_length)}")
        dv = DataVisualization(name)
        # dv.graph(0, int(len(dv.X)/10))
        dv.graph(0, int(len(y_long)*0.1))
        dv.segment_analysis()
        print("\n\n")
        