import numpy as np
import tensorflow as tf
import cv2
import scipy.ndimage as ndi
from scipy.ndimage import measurements
from scipy.ndimage.morphology import generate_binary_structure

class afm_segment():
    def __init__(self):
        self.model = self.load_model()

    def load_model(self, model_name='ConvNN_l2.h5'):
        model = tf.keras.models.load_model(model_name)
        return model

    def afm_import(self, path):
        img = cv2.imread(path)
        return img

    def whole_predict(self, img):
        assert img.shape == (107,107,3)
        img_expand = np.expand_dims(self, img, axis=0)
        prediction = self.model.predict(img_expand)
        prediction = prediction.reshape(107,107)
        return prediction

    def conv_predict(self, img, conv_num):
        def combine(l, n):
            length = int(len(l)/n)
            slice_range = [l[i*length:i*length+length] for i in range(n)]
            slice_range_small = []
            for i in slice_range:
                i = np.hstack(i)
                slice_range_small.append(i)
            whole = np.vstack(slice_range_small)
            return whole
        img_c = cv2.resize(img, (int(conv_num*107),int(conv_num*107)))
        print('The target img shape is',img_c.shape)
        x_window_range = np.linspace(0,img_c.shape[0]-107,conv_num)
        y_window_range = np.linspace(0,img_c.shape[0]-107,conv_num)
        slice_range = [(i,j) for i in x_window_range for j in y_window_range]
        out_list = []
        for (i,j) in  slice_range:
            slice_window = img_c[int(i) : int(i + 107), int(j) : int(j + 107), :]
            out_list.append(slice_window)
            #print(int((i*conv_num+j)/107),'/',conv_num**2,'is under prediction')   
        first = []
        for i in out_list:
            i = np.expand_dims(i, axis=0)
            i = self.model.predict(i)
            i = i.reshape(107,107)
            first.append(i)
        print(len(first),'sub-pictures obtained, with shape',first[0].shape)
        undo = combine(first, conv_num)
        return undo

    def activation(segment, threshold):
        return np.ceil(np.clip(segment,threshold,1)-threshold)

    def get_large_connect(self, img, area_thresh = 20):
        i = img.copy()
        s = generate_binary_structure(2,2)
        label, number = measurements.label(img, structure = s)
        area = measurements.sum(img, label, index=range(label.max() + 1))
        areaImg = area[label]
        for y in range(i.shape[0]):
            for x in range(i.shape[1]):
                if areaImg[y,x] <= area_thresh:
                    i[y,x] = 0
                else:
                    i[y,x] = 1
        return i

    def split_segment(self, img, num_list=(7,8,9,11,13)):
        split_list = []
        for conv_num in num_list:
            segment = afm_segment.conv_predict(self, img, conv_num)
            split_list.append(segment)
        return split_list

    def posterior(self, segment):
        post = ndi.distance_transform_edt(segment)
        return post

    def posterior_check(self, threshold, split_list):
        weight_map = 0
        for segment in split_list:
            segment = cv2.resize(segment, (500,500))
            segment = afm_segment.activation(segment=segment, threshold=threshold)
            segment_check = ndi.distance_transform_edt(segment)
            weight_map += segment_check/len(split_list)
        weight_map[weight_map < 1] = 0
        return np.sign(weight_map)

    def segment_save(self, segment, file_path):
        plt.imshow(segment, cmap='binary')
        plt.axis('off')
        plt.savefig(file_path,bbox_inches='tight',pad_inches=0,dpi=300,figsize=(3,3))

    def segment_transfer(self, segment, target_size=(900,900)):
        pic = np.concatenate([segment,segment,segment],axis=-1) * 255
        pic = cv2.resize(pic, target_size)
        return pic