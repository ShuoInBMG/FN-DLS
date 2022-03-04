import numpy as np

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Dropout, UpSampling2D, concatenate,Input, Add
from tensorflow.keras.layers import Conv2DTranspose, Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras import regularizers
import tensorflow as tf

# a metric function
def IoU_loss(ytrue, ypred):
    pred = tf.cast(tf.round(ypred),dtype=tf.float64)
    true = tf.cast(ytrue,dtype=tf.float64)
    differ = tf.subtract(pred, true)
    wrong_object = tf.clip_by_value(differ, 0, 1)
    #wrong_background = np.piecewise(differ, [differ==-1],[1,0])
    object_detected = tf.subtract(pred, wrong_object)
    object_all = tf.add(true, wrong_object)

    object_detected_sum = tf.reduce_sum(object_detected)
    object_all_sum = tf.reduce_sum(object_all)
    div = tf.divide(object_detected_sum, object_all_sum)
    out = tf.subtract(tf.constant(1, dtype=tf.float64), div)
    return div

# build the model
def segment_network_builder():
	input_ = Input(shape=(107,107,3),name='input')
	# initial → 105*105*8
	normalize = BatchNormalization(name='normalize')(input_)
	Conv2D_1_ = Conv2D(8, (3,3), padding='valid', strides=(1,1),name='conv_1_')(normalize)
	normalize_1_ = BatchNormalization(name='normalize_1')(Conv2D_1_)
	activation_1_ = Activation('tanh')(normalize_1_)

	# conv block size 105 no.1
	Conv2D_2_ = Conv2D(16, (3,3), padding='same', strides=(1,1),name='conv_2_')(activation_1_)
	normalize_2_ = BatchNormalization(name='normalize_2_')(Conv2D_2_)
	activation_2_ = Activation('tanh')(normalize_2_)

	# conv block size 105 no.2
	Conv2D_3_ = Conv2D(32, (3,3), padding='same', strides=(1,1),name='conv_3_')(activation_2_)
	normalize_3_ = BatchNormalization(name='normalize_3_')(Conv2D_2_)
	activation_3_ = Activation('tanh')(normalize_3_)

	# conv block size 105 no.3
	Conv2D_4_ = Conv2D(16, (3,3), padding='same', strides=(1,1),name='conv_4_')(activation_3_)
	normalize_4_ = BatchNormalization(name='normalize_4_')(Conv2D_4_)
	activation_4_ = Activation('tanh')(normalize_4_)

	# conv block size 105 no.4
	Conv2D_6_ = Conv2D(8, (3,3), padding='same', strides=(1,1),name='conv_6_',kernel_regularizer=regularizers.l2(0.02))(activation_4_)
	normalize_6_ = BatchNormalization(name='normalize_6_')(Conv2D_6_)

	# conv block size 105 no.1 identity
	Conv2D_5_ = Conv2D(8, (3,3), padding='valid', strides=(1,1),name='conv_5_',kernel_regularizer=regularizers.l2(0.02))(normalize)
	normalize_5_ = BatchNormalization(name='normalize_5')(Conv2D_5_)
	#activation_5_ = Activation('relu')(normalize_5_)

	add_1_ = Add()([normalize_6_, normalize_5_])
	activation_6_ = Activation('tanh')(add_1_)
	#normalize_1_ = BatchNormalization(name='normalize_1_')(add_1_)

	# convolution with size = 3
	MaxPooling_size3 = MaxPooling2D(pool_size=(3,3),name='maxpooling_size3')(activation_6_)

	# branch 3_1
	# conv block no.size3-1
	Conv2D_b31 = Conv2D(16, (7,7), padding='same', strides=(1,1),name='conv_b31')(MaxPooling_size3)
	normalize_b31 = BatchNormalization(name='normalize_b31')(Conv2D_b31)
	activation_b31 = Activation('tanh')(normalize_b31)
	# conv block no.size3-2
	Conv2D_b32 = Conv2D(32, (7,7), padding='same', strides=(1,1),name='conv_b32')(activation_b31)
	normalize_b32 = BatchNormalization(name='normalize_b32')(Conv2D_b32)
	activation_b32 = Activation('tanh')(normalize_b32)
	# conv block no.size3-3
	Conv2D_b33 = Conv2D(16, (7,7), padding='same', strides=(1,1),name='conv_b33')(activation_b32)
	normalize_b33 = BatchNormalization(name='normalize_b33')(Conv2D_b33)
	activation_b33 = Activation('tanh')(normalize_b33)

	Conv2D_b34 = Conv2D(8, (7,7), padding='same', strides=(1,1),name='conv_b34',kernel_regularizer=regularizers.l2(0.02))(activation_b33)
	normalize_b34 = BatchNormalization(name='normalize_b34')(Conv2D_b34)
	#activation_b34 = Activation('relu')(normalize_b34)

	# branch 3_2
	Conv2D_b35 = Conv2D(8, (7,7), padding='same', strides=(1,1),name='conv_b35',kernel_regularizer=regularizers.l2(0.02))(MaxPooling_size3)
	normalize_b35 = BatchNormalization(name='normalize_b35')(Conv2D_b35)

	# add residual
	add_3 = Add()([normalize_b34, normalize_b35])
	#normalize_b36 = BatchNormalization(name='normalize_b36')(add_3)
	activation_b35 = Activation('tanh')(add_3)
	# further pooling
	MaxPooling_size35 = MaxPooling2D(pool_size=(5,5),name='maxpooling_size35')(activation_b35)

	# convolution with size = 5
	MaxPooling_size5 = MaxPooling2D(pool_size=(5,5),name='maxpooling_size5')(activation_6_)

	# branch 3_1
	Conv2D_b51 = Conv2D(16, (7,7), padding='same', strides=(1,1),name='conv_b51')(MaxPooling_size5)
	normalize_b51 = BatchNormalization(name='normalize_b51')(Conv2D_b51)
	activation_b51 = Activation('tanh')(normalize_b51)

	Conv2D_b52 = Conv2D(32, (7,7), padding='same', strides=(1,1),name='conv_b52')(activation_b51)
	normalize_b52 = BatchNormalization(name='normalize_b52')(Conv2D_b52)
	activation_b52 = Activation('tanh')(normalize_b52)

	Conv2D_b53 = Conv2D(16, (7,7), padding='same', strides=(1,1),name='conv_b53')(activation_b52)
	normalize_b53 = BatchNormalization(name='normalize_b53')(Conv2D_b53)
	activation_b53 = Activation('tanh')(normalize_b53)

	Conv2D_b54 = Conv2D(8, (7,7), padding='same', strides=(1,1),name='conv_b54',kernel_regularizer=regularizers.l2(0.02))(activation_b53)
	normalize_b54 = BatchNormalization(name='normalize_b54')(Conv2D_b54)
	activation_b54 = Activation('tanh')(normalize_b54)

	# branch 3_2
	Conv2D_b55 = Conv2D(8, (7,7), padding='same', strides=(1,1),name='conv_b55',kernel_regularizer=regularizers.l2(0.02))(MaxPooling_size5)
	normalize_b55 = BatchNormalization(name='normalize_b55')(Conv2D_b55)
	#activation_b55 = Activation('relu')(normalize_b55)

	# add residual
	add_5 = Add()([normalize_b54, normalize_b55])
	#normalize_b56 = BatchNormalization(name='normalize_b56')(add_5)
	activation_b56 = Activation('tanh')(add_5)

	# further pooling
	MaxPooling_size53 = MaxPooling2D(pool_size=(3,3),name='maxpooling_size53')(activation_b56)

	# twist
	Twist_35 = concatenate([MaxPooling_size35, MaxPooling_size53],axis = -1)
	#normalize_t35 = BatchNormalization(name='twist_35')(Twist_35)
	Conv2D_t1 = Conv2D(8, (7,7), padding='same', strides=(1,1),name='conv_t1')(Twist_35)
	activation_t1 = Activation('tanh')(Conv2D_t1)

	# deconvolution with size = 5
	Upsampling_31 = UpSampling2D(size=(5,5),name='upsampling_31')(activation_t1)
	concatenate_3 = concatenate([Upsampling_31,activation_b35],axis=-1)

	# branch3_1
	Deconv_31 = Conv2DTranspose(16, (7,7), padding='same', strides=(1,1),name='deconv_31')(concatenate_3)
	normalize_b31d = BatchNormalization(name='normalize_b31d')(Deconv_31)
	activation_b31d = Activation('tanh')(normalize_b31d)

	Deconv_32 = Conv2DTranspose(32, (7,7), padding='same', strides=(1,1),name='deconv_32')(activation_b31d)
	normalize_b32d = BatchNormalization(name='normalize_b32d')(Deconv_32)
	activation_b32d = Activation('tanh')(normalize_b32d)

	Deconv_33 = Conv2DTranspose(16, (7,7), padding='same', strides=(1,1),name='deconv_33')(activation_b32d)
	normalize_b33d = BatchNormalization(name='normalize_b33d')(Deconv_33)
	activation_b33d = Activation('tanh')(normalize_b33d)

	Deconv_34 = Conv2DTranspose(8, (7,7), padding='same', strides=(1,1),name='deconv_34',kernel_regularizer=regularizers.l2(0.02))(activation_b33d)
	normalize_b34d = BatchNormalization(name='normalize_b34d')(Deconv_34)
	#activation_b34d = Activation('relu')(normalize_b34d)

	# branch 3_2
	Deconv_35 = Conv2DTranspose(8, (7,7), padding='same', strides=(1,1),name='deconv_35',kernel_regularizer=regularizers.l2(0.02))(concatenate_3)
	normalize_b35d = BatchNormalization(name='normalize_b35d')(Deconv_35)
	#activation_b35d = Activation('relu')(normalize_b35d)

	#add residual
	add_3d = Add()([normalize_b34d, normalize_b35d])
	#normallize_b34d = BatchNormalization(name='normalize_b34d')(add_3d)
	activation_b36d = Activation('tanh')(add_3d)

	#further upsampling to 105
	Upsampling_32 = UpSampling2D(size=(3,3),name='upsampling_32')(activation_b36d)

	# deconvolution with size = 3
	Upsampling_51 = UpSampling2D(size=(3,3),name='upsampling_51')(activation_t1)
	concatenate_5 = concatenate([Upsampling_51,activation_b56],axis=-1)

	# branch5_1
	Deconv_51 = Conv2DTranspose(16, (7,7), padding='same', strides=(1,1),name='deconv_51')(concatenate_5)
	normalize_b51d = BatchNormalization(name='normalize_b51d')(Deconv_51)
	activation_b51d = Activation('tanh')(normalize_b51d)

	Deconv_52 = Conv2DTranspose(32, (7,7), padding='same', strides=(1,1),name='deconv_52')(activation_b51d)
	normalize_b52d = BatchNormalization(name='normalize_b52d')(Deconv_52)
	activation_b52d = Activation('tanh')(normalize_b52d)

	Deconv_53 = Conv2DTranspose(16, (7,7), padding='same', strides=(1,1),name='deconv_53')(activation_b52d)
	normalize_b53d = BatchNormalization(name='normalize_b53d')(Deconv_53)
	activation_b53d = Activation('tanh')(normalize_b53d)

	Deconv_54 = Conv2DTranspose(8, (7,7), padding='same', strides=(1,1),name='deconv_54',kernel_regularizer=regularizers.l2(0.02))(activation_b53d)
	normalize_b54d = BatchNormalization(name='normalize_b54d')(Deconv_54)
	#activation_b54d = Activation('relu')(normalize_b54d)

	# branch 5_2
	Deconv_55 = Conv2DTranspose(8, (7,7), padding='same', strides=(1,1),name='deconv_55',kernel_regularizer=regularizers.l2(0.02))(concatenate_5)
	normalize_b55d = BatchNormalization(name='normalize_b55d')(Deconv_55)
	#activation_b55d = Activation('relu')(normalize_b55d)

	#add residual
	add_5d = Add()([normalize_b54d, normalize_b55d])
	#normallize_b54d = BatchNormalization(name='normalize_b54d')(add_5d)
	activation_b56d = Activation('tanh')(add_5d)

	#further upsampling to 105
	Upsampling_52 = UpSampling2D(size=(5,5),name='upsampling_33')(activation_b56d)

	# connect before and after to size 105*105*16
	concatenate_1 = concatenate([Upsampling_32,Upsampling_52],axis=-1)

	Deconv_c1 = Conv2DTranspose(16, (7,7), padding='same', strides=(1,1),name='deconv_c1',kernel_regularizer=regularizers.l2(0.02))(concatenate_1)
	normalize_c1 = BatchNormalization(name='normalize_c1')(Deconv_c1)
	activation_c1 = Activation('tanh')(normalize_c1)

	Deconv_c2 = Conv2DTranspose(8, (7,7), padding='same', strides=(1,1),name='deconv_c2',kernel_regularizer=regularizers.l2(0.02))(activation_c1)
	normalize_c2 = BatchNormalization(name='normalize_c2')(Deconv_c2)
	activation_c2 = Activation('tanh')(normalize_c2)

	Deconv_c3 = Conv2DTranspose(4, (5,5), padding='same', strides=(1,1),name='deconv_c3',kernel_regularizer=regularizers.l2(0.02))(activation_c2)
	normalize_c3 = BatchNormalization(name='normalize_c3')(Deconv_c3)

	concatenate_2 = concatenate([normalize_c3,activation_6_],axis=-1)

	Deconv_c4 = Conv2DTranspose(1, (3,3), padding='valid', strides=(1,1),name='deconv_c4',kernel_regularizer=regularizers.l2(0.02))(concatenate_2)
	#normalize_c4 = BatchNormalization(name='normalize_c4')(Deconv_c4)
	activation_c4 = Activation('sigmoid')(Deconv_c4)

	ConvNN_1 = Model(inputs=input_, outputs=activation_c4)
	ConvNN_1.compile(optimizer='adam',loss='binary_crossentropy',metrics=[IoU_loss,'accuracy'])

	return ConvNN_1

# train the model
def segment_network_trainer(image_path, label_path, val_size=0.3, batch_size=60, epochs=100, return_history=False):
	model = segment_network_builder()
	train_images = np.load(image_path)
	labels = np.load(label_path)
	train_images, labels = shuffle(train_images,labels)
	X_train, X_val, y_train, y_val = train_test_split(train_images,labels,test_size=val_size)

	history = model.fit(X_train, y_train,batch_size=batch_size ,epochs=epochs, validation_data=(X_val, y_val))

	model.save('segment_network.h5')

	if return_history == True:
		return history, model
	else:
		return model

def segment_network_loader(model_name='ConvNN_l2.h5'):
	print('You are going to use a pre-trained model')
	model = tf.keras.models.load_model(model_name)
	print(model.summary())
	print('--------')
	print('The model is loaded successfully')

	return model