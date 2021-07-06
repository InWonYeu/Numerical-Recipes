import tensorflow as tf
import numpy as np
from keras import layers
from keras import Input
from keras import backend
from keras import optimizers
from keras import initializers
from keras.models import Model
from keras.models import load_model
from keras import callbacks
from keras.utils import plot_model

####### preparation of fingerprint_list #######
infile1 = open("../../../GaAs_perturb_N500_cheb64_whitened_finger.dat" ,'r')
finger_list = infile1.readlines()
infile1.close()

num_PC = 64 # num of input node
PC_list = []
for finger in finger_list:
	PC_list.append([float(finger.split()[i]) for i in range(num_PC)]) # raw input
PC_matrix = np.asfarray(PC_list)

####### preparation of target matrix #######
infile2 = open("../../../num_vs_E.dat" ,'r')
data_list = infile2.readlines()
infile2.close()

num_Ga_list = []
num_As_list = []
data_E_list = []
for data in data_list:
        num_Ga_list.append(data.split()[0])
        num_As_list.append(data.split()[1])
        data_E_list.append(data.split()[2])

num_Ga_matrix = np.asfarray(num_Ga_list) # num of Ga atoms for each data
num_As_matrix = np.asfarray(num_As_list) # num of As atoms for each data
data_E_matrix = np.asfarray(data_E_list)

####### standardization of target matrix #######
num_data = data_E_matrix.shape[0] # num of data
data_E_mean = data_E_matrix.mean()
data_E_std = 0
for i in range(num_data):
        data_E_std += (data_E_matrix[i]-data_E_mean)**2

data_E_std = (data_E_std/(num_data-1))**0.5
data_stE_matrix = (data_E_matrix-data_E_mean)/data_E_std # standarized target E

####### preparation of input_list and test_list #######
f_test = 0.2 # fraction of test data
index_list = [[0],[0]]
input_Ga_matrix = np.empty([num_data, 20, num_PC])
input_As_matrix = np.empty([num_data, 20, num_PC])

for n in range(num_data):
	index_list[0].append(index_list[1][n]+int(num_Ga_matrix[n]))
	index_list[1].append(index_list[0][n+1]+int(num_As_matrix[n]))
	
	input_Ga_matrix[n] = PC_matrix[index_list[1][n]:index_list[0][n+1]].copy()
	input_As_matrix[n] = PC_matrix[index_list[0][n+1]:index_list[1][n+1]].copy()

test_index = np.sort(np.random.choice(range(1, num_data), int(num_data*f_test), replace = False))
train_index = np.delete(np.arange(num_data), test_index)
np.random.shuffle(train_index)

#test_index = np.arange(802, 1002, 1)
#train_index = np.arange(0, 802, 1)

num_test = test_index.shape[0]
num_train = train_index.shape[0]

train_Ga_matrix = input_Ga_matrix[train_index]
train_As_matrix = input_As_matrix[train_index]
test_Ga_matrix = input_Ga_matrix[test_index]
test_As_matrix = input_As_matrix[test_index]
train_stE_matrix = data_stE_matrix[train_index]
test_stE_matrix = data_stE_matrix[test_index]

def MSE(y_true, y_pred):
        return backend.mean(backend.square((y_pred - y_true)/40), axis=-1)

def create_model():
	n_inputs = num_PC
	n_hidden1 = 10
	n_hidden2 = 10
	n_outputs = 1
	Xavier_init = initializers.VarianceScaling(scale=1.0, mode='fan_avg', distribution='normal')

	with tf.name_scope("Ga"):
		hidden_Ga1 = layers.Dense(n_hidden1, name="hidden_Ga1", activation='tanh', kernel_initializer=Xavier_init)
		hidden_Ga2 = layers.Dense(n_hidden2, name="hidden_Ga2", activation='tanh', kernel_initializer=Xavier_init)
		output_Ga = layers.Dense(n_outputs, name="output_Ga", activation=None, kernel_initializer=Xavier_init)

		input_Ga = Input(shape=(20, n_inputs), dtype='float32', name='input_Ga')
		y_Ga = hidden_Ga1(input_Ga)
		y_Ga = hidden_Ga2(y_Ga)
		y_Ga = output_Ga(y_Ga)
		sum_Ga = layers.Lambda(lambda x: backend.sum(x, axis=1), name='sum_Ga')(y_Ga)

	with tf.name_scope("As"):
		hidden_As1 = layers.Dense(n_hidden1, name="hidden_As1", activation='tanh', kernel_initializer=Xavier_init)
		hidden_As2 = layers.Dense(n_hidden2, name="hidden_As2", activation='tanh', kernel_initializer=Xavier_init)
		output_As = layers.Dense(n_outputs, name="output_As", activation=None, kernel_initializer=Xavier_init)

		input_As = Input(shape=(20, n_inputs), dtype='float32', name='input_As')
		y_As = hidden_As1(input_As)
		y_As = hidden_As2(y_As)
		y_As = output_As(y_As)
		sum_As = layers.Lambda(lambda x: backend.sum(x, axis=1), name='sum_As')(y_As)

	total_E = layers.add([sum_Ga, sum_As], name="total_E")
	return  Model([input_Ga, input_As], total_E)

def train_model():
	model = create_model()
#	plot_model(model, show_shapes=True, show_layer_names=True, to_file='model.png')
	model.summary()
#	model.compile(loss=MSE, optimizer=optimizers.SGD(lr=0.05, decay=0.0, momentum=0.9, nesterov=True))
	model.compile(loss=MSE, optimizer=optimizers.Adam())
	callbacks_list = [callbacks.EarlyStopping(monitor='val_loss', patience=500), callbacks.ModelCheckpoint(filepath='my_model.{epoch:05d}.hdf5', monitor='val_loss', save_best_only=True, save_weights_only=False, period=1)]
	history = model.fit([train_Ga_matrix, train_As_matrix], train_stE_matrix, epochs=80000, batch_size=64, callbacks=callbacks_list, validation_data=([test_Ga_matrix, test_As_matrix], test_stE_matrix))

def predict_from_model(my_model):
	model = load_model(my_model, custom_objects={"backend": backend})
	activation_model = Model(model.input, [model.layers[6].output, model.layers[7].output])
#	print(model.layers)
	print(data_stE_matrix[1] * data_E_std + data_E_mean)
	print(model.predict([[input_Ga_matrix[1]], [input_As_matrix[1]]]) * data_E_std + data_E_mean)
	Ga, As = activation_model.predict([[input_Ga_matrix[1]], [input_As_matrix[1]]])
	print(Ga*data_E_std+data_E_mean/40, '\n', As*data_E_std+data_E_mean/40)

train_model()
#predict_from_model('./my_model.34287.hdf5')
