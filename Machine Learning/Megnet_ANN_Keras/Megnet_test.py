
##### by In Won Yeu (yiw0121@snu.ac.kr) #####
## 1. make slab graph network
## 2019.11.13
##########

##### import module #####
#import os
#import re
#import sys
from __future__ import print_function, division
#import glob
#import subprocess
import tensorflow as tf
import numpy as np
#import keras
#import math
from keras import layers
from keras import Input
import tensorflow.keras.backend as kb
from keras import callbacks
from keras import optimizers
from keras import initializers
from keras.models import Model
from megnet.layers import MEGNetLayer, CrystalGraphLayer, InteractionLayer, Set2Set
from megnet.activations import softplus2
from keras.utils import plot_model
from functions_GN_input import make_input, standarization, concat_bulk_surf
##########

##### define numbers #####
###########

##### read files #####
infile0 = open('tmp00','r')
bulk_file_list = infile0.readlines()
infile0.close()

infile1 = open('tmp01','r')
surf_file_list = infile1.readlines()
infile1.close()
######################

###############################################################
##### make training input list #####
n_data = 200
bulk_inputs, bulk_E = make_input(bulk_file_list[0:n_data])
surf_inputs, surf_E = make_input(surf_file_list[0:n_data])

E = bulk_E + surf_E
stand_E, E_mean, E_std = standarization(E)
print("shape of stand_E: ", np.asfarray(stand_E).shape)

###############################################################

###############################################################
def create_model():
	n_atom_feature= 8
	n_bond_feature = 1
	n_global_feature = 0
	Xavier_init = initializers.VarianceScaling(scale=1.0, mode='fan_avg', distribution='normal')

	x1 = Input(shape=(None, n_atom_feature)) # atom feature placeholder
	x2 = Input(shape=(None, n_bond_feature)) # bond feature placeholder
	x3 = Input(shape=(None, n_global_feature)) # global feature placeholder
	x4 = Input(shape=(None,), dtype='int32') # bond index1 placeholder
	x5 = Input(shape=(None,), dtype='int32') # bond index2 placeholder
	x6 = Input(shape=(None,), dtype='int32') # atom_ind placeholder
	x7 = Input(shape=(None,), dtype='int32') # bond_ind placeholder
	x8 = Input(shape=(None,), dtype='int32') # Ga_index placeholder to gather Ga nodes to a single tensor
	x9 = Input(shape=(None,), dtype='int32') # As_index placeholder to gather As nodes to a single tensor
	x10 = Input(shape=(None,), dtype='int32') # Ga_ind placeholder to sum of the atomic energies of Ga atoms for each single structure 
	x11 = Input(shape=(None,), dtype='int32') # As_ind placeholder to sum of the atomic energies of As atoms for each single structure

	with kb.name_scope("embedding"):
		embed_atom_fea = layers.Dense(16, name="embedding_atom_fea", activation='tanh', kernel_initializer=Xavier_init)
		embed_bond_fea = layers.Dense(16, name="embedding_bond_fea", activation='tanh', kernel_initializer=Xavier_init)
		GN_1 = MEGNetLayer([16, 16], [16, 16], [1], pool_method='mean', activation=softplus2)
		GN_2 = MEGNetLayer([16, 16], [16, 16], [1], pool_method='mean', activation=softplus2)

		x1_ = embed_atom_fea(x1)
		x2_ = embed_bond_fea(x2)

		out1 = GN_1([x1_, x2_, x3, x4, x5, x6, x7])
		x1__ = layers.Add()([x1_, out1[0]])
		x2__ = layers.Add()([x2_, out1[1]])

		out2 = GN_2([x1__, x2__, out1[2], x4, x5, x6, x7])
		x1___ = layers.Add()([x1__, out2[0]])
		x2___ = layers.Add()([x2__, out2[1]])
		
		Ga_idx = layers.Lambda(lambda x: tf.reshape(x, (-1,)), name="Ga_idx")(x8)
		As_idx = layers.Lambda(lambda x: tf.reshape(x, (-1,)), name="As_idx")(x9)
		Ga = layers.Lambda(lambda x: tf.gather(x, Ga_idx, axis=1), name="Ga")(x1___)
		As = layers.Lambda(lambda x: tf.gather(x, As_idx, axis=1), name="As")(x1___)

		Ga_grp = layers.Lambda(lambda x: tf.reshape(x, (-1,)), name="Ga_grp")(x10)
		As_grp = layers.Lambda(lambda x: tf.reshape(x, (-1,)), name="As_grp")(x11)

		#node = Set2Set(T=3, n_hidden=3)([x1___, x6])
		edge = Set2Set(T=3, n_hidden=3)([x2___, x7])
		edge = layers.Lambda(lambda x: kb.sum(x, axis=2, keepdims=True), name="sum_edge")(edge)
		zero = layers.Lambda(lambda x: tf.zeros_like(x), name="zero_like_edge")(edge)
		zero_edge = layers.Multiply(name="zero_edge")([edge, zero])
		zero_glob = layers.Multiply(name="zero_glob")([out2[2], zero])
		#final = layers.Concatenate(axis=-1)([node, edge, out2[2]])

	with kb.name_scope("Ga"):
		hidden_Ga1 = layers.Dense(10, name="hidden_Ga1", activation='tanh', kernel_initializer=Xavier_init)
		hidden_Ga2 = layers.Dense(10, name="hidden_Ga2", activation='tanh', kernel_initializer=Xavier_init)
		output_Ga = layers.Dense(1, name="output_Ga", activation=None, kernel_initializer=Xavier_init)

		E_Ga = hidden_Ga1(Ga)
		E_Ga = hidden_Ga2(E_Ga)
		E_Ga = output_Ga(E_Ga)

		E_Ga = layers.Lambda(lambda x: tf.reshape(x, (-1,1)), name="reshape_E_Ga")(E_Ga)
		sum_Ga = layers.Lambda(lambda x: tf.math.segment_sum(x, Ga_grp), name="sum_Ga")(E_Ga)
		#sum_Ga = layers.Lambda(lambda x: tf.reshape(x, glob.shape), name="reshape_sum_Ga")(sum_Ga)
		sum_Ga = layers.Lambda(lambda x: tf.expand_dims(x, axis=0), name="reshape_sum_Ga")(sum_Ga)

	with kb.name_scope("As"):
		hidden_As1 = layers.Dense(10, name="hidden_As1", activation='tanh', kernel_initializer=Xavier_init)
		hidden_As2 = layers.Dense(10, name="hidden_As2", activation='tanh', kernel_initializer=Xavier_init)
		output_As = layers.Dense(1, name="output_As", activation=None, kernel_initializer=Xavier_init)

		E_As = hidden_As1(As)
		E_As = hidden_As2(E_As)
		E_As = output_As(E_As)

		E_As = layers.Lambda(lambda x: tf.reshape(x, (-1,1)), name="reshape_E_As")(E_As)
		sum_As = layers.Lambda(lambda x: tf.math.segment_sum(x, As_grp), name="sum_As")(E_As)
		#sum_As = layers.Lambda(lambda x: tf.reshape(x, glob.shape), name="reshape_sum_As")(sum_As)
		sum_As = layers.Lambda(lambda x: tf.expand_dims(x, axis=0), name="reshape_sum_As")(sum_As)

	total_E = layers.Add(name="total_E")([sum_Ga, sum_As])
	final_E = layers.Add(name="final_E")([total_E, zero_edge, zero_glob])
#	total_E = layers.Lambda(lambda x: tf.expand_dims(x, axis=0))(total_E)
	return Model(inputs=[x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11], outputs=final_E)

def MSE(y_true, y_pred):
	return kb.mean(kb.square((y_pred[0] - y_true[0])/40), axis=-1)
###############################################################
def train_model(n_epochs, n_train, batch_size):
	model = create_model()
	model.summary()
	activation_model = Model(model.input, model.layers[35].output)
#	print(model.get_config())
	model.compile(loss='mse', optimizer=optimizers.Adam())
	callbacks_list = [callbacks.EarlyStopping(monitor='val_loss', patience=500), callbacks.ModelCheckpoint(filepath='my_model.{epoch:05d}.hdf5', monitor='val_loss', save_best_only=True, save_weights_only=False, period=1)]
	
	for epoch in range(n_epochs):
		for index in range(n_train // batch_size):
			bulk_inputs_train, bulk_E_train = make_input(bulk_file_list[index*batch_size:(index+1)*batch_size])
			surf_inputs_train, surf_E_train = make_input(surf_file_list[index*batch_size:(index+1)*batch_size])
			X_batch, y_batch = concat_bulk_surf(bulk_inputs_train, bulk_E_train, surf_inputs_train, surf_E_train, batch_size, E_mean, E_std)
			print("X_batch[0]'s shape: ", np.asfarray(X_batch[0]).shape, "y_batch's shape: ", y_batch.shape)
			model.fit(X_batch, y_batch, epochs=1, batch_size=1)

		if epoch==0:
			bulk_inputs_valid, bulk_E_valid = make_input(bulk_file_list[160:n_data])
			surf_inputs_valid, surf_E_valid = make_input(surf_file_list[160:n_data])
			X_valid, y_valid = concat_bulk_surf(bulk_inputs_valid, bulk_E_valid, surf_inputs_valid, surf_E_valid, n_data-160, E_mean, E_std)

		activation_model = Model(model.input, model.layers[35].output)
		out_valid = activation_model.predict(X_valid)
		val_loss=model.evaluate(X_valid, y_valid)
		print("epoch: ", epoch, "val_loss: ", val_loss)
		#print("out_valid: ", out_valid)
		#print("target: ", y_valid)

###############################################################

def predict_from_model():
	model = create_model()
	model.summary()
	activation_model = Model(model.input, [model.layers[34].output, model.layers[35].output])
	
	bulk_inputs_temp, bulk_E_temp = make_input(bulk_file_list[10:20])
	surf_inputs_temp, surf_E_temp = make_input(surf_file_list[10:20])
	X_temp, y_temp = concat_bulk_surf(bulk_inputs_temp, bulk_E_temp, surf_inputs_temp, surf_E_temp, 10, E_mean, E_std)
	print("X_temp[0]'s shape: ", np.asfarray(X_temp[0]).shape, "y_temp's shape: ", y_temp.shape)

	node, edge = activation_model.predict(X_temp)
	print(node.shape, '\n', edge.shape)

train_model(n_epochs=10, n_train=160, batch_size=16)
#predict_from_model()
