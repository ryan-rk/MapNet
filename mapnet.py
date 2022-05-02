import tfext as tx
tx.supp_warn()
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
import time
import math
import arrext as axt

class ConvBlock(tf.keras.layers.Layer):
	def __init__(self, kernel_num, kernel_size, strides, padding='same'):
		super(ConvBlock, self).__init__()
		# conv layer
		self.conv = tf.keras.layers.Conv2D(kernel_num, 
						kernel_size=kernel_size, 
						strides=strides, padding=padding)


	def call(self, input_tensor, training=False):
		x = self.conv(input_tensor)
		x = tf.nn.relu(x)
		
		return x

class DeConvBlock(tf.keras.layers.Layer):
	def __init__(self, kernel_num, kernel_size, padding='same'):
		super(DeConvBlock, self).__init__()
		# conv layer
		self.deconv = tf.keras.layers.Conv2DTranspose(filters=kernel_num, kernel_size=kernel_size, strides=(2, 2), padding="SAME")


	def call(self, input_tensor, training=False):
		x = self.deconv(input_tensor)
		x = tf.nn.relu(x)
		
		return x

class SEBlock(tf.keras.layers.Layer):
	def __init__(self, channels):
		super(SEBlock, self).__init__()
		self.squeeze = tf.keras.layers.GlobalAveragePooling2D()
		self.excitation1 = tf.keras.layers.Dense(channels//16)
		self.excitation2 = tf.keras.layers.Dense(channels)
		self.reshape = tf.keras.layers.Reshape((1,1,channels))


	def call(self, input_tensor):
		residual = input_tensor
		x = self.squeeze(input_tensor)
		x = self.excitation1(x)
		x = tf.nn.relu(x)
		x = self.excitation2(x)
		x = tf.nn.sigmoid(x)
		x = self.reshape(x)
		
		return tf.multiply(x, residual)

class ResBlock(tf.keras.layers.Layer):
	def __init__(self, channels, padding='same'):
		super(ResBlock, self).__init__()
		self.conv1 = tf.keras.layers.Conv2D(channels, kernel_size=(5,5), strides=(1,1), padding=padding)
		self.conv2 = tf.keras.layers.Conv2D(channels, kernel_size=(5,5), strides=(1,1), padding=padding)
		self.scale = SEBlock(channels)


	def call(self, input_tensor, training=False):
		residual = input_tensor
		x = self.conv1(input_tensor)
		x = tf.nn.relu(x)
		x = self.conv2(x)
		x = tf.nn.relu(x)
		x = self.scale(x)
		x = tf.keras.layers.add([x, residual])

		return x


class MapNet(tf.keras.Model):
	def __init__(self):
		super(MapNet, self).__init__()

		# the first conv module
		self.conv1 = ConvBlock(16, (5,5), (1,1))
		self.conv2 = ConvBlock(32, (5,5), (1,1))
		self.densconv1 = ConvBlock(16, (5,5), (2,2))
		self.densconv2 = ConvBlock(16, (5,5), (2,2))
		self.densconv3 = ConvBlock(32, (5,5), (2,2))
		self.densconv4 = ConvBlock(32, (5,5), (2,2))

		self.res1 = ResBlock(64)
		self.res2 = ResBlock(64)
		self.res3 = ResBlock(64)
		self.res4 = ResBlock(64)
		self.res5 = ResBlock(64)

		self.deconv1 = DeConvBlock(64, (5,5))
		self.deconv2 = DeConvBlock(64, (5,5))
		self.deconv3 = DeConvBlock(32, (5,5))
		self.deconv4 = DeConvBlock(16, (5,5))
		self.outConv = tf.keras.layers.Conv2D(1, kernel_size=(5,5), strides=(1,1), padding='same')


	def call(self, input_tensor, training=False, **kwargs):
		
		# forward pass 
		x = self.conv1(input_tensor[0], training=training)
		comp = self.conv2(x, training=training)
		dens1 = self.densconv1(input_tensor[1], training=training)
		dens2 = self.densconv2(dens1, training=training)
		dens3 = self.densconv3(dens2, training=training)
		dens4 = self.densconv4(dens3, training=training)
		x = tf.concat([comp,dens4],3)

		x = self.res1(x, training=training)
		x = self.res2(x, training=training)
		x = self.res3(x, training=training)
		x = self.res4(x, training=training)
		x = self.res5(x, training=training)

		x = tf.concat([x,dens4],3)
		x = self.deconv1(x, training=training)
		x = tf.concat([x,dens3],3)
		x = self.deconv2(x, training=training)
		x = tf.concat([x,dens2],3)
		x = self.deconv3(x, training=training)
		x = tf.concat([x,dens1],3)
		x = self.deconv4(x, training=training)
		x = self.outConv(x)

		return x

	def build_graph(self, shape):
		x = tf.keras.layers.Input(shape=shape)
		return tf.keras.Model(inputs=[x], outputs=self.call(x))


def custom_loss(y_true, y_pred):
    error = y_true-y_pred
    sqr_error = tf.math.square(error)
    mse = tf.math.reduce_mean(sqr_error)
    #return the error
    return mse


def main():
	is_train = False
	input_dim = 4
	output_dim = 64
	coarse_normfac = 1e-4
	fine_normfac = 1e-6
	frag_scale = 29*29
	frag_colrow = int(math.sqrt(frag_scale))
	total_train = 60*frag_scale
	if is_train:
		total_test = 1*frag_scale
	else:
		total_test = 100*frag_scale
	test_start_ind = 0*frag_scale
	test_end_ind = test_start_ind + total_test
	batch_size = 1*frag_scale
	num_batches = total_train//batch_size
	epochs = 1000
	learn_rate = 1e-4
	savefileid = 'save/mapnet/ckpt'

	coarse_data = np.load('dataset/coarse_compliance_fragmented.npy')
	fine_data = np.load('dataset/fine_compliance_fragmented.npy')
	dens_data = np.load('dataset/fine_density_fragmented.npy')
	x_data = 1/coarse_normfac*coarse_data.reshape(-1,input_dim,input_dim,1)
	y_data = 1/fine_normfac*fine_data.reshape(-1,output_dim,output_dim,1)
	d_data = dens_data.reshape(-1,output_dim,output_dim,1)
	x_train = x_data[0:total_train]
	y_train = y_data[0:total_train]
	d_train = d_data[0:total_train]
	x_test = x_data[test_start_ind:test_end_ind]
	y_test = y_data[test_start_ind:test_end_ind]
	d_test = d_data[test_start_ind:test_end_ind]
	
	np.random.seed(0)
	perm_train = np.random.permutation(total_train)

	x_train = x_train[perm_train]
	y_train = y_train[perm_train]
	d_train = d_train[perm_train]
	print('x_train shape = {}, y_train shape = {}'.format(x_train.shape,y_train.shape))

	x_test = tf.convert_to_tensor(x_test,dtype=tf.float32)
	y_test = tf.convert_to_tensor(y_test,dtype=tf.float32)
	d_test = tf.convert_to_tensor(d_test,dtype=tf.float32)
	print('x_test shape = {}, y_test shape = {}'.format(x_test.shape,y_test.shape))

	print("x train max min: {0} {1}, y train max min: {2} {3}".format(np.amax(x_train),np.amin(x_train),np.amax(y_train),np.amin(y_train)))
	print("x test max min: {0} {1}, y test max min: {2} {3}".format(np.amax(x_test),np.amin(x_test),np.amax(y_test),np.amin(y_test)))

	# define model 
	# init model object
	model = MapNet()
	if is_train == False:
		load_status = model.load_weights(savefileid)

	# print summary
	# raw_input = (512, 512, 2)
	# model.build_graph(raw_input).summary()

	# Instantiate an optimizer to train the model.
	optimizer = tf.keras.optimizers.Adam(learning_rate=learn_rate)
	# Instantiate a loss function.

	# Prepare the metrics.
	train_acc_metric = tf.keras.metrics.CosineSimilarity(axis=1)
	val_acc_metric   = tf.keras.metrics.CosineSimilarity(axis=1)

	@tf.function
	def train_step(x, y, d):
		with tf.GradientTape() as tape:
			prediction = model([x, d], training=True)
			train_loss_value = custom_loss(y, prediction)
		grads = tape.gradient(train_loss_value, model.trainable_weights)
		optimizer.apply_gradients(zip(grads, model.trainable_weights))
		train_acc_metric.update_state(tf.math.abs(tf.reshape(y,[-1,output_dim*output_dim])), tf.math.abs(tf.reshape(prediction,[-1,output_dim*output_dim])))
		
		return train_loss_value

	@tf.function
	def test_step(x, y, d):
		val_pred = model([x, d], training=False)
		# Compute the loss value f
		val_loss_value = custom_loss(y, val_pred)
		# Update val metrics
		val_acc_metric.update_state(tf.math.abs(tf.reshape(y,[-1,output_dim*output_dim])), tf.math.abs(tf.reshape(val_pred,[-1,output_dim*output_dim])))

		return val_loss_value

	if is_train == True:
		# #tensorboard writer 
		curr_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
		train_writer = tf.summary.create_file_writer('logs/'+curr_timestamp+'/train/')
		test_writer  = tf.summary.create_file_writer('logs/'+curr_timestamp+'/test/')
			
		# custom training loop 
		for epoch in range(epochs):
			# batch training 
			# Iterate over the batches of the dataset.
			timestart = time.time()
			for train_batch_step in range(num_batches):
				train_batch_step = tf.convert_to_tensor(train_batch_step, dtype=tf.int64)
				x_batch_train = tf.convert_to_tensor(x_train[train_batch_step*batch_size:(train_batch_step+1)*batch_size],dtype=tf.float32)
				y_batch_train = tf.convert_to_tensor(y_train[train_batch_step*batch_size:(train_batch_step+1)*batch_size],dtype=tf.float32)
				d_batch_train = tf.convert_to_tensor(d_train[train_batch_step*batch_size:(train_batch_step+1)*batch_size],dtype=tf.float32)
				train_loss_value = train_step(x_batch_train, y_batch_train, d_batch_train)

			timeend = time.time()

			# write training loss and accuracy to the tensorboard
			with train_writer.as_default():
				tf.summary.scalar('loss', train_loss_value, step=epoch)
				tf.summary.scalar('accuracy', train_acc_metric.result(), step=epoch) 

			# evaluation on validation set 
			# Run a validation loop at the end of each epoch.
			val_loss_value = test_step(x_test, y_test, d_test)

			# write test loss and accuracy to the tensorboard
			with test_writer.as_default():
				tf.summary.scalar('val loss', val_loss_value, step=epoch)
				tf.summary.scalar('val accuracy', val_acc_metric.result(), step=epoch) 

			template = 'epoch: {} loss: {}  acc: {} val loss: {} val acc: {} time/iter: {}'
			if (epoch+1)%100 == 0:
				print(template.format(
					epoch + 1,
					train_loss_value, float(train_acc_metric.result()),
					val_loss_value, float(val_acc_metric.result()),
					timeend-timestart
				))
				model.save_weights(savefileid)
			
			# Reset metrics at the end of each epoch
			train_acc_metric.reset_states()
			val_acc_metric.reset_states()
		model.save_weights(savefileid)

	prediction = model.predict([x_test, d_test])

	error = np.mean(np.square(prediction-y_test))
	real_outputdim = 512
	ori_defrag = np.zeros((total_test//frag_scale,real_outputdim*real_outputdim))
	pred_defrag = np.zeros((total_test//frag_scale,real_outputdim*real_outputdim))
	for i in range(total_test//frag_scale):
		ori_todefrag = y_test[i*frag_scale:(i+1)*frag_scale].numpy().reshape(frag_scale,output_dim,output_dim)
		pred_todefrag = prediction[i*frag_scale:(i+1)*frag_scale].reshape(frag_scale,output_dim,output_dim)
		ori_defrag_tmp = axt.defrag_overlap_more(ori_todefrag, frag_colrow, frag_colrow)
		pred_defrag_tmp = axt.defrag_overlap_more(pred_todefrag, frag_colrow, frag_colrow)
		ori_defrag[i] = ori_defrag_tmp.reshape(1,real_outputdim*real_outputdim) 
		pred_defrag[i] = pred_defrag_tmp.reshape(1,real_outputdim*real_outputdim) 
	val_acc_metric.update_state(tf.math.abs(ori_defrag), tf.math.abs(pred_defrag))

	print("mean square error is {0}, accuracy is {1}".format(error,val_acc_metric.result()))

	fig = plt.figure()
	perm_test = np.random.permutation(total_test//frag_scale)
	for i in range(8):
		ax = fig.add_subplot(2,8,i+1)
		ax.imshow(np.log(np.maximum(fine_normfac*ori_defrag[perm_test[i]],0)+1e-8).reshape(real_outputdim,real_outputdim), cmap='gray')
		ax.set_xticks([]) 
		ax.set_yticks([]) 
		ax = fig.add_subplot(2,8,i+9)
		ax.imshow(np.log(np.maximum(fine_normfac*pred_defrag[perm_test[i]],0)+1e-8).reshape(real_outputdim,real_outputdim), cmap='gray')
		ax.set_xticks([]) 
		ax.set_yticks([]) 
	plt.show()

if __name__=='__main__':
	main()
