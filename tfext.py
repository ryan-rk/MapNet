# Module for tf 2.3 extension

# Suppress CUDA library error
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

def supp_warn():
	
	# Suppress deprecation error
	import tensorflow.python.util.deprecation as deprecation
	deprecation._PRINT_DEPRECATION_WARNINGS = False

	# Prevent CUBLAS_STATUS_ALLOC_FAILED error
	gpus = tf.config.experimental.list_physical_devices('GPU')
	if gpus:
		try:
			# Currently, memory growth needs to be the same across GPUs
			for gpu in gpus:
				tf.config.experimental.set_memory_growth(gpu, True)
			logical_gpus = tf.config.experimental.list_logical_devices('GPU')
		except RuntimeError as e:
			# Memory growth must be set before GPUs have been initialized
			print(e)

def mixed_precision():
	# Use mixed precision on RTX GPU
	from tensorflow.keras import mixed_precision
	mixed_precision.set_global_policy('mixed_float16')


# Pre-built network

# MLP
def dense_net(archi, activation=None):
	x_in = tf.keras.Input()
	for layers in (len(archi)-2):
		x = tf.keras.layers.Dense(archi[layers+1], activation)(x)
	x_out = tf.keras.layers.Dense(archi[-1], activation)(x)
	model = tf.keras.Model(inputs=x_in, outputs=x_out)

	return model

# CNN
def conv2d_net(archi, kernel_size=3, strides=(1,1), activation=None, use_bn=False):

	x_in = tf.keras.Input()
	for layers in (len(archi)-2):
		x = tf.keras.layers.Conv2D(archi[layers+1], kernel_size=kernel_size, strides=strides, padding='same', activation=activation)(x)
		if use_bn != False:
			x = tf.keras.layers.BatchNormalization()(x)
	x_out = tf.keras.layers.Conv2D(archi[-1], kernel_size, strides, padding='same', activation=activation)(x)
	model = tf.keras.Model(inputs=x_in, outputs=x_out)

	return model

def train_net(custom_model, input_train, output_train, input_test, output_test, batch_size, num_batches, epochs, learn_rate=1e-3):
	model = custom_model()

	# Instantiate an optimizer to train the model.
	optimizer = tf.keras.optimizers.Adam(learning_rate=learn_rate)
	# Instantiate a loss function.
	loss_fn = tf.keras.losses.MeanSquaredError()

	# Prepare the metrics.
	train_acc_metric = tf.keras.metrics.Accuracy()
	val_acc_metric   = tf.keras.metrics.Accuracy()

	#tensorboard writer 
	train_writer = tf.summary.create_file_writer('logs/train/')
	test_writer  = tf.summary.create_file_writer('logs/test/')

	@tf.function
	def train_step(step, x, y):
		with tf.GradientTape() as tape:
			prediction = model(x, training=True)
			train_loss_value = loss_fn(y, prediction)
		grads = tape.gradient(train_loss_value, model.trainable_weights)
		optimizer.apply_gradients(zip(grads, model.trainable_weights))
		train_acc_metric.update_state(y, prediction)
		
		# write training loss and accuracy to the tensorboard
		with train_writer.as_default():
			tf.summary.scalar('loss', train_loss_value, step=step)
			tf.summary.scalar('accuracy', train_acc_metric.result(), step=step) 
		
		return train_loss_value

	@tf.function
	def test_step(step, x, y):
		val_pred = model(x, training=False)
		# Compute the loss value
		val_loss_value = loss_fn(y, val_pred)
		# Update val metrics
		val_acc_metric.update_state(y, val_pred)
		
		# write test loss and accuracy to the tensorboard
		with train_writer.as_default():
			tf.summary.scalar('val loss', val_loss_value, step=step)
			tf.summary.scalar('val accuracy', val_acc_metric.result(), step=step) 

		return val_loss_value

	# custom training loop 
	print_step = epochs//10
	import time
	for epoch in range(epochs):
		t = time.time()
		# batch training, iterate over the batches of the dataset.
		for train_batch_step in range(num_batches):
			train_batch_step = tf.convert_to_tensor(train_batch_step, dtype=tf.int64)
			x_batch_train = input_train[train_batch_step*batch_size:(train_batch_step+1)*batch_size]
			y_batch_train = output_train[train_batch_step*batch_size:(train_batch_step+1)*batch_size]
			train_loss_value = train_step(train_batch_step, x_batch_train, y_batch_train)

		# Run a validation loop at the end of each epoch.
		val_loss_value = test_step(tf.convert_to_tensor(epoch, dtype=tf.int64), input_test, output_test)

		template = 'Time each epoch: {} epoch: {} loss: {}  acc: {} val loss: {} val acc: {}'
		if (epoch+1)%print_step == 0:
			print(template.format(time.time()-t, epoch + 1, train_loss_value, float(train_acc_metric.result()), val_loss_value, float(val_acc_metric.result())))
		
		# Reset metrics at the end of each epoch
		train_acc_metric.reset_states()
		val_acc_metric.reset_states()

	model.save('saved_model', save_format='tf')