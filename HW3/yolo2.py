import sys
import tensorflow as tf
from tensorflow.keras.layers import Input,Conv2D,LeakyReLU,Dense,Reshape,BatchNormalization,Lambda,MaxPooling2D
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import h5py
import pandas as pd
import utils as ut

IMAGE_H, IMAGE_W = 208, 208
GRID_H, GRID_W = 13, 13
BOX = 5
LABELS = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
TRUE_BOX_BUFFER = 10

CLASS = 10
CLASS_WEIGHTS = np.ones(CLASS, dtype='float32')
OBJ_THRESHOLD = 0.3  # 0.5
NMS_THRESHOLD = 0.1  # 0.45
#ANCHORS = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]
# 8*16 grid anchors
# ANCHORS = [1.903865227092189, 4.099548436908577, 2.3514071459696275, 5.755811221969188, 1.1040676087236638, 2.9041994848101864, 1.508090979843708, 5.352452608112024, 3.3192913603990215, 6.251185195162513]
# 4*8 grid anchors
# ANCHORS = [0.95193261, 2.04977422, 1.17570357, 2.87790561, 0.5520338 , 1.45209974, 0.75404549, 2.6762263 , 1.65964568, 3.1255926 ]
# 176 / 11*11 grid anchors
#ANCHORS = [1.4231116415164426,5.893868168549424,0.9869430092495134,6.525999380980469,0.7444686034080288,3.70592714021776,2.2532861105277444,8.535785211641143,1.5608500273738442,8.081473691342476]
# 240 / 15*15 grid anchors
# ANCHORS = [3.0726628779923804,11.639707106783405,2.1283879507370824,11.020300640352712,1.3458664044748565,8.89856637970794,1.9407349899582447,8.037330691126144,1.0151844591927626,5.0535370093878615]
# 208 / 13*13 grid anchors
ANCHORS = [2.6969745235162783,10.15838663106423,0.8970549320879804,4.719324162816479,1.2253239211229794,8.697735488181804,1.9105766137560825,9.353236207142563,1.5469048766842433,6.661990294797252]

# TODO:Duplicate boxes two times


anchor_boxs = [(ANCHORS[2 * i], ANCHORS[2 * i + 1]) for i in range(int(len(ANCHORS) // 2))]




NO_OBJECT_SCALE = 1.0
OBJECT_SCALE = 5.0
COORD_SCALE = 1.0
CLASS_SCALE = 1.0

BATCH_SIZE = 32
WARM_UP_BATCHES = 0

img_path = './train/'


#RCONF = False
RCONF = True
TEST = True
#TEST = False
SHOW = False
T_TRAIN = False

# Model def
input_image = Input(shape=(IMAGE_H, IMAGE_W, 3))
true_boxes = Input(shape=(1, 1, 1, TRUE_BOX_BUFFER, 4))
# Layer 1
x = Conv2D(32, (3,3), strides=(1,1), padding='same', name='conv_1', use_bias=False)(input_image)
x = BatchNormalization(name='norm_1')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
# Layer 2
x = Conv2D(64, (3,3), strides=(1,1), padding='same', name='conv_2', use_bias=False)(x)
x = BatchNormalization(name='norm_2')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
# Layer 3
x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_3', use_bias=False)(x)
x = BatchNormalization(name='norm_3')(x)
x = LeakyReLU(alpha=0.1)(x)
# Layer 4
x = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_4', use_bias=False)(x)
x = BatchNormalization(name='norm_4')(x)
x = LeakyReLU(alpha=0.1)(x)
# Layer 5
x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_5', use_bias=False)(x)
x = BatchNormalization(name='norm_5')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
# Layer 6
x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_6', use_bias=False)(x)
x = BatchNormalization(name='norm_6')(x)
x = LeakyReLU(alpha=0.1)(x)
# Layer 7
x = Conv2D(128, (1,1), strides=(1,1), padding='same', name='conv_7', use_bias=False)(x)
x = BatchNormalization(name='norm_7')(x)
x = LeakyReLU(alpha=0.1)(x)
# Layer 8
x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_8', use_bias=False)(x)
x = BatchNormalization(name='norm_8')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
# Layer 9
x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_9', use_bias=False)(x)
x = BatchNormalization(name='norm_9')(x)
x = LeakyReLU(alpha=0.1)(x)
# Layer 10
x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_10', use_bias=False)(x)
x = BatchNormalization(name='norm_10')(x)
x = LeakyReLU(alpha=0.1)(x)
# Layer 11
x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_11', use_bias=False)(x)
x = BatchNormalization(name='norm_11')(x)
x = LeakyReLU(alpha=0.1)(x)
# Layer 12
x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_12', use_bias=False)(x)
x = BatchNormalization(name='norm_12')(x)
x = LeakyReLU(alpha=0.1)(x)
# Layer 13
x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_13', use_bias=False)(x)
x = BatchNormalization(name='norm_13')(x)
x = LeakyReLU(alpha=0.1)(x)
skip_connection = x
#x = MaxPooling2D(pool_size=(2, 2))(x)
# Layer 14
x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_14', use_bias=False)(x)
x = BatchNormalization(name='norm_14')(x)
x = LeakyReLU(alpha=0.1)(x)
# Layer 15
x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_15', use_bias=False)(x)
x = BatchNormalization(name='norm_15')(x)
x = LeakyReLU(alpha=0.1)(x)
# Layer 16
x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_16', use_bias=False)(x)
x = BatchNormalization(name='norm_16')(x)
x = LeakyReLU(alpha=0.1)(x)
# Layer 17
x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_17', use_bias=False)(x)
x = BatchNormalization(name='norm_17')(x)
x = LeakyReLU(alpha=0.1)(x)
# Layer 18
x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_18', use_bias=False)(x)
x = BatchNormalization(name='norm_18')(x)
x = LeakyReLU(alpha=0.1)(x)
# Layer 19
x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_19', use_bias=False)(x)
x = BatchNormalization(name='norm_19')(x)
x = LeakyReLU(alpha=0.1)(x)
# Layer 20
x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_20', use_bias=False)(x)
x = BatchNormalization(name='norm_20')(x)
x = LeakyReLU(alpha=0.1)(x)
# Layer 21
skip_connection = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_21', use_bias=False)(skip_connection)
skip_connection = BatchNormalization(name='norm_21')(skip_connection)
skip_connection = LeakyReLU(alpha=0.1)(skip_connection)
print (skip_connection)
print (x)
x = tf.concat([skip_connection, x], -1)

'''# Layer 22.5
x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_225', use_bias=False)(x)
x = BatchNormalization(name='norm_225')(x)
x = LeakyReLU(alpha=0.1)(x)'''

# Layer 22
x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_22', use_bias=False)(x)
x = BatchNormalization(name='norm_22')(x)
x = LeakyReLU(alpha=0.1)(x)


# Layer 23
x = Conv2D(BOX * (4 + 1 + CLASS), (1,1), strides=(1,1), padding='same', name='conv_23')(x)
output = Reshape((GRID_H, GRID_W, BOX, 4 + 1 + CLASS))(x)

# small hack to allow true_boxes to be registered when Keras build the model
# for more information: https://github.com/fchollet/keras/issues/2790
output = Lambda(lambda args: args[0])([output, true_boxes])

model = tf.keras.Model([input_image, true_boxes], output)
model.summary()




def get_name(index, hdf5_data):
	name = hdf5_data['/digitStruct/name']
	return ''.join([chr(v[0]) for v in hdf5_data[name[index][0]].value])

def get_bbox(index, hdf5_data):
	attrs = {}
	item = hdf5_data['digitStruct']['bbox'][index].item()
	for key in ['label', 'left', 'top', 'width', 'height']:
		attr = hdf5_data[item][key]
		values = [hdf5_data[attr.value[i].item()].value[0][0]
				  for i in range(len(attr))] if len(attr) > 1 else [attr.value[0][0]]
		attrs[key] = values
	return attrs



def iou_test(real_box, anchor_box):
	w_lap = min(real_box[0],anchor_box[0])
	h_lap = min(real_box[1],anchor_box[1])
	lap = w_lap * h_lap
	union = (real_box[0] * real_box[1]) + (anchor_box[0] * anchor_box[1]) - lap
	return float(lap) / union

# the function to implement the orgnization layer (thanks to github.com/allanzelener/YAD2K)
def space_to_depth_x2(x):
	return tf.space_to_depth(x, block_size=2)

def custom_loss(y_true, y_pred):
	mask_shape = tf.shape(y_true)[:4]

	cell_x = tf.cast(tf.reshape(tf.tile(tf.range(GRID_W), [GRID_H]), (1, GRID_H, GRID_W, 1, 1)), dtype=tf.float32)
	cell_y = tf.cast(tf.reshape(tf.transpose(tf.tile(tf.range(GRID_H), [GRID_W])), (1, GRID_H, GRID_W, 1, 1)), dtype=tf.float32)

	print(cell_x)
	print(cell_y)
	cell_grid = tf.tile(tf.concat([cell_x, cell_y], -1), [BATCH_SIZE, 1, 1, 5, 1])

	coord_mask = tf.zeros(mask_shape)
	conf_mask = tf.zeros(mask_shape)
	class_mask = tf.zeros(mask_shape)


	#seen = tf.Variable(0.)
	#total_recall = tf.Variable(0.)

	"""
	Adjust prediction
	"""
	### adjust x and y
	pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid
	### adjust w and h
	pred_box_wh = tf.exp(y_pred[..., 2:4]) * np.reshape(ANCHORS, [1, 1, 1, BOX, 2])
	### adjust confidence
	pred_box_conf = tf.sigmoid(y_pred[..., 4])
	### adjust class probabilities
	pred_box_class = y_pred[..., 5:]

	"""
	Adjust ground truth
	"""
	### adjust x and y
	true_box_xy = y_true[..., 0:2]  # relative position to the containing cell

	### adjust w and h
	true_box_wh = y_true[..., 2:4]  # number of cells accross, horizontally and vertically

	### adjust confidence
	true_wh_half = true_box_wh / 2.
	true_mins = true_box_xy - true_wh_half
	true_maxes = true_box_xy + true_wh_half

	pred_wh_half = pred_box_wh / 2.
	pred_mins = pred_box_xy - pred_wh_half
	pred_maxes = pred_box_xy + pred_wh_half

	intersect_mins = tf.maximum(pred_mins, true_mins)
	intersect_maxes = tf.minimum(pred_maxes, true_maxes)
	intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
	intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

	true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
	pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]

	union_areas = pred_areas + true_areas - intersect_areas
	iou_scores = tf.truediv(intersect_areas, union_areas)

	true_box_conf = iou_scores * y_true[..., 4]

	### adjust class probabilities
	true_box_class = tf.argmax(y_true[..., 5:], -1)

	"""
	Determine the masks
	"""
	### coordinate mask: simply the position of the ground truth boxes (the predictors)
	coord_mask = tf.expand_dims(y_true[..., 4], axis=-1) * COORD_SCALE

	### confidence mask: penelize predictors + penalize boxes with low IOU
	# penalize the confidence of the boxes, which have IOU with some ground truth box < 0.6
	true_xy = true_boxes[..., 0:2]
	true_wh = true_boxes[..., 2:4]

	true_wh_half = true_wh / 2.
	true_mins = true_xy - true_wh_half
	true_maxes = true_xy + true_wh_half

	pred_xy = tf.expand_dims(pred_box_xy, 4)
	pred_wh = tf.expand_dims(pred_box_wh, 4)

	pred_wh_half = pred_wh / 2.
	pred_mins = pred_xy - pred_wh_half
	pred_maxes = pred_xy + pred_wh_half

	intersect_mins = tf.maximum(pred_mins, true_mins)
	intersect_maxes = tf.minimum(pred_maxes, true_maxes)
	intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
	intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

	true_areas = true_wh[..., 0] * true_wh[..., 1]
	pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

	union_areas = pred_areas + true_areas - intersect_areas
	iou_scores = tf.truediv(intersect_areas, union_areas)

	best_ious = tf.reduce_max(iou_scores, axis=4)
	conf_mask = conf_mask + tf.cast(best_ious < 0.6, dtype=tf.float32) * (1 - y_true[..., 4]) * NO_OBJECT_SCALE

	# penalize the confidence of the boxes, which are reponsible for corresponding ground truth box
	conf_mask = conf_mask + y_true[..., 4] * OBJECT_SCALE

	### class mask: simply the position of the ground truth boxes (the predictors)
	class_mask = y_true[..., 4] * tf.gather(CLASS_WEIGHTS, true_box_class) * CLASS_SCALE

	"""
	Warm-up training
	"""
	no_boxes_mask = tf.cast(coord_mask < COORD_SCALE / 2., dtype=tf.float32)
	#seen = tf.assign_add(seen, 1.)
	#seen = seen + 1.

	true_box_xy, true_box_wh, coord_mask = tf.cond(tf.less(10, WARM_UP_BATCHES),
												   lambda: [true_box_xy + (0.5 + cell_grid) * no_boxes_mask,
															true_box_wh + tf.ones_like(true_box_wh) * np.reshape(
																ANCHORS, [1, 1, 1, BOX, 2]) * no_boxes_mask,
															tf.ones_like(coord_mask)],
												   lambda: [true_box_xy,
															true_box_wh,
															coord_mask])

	"""
	Finalize the loss
	"""
	nb_coord_box = tf.reduce_sum(tf.cast(coord_mask > 0.0, dtype=tf.float32))
	nb_conf_box = tf.reduce_sum(tf.cast(conf_mask > 0.0, dtype=tf.float32))
	nb_class_box = tf.reduce_sum(tf.cast(class_mask > 0.0, dtype=tf.float32))

	loss_xy = tf.reduce_sum(tf.square(true_box_xy - pred_box_xy) * coord_mask) / (nb_coord_box + 1e-6) / 2.
	loss_wh = tf.reduce_sum(tf.square(true_box_wh - pred_box_wh) * coord_mask) / (nb_coord_box + 1e-6) / 2.
	loss_conf = tf.reduce_sum(tf.square(true_box_conf - pred_box_conf) * conf_mask) / (nb_conf_box + 1e-6) / 2.
	loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class)
	loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6)

	loss = loss_xy + loss_wh + loss_conf + loss_class

	nb_true_box = tf.reduce_sum(y_true[..., 4])
	nb_pred_box = tf.reduce_sum(tf.cast(true_box_conf > 0.5, dtype=tf.float32) * tf.cast(pred_box_conf > 0.3, dtype=tf.float32))

	"""
	Debugging code
	"""
	current_recall = nb_pred_box / (nb_true_box + 1e-6)
	#loss = tf.keras.backend.print_tensor(loss, message='\trecall:{}\t'.format(tf.keras.backend.get_value(current_recall)))
	#total_recall = tf.assign_add(total_recall, current_recall)
	# total_recall = total_recall + current_recall
	#tf.print(current_recall, output_stream=sys.stderr)

	'''
	loss = tf.Print(loss, [tf.zeros((1))], message='Dummy Line \t', summarize=1000)
	loss = tf.Print(loss, [loss_xy], message='Loss XY \t', summarize=1000)
	loss = tf.Print(loss, [loss_wh], message='Loss WH \t', summarize=1000)
	loss = tf.Print(loss, [loss_conf], message='Loss Conf \t', summarize=1000)
	loss = tf.Print(loss, [loss_class], message='Loss Class \t', summarize=1000)
	loss = tf.Print(loss, [loss], message='Total Loss \t', summarize=1000)
	loss = tf.Print(loss, [current_recall], message='Current Recall \t', summarize=1000)
	loss = tf.Print(loss, [total_recall / seen], message='Average Recall \t', summarize=1000)
	'''
	#print(loss)
	return loss

# img_boundingbox_data_constructor()
# Try load weight
# ......
optimizer = tf.keras.optimizers.Adam(learning_rate=0.5e-4)
#
checkpoint_path = "weights/cp.ckpt"

TRANSFER = True
try:
	model.load_weights(checkpoint_path)
	print('Weight loaded')
except:
	print('Weight not found!!')
	input()
	wt_path = 'yolo.weights'
	if TRANSFER:
		weight_reader = ut.WeightReader(wt_path)
		weight_reader.reset()
		nb_conv = 23
		for i in range(1, nb_conv + 1):
			conv_layer = model.get_layer('conv_' + str(i))

			if i < nb_conv:
				norm_layer = model.get_layer('norm_' + str(i))

				size = np.prod(norm_layer.get_weights()[0].shape)

				beta = weight_reader.read_bytes(size)
				gamma = weight_reader.read_bytes(size)
				mean = weight_reader.read_bytes(size)
				var = weight_reader.read_bytes(size)

				weights = norm_layer.set_weights([gamma, beta, mean, var])

			if len(conv_layer.get_weights()) > 1:
				bias = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
				kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
				kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
				kernel = kernel.transpose([2, 3, 1, 0])
				conv_layer.set_weights([kernel, bias])
			else:
				kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
				kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
				kernel = kernel.transpose([2, 3, 1, 0])
				conv_layer.set_weights([kernel])
		layer = model.layers[-4]  # the last convolutional layer
		weights = layer.get_weights()
		new_kernel = np.random.normal(size=weights[0].shape) / (GRID_H * GRID_W)
		new_bias = np.random.normal(size=weights[1].shape) / (GRID_H * GRID_W)
		layer.set_weights([new_kernel, new_bias])



if TEST:
	x_batch = np.zeros((1, IMAGE_H, IMAGE_W, 3))
	dummies_b = np.zeros((1, 1, 1, 1, TRUE_BOX_BUFFER, 4))
	if T_TRAIN:
		img_path = './train/'
	else:
		img_path = './test/'
	# Testing data
	kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
	out_lst = []
	for img_num in range(13068):
		if img_num % 1000 == 0:
			print(img_num)
		img_name = str(img_num + 1) + '.png'
		img_ar = cv2.imread(img_path + img_name)
		#img_ar = cv2.filter2D(img_ar, -1, kernel=kernel)

		(img_h, img_w, _) = img_ar.shape
		# 0.Get image in x_batch
		x_batch[0] = cv2.resize(img_ar, (IMAGE_W, IMAGE_H))
		# test on testing data
		y = model.predict(x=(x_batch, dummies_b))
		vv = (ut._sigmoid(y[0,:,:,:,4]))
		#print (ut._softmax(y[0,:,:,:,5:]))
		#print (ut._sigmoid(y[0,:,:,:,4]).max())
		boxes = ut.decode_netout(y[0],
                      obj_threshold=OBJ_THRESHOLD,
                      nms_threshold=NMS_THRESHOLD,
                      anchors=ANCHORS,
                      nb_class=CLASS)
		if SHOW:
			print(boxes)
			fig, ax = plt.subplots(1)
			ax.imshow(img_ar)

		dic = {"bbox": [], "label": [], "score": []}
		if boxes:
			for box in boxes:
				dic["bbox"].append([box.ymin * IMAGE_H/(IMAGE_H/img_h), box.xmin * IMAGE_W/(IMAGE_W/img_w), box.ymax * IMAGE_H/(IMAGE_H/img_h), box.xmax * IMAGE_W/(IMAGE_W/img_w)])
				lb = box.get_label()
				lb = 10 if lb==0 else lb
				dic["label"].append(lb)
				dic["score"].append(box.get_score())
				wid = (box.xmax - box.xmin) * IMAGE_W
				hei = (box.ymax - box.ymin) * IMAGE_H
				if SHOW:
					rect = patches.Rectangle((box.xmin*IMAGE_W/(IMAGE_W/img_w), box.ymin*IMAGE_H/(IMAGE_H/img_h)), wid/(IMAGE_W/img_w), hei/(IMAGE_H/img_h),linewidth=1,edgecolor='r',facecolor='none')
					ax.add_patch(rect)
					print(box.get_label())
					print(box.get_score())
		out_lst.append(dic)
		if SHOW:
			plt.show()
	print('Finish predicting, total data:',len(out_lst))
	with open('output.json','w') as qq:
		qq.write(str(out_lst).replace('\'','\"'))
	print('Done')
	exit()

class LossHistory(tf.keras.callbacks.Callback):
	def on_train_begin(self, logs={}):
		self.losses = []

	def on_batch_end(self, batch, logs={}):
		self.losses.append(logs.get('loss'))



model.compile(loss=custom_loss, optimizer=optimizer, experimental_run_tf_function=False)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
												 save_weights_only=True,
												 verbose=1,
												 period=1)
history = LossHistory()




# Load training data
mat_file = './train/digitStruct.mat'
f = h5py.File(mat_file, 'r')
img_amount = f['/digitStruct/bbox'].shape[0]






# range(f['/digitStruct/bbox'].shape[0])
l_ldst = [0, 3200, 6400, 9600, 12800, 16000, 19200, 22400, 25600, 28800, 32000]
l_ldsz = [3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 1376]

#l_ldst = [0, 6400, 12800, 19200, 25600]
#l_ldsz = [6400, 6400, 6400, 6400, 7776]

# l_ldst = [0, 12800, 25600]
# l_ldsz = [12800, 12800, 7776]

for epochs in range(1000):
	for idk in range(len(l_ldst)):
		ldst = l_ldst[idk]
		ldsz = l_ldsz[idk]

		x_batch = np.zeros((ldsz, IMAGE_H, IMAGE_W, 3))
		b_batch = np.zeros((ldsz, 1, 1, 1, TRUE_BOX_BUFFER, 4))
		y_batch = np.zeros((ldsz, GRID_H, GRID_W, BOX, 4 + 1 + CLASS))
		for img_num in range(ldsz):
			if img_num % 1000 == 0:
				print(img_num + ldst)
			img_name = str(img_num + ldst + 1) + '.png'
			img_ar = cv2.imread(img_path + img_name)
			(img_h, img_w, _) = img_ar.shape

			# 0.Get image in x_batch
			img_ar = cv2.resize(img_ar, (IMAGE_W, IMAGE_H))
			x_batch[img_num] = img_ar

			# DEBUG:PLOT BBOX
			# fig,ax = plt.subplots(1)
			# ax.imshow(img_ar)

			# 1.Get bounding box information (true_box)
			# in dic[dic['name']==img_name]
			tbcnt = 0
			p_img_name = get_name(img_num + ldst, f)
			assert p_img_name == img_name
			row_dict = get_bbox(img_num + ldst, f)
			# row_dict['img_name'] = img_name
			for tbcnt in range(len(row_dict['label'])):
				# Relocate boxes
				(ratio_x, ratio_y) = (float(IMAGE_W) / img_w, float(IMAGE_H) / img_h)
				(cx, cy) = (row_dict['left'][tbcnt] + (row_dict['width'][tbcnt] / 2),
							row_dict['top'][tbcnt] + (row_dict['height'][tbcnt] / 2))
				lb = row_dict['label'][tbcnt]
				obj_indx = int(lb)
				obj_indx = 0 if obj_indx == 10 else obj_indx
				assert cx < img_w
				assert cy < img_h
				(cx, cy) = (cx / (IMAGE_W / GRID_W), cy / (IMAGE_H / GRID_H))
				(cw, ch) = (row_dict['width'][tbcnt] / (IMAGE_W / GRID_W), row_dict['height'][tbcnt] / (IMAGE_H / GRID_H))
				(nx, ny, nw, nh) = (ratio_x * cx, ratio_y * cy, ratio_x * cw, ratio_y * ch)
				box = np.array((nx, ny, nw, nh))
				grid_x = int(np.floor(nx))
				grid_y = int(np.floor(ny))

				# Add the patch to the Axes
				# rect = patches.Rectangle((ratio_x*row_dict['left'][tbcnt],ratio_y*row_dict['top'][tbcnt]),ratio_x*row_dict['width'][tbcnt],ratio_y*row_dict['height'][tbcnt],linewidth=1,edgecolor='r',facecolor='none')
				# ax.add_patch(rect)
				# print(obj_indx)

				# 2.Mark grids (y_true)
				# find the anchor that best predicts this box(least w,h to fit)
				best_anchor = -1
				max_iou = -1
				shifted_box = (nw, nh)
				for i in range(len(anchor_boxs)):
					anchor = anchor_boxs[i]
					iou = iou_test(shifted_box, anchor)
					if max_iou < iou and y_batch[img_num, grid_y, grid_x, i, 4]==0:
						best_anchor = i
						max_iou = iou
				# assign ground truth x, y, w, h, confidence and class probs to y_batch
				#print (best_anchor)
				assert best_anchor != -1
				if RCONF and y_batch[img_num, grid_y, grid_x, best_anchor, 4] != 0:
					print('Conflict boxes!!',img_name)
					print('Now with label:',obj_indx)
					input()
				y_batch[img_num, grid_y, grid_x, best_anchor, 0:4] = box
				y_batch[img_num, grid_y, grid_x, best_anchor, 4] = 1.
				y_batch[img_num, grid_y, grid_x, best_anchor, 5 + obj_indx] = 1

				# assign the true box to b_batch
				b_batch[img_num, 0, 0, 0, tbcnt] = box
			#print(y_batch[img_num])
			#print(b_batch[img_num])
			#input()
		print('Data loaded')
		model.fit(x=(x_batch[:ldsz], b_batch[:ldsz]), y=y_batch[:ldsz], batch_size=BATCH_SIZE, epochs=3, verbose=1,
					  callbacks=[cp_callback, history])
	#with open('log.txt','w') as www:
		#www.write(str(history))
# plt.show()
# print(b_batch[img_num])

