import tensorflow as tf
import tensorflow_hub as hub
import pathlib
import numpy as np
import cv2

#Input
pX = tf.placeholder(tf.float32,shape=[None,299,299,1])
pY = tf.placeholder(tf.int32,shape=[None])
b_size = tf.placeholder(tf.int64)
dataset = tf.data.Dataset.from_tensor_slices((pX,pY))
dataset = dataset.shuffle(3000).batch(b_size)
it = dataset.make_initializable_iterator()
(batch_X, batch_Y) = it.get_next()

#Net
image = tf.image.grayscale_to_rgb(batch_X)/255
module = hub.Module("https://tfhub.dev/google/imagenet/inception_v3/feature_vector/3")
features = module(image)
'''
h1 = tf.layers.conv2d(batch_X,filters=60,kernel_size=[3,3],strides=[1,1],padding='same',activation=tf.nn.leaky_relu)
h1p = tf.layers.max_pooling2d(h1,pool_size=[2,2],strides=[2,2])
h2 = tf.layers.conv2d(h1p,filters=128,kernel_size=[3,3],strides=[1,1],padding='same',activation=tf.nn.leaky_relu)
h2p = tf.layers.max_pooling2d(h2,pool_size=[2,2],strides=[2,2])	
h3 = tf.layers.conv2d(h2p,filters=192,kernel_size=[3,3],strides=[1,1],padding='same',activation=tf.nn.leaky_relu)
#h3p = tf.layers.max_pooling2d(h3,pool_size=[2,2],strides=[2,2])	
h4 = tf.layers.conv2d(h3,filters=192,kernel_size=[3,3],strides=[1,1],padding='same',activation=tf.nn.leaky_relu)
#h4p = tf.layers.max_pooling2d(h4,pool_size=[2,2],strides=[2,2])	
h8 = tf.layers.conv2d(h4,filters=192,kernel_size=[3,3],strides=[1,1],padding='same',activation=tf.nn.leaky_relu)
#h3p = tf.layers.max_pooling2d(h3,pool_size=[2,2],strides=[2,2])	
h9 = tf.layers.conv2d(h8,filters=192,kernel_size=[3,3],strides=[1,1],padding='same',activation=tf.nn.leaky_relu)
#h9p = tf.layers.max_pooling2d(h9,pool_size=[2,2],strides=[2,2])	
h5 = tf.layers.conv2d(h9,filters=128,kernel_size=[3,3],strides=[1,1],padding='same',activation=tf.nn.leaky_relu)
h5p = tf.layers.max_pooling2d(h5,pool_size=[2,2],strides=[2,2])	
#print (h5p)
f5 = tf.reshape(h5p,shape=[-1,9*9*128])
'''
h6 = tf.layers.dense(features,units=100,activation=tf.nn.leaky_relu)
h6d = tf.layers.dropout(h6,rate=0.5)
h7 = tf.layers.dense(h6,units=100,activation=tf.nn.leaky_relu)
h7d = tf.layers.dropout(h7,rate=0.5)
out = tf.layers.dense(h7,units=13)
#print (out)
err = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out,labels=batch_Y)
train = tf.train.AdamOptimizer().minimize(err)
#train = tf.train.GradientDescentOptimizer(0.001).minimize(err)
#train = tf.train.MomentumOptimizer(0.0001,0.85).minimize(err)


#Code
init = tf.global_variables_initializer()
s = tf.Session()
s.run(init)
saver = tf.train.Saver()
#Load weight
try:
	saver.restore(s,"./model.ckpt")
except:
	pass

tf_p = pathlib.Path('dataset/dataset/train')
t_path = [str(p) for p in tf_p.glob('*/*')]
#lb_map = dict((cate.name,index) for index,cate in enumerate(tf_p.glob('*')))
lb_map = {'tallbuilding': 9, 'kitchen': 3, 'insidecity': 4, 'bedroom': 5, 'highway': 7, 'coast': 12, 'office': 10, 'opencountry': 11, 'forest': 1, 'mountain': 6, 'street': 2, 'livingroom': 8, 'suburb': 0}
rlb_map = dict((lb_map[cate],cate) for cate in lb_map.keys())
t_lb = [lb_map[i.split('/')[3]] for i in t_path]
#Set this to False if training
test = True
if not test:
	train_data = []
	for i in t_path:
		train_data.append(cv2.resize(cv2.imread(i,cv2.IMREAD_GRAYSCALE), (299, 299), interpolation=cv2.INTER_CUBIC))
	train_data = np.expand_dims(train_data,axis=3)
	t_lb = np.array(t_lb).astype(np.int32)
	epoch = 50
	for i in range(epoch):
		s.run(it.initializer,feed_dict={pX:train_data,pY:t_lb,b_size:16})
		errsum = 0.0
		#Go through batchs
		try:
			while True:
				#error,o,by,_ = s.run([err,out,batch_Y,train])
				error,_ = s.run([err,train])
				errsum += error.sum()
				print (error.sum())
				
				'''
				#Debug
				total = 0.0
				tp = 0.0
				#adj = {}
				for j in range(len(o)):
					total += 1
					if np.argmax(o[j]) == by[j]:
						tp += 1
					else:
						pass
						
						#if((rlb_map[by[j]]+'->'+rlb_map[np.argmax(o[j])]) in adj.keys()) :
						#	adj[(rlb_map[by[j]]+'->'+rlb_map[np.argmax(o[j])])] += 1
						#else:
						#	adj[(rlb_map[by[j]]+'->'+rlb_map[np.argmax(o[j])])] = 1
						
				print ('Accuracy: %f' % (tp/total))
				'''
				#print (adj)
				#print (np.argmax(o[0]),'/',by[0])

		except tf.errors.OutOfRangeError:
			pass
		print('Epoch %d error:%f' % (i,errsum))
		saver.save(s,"./model.ckpt")
else:
	testf_p = pathlib.Path('dataset/dataset/test')
	test_path = [str(p) for p in testf_p.glob('*')]
	test_data = []
	sz = 0
	for i in test_path:
		test_data.append(cv2.resize(cv2.imread(i,cv2.IMREAD_GRAYSCALE), (299, 299), interpolation=cv2.INTER_CUBIC))
		sz += 1
	test_data = np.expand_dims(test_data,axis=3)

	#pY is now filename index of the list: test_path
	s.run(it.initializer,feed_dict={pX:test_data,pY:np.array(range(sz),dtype=np.int32),b_size:1})
	with open('test.csv','w') as w:
		w.write('id,label')
		for i in range(sz):	
			o,ix = s.run([out,batch_Y])
			#Output csv
			filename = ((test_path[ix[0]]).split('/')[-1].split('.'))[0]
			fclass = rlb_map[np.argmax(o[0])]
			w.write('\n'+filename+','+fclass)
				
