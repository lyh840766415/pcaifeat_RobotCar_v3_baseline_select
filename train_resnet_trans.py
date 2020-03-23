import tensorflow as tf
import numpy as np
from loading_input_v3 import *
import nets_v3.resnet_v1_trans as resnet
from pointnetvlad_v3.pointnetvlad_trans import *
from multiprocessing.dummy import Pool as ThreadPool
import matplotlib.pyplot as plt

#thread pool
pool = ThreadPool(10)

BATCH_REACH_END = False
CUR_LOAD = 0
LOAD_BATCH_SIZE = 2
FEAT_BARCH_SIZE = 2
#pos num,neg num,other neg num,all_num
POS_NUM = 2
NEG_NUM = 5
OTH_NUM = 1
BATCH_DATA_SIZE = 1 + POS_NUM + NEG_NUM + OTH_NUM

# Margin
MARGIN1 = 0.5
MARGIN2 = 0.2

BASE_LEARNING_RATE = 5e-5
EPOCH = 20


TRAIN_FILE = 'generate_queries_v3/training_queries_RobotCar_trans_matrix.pickle'
TRAINING_QUERIES = get_queries_dict(TRAIN_FILE)

def get_learning_rate(epoch):
	learning_rate = BASE_LEARNING_RATE*((0.9)**(epoch//5))
	learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
	return learning_rate

def init_imgnetwork(pc_trans_feat):
	step = tf.Variable(0)
	
	with tf.variable_scope("img_var"):
		img_placeholder = tf.placeholder(tf.float32,shape=[FEAT_BARCH_SIZE*BATCH_DATA_SIZE,256,256,3])
		img_feat,_= resnet.endpoints(img_placeholder,None,is_training=True)
		
	
	img_feat = tf.reshape(img_feat,[FEAT_BARCH_SIZE,BATCH_DATA_SIZE,img_feat.shape[1]])
	q_img_vec, pos_img_vec, neg_img_vec, oth_img_vec = tf.split(img_feat, [1,POS_NUM,NEG_NUM,OTH_NUM],1)
	img_loss = lazy_quadruplet_loss(q_img_vec, pos_img_vec, neg_img_vec, oth_img_vec, MARGIN1, MARGIN2)
	tf.summary.scalar('img_loss', img_loss)
	
	epoch_num_placeholder = tf.placeholder(tf.float32, shape=())
	learning_rate = get_learning_rate(epoch_num_placeholder)
	tf.summary.scalar('learning_rate', learning_rate)
	optimizer = tf.train.AdamOptimizer(learning_rate)
	img_train_op = optimizer.minimize(img_loss, global_step=step)
	merged = tf.summary.merge_all()
		
	ops = {
		"img_placeholder":img_placeholder,
		"epoch_num_placeholder":epoch_num_placeholder,
		"img_loss":img_loss,
		"img_train_op":img_train_op,
		"merged":merged,
		"step":step}
	return ops

def get_batch_keys(train_file_idxs,train_file_num):
	global CUR_LOAD
	load_batch_keys = []
	
	while len(load_batch_keys) < LOAD_BATCH_SIZE:
		skip_num = 0
		#make sure cur_load is valid
		if CUR_LOAD >= train_file_num:
			return True,None
			
		cur_key = train_file_idxs[CUR_LOAD]
		if len(TRAINING_QUERIES[cur_key]["positives"]) < POS_NUM:
			CUR_LOAD = CUR_LOAD + 1
			skip_num = skip_num + 1
			continue
		
		filename = "%s_mono_left.png"%(TRAINING_QUERIES[cur_key]["query"][:-4])
		if not os.path.exists(filename):
			#print(TRAINING_QUERIES[cur_key]["query"])
			CUR_LOAD = CUR_LOAD + 1
			skip_num = skip_num + 1
			continue
		
		valid_pos = 0
		for i in range(len(TRAINING_QUERIES[cur_key]["positives"])):
			filename = "%s_mono_left.png"%(TRAINING_QUERIES[TRAINING_QUERIES[cur_key]["positives"][i]]["query"][:-4])
			if os.path.exists(filename):
					valid_pos = valid_pos + 1
				
		if valid_pos < POS_NUM:
			skip_num = skip_num + 1
			CUR_LOAD = CUR_LOAD + 1
			continue
						
		load_batch_keys.append(train_file_idxs[CUR_LOAD])
		CUR_LOAD = CUR_LOAD + 1
		
	return False,load_batch_keys

def is_negative(query,not_negative):
	return not query in not_negative
	
def get_load_batch_filename(load_batch_keys,quadruplet):		
	pc_files = []
	img_files = []
	for key_cnt ,key in enumerate(load_batch_keys):
		pc_files.append(TRAINING_QUERIES[key]["query"])
		img_files.append("%s_mono_left.png"%(TRAINING_QUERIES[key]["query"][:-4]))
		random.shuffle(TRAINING_QUERIES[key]["positives"])
		
		#print(TRAINING_QUERIES[key])
		cur_pos = 0;
		for i in range(POS_NUM):
			while True:
				filename = "%s_mono_left.png"%(TRAINING_QUERIES[TRAINING_QUERIES[key]["positives"][cur_pos]]["query"][:-4])

				if filename in img_files[BATCH_DATA_SIZE*(key_cnt)+1:BATCH_DATA_SIZE*(key_cnt)+1+i]:
					cur_pos = cur_pos+1
					continue
				if os.path.exists(filename):
					break
				cur_pos = cur_pos+1
				if cur_pos>len(TRAINING_QUERIES[key]["positives"]):
					print("line 259, error in positive number")
					exit()
			
			pc_files.append(TRAINING_QUERIES[TRAINING_QUERIES[key]["positives"][cur_pos]]["query"])
			img_files.append(filename)		
		
		neg_indices = []
		for i in range(NEG_NUM):
			while True:
				while True:
					neg_ind = random.randint(0,len(TRAINING_QUERIES.keys())-1)
					if is_negative(neg_ind,TRAINING_QUERIES[key]["not_negative"]):
						break
				

				filename = "%s_mono_left.png"%(TRAINING_QUERIES[neg_ind]["query"][:-4])
				if filename in img_files[BATCH_DATA_SIZE*(key_cnt)+1+POS_NUM:BATCH_DATA_SIZE*(key_cnt)+1+POS_NUM+i]:
					continue
				if os.path.exists(filename):
					break
					
			neg_indices.append(neg_ind)
			pc_files.append(TRAINING_QUERIES[neg_ind]["query"])
			img_files.append(filename)
		
		'''
		tmp_list = img_files[9*(key_cnt)+1+POS_NUM:9*(key_cnt)+1+POS_NUM+NEG_NUM]
		if len(tmp_list)!=len(set(tmp_list)):
			print("neg_duplicate")
			input()
		'''
		
		if quadruplet:
			neighbors=[]
			for pos in TRAINING_QUERIES[key]["positives"]:
				neighbors.append(pos)
			for neg in neg_indices:
				for pos in TRAINING_QUERIES[neg]["positives"]:
					neighbors.append(pos)
					
			#print("neighbors size = ",len(neighbors))
			while True:
				neg_ind = random.randint(0,len(TRAINING_QUERIES.keys())-1)
				if is_negative(neg_ind,neighbors):
					filename = "%s_mono_left.png"%(TRAINING_QUERIES[neg_ind]["query"][:-4])
					if os.path.exists(filename):
						break						
									
			pc_files.append(TRAINING_QUERIES[neg_ind]["query"])
			img_files.append(filename)
	
	return pc_files,img_files

	
def main():
	global BATCH_REACH_END
	global EPOCH
	global CUR_LOAD
	
	ops = init_imgnetwork(None)
	
	variables = tf.trainable_variables()
	#img_variable = [v for v in variables if v.name.split('/')[0] =='img_var' and v.name.split('/')[1] == 'resnet_v1_50']
	img_variable = [v for v in variables if v.name.split('/')[0] =='img_var']
	all_saver = tf.train.Saver(img_variable)
	
	'''
	for v in img_variable:
		print(v)
	exit()
	'''
	
	
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	
	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())
		print("initialized")
		#all_saver.restore(sess,"/data/lyh/lab/pcaifeat_RobotCar_v3_1/log/train_save_trans_img/img_model_00174058.ckpt")
		#print("restore")
		train_writer = tf.summary.FileWriter("./log/train_save_resnet_9", sess.graph)
		for ep in range(EPOCH):
			train_file_num = len(TRAINING_QUERIES.keys())
			train_file_idxs = np.arange(0,train_file_num)
			np.random.shuffle(train_file_idxs)
			print('Eppch = %d, train_file_num = %f , FEAT_BATCH_SIZE = %f , iteration per batch = %f' %(ep,len(train_file_idxs), FEAT_BARCH_SIZE,len(train_file_idxs)//FEAT_BARCH_SIZE))
			BATCH_REACH_END = False
			CUR_LOAD = 0
			
			while True:
				BATCH_REACH_END,load_batch_keys = get_batch_keys(train_file_idxs,train_file_idxs.shape[0])
				if BATCH_REACH_END:
					break
				
				load_pc_filenames,load_img_filenames = get_load_batch_filename(load_batch_keys,True)
				
				
				pc_data,img_data = load_img_pc(load_pc_filenames,load_img_filenames,pool)
				
				train_feed_dict = {
					ops["img_placeholder"]:img_data,
					ops["epoch_num_placeholder"]:ep}
						
				summary,step,img_loss,_,= sess.run([ops["merged"],ops["step"],ops["img_loss"],ops["img_train_op"]],feed_dict = train_feed_dict)
				print("batch num = %d , img_loss = %f"%(step, img_loss))
				
				train_writer.add_summary(summary, step)
				



if __name__ == "__main__":
	main()