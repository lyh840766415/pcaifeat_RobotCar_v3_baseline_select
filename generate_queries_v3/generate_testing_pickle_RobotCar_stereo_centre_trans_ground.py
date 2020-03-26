import pandas as pd
import numpy as np
import os
import pandas as pd
from sklearn.neighbors import KDTree
import pickle
import random
import sys
sys.path.append('..')
from loading_input_v3 import *


###Building database and query files for evaluation
base_path= "/data/lyh/RobotCar/pc_img_ground_0310/"


#####For training and test data split#####
x_width=150
y_width=150

#For Oxford
p1=[5735712.768124,620084.402381]
p2=[5735611.299219,620540.270327]
p3=[5735237.358209,620543.094379]
p4=[5734749.303802,619932.693364]   

#For University Sector
p5=[363621.292362,142864.19756]
p6=[364788.795462,143125.746609]
p7=[363597.507711,144011.414174]

#For Residential Area
p8=[360895.486453,144999.915143]
p9=[362357.024536,144894.825301]
p10=[361368.907155,145209.663042]

p_dict={"oxford":[p1,p2,p3,p4], "university":[p5,p6,p7], "residential": [p8,p9,p10], "business":[]}

def check_in_test_set(northing, easting, points, x_width, y_width):
	in_test_set=False
	for point in points:
		if(point[0]-x_width<northing and northing< point[0]+x_width and point[1]-y_width<easting and easting<point[1]+y_width):
			in_test_set=True
			break
	return in_test_set
##########################################

def output_to_file(output, filename):
	with open(filename, 'wb') as handle:
		pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
	print("Done ", filename)

def construct_query_and_database_sets(base_path, runs_folder, folders, pointcloud_fols, filename, p, output_name):
	database_trees=[]
	test_trees=[]
	for folder in folders:
		print(folder)
		df_database= pd.DataFrame(columns=['file','northing','easting'])
		df_test= pd.DataFrame(columns=['file','northing','easting'])
		
		df_locations= pd.read_csv(os.path.join(base_path,runs_folder,folder,filename),sep=',')
		for index, row in df_locations.iterrows():
			#entire business district is in the test set
			#get correspond image
			pc_filename = os.path.join(base_path,runs_folder,folder,pointcloud_fols[1:],"%d.bin"%(row['timestamp']))
			img_filename = "%s_stereo_centre.png"%(pc_filename[:-4])
						
			if(output_name=="business"):
				df_test=df_test.append(row, ignore_index=True)
			elif not os.path.exists(img_filename):
				continue
			elif(check_in_test_set(row['northing'], row['easting'], p, x_width, y_width)):
				df_test=df_test.append(row, ignore_index=True)
			df_database=df_database.append(row, ignore_index=True)
		
		#print("df_test len = ",len(df_test))	
		#print("df_database len = ",len(df_database))
		
		database_tree = KDTree(df_database[['northing','easting']])
		test_tree = KDTree(df_test[['northing','easting']])
		database_trees.append(database_tree)
		test_trees.append(test_tree)

	test_sets=[]
	database_sets=[]
	for folder in folders:
		database={}
		test={} 
		df_locations= pd.read_csv(os.path.join(base_path,runs_folder,folder,filename),sep=',')
		df_locations['timestamp']=base_path+runs_folder+folder+pointcloud_fols+df_locations['timestamp'].astype(str)+'.bin'
		df_locations=df_locations.rename(columns={'timestamp':'file'})
		for index,row in df_locations.iterrows():	
			pc_filename = row['file']
			img_filename = "%s_stereo_centre.png"%(pc_filename[:-4])			
			#entire business district is in the test set
			if(output_name=="business"):
				test[len(test.keys())]={'query':row['file'],'northing':row['northing'],'easting':row['easting']}
			elif not os.path.exists(img_filename):
				continue
			elif(check_in_test_set(row['northing'], row['easting'], p, x_width, y_width)):
				test[len(test.keys())]={'query':row['file'],'northing':row['northing'],'easting':row['easting']}
			database[len(database.keys())]={'query':row['file'],'northing':row['northing'],'easting':row['easting']}
		
		#print("test len = ",len(test.keys()))	
		#print("database len = ",len(database.keys()))
		#exit()
		database_sets.append(database)
		test_sets.append(test)		

	for i in range(len(database_sets)):
		tree=database_trees[i]
		for j in range(len(test_sets)):
			if(i==j):
				continue
			for key in range(len(test_sets[j].keys())):
				coor=np.array([[test_sets[j][key]["northing"],test_sets[j][key]["easting"]]])
				index = tree.query_radius(coor, r=25)
				#indices of the positive matches in database i of each query (key) in test set j
				test_sets[j][key][i]=index[0].tolist()

	#print(database_sets)
	#print(test_sets)
	output_to_file(database_sets, "stereo_centre_trans_RobotCar_ground_"+output_name+'_evaluation_database.pickle')
	output_to_file(test_sets, "stereo_centre_trans_RobotCar_ground_"+output_name+'_evaluation_query.pickle')


#For Oxford
folders=[]
runs_folder = "20m_20dis_color_resize/"
all_folders=sorted(os.listdir(os.path.join(base_path,runs_folder)))
index_list=[5,6,7,9,10,11,12,13,14,15,16,17,18,19,22,24,31,32,33,38,39,43,44]
print(len(index_list))
for index in index_list:
	folders.append(all_folders[index])

print(folders)
construct_query_and_database_sets(base_path, runs_folder, folders, "/pointclouds/", "pointcloud_locations_20m_20dis.csv", p_dict["oxford"], "oxford")