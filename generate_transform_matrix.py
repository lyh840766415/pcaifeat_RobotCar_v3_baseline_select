import os
import sys
sys.path.append('/data/lyh/lab/robotcar-dataset-sdk/python')
from camera_model import CameraModel
from transform import build_se3_transform
import numpy as np
from image import load_image
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates

def main():
	#read the pointcloud
	pointcloud_filename = "/data/lyh/lab/pcaifeat_RobotCar_v3_1/1400505794141322.txt"
	pointcloud = np.loadtxt(pointcloud_filename, delimiter=' ')
	pointcloud = np.hstack([pointcloud, np.ones((pointcloud.shape[0],1))])
	
	'''
	for i in range(pointcloud.shape[0]):
		print(" %.3f %.3f %.3f %.3f"%(pointcloud[i,0],pointcloud[i,1],pointcloud[i,2],pointcloud[i,3]))
	exit()
	'''
	
	#load the camera model
	models_dir = "/data/lyh/lab/robotcar-dataset-sdk/models/"
	model = CameraModel(models_dir, "mono_left")	
	
	#load the camera global pose
	imgpos_path = "/data/lyh/lab/pcaifeat_RobotCar_v3_1/1400505794141322_imgpos.txt"
	print(imgpos_path)
	imgpos = {}
	with open(imgpos_path) as imgpos_file:
		for line in imgpos_file:
			pos = [x for x in line.split(' ')]
			for i in range(len(pos)-2):
				pos[i+1] = float(pos[i+1])
			imgpos[pos[0]] = np.reshape(np.array(pos[1:-1]),[4,4])
			
	'''
	for key in imgpos.keys():
		print(key)
		print(imgpos[key])
	exit()
	'''
	
	#read the camera and ins extrinsics
	extrinsics_path = "/data/lyh/lab/robotcar-dataset-sdk/extrinsics/mono_left.txt"
	print(extrinsics_path)
	with open(extrinsics_path) as extrinsics_file:
		extrinsics = [float(x) for x in next(extrinsics_file).split(' ')]
	G_camera_vehicle = build_se3_transform(extrinsics)
	print(G_camera_vehicle)
	
	extrinsics_path = "/data/lyh/lab/robotcar-dataset-sdk/extrinsics/ins.txt"
	print(extrinsics_path)
	with open(extrinsics_path) as extrinsics_file:
		extrinsics = [float(x) for x in next(extrinsics_file).split(' ')]
	G_ins_vehicle = build_se3_transform(extrinsics)
	print(G_ins_vehicle)
	G_camera_posesource = G_camera_vehicle*G_ins_vehicle
	
	#translate pointcloud to image coordinate
	pointcloud = np.dot(np.linalg.inv(imgpos["mono_left"]),pointcloud.T)
	pointcloud = np.dot(G_camera_posesource, pointcloud)
		
	#project pointcloud to image
	image_path = "/data/lyh/lab/pcaifeat_RobotCar_v3_1/1400505794141322_mono_left.png"
	#image = load_image(image_path, model)
	image = load_image(image_path)
	
	uv= model.project(pointcloud, [1024,1024])
	
	lut = model.bilinear_lut[:, 1::-1].T.reshape((2, 1024, 1024))
	u = map_coordinates(lut[0, :, :], uv, order=1)
	v = map_coordinates(lut[1, :, :], uv, order=1)
	uv = np.array([u,v])
	print(uv.shape)
	transform_matrix = np.zeros([64,4096])
	for i in range(uv.shape[1]):
		if uv[0,i] == 0 and uv[1,i] == 0:
			continue
		cur_uv = np.rint(uv[:,i]/128)
		row = cur_uv[1]*8 + cur_uv[0]
		transform_matrix[int(row),i] = 1
	
	aa = np.sum(transform_matrix,1).reshape([8,8])
	print(np.sum(aa))
	plt.figure(1)
	plt.imshow(aa)
	#plt.show()
	
	#exit()
	plt.figure(2)
	#plt.imshow(image)	
	#uv = np.rint(uv/32)
	plt.scatter(np.ravel(uv[0, :]), np.ravel(uv[1, :]), s=2, edgecolors='none', cmap='jet')
	plt.xlim(0, 1024)
	plt.ylim(1024, 0)
	plt.xticks([])
	plt.yticks([])
	plt.show()
	
	
	
	

if __name__ == "__main__":
	main()