import torch
import numpy as np

from torchvision import transforms

from skimage import color
from skimage.draw import line, set_color, disk

from model import Model
import click

from ngdsac import NGDSAC
from loss import Loss

import cv2

from cyclonedds.topic import Topic
from cyclonedds.sub import DataReader
from cyclonedds.pub import DataWriter
from cyclonedds.domain import DomainParticipant
from dds_data_structures import MainCameraImage, Point2D
from msgs import Points2D

def process_frame(image, imagesize, device, uniform, nn, ngdsac, score_thr):
	'''
	Estimate horizon line for an image and return a visualization.

	image -- 3 dim numpy image tensor
	'''

	# determine image scaling factor
	image_scale = max(image.shape[0], image.shape[1])
	image_scale = imagesize / image_scale 

	# convert image to RGB
	if len(image.shape) < 3:
		image = color.gray2rgb(image)

	# store original image dimensions		
	src_h = int(image.shape[0] * image_scale)
	src_w = int(image.shape[1] * image_scale)

	# resize and to gray scale
	image = transforms.functional.to_pil_image(image)
	image = transforms.functional.resize(image, (src_h, src_w))
	image = transforms.functional.adjust_saturation(image, 0)
	image = transforms.functional.to_tensor(image)

	# make image square by zero padding
	padding_left = int((imagesize - image.size(2)) / 2)
	padding_right = imagesize - image.size(2) - padding_left
	padding_top = int((imagesize - image.size(1)) / 2)
	padding_bottom = imagesize - image.size(1) - padding_top

	padding = torch.nn.ZeroPad2d((padding_left, padding_right, padding_top, padding_bottom))
	image = padding(image)

	image_src = image.clone().unsqueeze(0)
	image_src = image_src.to(device)

	# normalize image (mean and variance), values estimated offline from HLW training set
	img_mask = image.sum(0) > 0
	image[:,img_mask] -= 0.45
	image[:,img_mask] /= 0.25
	image = image.unsqueeze(0)

	with torch.no_grad():
		#predict data points and neural guidance
		points, log_probs = nn(image)
	
		if uniform:
			# overwrite neural guidance with uniform sampling probabilities
			log_probs.fill_(1/log_probs.size(1))
			log_probs = torch.log(log_probs)

		# fit line with NG-DSAC, providing dummy ground truth labels
		ngdsac(points, log_probs, torch.zeros((1,2)), torch.zeros((1)), torch.ones((1)), torch.ones((1))) 

 	# normalized inlier score of the estimated line
	score = ngdsac.batch_inliers[0].sum() / points.shape[2]

	image_src = image_src.cpu().permute(0,2,3,1).numpy() #Torch to Numpy
	viz_probs = image_src.copy() * 0.2 # make a faint copy of the input image
	
	# draw estimated line
	if score > score_thr:
		image_src = draw_models(ngdsac.est_parameters, clr=(0,0,1), data=image_src, imagesize=imagesize)

	viz = [image_src]

	# if verbose:	
	# 	# create additional visualizations

	# 	# draw faint estimated line 
	# 	viz_score = viz_probs.copy()
	# 	viz_probs = draw_models(ngdsac.est_parameters, clr=(0.3,0.3,0.3), data=viz_probs)
	# 	viz_inliers = viz_probs.copy()

	# 	# draw predicted points with neural guidance and soft inlier count, respectively
	# 	viz_probs = draw_wpoints(points, viz_probs, weights=torch.exp(log_probs), clrmap=cv2.COLORMAP_PLASMA)
	# 	viz_inliers = draw_wpoints(points, viz_inliers, weights=ngdsac.batch_inliers, clrmap=cv2.COLORMAP_WINTER)

	# 	# create a explicit color map for visualize score of estimate line
	# 	color_map = np.arange(256).astype('u1')
	# 	color_map = cv2.applyColorMap(color_map, cv2.COLORMAP_HSV)	
	# 	color_map = color_map[:,:,::-1]

	# 	# map score to color
	# 	score = int(score*100) #using only the first portion of HSV to get a nice (red, yellow, green) gradient
	# 	clr = color_map[score, 0] / 255

	# 	viz_score = draw_models(ngdsac.est_parameters, clr=clr, data=viz_score)

	# 	viz = viz + [viz_probs, viz_inliers, viz_score]

	#undo zero padding of inputs
	if padding_left > 0:
		viz = [img[:,:,padding_left:,:] for img in viz]
	if padding_right > 0:
		viz = [img[:,:,:-padding_right,:] for img in viz]
	if padding_top > 0:
		viz = [img[:,padding_top:,:,:] for img in viz]
	if padding_bottom > 0:
		viz = [img[:,:-padding_bottom,:,:] for img in viz]		

	# convert to a single uchar image
	viz = np.concatenate(viz, axis=2)
	viz = viz * 255
	viz = viz.astype('u1')

	img = cv2.resize(viz[0], (0,0), fx = 1/image_scale, fy = 1/image_scale)
	return img

def draw_line(data, lX1, lY1, lX2, lY2, clr):
	'''
	Draw a line with the given color and opacity.

	data -- image to draw to
	lX1 -- x value of line segment start point
	lY1 -- y value of line segment start point
	lX2 -- x value of line segment end point
	lY2 -- y value of line segment end point
	clr -- line color, triple of values
	'''

	rr, cc = line(lY1, lX1, lY2, lX2)
	set_color(data, (rr, cc), clr)

def draw_models(labels, clr, data, imagesize):
	'''
	Draw disks for a batch of images.

	labels -- line parameters, array shape (Nx2) where 
		N is the number of images in the batch
		2 is the number of line parameters (offset,  slope)
	data -- batch of images to draw to
	'''

	# number of image in batch
	n = labels.shape[0]

	for i in range (n):

		#line
		lY1 = int(labels[i, 0] * imagesize)
		lY2 = int(labels[i, 1] * imagesize + labels[i, 0] * imagesize)
		draw_line(data[i], 0, lY1, imagesize, lY2, clr)

		return data	

def draw_wpoints(points, data, weights, clrmap, score_thr):
	'''
	Draw 2D points for a batch of images.

	points -- 2D points, array shape (Nx2xM) where 
		N is the number of images in the batch
		2 is the number of point dimensions (x, y)
		M is the number of points
	data -- batch of images to draw to
	weights -- array shape (NxM), one weight per point, for visualization
	clrmap -- OpenCV color map for visualizing weights
		
	'''

	# create explicit color map
	color_map = np.arange(256).astype('u1')
	color_map = cv2.applyColorMap(color_map, clrmap)
	color_map = color_map[:,:,::-1] # BGR to RGB

	n = points.shape[0] # number of images
	m = points.shape[2] # number of points

	for i in range (0, n):

		s_idx = weights[i].sort(descending=False)[1] # draw low weight points first
		weights[i] = weights[i] / weights[i].max() # normalize weights for visualization

		for j in range(0, m):

			idx = int(s_idx[j])

			# convert weight to color
			clr_idx = float(min(1, weights[i,idx]))
			clr_idx = int(clr_idx * 255)
			clr = color_map[clr_idx, 0] / 255

			# draw point
			r = int(points[i, 0, idx] * imagesize)
			c = int(points[i, 1, idx] * imagesize)
			rr, cc = disk(r, c, 2)
			set_color(data[i], (rr, cc), clr)

	return data


@click.command()
@click.option(
    "--topic_in_img", default="camera_images", help="name of the input image topic"
)
@click.option(
    "--topic_out_debug_img",
    default="test_image/front_camera_horizon",
    help="name of the output debug image topic",
)
@click.option(
    "--topic_out_hor_coeff",
    default="front_camera/horizon_coeffs",
    help="name of the horizon Line coeffs topic",
)
@click.option('--model', '-m', default='models/weights_ngdsac_pretrained.net', 
	help='a trained network')

@click.option('--capacity', '-c', type=int, default=4, 
	help='controls the model capactiy of the network, must match the model to load (multiplicative factor for number of channels)')

@click.option('--imagesize', '-is', type=int, default=256, 
	help='size of input images to the network, must match the model to load')

@click.option('--inlier_thr', '-it', type=float, default=0.05, 
	help='threshold used in the soft inlier count, relative to image size')

@click.option('--inlier_alpha', '-ia', type=float, default=0.1, 
	help='scaling factor for the soft inlier scores (controls the peakiness of the hypothesis distribution)')

@click.option('--inlier_beta', '-ib', type=float, default=100.0, 
	help='scaling factor within the sigmoid of the soft inlier count')

@click.option('--hypotheses', '-hyps', type=int, default=16, 
	help='number of line hypotheses sampled for each image')

@click.option('--uniform', '-u', 
	help='disable neural-guidance and sample data points uniformely, use with a DSAC model')

@click.option('--score_thr', '-st', type=float, default=0.4, 
	help='threshold on soft inlier count for drawing the estimate (range 0-1)')
@click.option('--device', '-d', type=str, default="cuda", 
	help='Device to run the neural network (cuda, cpu)')
def main(topic_in_img, topic_out_debug_img, topic_out_hor_coeff, model, capacity, imagesize, inlier_thr, 
         inlier_alpha, inlier_beta, hypotheses, uniform, score_thr, device):

	# setup ng dsac estimator
	ngdsac = NGDSAC(hypotheses, inlier_thr, inlier_beta, inlier_alpha, Loss(imagesize), 1)

	# load network
	nn = Model(capacity)
	nn.load_state_dict(torch.load(model, map_location=torch.device(device)))
	nn.eval()
	nn = nn.to(device)
 
	domain = DomainParticipant()

	# input topics
	in_topic_img = Topic(domain, topic_in_img, MainCameraImage)

	# output topics
	topic_out_debug_img = Topic(domain, topic_out_debug_img, MainCameraImage)
	topic_out_hor_coeff = Topic(domain, topic_out_hor_coeff, Points2D)

	# Subs-Pubs
	img_reader = DataReader(domain, in_topic_img)

	debug_img_writer = DataWriter(domain, topic_out_debug_img)
	debug_hor_coff_writed = DataWriter(domain, topic_out_hor_coeff)

	while True:
		# # Read input data (eulers from imu, image from camera) deleting them from queue
		image = img_reader.take_one()

		if not (image):
			continue

		#  Pre-process image
		frame = image.to_numpy()

		# Process frame
		viz = process_frame(frame, imagesize, device, uniform, nn, ngdsac, score_thr)
		# Prepare horizon line msg
		# p1 = Point2D(x=0, y=horizon_p_0_y)
		# p2 = Point2D(x=frame.shape[1], y=horizon_p_w_y)
		# points = Points2D([p1, p2])

		# Publish all
		# debug_hor_coff_writed.write(points)
		# print(viz.shape)
		cv2.imwrite('viz.png', viz)
		debug_img_writer.write(MainCameraImage.from_numpy(viz))
 
if __name__ == "__main__":
    main()