import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from moviepy.editor import VideoFileClip
from scipy import interp, arange
from camcal import CameraCalibrator
from imgbin import ImageToBinary

class Line:
	def __init__(self, fit):
		# was the line detected in the last iteration?
		self.detected = False     
		#polynomial coefficients for the most recent fit
		self.fit = fit.copy() 
		#radius of curvature of the line in some units
		self.radius_of_curvature = self.get_curvature(fit, 0) 
		#distance in meters of vehicle center from the line
		self.line_base_pos = fit[2]

		self.direction = 2*fit[0]*520 + fit[1]
		
		a = 30/720.0
		b = 3.7/700.0
		print (self.direction, self.radius_of_curvature*b/a, fit)

	def get_curvature(self, fit, y_eval):
		return ((1 + (2*fit[0]*y_eval + fit[1])**2)**1.5) / np.absolute(2*fit[0])

	def get_curvature_real(self, y_eval):
		r = self.get_curvature(self.fit, y_eval)
		a = 30/720.0
		b = 3.7/700.0
		return r*b/a

	def get_x(self, y_eval):
		return self.fit[0]*y_eval*y_eval + self.fit[1]*y_eval + self.fit[2]

class LaneFinder:
	
	def __init__(self):
		# window settings
		self.window_width = 50 
		self.window_height = 120 # Break image into 9 vertical layers since image height is 720
		self.margin = 50 # How much to slide left and right for searching

		self.camCalibrator = CameraCalibrator()
		self.img2bin = ImageToBinary()
		
		# Dta of previous detection to smooth the results and prevent outliers
		self.last_left_lanes = []
		self.last_right_lanes = []

		self.n_lanes = 6

		self.last_right_lane = None
		self.last_left_lane = None

	def window_mask(self, img_ref, center,level):
		output = np.zeros_like(img_ref)
		output[int(img_ref.shape[0]-(level+1)*self.window_height):int(img_ref.shape[0]-level*self.window_height),max(0,int(center-self.window_width/2)):min(int(center+self.window_width/2),img_ref.shape[1])] = 1
		return output

	def find_window_centroids(self, warped):
		
		window_centroids = [] # Store the (left,right) window centroid positions per level
		window = np.ones(self.window_width) # Create our window template that we will use for convolutions
		
		# First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
		# and then np.convolve the vertical image slice with the window template 
		
		# Sum quarter bottom of image to get slice, could use a different ratio
		l_sum = np.sum(warped[int(3*warped.shape[0]/4):,:int(warped.shape[1]/2)], axis=0)
		l_center = np.argmax(np.convolve(window,l_sum))-self.window_width/2
		r_sum = np.sum(warped[int(3*warped.shape[0]/4):,int(warped.shape[1]/2):], axis=0)
		r_center = np.argmax(np.convolve(window,r_sum))-self.window_width/2+int(warped.shape[1]/2)
		
		# Add what we found for the first layer
		window_centroids.append((l_center,r_center))

		dif_l = 0
		prev_l = l_center
		dif_r = 0
		prev_r = r_center
		
		# Go through each layer looking for max pixel locations
		for level in range(1,(int)(warped.shape[0]/self.window_height)):
			# convolve the window into the vertical slice of the image
			image_layer = np.sum(warped[int(warped.shape[0]-(level+1)*self.window_height):int(warped.shape[0]-level*self.window_height),:], axis=0)
			conv_signal = np.convolve(window, image_layer)

			# Find the best left centroid by using past left center as a reference
			# Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
			margin = self.margin
			found = False
			while margin <= 250 and not found:
				offset = self.window_width/2
				l_min_index = int(max(l_center+offset-margin,0))
				l_max_index = int(min(l_center+offset+margin,warped.shape[1]))

				index = np.argmax(conv_signal[l_min_index:l_max_index])
			
				#print (conv_signal[l_min_index + index])
				if conv_signal[l_min_index + index] > 30000 or len(window_centroids) == 0:
					#print (dif_l, index+l_min_index-offset - prev_l)
					dif_l = index+l_min_index-offset - prev_l
					l_center = index+l_min_index-offset
					prev_l = l_center
					found = True
				
				margin += self.margin
			
			if not found:
				prev_l = l_center
				l_center = window_centroids[-1][0]
	
			# Find the best right centroid by using past right center as a reference
			margin = self.margin
			found = False
			while margin <= 250 and not found:
				r_min_index = int(max(r_center+offset-margin,0))
				r_max_index = int(min(r_center+offset+margin,warped.shape[1]))
				index = np.argmax(conv_signal[r_min_index:r_max_index])
			
				#print ([r_min_index,r_max_index])
				#print (level,margin, index, index+r_min_index-offset, conv_signal[r_min_index + index])
				#print (conv_signal[r_min_index + index])

				if conv_signal[r_min_index + index] > 30000 or len(window_centroids) == 0:
					#print (dif_r, index+r_min_index-offset - prev_r)
					r_center = index+r_min_index-offset
					prev_r = r_center
					found = True

				margin += self.margin

			if not found:
				prev_r = r_center
				r_center = window_centroids[-1][1]

			# Add what we found for that layer
			window_centroids.append((l_center,r_center))

		return window_centroids
	
	# Returns the binary image with the centroids as rectangles
	def draw_lanes_rect(self, warped):
		window_centroids = self.find_window_centroids(warped)

		# If we found any window centers
		if len(window_centroids) > 0:

			# Points used to draw all the left and right windows
			l_points = np.zeros_like(warped)
			r_points = np.zeros_like(warped)

			# Go through each level and draw the windows 	
			for level in range(0,len(window_centroids)):
				# Window_mask is a function to draw window areas
				l_mask = self.window_mask(warped,window_centroids[level][0],level)
				r_mask = self.window_mask(warped,window_centroids[level][1],level)
				# Add graphic points from window mask here to total pixels found 
				l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
				r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

			# Draw the results
			template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
			zero_channel = np.zeros_like(template) # create a zero color channel
			template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
			warpage = np.array(cv2.merge((warped,warped,warped)),np.uint8) # making the original road pixels 3 color channels
			output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results
		 
		# If no window centers found, just display orginal road image
		else:
			output = np.array(cv2.merge((warped,warped,warped)),np.uint8)

		return output
	
	# Function to calculate the lanes polynomials
	def get_lanes(self, warped):

		window_centroids = self.find_window_centroids(warped)

		if len(window_centroids) > 0:

			leftx = np.zeros(len(window_centroids))
			lefty = np.zeros(len(window_centroids))
			rightx = np.zeros(len(window_centroids))
			righty = np.zeros(len(window_centroids))

			# Go through each level and draw the windows 	
			for level in range(0,len(window_centroids)):
				leftx[level] = window_centroids[level][0]
				rightx[level] = window_centroids[level][1]
				lefty[level] = warped.shape[0]-level*self.window_height
				righty[level] = warped.shape[0]-level*self.window_height

			nl = 0
			nr = 0
			while nl < len(window_centroids)-3:
				if window_centroids[-nl-1][0] == window_centroids[-nl-2][0]:
					nl += 1
				else:
					break

			while nr < len(window_centroids)-3:
				if window_centroids[-nr-1][1] == window_centroids[-nr-2][1]:
					nr += 1
				else:
					break

			nl = len(window_centroids) - nl
			nr = len(window_centroids) - nr

			#print (nl, len(window_centroids))
			#print (window_centroids)
			#print (lefty[0:nl-1], leftx[0:nl-1])
			#print (np.polyfit(lefty[0:nl-1], leftx[0:nl-1], 2))
			#print (np.polyfit(lefty, leftx, 2))

			# Fit a second order polynomial to each
			left_fit = np.polyfit(lefty[0:nl], leftx[0:nl], 2)
			right_fit = np.polyfit(righty[0:nr], rightx[0:nr], 2)
		
			return left_fit, right_fit

		else:
			return np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])

	# Returns the binary image with the fitted polynomial lines
	def draw_lanes_poly(self, img):
		out_img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGRA)
		window_img = np.zeros_like(out_img)
		
		left_fit, right_fit = self.get_lanes(img)

		ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
		left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
		right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

		# Generate a polygon to illustrate the search window area
		# And recast the x and y points into usable format for cv2.fillPoly()
		left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-self.margin, ploty]))])
		left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+self.margin, ploty])))])
		left_line_pts = np.hstack((left_line_window1, left_line_window2))
		right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-self.margin, ploty]))])
		right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+self.margin, ploty])))])
		right_line_pts = np.hstack((right_line_window1, right_line_window2))

		# Draw the lane onto the warped blank image
		cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
		cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
		result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

		# Lanes
		window_img = np.zeros_like(out_img)

		left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-2, ploty]))])
		left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+2, ploty])))])
		left_line_pts = np.hstack((left_line_window1, left_line_window2))
		right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-2, ploty]))])
		right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+2, ploty])))])
		right_line_pts = np.hstack((right_line_window1, right_line_window2))

		# Draw the lane onto the warped blank image
		cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 255))
		cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 255))
		result = cv2.addWeighted(result, 1, window_img, 1.0, 0)

		return result

	def calc_next_lane_warped(self, img):
		out_img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGRA)
		
		left_fit, right_fit = self.get_lanes(img)

		left_fit = self.process_left_lane(left_fit)
		right_fit = self.process_right_lane(right_fit)

		ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
		left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
		right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

		warp_zero = np.zeros_like(img).astype(np.uint8)
		color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

		pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
		pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
		pts = np.hstack((pts_left, pts_right))

		# Draw the lane onto the warped blank image
		cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

		return color_warp

	def process_left_lane(self, lane):
		
		line = Line(lane.copy())
		line.detected = True

		self.last_left_lanes.append(line)
		if len(self.last_left_lanes) > self.n_lanes:
			self.last_left_lanes = self.last_left_lanes[1:]


		points = [[] for i in range((int)(720/self.window_height))]

		Y = []
		for i in range((int)(720/self.window_height)):
			y = i*self.window_height + self.window_height/2
			Y.append(y)

		for elem in self.last_left_lanes:
			if elem.detected:
				for i in range((int)(720/self.window_height)):
					y = i*self.window_height + self.window_height/2
					points[i].append(elem.fit[0]*y**2 + elem.fit[1]*y + elem.fit[2])

		X = []
		for i in range((int)(720/self.window_height)):
			x = np.median(np.array(points[i]))
			X.append(x)

		#print (lane, np.polyfit(Y,X,2))

		self.last_left_lane = Line(np.polyfit(Y,X,2))
		
		return self.last_left_lane.fit

	def process_right_lane(self, lane):
		
		line = Line(lane)
		line.detected = True

		self.last_right_lanes.append(line)
		if len(self.last_right_lanes) > self.n_lanes:
			self.last_right_lanes = self.last_right_lanes[1:]


		points = [[] for i in range((int)(720/self.window_height))]

		Y = []
		for i in range((int)(720/self.window_height)):
			y = i*self.window_height + self.window_height/2
			Y.append(y)

		for elem in self.last_right_lanes:
			if elem.detected:
				for i in range((int)(720/self.window_height)):
					y = i*self.window_height + self.window_height/2
					points[i].append(elem.fit[0]*y**2 + elem.fit[1]*y + elem.fit[2])

		X = []
		for i in range((int)(720/self.window_height)):
			x = np.median(np.array(points[i]))
			X.append(x)

		#print (lane, np.polyfit(Y,X,2))

		self.last_right_lane = Line(np.polyfit(Y,X,2))
		
		return self.last_right_lane.fit

	def find_lanes_video(self, path):
		white_output = 'output_videos/' + path.split('/')[-1]
		## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
		## To do so add .subclip(start_second,end_second) to the end of the line below
		## Where start_second and end_second are integer values representing the start and end of the subclip
		## You may also uncomment the following line for a subclip of the first 5 seconds
		##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
		clip1 = VideoFileClip(path)
		white_clip = clip1.fl_image(self.process_image) #NOTE: this function expects color images!!
		white_clip.write_videofile(white_output, audio=False)

	def process_image(self, dist):
		
		global idx
		#cv2.imwrite("tmp/test_" + str(idx) + ".jpg", cv2.cvtColor(dist, cv2.COLOR_RGB2BGR))
		#idx += 1

		undist = self.camCalibrator.cal_undistort(dist)
		bin_img = self.img2bin.convert_to_binary(undist)
		
		shape = bin_img.shape[0:2]

		src = np.float32([[0.12*shape[1], shape[0]-10],[shape[1]*0.44, shape[0]*0.64],[shape[1]*0.56, shape[0]*0.64],[shape[1]*0.88, shape[0]-10]])

		dest = np.float32([[0.10*shape[1], shape[0]-1],[shape[1]*0.10, 0],[shape[1]*0.90, 0],[shape[1]*0.90, shape[0]-1]])

		warped = self.camCalibrator.change_perspective(bin_img, src, dest, bin_img.shape[0:2])
		
		output = self.calc_next_lane_warped(warped)

		newwarp = self.camCalibrator.change_perspective(output, dest, src, output.shape[0:2]) 
		# Combine the result with the original image
		result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
	
		leftCurv = self.last_left_lane.get_curvature_real(shape[0])
		rightCurv = self.last_right_lane.get_curvature_real(shape[0])

		text = 'Radius of curvature: (L): {left} m (R): {right} m'.format(left=leftCurv, right=rightCurv)
		cv2.putText(result,text, (0,50), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
		
		left = self.last_left_lane.get_x(shape[0])
		right = self.last_right_lane.get_x(shape[0])
		offset = (right + left)/2.0 - shape[1]/2
		offset *= 3.7/700.0

		text = 'Offset from center: {offset} m'.format(offset=offset)
		cv2.putText(result,text, (0,100), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)

		#cv2.imwrite("tmp/testout_" + str(idx) + ".jpg", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
		#idx += 1

		return result

idx = 0		
	
if __name__ == "__main__":
	
	testing = False

	if testing:
		# Object to correct the camera images
		camCalibrator = CameraCalibrator()

		# Examples
		#camCalibrator.save_undistort_image("./camera_cal/calibration1.jpg")
	
		#for filename in os.listdir("./test_images"):
		#	camCalibrator.save_undistort_image("./test_images/" + filename)

		img2bin = ImageToBinary()

		test = "./tmp/test_1.jpg"
		img = cv2.imread(test)
		img = camCalibrator.cal_undistort(img)
		bin_img = img2bin.convert_to_binary(img)

		# Plotting thresholded images
		cv2.imshow("test", bin_img)
		cv2.waitKey(0)

	
		# Points for bird-view transformation
		shape = bin_img.shape[0:2]
		src = np.float32([[0.12*shape[1], shape[0]-10],[shape[1]*0.44, shape[0]*0.64],[shape[1]*0.56, shape[0]*0.64],[shape[1]*0.88, shape[0]-10]])

		# We draw the points in the image to check them
		img2 = bin_img
		img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
		for i in range(len(src)):
			cv2.line(img2,tuple(src[i]),tuple(src[(i+1)%len(src)]),(0,0,255),3)
		
		cv2.imwrite("./examples/test1_bin_trans1.jpg", img2)
		cv2.imshow("test", img2)
		cv2.waitKey(0)

		dest = np.float32([[0.10*shape[1], shape[0]-1],[shape[1]*0.10, 0],[shape[1]*0.90, 0],[shape[1]*0.90, shape[0]-1]])

		print (src)
	
		img2 = camCalibrator.change_perspective(bin_img, src, dest, img.shape[0:2])
	
		out = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
		for i in range(len(dest)):
			cv2.line(out,tuple(dest[i]),tuple(dest[(i+1)%len(dest)]),(0,0,255),3)
		
		cv2.imwrite("./examples/test1_bin_trans2.jpg", out)
		cv2.imshow("test", out)
		cv2.waitKey(0)
	
		solver = LaneFinder()
		warped = img2
	
		output = solver.draw_lanes_rect(warped)

		# Display the final results
		plt.imshow(output)
		#plt.title('window fitting results')
		plt.show()
		cv2.imwrite("./examples/test1_bin_rect.jpg", output)

		#print (solver.get_lanes(warped))

		output = solver.draw_lanes_poly(warped)
		
		#cv2.imwrite("./examples/test1_bin_pol.jpg", output)
		# Display the final results
		#cv2.imshow("test", output)
		#cv2.waitKey(0)

		output = solver.calc_next_lane_warped(warped)
	
		newwarp = camCalibrator.change_perspective(output, dest, src, img.shape[0:2]) 
		# Combine the result with the original image
		result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
		cv2.imshow("test", result)
		cv2.waitKey(0)
		#cv2.imwrite("./examples/test1_result.jpg", result)
	else:
		solver = LaneFinder()

		#import cProfile
		#cProfile.run('solver.find_lanes_video("project_video.mp4")')

		solver.find_lanes_video('project_video.mp4')

	
		

