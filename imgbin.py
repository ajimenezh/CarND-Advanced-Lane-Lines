import numpy as np
import cv2
from camcal import CameraCalibrator

class ImageToBinary:
	
	def convert_to_binary(self, img):
		# Convert to HLS color space and separate the S channel
		hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
		s_channel = hls[:,:,0]
		# Threshold color channel
		s_thresh_min = 20
		s_thresh_max = 255
		s_binary = np.zeros_like(s_channel)
		s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

		s_channel = hls[:,:,2]
		# Threshold color channel
		s_thresh_min = 40
		s_thresh_max = 255
		s_binary2 = np.zeros_like(s_channel)
		s_binary2[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

		s_channel = hls[:,:,1]
		# Threshold color channel
		s_thresh_min = 180
		s_thresh_max = 255
		s_binary3 = np.zeros_like(s_channel)
		s_binary3[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
		
		# Convert to HSV color space and separate the V channe
		hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
		s_channel = hsv[:,:,2]
		# Threshold color channel
		s_thresh_min = 160
		s_thresh_max = 255
		s_binary_v = np.zeros_like(s_channel)
		s_binary_v[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
		
		# Apply each of the thresholding functions
		sxbinary = self.abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(20, 100))
		sxbinarx = self.abs_sobel_thresh(img, orient='y', sobel_kernel=3, thresh=(20, 100))
		mag_binary = self.mag_thresh(img, sobel_kernel=3, thresh=(30, 100))
		dir_binary = self.dir_threshold(img, sobel_kernel=3, thresh=(0.7, 1.3))

		combined = np.zeros_like(dir_binary)
		combined[((sxbinary == 1) & (sxbinarx == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

		combined_binary = np.zeros_like(sxbinary)
		#combined_binary[(s_binary2 == 1) | ((s_binary == 0) & (combined == 1))] = 255
		combined_binary[((s_binary2 == 1) & (s_binary_v==1)) | (s_binary3 == 1)] = 255
		#combined_binary[((s_binary2 == 1) )] = 255

		return combined_binary

	def abs_sobel_thresh(self, img, orient='x', sobel_kernel=3, thresh=(0, 255)):
		
		# Convert to grayscale
		gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		# Take the derivative in x or y given orient = 'x' or 'y'
		if orient == 'x':
		    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
		else:
		    sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
		# Take the absolute value of the derivative or gradient
		abs_sobel = np.absolute(sobel)
		# Scale to 8-bit (0 - 255)
		scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
		# mask of 1's where the scaled gradient magnitude 
		        # is > thresh_min and < thresh_max
		sbinary = np.zeros_like(scaled_sobel)
		sbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

		return sbinary

	def mag_thresh(self, img, sobel_kernel=3, thresh=(0, 255)):
    
		# Convert to grayscale
		gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		# gradient in x and y separately
		sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
		sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
		# magnitude 
		mag_sobel = np.sqrt(np.add(np.multiply(sobelx, sobelx), np.multiply(sobely, sobely)))
		# Scale to 8-bit (0 - 255)
		scaled_sobel = np.uint8(255*mag_sobel/np.max(mag_sobel))
		# binary mask where mag thresholds are met
		binary_output = np.zeros_like(scaled_sobel)
		binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

		return binary_output
    
	def dir_threshold(self, img, sobel_kernel=3, thresh=(0, np.pi/2)):
    
		# Convert to grayscale
		gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		# Take the gradient in x and y separately
		sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
		sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
		# Take the absolute value of the x and y gradients
		abs_sobelx = np.absolute(sobelx)
		abs_sobely = np.absolute(sobely)
		# calculate the direction of the gradient 
		dir = np.arctan2(abs_sobely, abs_sobelx)
		# binary mask where direction thresholds are met
		binary_output = np.zeros_like(dir)
		binary_output[(dir >= thresh[0]) & (dir <= thresh[1])] = 1

		return binary_output

if __name__ == "__main__":
	
	# Object to correct the camera images
	camCalibrator = CameraCalibrator()

	img2bin = ImageToBinary()

	test = "./examples/test_1.jpg"
	img = cv2.imread(test)
	img = camCalibrator.cal_undistort(img)
	bin_img = img2bin.convert_to_binary(img)

	# Plotting thresholded images
	out = "./examples/" + test.split("/")[-1][:-4] + "_bin.jpg"
	cv2.imwrite(out, bin_img)

