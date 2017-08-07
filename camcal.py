import glob
import pickle
import cv2

class CameraCalibrator:
	
	def __init__(self):
		self.objpoints = []
		self.imgpoints = [] #corners

		self.load_image_points()
		
		self.calibrate_camera()
		

	def load_image_points(self):
		# First, we extract the images points to find the calibration 
		# matrix and distortion coefficients
		# We only calculate the points the first time
		calcPoints = False

		try:
			dist_pickle = pickle.load(open("wide_dist_pickle.p", "rb"))
			self.objpoints = dist_pickle["objpoints"]
			self.imgpoints = dist_pickle["imgpoints"]
		except (OSError, IOError) as e:
			calcPoints = True
		
		img = cv2.imread("./camera_cal/calibration1.jpg")
		self.img_shapes = img.shape[0:2]

		if calcPoints:
			images = sorted(glob.glob("./camera_cal/calibration*.jpg"))
	
			# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
			objp = np.zeros((6*9,3), np.float32)
			objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)
		
			# boolean to save image to file, so we can check the results
			save_img = True
	
			for file in images:
		
				img = cv2.imread(file)

				gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

				ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
			
				if ret == True:
					self.imgpoints.append(corners)
					self.objpoints.append(objp)
			
					if save_img:
						undist = cv2.drawChessboardCorners(img, (9,6), corners, ret)
						cv2.imwrite("./output_images/corrected.jpg", undist)
						cv2.imwrite("./output_images/distorted.jpg", img)
						save_img = False
		
			data = {"objpoints" : self.objpoints, "imgpoints" : self.imgpoints}
			pickle.dump(data, open("wide_dist_pickle.p", "wb"))

	def calibrate_camera(self):
		self.t, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, self.img_shapes, None, None)

	# Function that given an image, returns the undistorted image
	def cal_undistort(self, img):

		undist = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
		return undist
	
	# Function that saves the undistorted image
	def save_undistort_image(self, path):
		img = cv2.imread(path)
		out = "./examples/" + path.split("/")[-1][:-4] + "_undist.jpg"
		undistorted = self.cal_undistort(img)
		cv2.imwrite(out, undistorted)

	
	def change_perspective(self, img, src, dst, shape):
		M = cv2.getPerspectiveTransform(src, dst)
		warped = cv2.warpPerspective(img, M, (shape[1], shape[0]), flags=cv2.INTER_LINEAR)
		return warped

if __name__ == "__main__":
	
	# Object to correct the camera images
	camCalibrator = CameraCalibrator()

	# Examples
	camCalibrator.save_undistort_image("./examples/cal1.jpg")
	camCalibrator.save_undistort_image("./examples/test_1.jpg")
