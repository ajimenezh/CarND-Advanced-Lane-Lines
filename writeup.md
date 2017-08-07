## Writeup 

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[board_dist]: ./examples/cal1.jpg "Distorted"
[board_undist]: ./examples/cal1_undist.jpg "Undistorted"
[road_dist]: ./examples/test_1.jpg "Road Distorted"
[road_undist]: ./examples/test_1_undist.jpg "Road Undistorted"
[road_bin]: ./examples/test_1_bin.jpg "Road Binary"
[perspective1]: ./examples/test1_bin_trans1.jpg "Road Binary Lines"
[perspective2]: ./examples/test1_bin_trans2.jpg "Road Binary Bird-View"
[road_rect]: ./examples/test1_bin_rect.jpg "Road Binary Centroids"
[road_pol]: ./examples/test1_bin_pol.jpg "Road Binary Polynomial Lines"
[road_result]: ./examples/test1_result.jpg "Road Result"

[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the file "camcal.py", which contains a class called CameraCalibrator, responsible for the correction of the distortion of the images.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. These points are only calculated the first time, and then they are store in a file with pickle, so the next time we don't need to load all the images.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image

![alt text][board_dist]

and using the `cv2.undistort()` function, I obtained this result: 

![alt text][board_undist]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

With the same function mentioned earlier, we can correct the distortion of any other camera image, for example, we can use a frame from one of the videos to show how it works:
![alt text][road_dist]

And using the function `cal_undistort()` from the class `CameraCalibrator`, which uses `cv2.undistort()` internally, we correct the distortion.

![alt text][road_undist]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The class in charge of creating the thresholded binary image is `ImageToBinary` in the file `imgbin.py`. The binary image is created by a combination of the S channel of a HLS image and the V channel of HSV image with an and operation, and then we add the L channel of the HLS image with an or operation. I have found that this combination is the one that works best with different illuminations and shadows. The other operations using the Sobel operation didn't differentiate well enough shadows on the road.

To obtain the binary image, we call `convert_to_binary()` with the undistorted image as the input, for example, for the previous image:

![alt text][road_bin]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for the perspective transformation is contained in the class `CameraCalibrator` in the file `camcal.py` by the name `change_perspective()`. This function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points. I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32([
[0.14*shape[1], shape[0]-10],
[shape[1]*0.46, shape[0]*0.62],
[shape[1]*0.56, shape[0]*0.62],
[shape[1]*0.88, shape[0]-10]])

dest = np.float32([
	[0.10*shape[1], shape[0]-1],
	[shape[1]*0.10, 0],
	[shape[1]*0.90, 0],
	[shape[1]*0.90, shape[0]-1]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 281, 710      | 128, 719      | 
| 563, 489      | 128, 0      	|
| 716, 489     	| 1152, 0      	|
| 998, 710      | 1152, 719     |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][perspective1]
![alt text][perspective2]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The lanes are fitted with a 2nd order polynomial in the function `get_lanes()` with the bird-view image as input. This function is inside the class `LaneFinder` in the file `program.py`.

To find the lanes, we first divide the image in horizontal windows, and with a convolution of the data, we find were are more white pixels, which means that it is probable that the lane is there. In order to find the lane in the convolution data, we start with a sum of all the data to obtain a first prediction, and then, we can search in windows near the last ones found the next window. Because sometimes, when there is a curve, a small margin is not enough, I've implemented a loop that increases the search margin if we have not found a good solution. We define a good solutions, the one were the convolution value is larger than 30000.

Once we have found the window centroids, we fit the data to a 2nd order polynomial, and these will be our lanes.

In this image we can see the result of the search of window centroids:

![alt text][road_rect]

And here, I have plotted the polynomial:

![alt text][road_pol]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The curvature is calculated in the function `get_curvature()` in the class `Line` in `program.py`. But I have not used the curvature in my algorithm, because it varies a lot between frames. Also we can find the position of the car with respect to the center by multiplying by the appropiate scale factors to the data before fitting the polynomial.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The function that plots the lane area is `process_image()` in the class `LaneFinder` in `program.py`. This functions does all the operations mentioned earlier, and then changes the perspective again, to obtain the original.

![alt text][road_result]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_videos/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

One problem I had, is that, when fitting the data to find the lane, sometimes, especially when the lane is not continuous, I don't have data for all the area that I want to plot, and the polynomial fitting is very bad at extrapolation, so if the discontinuous line is near the car, there the fit is good, but when we go further away and we don't have data, the curvature of the line can be wrong. A possible solution could be to use a line, instead of a 2nd order polynomial (even if the input are only two points, the function polyfit does not return a 1st order line).

Also, another problem I found is that it is hard to find a good binary threshold that works in every case. One possible solution may be to change the parameters automatically to find a good lane.

In addition, I have found that in the challenge video my algorithm fails in a place where a shadow hides the lane. The smoothing helps this, because if the pipeline can't detect a lane, it takes the median of the previous detections, but it is not a perfect solution.

I think that the biggest inestability comes from the binary threshold, and a way the make the program more robust would be to have a few functions with different parameters, each one trying to detect the lanes, and picking the best ones.

