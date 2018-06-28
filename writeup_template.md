**Vehicle Detection Project**

Goals / steps of this project:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./carNotCar.png
[image2]: ./cspaceHOG.png
[image3]: ./slidingWindows.png
[image4]: ./slidingWindows.jpg
[image5]: ./heatmap.png
[video1]: ./test_video_0.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in code cell 5 of the IPython notebook: 
```
# Define a function to return HOG features and visualization
# Vis == False means we do not want to get an image back, True produces output image.
def get_hog_features(img, 
                     orient, 
                     pix_per_cell, 
                     cell_per_block, 
                     vis=False, 
                     feature_vec=True):
    if vis == True:
        features, hog_image = hog(img, 
                                  orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
#                                   block_norm= 'L2-Hys', 
                                  transform_sqrt=False, 
                                  visualise=vis, 
                                  feature_vector=feature_vec)
        return features, hog_image
    else:      
        features = hog(img, 
                       orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
#                        block_norm= 'L2-Hys', 
                       transform_sqrt=False, 
                       visualise=vis, 
                       feature_vector=feature_vec)
        return features
```        
        
I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

Using trail and error, and running HOG with different parameters led me to pick the chosen HOG parameters:

Using :  9 Orientations, 8 Pixels per cell 2 Cells per block 32 Histogram bins, and (32, 32) Spatial sampling


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the code from the course work and the walk through video (Cell 7 of the accompanying Jupyter Notebook):
```
# Define feature parameters
color_space = 'YCrCb'
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL'
spatial_size = (32, 32)
hist_bins = 32
spatial_feat = True
hist_feat = True
hog_feat = True

t = time.time()
n_samples = 8000
# Generate 1000 random indices
random_idxs = np.random.randint(0 , len(cars), n_samples)
test_cars = cars # np.array(cars)[random_idxs]
test_notcars = notcars # np.array(notcars)[random_idxs]

car_features = extract_features(test_cars,
                               color_space = color_space,
                               spatial_size = spatial_size,
                               hist_bins = hist_bins,
                               orient = orient,
                               pix_per_cell = pix_per_cell,
                               cell_per_block = cell_per_block,
                               hog_channel = hog_channel,
                               spatial_feat = spatial_feat,
                               hist_feat = hist_feat,
                               hog_feat = hog_feat
                               )

notcar_features = extract_features(test_notcars,
                               color_space = color_space,
                               spatial_size = spatial_size,
                               hist_bins = hist_bins,
                               orient = orient,
                               pix_per_cell = pix_per_cell,
                               cell_per_block = cell_per_block,
                               hog_channel = hog_channel,
                               spatial_feat = spatial_feat,
                               hist_feat = hist_feat,
                               hog_feat = hog_feat
                               )

print(time.time()-t, 'Seconds to compute features...')

X = np.vstack((car_features, notcar_features)).astype(np.float64)
X_scaler = StandardScaler().fit(X)
scaled_X = X_scaler.transform(X)

y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X,
                                                   y,
                                                   test_size = 0.1,
                                                   random_state = rand_state
                                                   )
print('Using : ', orient,'Orientations,',pix_per_cell,'Pixels per cell', cell_per_block, 'Cells per block', 
      hist_bins,'Histogram bins, and', spatial_size, 'Spatial sampling')
print('Feature vector length : ', len(X_train[0]))

# Use SVC
svc = LinearSVC()

t = time.time()
svc.fit(X_train, y_train)
print(round(time.time() -t, 2 ), "Seconds to train SVC...")

print('Test accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
```
`221.21797585487366 Seconds to compute features...`
`Using :  9 Orientations, 8 Pixels per cell 2 Cells per block 32 Histogram bins, and (32, 32) Spatial sampling`
`Feature vector length :  8460`
`83.67 Seconds to train SVC...`
`Test accuracy of SVC =  0.9955`

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I used the code from the lessons and the walk through, as in cell 21 of the accompanying Jupyter Notebook. Overlaps of 0.3 through 0.75 were tried, with 0.5 producing the best results:
```
searchpath = 'test_images/*'
example_images = glob.glob(searchpath)
images = []
titles = []
y_start_stop = [400, 656]
overlap = 0.5

for img_src in example_images:
    t1 = time.time()
    img = mpimg.imread(img_src)
    draw_img = np.copy(img)
    img = img.astype(np.float32)/255
    print(np.min(img), np.max(img))
    
    windows = slide_window(img,
                          x_start_stop= [None, None],
                          y_start_stop = y_start_stop,
                          xy_window = (64, 64),
                          xy_overlap = (overlap, overlap)
                          )
    
    hot_windows = search_windows(img,
                                windows,
                                svc,
                                X_scaler,
                                color_space = color_space,
                                spatial_size = spatial_size,
                                hist_bins = hist_bins,
                                orient = orient,
                                pix_per_cell = pix_per_cell,
                                cell_per_block = cell_per_block,
                                hog_channel = hog_channel,
                                spatial_feat = spatial_feat,
                                hist_feat = hist_feat,
                                hog_feat = hog_feat 
                                )
    
    window_img = draw_boxes(draw_img, hot_windows, color = (0, 0, 255), thick = 6)
    images.append(window_img)
    titles.append('')
    print(time.time() - t1, 'seconds to process one image searching', len(windows),' windows')
    
fig = plt.figure(figsize=(12,18), dpi=300)
visualize(fig,5,2,images,titles)
```
`4.681360960006714 seconds to process one image searching 273  windows
0.0 1.0
2.67714786529541 seconds to process one image searching 273  windows
0.0 1.0
2.6031229496002197 seconds to process one image searching 273  windows
0.0 1.0
2.850148916244507 seconds to process one image searching 273  windows
0.0 1.0
4.27184796333313 seconds to process one image searching 273  windows
0.0 1.0
2.8198859691619873 seconds to process one image searching 273  windows`



![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, a reasonable result was produced.  Some example images of the results before a heat map and thersholding was introduced:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

False positives were reduced by adjusting the scale, ystart and ystop variables and scale. Thresholding and a Heatmap was applied using the code below, from cell 10 of the accompanying Jupyter Notebook:

```
def apply_threshold(heatmap, threshold):
    
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return the image after applying threshold
    return heatmap

def draw_labeled_bboxes(img, labels):
    
    # Iterate over all detected cars
    for car_number in range(1, labels[1]+1):
        nonzero = (labels[0] == car_number).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        cv2.rectangle(img, bbox[0], bbox[1],(0,255,255), 6)
    return img
```

Thresholding was applied using the code below, at cell 23 of the accompanying Jupyter Notebook. The threshold was set to 0.

```
def apply_threshold(heatmap, threshold):
    
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return the image after applying threshold
    return heatmap
```
Here is the code using scipy.ndimage.measurements.label() on the images being processed:
```
def process_image(img):
    out_img, heat_map = find_cars(img, scale)
    heat_map = apply_threshold(heat_map, 0)
    labels = label(heat_map)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)
    return draw_img
```

### Here are some frames and their corresponding heatmaps:

![alt text][image5]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Overall, this is a good starting point in vehicle detection and tracking, however there are still more false positives to eliminate and the bounding boxes could be less jittery.

Improvements

Adjust scale to detect images further away.
*Smoothing out the boxes in order to eliminate jittery bounding boxes.
*I did develop a very basic YOLO solution that worked reasonably well, with better accuracy and less jitter, however this was at deadline time not ready to be included in this Project submission. My intuition is that a Deep Neural Network, implementing YOLO v3 should be a faster solution, due to the fact that You Only Look Once.
