from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler 
import numpy as np
from parse import parse
from cv2 import imshow, waitKey

# read data
x = parse('./data/2d_images')
y = parse('./data/2d_masks')

# scale the image
#scaler = StandardScaler()
#scaler.fit(x)

# build and train the MLP classifier
mlp = MLPClassifier(alpha = 1e-4, hidden_layer_sizes = (262144,512,), 
		random_state = 12, max_iter = 500, activation = 'relu',
		verbose = True, early_stopping = True, learning_rate_init = 0.001)
#mlp.fit(scaler.transform(x), y)
mlp.fit(x, y)

# test the MLP classifier
x_test = parse('./data/test_data/image')
test_mask = np.array(mlp.predict(scaler.transform(test[0])), np.uint8).reshape([512,512])
imshow('test mask', test_mask)
waitKey()