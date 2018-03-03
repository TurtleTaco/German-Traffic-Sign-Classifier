# Load pickled data
import pickle
import matplotlib.pyplot as plt
import cv2
import numpy as np
import random
import scipy
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import zoom
import random
import warnings
from tensorflow.contrib.layers import flatten
import tensorflow as tf
from sklearn.utils import shuffle

warnings.filterwarnings('ignore', '.*output shape of zoom.*')

__weightRegularize__ = False

def plot_num_data(X_train, y_train, X_valid, y_valid, X_test, y_test):
    # Plot training set
    sample_num_train_class = [0] * 43
    for i in range(X_train.shape[0]):
        target_class = y_train[i]
        sample_num_train_class[target_class] += 1

    print(sample_num_train_class)
    print("mean train sample number: ", np.mean(sample_num_train_class))
    target_class_list = list(range(43))
    plt.xlabel('Target Class')
    plt.ylabel('Number of Samples')
    plt.title('Class Sample Number in Training Set')
    plt.grid(True)
    plt.bar(target_class_list, sample_num_train_class)
    plt.show()

    # Plot validation set
    sample_num_valid_class = [0] * 43
    for i in range(X_valid.shape[0]):
        target_class = y_valid[i]
        sample_num_valid_class[target_class] += 1

    print(sample_num_valid_class)
    print("mean valid sample number: ", np.mean(sample_num_valid_class))
    target_class_list = list(range(43))
    plt.xlabel('Target Class')
    plt.ylabel('Number of Samples')
    plt.title('Class Sample Number in Validation Set')
    plt.grid(True)
    plt.bar(target_class_list, sample_num_valid_class)
    plt.show()

    # Plot test set
    sample_num_test_class = [0] * 43
    for i in range(X_test.shape[0]):
        target_class = y_test[i]
        sample_num_test_class[target_class] += 1

    print(sample_num_test_class)
    print("mean test sample number: ", np.mean(sample_num_test_class))
    target_class_list = list(range(43))
    plt.xlabel('Target Class')
    plt.ylabel('Number of Samples')
    plt.title('Class Sample Number in Test Set')
    plt.grid(True)
    plt.bar(target_class_list, sample_num_test_class)
    plt.show()

def grey_scale_batch(X_train):
    # take only 1 channel from X_train because greyscaled images only has one channel
    grey_X_train = X_train[:, :, :, 1]

    # double check shape of receiving container is (34799, 32, 32) ?
    # print(grey_X_train.shape)
    for i in range(X_train.shape[0]):
        grey_X_train[i, :, :] = cv2.cvtColor(X_train[i, :, :, :], cv2.COLOR_RGB2GRAY)
    return grey_X_train


def normalization(grey_X_train):
    # must specify the data type as float, otherwise it will inherite data type as int and cause the container to be int
    norm_X_train = grey_X_train[:, :, :].astype(float)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    for i in range(grey_X_train.shape[0]):
        norm_X_train[i, :, :] = (grey_X_train[i, :, :].astype(float) - 128) / 128
    return norm_X_train


def standardization(norm_X_train):
    stand_X_train = norm_X_train[:, :, :].astype(float)
    for i in range(stand_X_train.shape[0]):
        scaler = StandardScaler().fit(norm_X_train[i, :, :].astype(float))
        rescaled_image = scaler.transform(norm_X_train[i, :, :].astype(float))
        stand_X_train[i, :, :] = rescaled_image
    return stand_X_train

def rotate90(gray_image):
    return np.rot90(gray_image)


def flip_lr(gray_image):
    return np.fliplr(gray_image)


def flip_ud(gray_image):
    return np.flipud(gray_image)


def clipped_zoom(img, zoom_factor, **kwargs):
    h, w = img.shape[:2]

    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out
    if zoom_factor < 1:

        # Bounding box of the zoomed-out image within the output array
        zh = int(h / zoom_factor)
        zw = int(w / zoom_factor)
        top = (h - zh) // 2
        left = (w - zw) // 2

        # Zero-padding
        out = np.zeros_like(img)
        out[top:top + zh, left:left + zw] = zoom(img, zoom_tuple, **kwargs)

    # Zooming in
    elif zoom_factor > 1:

        # Bounding box of the zoomed-in region within the input array
        zh = int(h / zoom_factor)
        zw = int(w / zoom_factor)
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = zoom(img[top:top + zh, left:left + zw], zoom_tuple, **kwargs)

        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top + h, trim_left:trim_left + w]

    # If zoom_factor == 1, just return the input array
    else:
        out = img
    return out


def recreate_data(X_train, y_train, N_test):
    train_sample_number = [0] * 43

    # gray scale
    gray_X_train = grey_scale_batch(X_train)  # grey_X_train is (34799, 32, 32)
    # normalization
    norm_X_train = normalization(gray_X_train)
    # standardize image
    stand_X_train = standardization(norm_X_train)

    # Cut the samples from classes with more than N_test images
    X_train_recreate = np.ndarray(shape=(N_test * 43, 32, 32), dtype=float)
    y_train_recreate = np.ndarray(shape=(N_test * 43), dtype=int)
    print(X_train_recreate.shape)
    index_in_train_recreate = -1
    for i in range(stand_X_train.shape[0]):
        current_class = y_train[i]
        if train_sample_number[current_class] < N_test:
            train_sample_number[current_class] += 1
            index_in_train_recreate += 1
            X_train_recreate[index_in_train_recreate] = stand_X_train[i]
            y_train_recreate[index_in_train_recreate] = current_class

    print(train_sample_number)
    # create fake data for classes with less then 900 images
    for i in range(len(train_sample_number)):
        if train_sample_number[i] < N_test:
            # create fake images on "class i" images
            current_class_image_occurance_index = np.where(y_train_recreate == i)[0]
            current_num_image = train_sample_number[i]
            #         print(len(current_class_image_occurance_index))
            while (current_num_image < N_test):
                current_image = X_train_recreate[random.choice(current_class_image_occurance_index)]
                #             rotate_image = rotate90(current_image)
                flip_left_right = flip_lr(current_image)
                flip_up_down = flip_ud(current_image)
                zoom_image = clipped_zoom(current_image, 1.5)

                # index_in_train_recreate += 1
                # if index_in_train_recreate >= N_test * 43:
                #   break
                # else:
                #   X_train_recreate[index_in_train_recreate] = rotate_image
                #   y_train_recreate[index_in_train_recreate] = i

                index_in_train_recreate += 1
                if index_in_train_recreate >= N_test * 43:
                    break
                else:
                    X_train_recreate[index_in_train_recreate] = flip_left_right
                    y_train_recreate[index_in_train_recreate] = i
                    current_num_image += 1

                index_in_train_recreate += 1
                if index_in_train_recreate >= N_test * 43:
                    break
                else:
                    X_train_recreate[index_in_train_recreate] = flip_up_down
                    y_train_recreate[index_in_train_recreate] = i
                    current_num_image += 1

                index_in_train_recreate += 1
                if index_in_train_recreate >= N_test * 43:
                    break
                else:
                    X_train_recreate[index_in_train_recreate] = zoom_image
                    y_train_recreate[index_in_train_recreate] = i
                    current_num_image += 1

            train_sample_number[i] = current_num_image
    print(train_sample_number)
    return X_train_recreate, y_train_recreate, stand_X_train

def LeNet(x, keep_prob):
    # Hyperparameters
    mu = 0
    sigma = 0.1

    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x20.
    ''' output width = (input width - filter width + 1)/ stride '''
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 20), mean=mu, stddev=sigma))
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, conv1_W)
    conv1_b = tf.Variable(tf.zeros(20))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # Activation.
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input = 28x28x20. Output = 14x14x20.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 2: Convolutional. Output = 10x10x80.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 20, 80), mean=mu, stddev=sigma))
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, conv2_W)
    conv2_b = tf.Variable(tf.zeros(80))
    ''' conv1 [14, 14, 20] conv2_W [5, 5, 20, 40] output width (14 - 5 + 1)/1 = 10 output: [10, 10, 40] '''
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    # Layer 3: Convolutional. Output = 6x6x120
    conv3_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 80, 120), mean=mu, stddev=sigma))
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, conv3_W)
    conv3_b = tf.Variable(tf.zeros(120))
    conv3 = tf.nn.conv2d(conv2, conv3_W, strides=[1, 1, 1, 1], padding='VALID') + conv3_b

    # Activation.
    conv3 = tf.nn.relu(conv3)

    # Dropout
    conv3 = tf.nn.dropout(conv3, keep_prob)

    # Pooling. Input = 6x6x120. Output = 3x3x120.
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten. Input = 3x3x120. Output = 1080.
    ''' 3*3*120 = 1080 '''
    fc0 = flatten(conv3)

    # Layer 3: Fully Connected. Input = 1080. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(1080, 120), mean=mu, stddev=sigma))
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, fc1_W)
    fc1_b = tf.Variable(tf.zeros(120))
    ''' fc0 [1, 1080], fc1_W [1080, 120] output fc1 [1, 120] '''
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b

    # Activation.
    fc1 = tf.nn.relu(fc1)

    # Layer 4: Fully Connected. Input = 120. Output = 100.
    fc2_W = tf.Variable(tf.truncated_normal(shape=(120, 100), mean=mu, stddev=sigma))
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, fc2_W)
    fc2_b = tf.Variable(tf.zeros(100))
    ''' [1, 120] * [120, 100] = [1, 100] '''
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b

    # Activation.
    fc2 = tf.nn.relu(fc2)

    fc25_W = tf.Variable(tf.truncated_normal(shape=(100, 84), mean=mu, stddev=sigma))
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, fc25_W)
    fc25_b = tf.Variable(tf.zeros(84))
    fc25 = tf.matmul(fc2, fc25_W) + fc25_b

    # Activation
    fc25 = tf.nn.relu(fc25)

    # Layer 5: Fully Connected. Input = 84. Output = 43.
    fc3_W = tf.Variable(tf.truncated_normal(shape=(84, 43), mean=mu, stddev=sigma))
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, fc3_W)
    fc3_b = tf.Variable(tf.zeros(43))
    ''' [1, 84] * [84, 43] = [1, 43] '''
    logits = tf.matmul(fc25, fc3_W) + fc3_b

    return logits


def LeNet_x(x, keep_prob):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x20.
    conv1_w = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 20), mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(20))
    conv1 = tf.nn.bias_add(tf.nn.conv2d(x, conv1_w, strides=[1, 1, 1, 1], padding='VALID'), conv1_b)
    # TODO: Activation.
    conv1 = tf.nn.relu(conv1)

    # TODO: Pooling. Input = 28x28x20. Output = 14x14x20.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # add a internal layer 1.5 between layer 1 and layer 2
    # Convolutional. Input 14x14x20 Output = 12x12x40
    conv15_w = tf.Variable(tf.truncated_normal(shape=(3, 3, 20, 40), mean=mu, stddev=sigma))
    conv15_b = tf.Variable(tf.zeros(40))
    conv15 = tf.nn.bias_add(tf.nn.conv2d(conv1, conv15_w, strides=[1, 1, 1, 1], padding='VALID'), conv15_b)
    # TODO: Activation.
    conv15 = tf.nn.relu(conv15)

    conv15 = tf.nn.dropout(conv15, keep_prob)

    # TODO: Layer 2: Convolutional. Input = 12x12x40. Output = 10x10x80.
    conv2_w = tf.Variable(tf.truncated_normal(shape=(3, 3, 40, 80), mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(80))
    conv2 = tf.nn.bias_add(tf.nn.conv2d(conv15, conv2_w, strides=[1, 1, 1, 1], padding='VALID'), conv2_b)

    # TODO: Activation.
    conv2 = tf.nn.relu(conv2)

    # TODO: Pooling. Input = 10x10x80. Output = 5x5x80.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # TODO: Flatten. Input = 5x5x80. Output = 2000.
    fc0 = flatten(conv2)
    # TODO: Layer 3: Fully Connected. Input = 2000. Output = 120.
    fc1_w = tf.Variable(tf.truncated_normal(shape=(2000, 120), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.nn.bias_add(tf.matmul(fc0, fc1_w), fc1_b)
    # TODO: Activation.
    fc1 = tf.nn.relu(fc1)

    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_w = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.nn.bias_add(tf.matmul(fc1, fc2_w), fc2_b)
    # TODO: Activation.
    fc2 = tf.nn.relu(fc2)

    # TODO: Layer 5: Fully Connected. Input = 84. Output = 43.
    fc3_w = tf.Variable(tf.truncated_normal(shape=(84, 43), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(43))
    logits = tf.nn.bias_add(tf.matmul(fc2, fc3_w), fc3_b)

    return logits

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
        # Keep all weights during evaluation, dropout keeps all (1.0)
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

'''--------------------------------- Main ------------------------------------'''

training_file = "traffic-signs-data/train.p"
validation_file = "traffic-signs-data/valid.p"
testing_file = "traffic-signs-data/test.p"

EPOCHS = 50
BATCH_SIZE = 128
keep_prob = tf.placeholder(tf.float32)

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

# record shape of data
print("X_train: ", X_train.shape)
print("y_train: ", y_train.shape)
print("X_valid: ", X_valid.shape)
print("y_valid: ", y_valid.shape)
print("X_test: ", X_test.shape)
print("y_test: ", y_test.shape)

# plot_num_data(X_train, y_train, X_valid, y_valid, X_test, y_test)

''' Create more training samples because if one class has 20 times more examples of the other, 
    most of the training procedure will be spent on minimising the classification error on the 
    most numerous class. '''
print("Generating Training Data")
# Extending/reducing training set to 810 images per class

# X_train:  (34799, 32, 32, 3)
# y_train:  (34799,)
# X_valid:  (4410, 32, 32, 3)
# y_valid:  (4410,)
# X_test:  (12630, 32, 32, 3)
# y_test:  (12630,)

X_train_recreate, y_train_recreate, gray_X_train = recreate_data(X_train, y_train, 900)
# X_valid_recreate, y_valid_recreate, gray_X_valid = recreate_data(X_valid, y_valid, 102)
# gray scale
gray_X_valid = grey_scale_batch(X_valid)  # grey_X_train is (34799, 32, 32)
# normalization
norm_X_valid = normalization(gray_X_valid)
# standardize image
stand_X_valid = standardization(norm_X_valid)
X_valid_recreate = stand_X_valid
y_valid_recreate = y_valid
X_test_recreate, y_test_recreate, gray_X_test = recreate_data(X_test, y_test, 294)

# plot_num_data(X_train_recreate, y_train_recreate, X_valid_recreate, y_valid_recreate, X_test_recreate, y_test_recreate)

# Prepare input data
x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)

# Model completion
rate = 0.001

regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
logits = LeNet(x, keep_prob)

if __weightRegularize__:
    reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
    cross_entropy += reg_term
else:
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)

loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=rate)
training_operation = optimizer.minimize(loss_operation)
# Finish training process

# Precision analysis part
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

# Reshape input data and label to be suitable for architecture
# y_train_recreate = np.reshape(y_train_recreate, (y_train_recreate.shape[0], 1))
# y_valid_recreate = np.reshape(y_valid_recreate, (y_valid_recreate.shape[0], 1))
# y_test_recreate = np.reshape(y_test_recreate, (y_test_recreate.shape[0], 1))

X_train_recreate = np.reshape(X_train_recreate, (X_train_recreate.shape[0], X_train_recreate.shape[1], X_train_recreate.shape[2], 1))
X_valid_recreate = np.reshape(X_valid_recreate, (X_valid_recreate.shape[0], X_valid_recreate.shape[1], X_valid_recreate.shape[2], 1))
# X_test_recreate = np.reshape(X_test_recreate, (X_test_recreate.shape[0], X_test_recreate.shape[1], X_test_recreate.shape[2], 1))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train_recreate)

    print("Training...")
    for i in range(EPOCHS):
        X_train_recreate, y_train_recreate = shuffle(X_train_recreate, y_train_recreate)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train_recreate[offset:end], y_train_recreate[offset:end]

            # print(sess.run(bla, feed_dict={y : batch_y}))

            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.7})

        validation_accuracy = evaluate(X_valid_recreate, y_valid_recreate)
        print("EPOCH {} ...".format(i + 1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    saver.save(sess, './traffic_sign_classifier')
    print("Model saved")