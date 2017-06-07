import os
#import scipy.io
import numpy as np
import statistics, math
import matplotlib.pyplot as plt
from libs import util_matlab as umatlab
from libs import datasets, dataset_utils, utils
import tensorflow as tf
import datetime

def main(winLSecs):
    data_dir = "/Users/alfonso/matlab/IndirectAcquisition/keras/dataforMarius/export"
    files = [os.path.join(data_dir, file_i) for file_i in os.listdir(data_dir) if file_i.endswith('.mat')]

    matlabStruct=umatlab.loadmat(files[1]).get('data')
    energyBand=matlabStruct.get('residualEnergyBand')
    energyBand=(energyBand /120 )+1 #normalize [0-1]
    totalSecs=matlabStruct.get('waveIn').shape[0]/matlabStruct.get('audioSR')
    energyBands_sr=240 #energyBand.shape[1]/totalSecs #This is around 240Hz- around 5ms at 44100Hz
    controlNames=matlabStruct.get('controlNames')
    controlData=matlabStruct.get('controlData')
    indexVel=[i for i in range(controlNames.shape[0]) if controlNames[i] == 'abs(velocity)'][0]
    indexForce=[i for i in range(controlNames.shape[0]) if controlNames[i] == 'forceN'][0]
    velocity=controlData[indexVel,:]/150
    force=(controlData[indexForce,:]+0.2)/2
    #indexString=[i for i in range(controlNames.shape[0]) if controlNames[i] == 'string'][0]
    #string=controlData[indexString,:]
    #pitch=controlData[6,:]/1500

    # We want winLSecs seconds of audio in our window
    #winLSecs = 0.05
    windowSize = int((winLSecs * energyBands_sr) // 2 * 2)
    # And we'll move our window by windowSize/2
    hopSize = windowSize // 2
    n_hops = (energyBand.shape[1]) // hopSize
    print('windowSize', windowSize)



    # ------------- prepare dataset
    Xs = []
    ys = []

    # Let's start with the music files
    for filename in files:
        # print(filename)
        matlabStruct = umatlab.loadmat(filename).get('data')
        energyBand = (matlabStruct.get('energyBand') / 120) + 1
        # energyBand=(matlabStruct.get('residualEnergyBand')/120)+1
        controlData = matlabStruct.get('controlData')
        controlNames = matlabStruct.get('controlNames')
        target = controlData[indexVel, :] / 150
        # target=(controlData[indexForce,:]+0.2)/2

        n_hops = (energyBand.shape[1]) // hopSize

        # print(n_frames_per_second, n_frames, frame_hops, n_hops)
        n_hops = int(n_hops) - 1
        for hop_i in range(n_hops):
            # Creating our sliding window
            frames = energyBand[:, (hop_i * hopSize):(hop_i * hopSize + windowSize)]
            Xs.append(frames[..., np.newaxis])
            # And then store the vel
            ys.append(target[(hop_i * hopSize):(hop_i * hopSize + windowSize)])

    Xs = np.array(Xs)
    ys = np.array(ys)
    print(Xs.shape, ys.shape)

    ds = datasets.Dataset(Xs=Xs, ys=ys, split=[0.8, 0.1, 0.1], n_classes=0)

    #---------- create ConvNet
    tf.reset_default_graph()

    # Create the input to the network.  This is a 4-dimensional tensor (batch_size, height(freq), widht(time), channels?)!
    # Recall that we are using sliding windows of our magnitudes (TODO):
    X = tf.placeholder(name='X', shape=(None, Xs.shape[1], Xs.shape[2], Xs.shape[3]), dtype=tf.float32)

    # Create the output to the network.  This is our one hot encoding of 2 possible values (TODO)!
    Y = tf.placeholder(name='Y', shape=(None, windowSize), dtype=tf.float32)

    # TODO:  Explore different numbers of layers, and sizes of the network
    n_filters = [9, 9, 9]

    # Now let's loop over our n_filters and create the deep convolutional neural network
    H = X
    for layer_i, n_filters_i in enumerate(n_filters):
        # Let's use the helper function to create our connection to the next layer:
        # TODO: explore changing the30 parameters here:
        H, W = utils.conv2d(
            H, n_filters_i, k_h=2, k_w=2, d_h=2, d_w=2,
            name=str(layer_i))

        # And use a nonlinearity
        # TODO: explore changing the activation here:
        # H = tf.nn.relu(H)
        H = tf.nn.softplus(H)
        # H 4D tensor [batch, height, width, channels]
        #    H=tf.nn.max_pool(value=H, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME', data_format='NHWC', name=None)

        # Just to check what's happening:
        print(H.get_shape().as_list())

    # Connect the last convolutional layer to a fully connected network
    fc1, W = utils.linear(H, n_output=100, name="fcn1", activation=tf.nn.relu)
    # fc2, W = utils.linear(fc, n_output=50, name="fcn2", activation=tf.nn.relu)
    # fc3, W = utils.linear(fc2, n_output=10, name="fcn3", activation=tf.nn.relu)


    # And another fully connceted network, now with just n_classes outputs, the number of outputs
    Y_pred, W = utils.linear(fc1, n_output=windowSize, name="pred", activation=tf.nn.sigmoid)

    loss = tf.squared_difference(Y_pred, Y)
    cost = tf.reduce_mean(tf.reduce_sum(loss, 1))
    learning_rate = 0.001
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # predicted_y = tf.argmax(Y_pred,1)
    # actual_y = tf.argmax(Y,1)
    # correct_prediction = tf.equal(predicted_y, actual_y)
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))



    #-----TRAIN ConvNet
    # Explore these parameters: (TODO)
    batch_size = 400

    # Create a session and init!
    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())

    # Now iterate over our dataset n_epoch times
    n_epochs = 100
    for epoch_i in range(n_epochs):
        print('Epoch: ', epoch_i)

        # Train
        this_cost = 0
        its = 0

        # Do our mini batches:
        for Xs_i, ys_i in ds.train.next_batch(batch_size):
            # Note here: we are running the optimizer so
            # that the network parameters train!
            this_cost += sess.run([cost, optimizer], feed_dict={
                X: Xs_i, Y: ys_i})[0]
            its += 1
            # print(this_cost / its)
        print('Training cost: ', this_cost / its)

        # Validation (see how the network does on unseen data).
        this_cost = 0
        its = 0

        # Do our mini batches:
        for Xs_i, ys_i in ds.valid.next_batch(batch_size):
            # Note here: we are NOT running the optimizer!
            # we only measure the accuracy!
            this_cost += sess.run(cost, feed_dict={
                X: Xs_i, Y: ys_i})  # , keep_prob: 1.0
            its += 1
        print('Validation cost: ', this_cost / its)

    # #-----plot convolutional Kernels learned
    # g = tf.get_default_graph()
    # for layer_i in range(len(n_filters)):
    #     W = sess.run(g.get_tensor_by_name('{}/W:0'.format(layer_i)))
    #     plt.figure(figsize=(5, 5))
    #     plt.imshow(utils.montage_filters(W))
    #     plt.title('Layer {}\'s Learned Convolution Kernels'.format(layer_i))

    modelFileName = './models/velocity_wL' + str(winLSecs) + '_' + datetime.datetime.now().strftime(
        "%Y%m_d_%H%M") + '.chkp'
    saver.save(sess, modelFileName)

if __name__ == "__main__":
    winLSecs = 0.05
    main(winLSecs)