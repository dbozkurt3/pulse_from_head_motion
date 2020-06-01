"""
Detecting Pulse from Head Motions in Video

This is a pipeline for detecting average pulse rate from video
based on the paper by Balakrishnan et al.
"""

import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.stats
from scipy.fftpack import fft, fftfreq
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA

# Haar cascade face classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Parameters for Shi-Tomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.05,
                       minDistance = 5,
                       blockSize = 3 )

# Parameters for Lucas-Kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Object for principal component analysis
pca = PCA(n_components=5)


def getProcessRegion(frame):
    """
    Gets the upper and lower facial regions of interest from the frame.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        # Calculate subrectangle based on dimensions given in the paper
        xx = int(x+w*0.25)
        yy = int(y+h*0.05)
        ww = int(w/2)
        hh = int(h*0.9)

        # Calculate the middle rectangle (eye region)
        ex = xx
        ey = int(yy+hh*0.20)
        ew = ww
        eh = int(hh*0.35)

        # Calculate the upper rectangle (forehead region)
        ux = xx
        uy = ey+eh
        uw = ww
        uh = int(hh*0.55)

        # Calculate the lower rectangle (mouth region)
        dx = xx
        dy = yy
        dw = ww
        dh = int(hh*0.20)

    return (ux,uy,uw,uh, ex,ey,ew,eh, dx,dy,dw,dh)


def opticalFlow(old_gray, current_frame, p0):
    """
    Given an old frame, old feature points, and the current frame,
    temporally track the feature points in the current frame using
    the Lucas-Kanade algorithm and return the new points.
    """
    frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]

    # Draw the points
    for i, (new,old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        current_frame = cv2.circle(current_frame, (a,b), 1, (255,0,0), -1) # THIS MODIFIES FRAME

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)

    return (p0, old_gray, good_old, good_new)


def cubicSplineInterpolation(x):
    """
    Performs cubic spline interpolation on an input time series to increase
    the sampling rate to that of an ECG device.
    """
    outFr = 250  # Desired output resolution = 250 Hz
    inFr = 30    # The videos were captured at a frame rate of 30 Hz
    fr = outFr/inFr

    t = np.linspace(0, len(x)-1, num=len(x))
    f = interp1d(t, x, kind='cubic')

    t_new = np.linspace(0, len(x)-1, num=round(fr*len(x)))
    resized = f(t_new)

    return resized


def stable_features(data):
    """
    Filters out erratic points from the data. Erratic points are
    defined to be points which have a maximum displacement that exceeds
    the mode of maximum displacements for all points.
    """
    motions = np.diff(data, axis=1)
    max_motions = np.max(motions, axis=1)
    mode = scipy.stats.mode(max_motions)[0]

    return data[max_motions <= mode, :]


def temporalFilter(x):
    """
    Applies a 5th order Butterworth filter to a time series
    in order to filter the signal to a passband of [0.8, 3] Hz.
    This filter removes low frequency movements like respiration
    and changes in posture and retains higher frequency movements
    like pulse and harmonics.
    """
    fs = 250
    nyq = 0.5 * fs
    f_low = 0.8
    f_high = 3
    f_low_dig = f_low/nyq
    f_high_dig = f_high/nyq

    # Design a Butterworth filter using second-order sections (SOS) for accuracy
    sos = signal.butter(3, [f_low_dig, f_high_dig], btype='band', output='sos')

    # Apply the filter forward and backwards to remove phase shift
    y = signal.sosfiltfilt(sos, x)

    return y


def PCA_compute(data):
    """
    Perform PCA on the time series data and project the points onto the
    principal axes of variation (eigenvectors of covariance matrix) to get
    the principal 1-D position signals
    """
    temp = data.T  # Arrange the time series data in the format given in the paper

    l2_norms = np.linalg.norm(temp, ord=2, axis=1)  # Get L2 norms of each m_t

    # Discard the points that have L2 norms in the top 25%
    temp_with_abnormalities_removed = temp[l2_norms < np.percentile(l2_norms, 75)]

    # Fit the PCA model
    pca.fit(temp_with_abnormalities_removed)

    # Project the tracked point movements on to the principle component vectors
    projected = pca.transform(temp)

    return projected


def signal_selection(s):
    """
    Heart rate is calculated by selecting the maximal frequency of the
    most periodic signal. The periodicity of the signal is defined as the
    percentage of total spectral power accounted for by the frequency
    with maximal power and its first harmonic.
    """
    fs = 250

    max_percentages = []
    max_freqs = []

    # Only the first 5 source signals are analyzed per the paper
    for i in range(5):
        s_i = s[:, i]
        N = len(s_i)
        T = 1.0 / fs

        # Compute the power spectrum of the source signal
        spectrum = np.abs(fft(s_i))
        spectrum *= spectrum
        freqs = fftfreq(N, T)

        # Get frequency with max power and its first harmonic
        maxInd = np.argmax(spectrum)
        maxPower = spectrum[maxInd]
        maxFreq = np.abs(freqs[maxInd])
        firstHarmonic = 2*maxFreq
        firstHarmonicInd = np.where(freqs == firstHarmonic)
        firstHarmonicPower = spectrum[firstHarmonicInd]

        # Calculate percentage of total power the max frequency accounts for
        total_power = np.sum(spectrum)
        percentage = (maxPower + firstHarmonicPower) / total_power

        # Plot signal along with its DFT
        plt.figure()
        t = np.linspace(0, T*N, N)
        plt.subplot(2,1,1)
        plt.title('s{}'.format(i+1))
        plt.plot(t, s_i)

        plt.subplot(2,1,2)
        plt.title('FFT')
        plt.xlabel('Frequency')
        plt.plot(freqs, 1.0 / N * spectrum)

        max_percentages.append(percentage)
        max_freqs.append(maxFreq)

    # Calculate BPM from the most periodic signal
    idx = np.argmax(max_percentages)
    selected_signal = s[:, idx]
    peaks, _ = signal.find_peaks(selected_signal, height=0)
    plt.figure()
    plt.plot(selected_signal)
    plt.plot(peaks, selected_signal[peaks], "x")

    bpm = 60 * max_freqs[idx]

    return bpm


##########
#  MAIN  #
##########
if __name__ == '__main__':
    # Read input video file as first command line argument
    if len(sys.argv) < 2:
        print('ERROR: Must provide filename of mp4 file as an argument')
        sys.exit(1)

    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture(sys.argv[1])

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")

    # Track face until video is completed
    isFirstFrame = True
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret == False:
            print('End of video')
            break

        if isFirstFrame:
            # On the first frame, get the face region
            ux,uy,uw,uh, ex,ey,ew,eh, dx,dy,dw,dh = getProcessRegion(frame)

            # Create a mask for the regions of interest (forehead and mouth)
            old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_mask = np.zeros_like(old_gray)
            face_mask[uy:uy+uh, ux:ux+uw] = 1
            face_mask[dy:dy+dh, dx:dx+dw] = 1

            # Get points to track in the regions of interest
            p0 = cv2.goodFeaturesToTrack(old_gray, mask=face_mask, **feature_params)

            # Create a row of data for each feature point
            num_feature_points = len(p0)
            data = np.zeros((num_feature_points, 1))

            isFirstFrame = False

        else:
            # On subsequent frames, track feature points with Lucas-Kanade algorithm
            p0, old_gray, good_old, good_new = opticalFlow(old_gray, frame, p0)

            # Add each feature point's y-value to a corresponding time series y(t)
            new_column = np.zeros((num_feature_points, 1))
            for i in range(len(good_old)):
                new_column[i, 0] = good_old[i, 1]  # Set a value of column to y-value of the tracked point
            data = np.append(data, new_column, axis=1)

            # Show the video with tracking points
            cv2.imshow('Face', frame)

        # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # When video is done, release the video capture object
    cap.release()

    # Close all the frames
    cv2.destroyAllWindows()

    # Delete first dummy column from data
    data = data[:, 1:]

    # Remove erratic points
    # data = stable_features(data)

    # Plot the original time series of a point
    r, c = data.shape
    t = np.linspace(0, c-1, num=c)
    plt.figure(1)
    plt.title('Original Time Series of a Point')
    plt.ylabel('Y-Position')
    plt.xlabel('Time')
    plt.plot(t, data[0, :], 'b')

    # Signal processing
    num_row, num_col = data.shape
    outFr = 250
    inFr = 30
    fr = outFr/inFr
    interpolated_data = np.zeros((num_feature_points, round(num_col*fr)))
    filtered_data = np.zeros((num_feature_points, round(num_col*fr)))

    for time_series in range(num_row):
        # Increase the sampling frequency of the data using cubic spline interpolation
        interpolated_data[time_series, :] = cubicSplineInterpolation(data[time_series, :])

    # Plot interpolated time series of the point
    r, c = interpolated_data.shape
    t = np.linspace(0, c-1, num=c)
    plt.figure(2)
    plt.title('Interpolated Time Series')
    plt.ylabel('Y-Position')
    plt.xlabel('Time')
    plt.plot(t, interpolated_data[0, :], 'b')

    for time_series in range(num_row):
        # Filter out the low frequency movement with Butterworth bandpass filter
        filtered_data[time_series, :] = temporalFilter(interpolated_data[time_series, :])

    plt.figure(3)
    plt.title('Interpolated Data Passed Through Filter')
    plt.plot(t, filtered_data[0, :], 'b')

    # Compute PCA projection
    s = PCA_compute(filtered_data)

    # Calculate the heart rate in BPM
    bpm = signal_selection(s)
    print('Heart rate = {:.2f} BPM'.format(bpm))

    # Display the graphs
    plt.show()
