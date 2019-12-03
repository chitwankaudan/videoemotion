from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.animation import FuncAnimation
import seaborn as sns
import cv2
from livepredicts import livepredicts
import imutils

"""
Running this script launches a "Face" frame to show you where a face has been detected
and live chart that displays the emotions that are detected in real time
"""

sns.set(style="darkgrid")

pause = False
maxSeconds = 20
maxProb= 1

# Use deque to hold the last maxSeconds number of predictions
deques = []
for i in range(7):
	deques.append(deque(np.zeros(maxSeconds), maxlen=maxSeconds))  # initialize deque with 0's

# Create plot, set axis titles
fig, ax = plt.subplots()
ax.set_ylim(0, maxProb)
ax.set_xlim(0, maxSeconds-1)
x = np.arange(0, maxSeconds)
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: '{:.0f}s'.format(maxSeconds - x - 1)))
#ax.legend('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
plt.xlabel('Seconds ago')

# Initialize lines for each emotion class 
lines = ax.plot(x, np.empty(maxSeconds), \
				x, np.empty(maxSeconds), \
				x, np.empty(maxSeconds), \
				x, np.empty(maxSeconds), \
				x, np.empty(maxSeconds), \
				x, np.empty(maxSeconds), \
				x, np.empty(maxSeconds))

labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# Shrink current axis's height by 10% on the bottom
box = ax.get_position()
ax.set_position([box.x0, box.y0,
				 box.width, box.height * 0.9])

# Put a legend below current axis
ax.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.2),
		ncol=4, prop={'size': 8})

# Create livepredicts
faceDetectionPath = 'detector/haarcascade_frontalface_default.xml' # may switch out for MTCNN later
emotionClassifierPath =  './models/model.hdf5'
predicts = livepredicts(faceDetectionPath, emotionClassifierPath, 'vgg')

# Start video
capture = cv2.VideoCapture(0)

# Define animate function to handling updating predictions over time
def animate(t):

	ret, frame = capture.read()
	frame = imutils.resize(frame, width=500) # resize frame to make processing for face detector
	
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Find face and return cropped frame
	face = predicts.crop(gray, frame, "demo.py")

	cv2.imshow('Face', frame)

	# Run emotion classifier
	predict = predicts.predict(face) #how to viz? need to figure that out
	print("Predictions: ", [str(labels[i]) + ": " + str(predict[i]) for i in range(7)])

	# Add next value to each line
	for i in range(7):
		deques[i].append(predict[i])
		lines[i].set_ydata(deques[i])

	return lines

# Pauses/plays graph on click
def onClick(event):
	global pause
	if not pause:
		pause = True
		ani.event_source.stop()
	else:
		pause = False
		ani.event_source.start()

# Run animation
fig.canvas.mpl_connect('button_press_event', onClick)
ani = FuncAnimation(
	fig, animate, interval=500, blit=True)

plt.show()

capture.release()
cv2.destroyAllWindows()