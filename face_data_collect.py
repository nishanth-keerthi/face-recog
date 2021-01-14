# Write a Python Script that captures images from your webcam video stream
# Extracts all Faces from the image frame (using haarcascades)
# Stores the Face information into numpy arrays

# 1. Read and show video stream, capture images
# 2. Detect Faces and show bounding box (haarcascade)
# 3. Flatten the largest face image(gray scale) and save in a numpy array
# 4. Repeat the above for multiple people to generate training data


import cv2
import numpy as np

#Init Camera
cap = cv2.VideoCapture(0)

# Face Detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0
face_data = []
#giving a dataset path 
dataset_path = './data/'
#asking user to enter name of the person who is getting scanned
file_name = input("Enter the name of the person : ")

while True:
	ret,frame = cap.read()#reading frame and return value from stream

	if ret==False:
		continue

	gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)#converting image to grayscale to save memory
	

	faces = face_cascade.detectMultiScale(frame,1.3,5) #detecting image scale and reducing it by 30% until it becomes <=100*100 and 5=k in knn
	if len(faces)==0:#continue detecting until face is found
		continue
		
	faces = sorted(faces,key=lambda f:f[2]*f[3])

	# Pick the last face (because it is the largest face acc to area(f[2]*f[3]))
	for face in faces[-1:]:
		x,y,w,h = face #x and y are starting co-ordinates of x whereas it travels a distance of w and h respectively to reach the other end of the box
		cv2.rectangle(frame,(x,y),(x+w,y+h),(245,249,239),2)#specifying the rectangle dimensions and colour

		#Extract (Crop out the required face) : Region of Interest
		offset = 10 #the padding which selects the region outside the face to be used into the rectangular frame
		face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset] #reframing the box keeping offset in mind
		face_section = cv2.resize(face_section,(100,100))#giving the face section a resize of 100*100
#intially skip=0 and then we skip frame to every 10th value so that the face is captured every 10th frame
		skip += 1
		if skip%10==0:
			face_data.append(face_section)
			print(len(face_data))


	cv2.imshow("Frame",frame)
	cv2.imshow("Face Section",face_section)

	key_pressed = cv2.waitKey(1) & 0xFF #converting 32 bit 1 to 8 bit using bitwise and
	if key_pressed == ord('q'): #stop video stream if q is pressed
		break

# Convert our face list array into a numpy array
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

# Save this data into file system
np.save(dataset_path+file_name+'.npy',face_data)
print("Data Successfully save at "+dataset_path+file_name+'.npy')

cap.release()
cv2.destroyAllWindows()


