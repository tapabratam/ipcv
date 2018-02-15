import cv2, sys, numpy, os,time
size = 4
fn_haar = 'haarcascade_frontalface_default.xml'
fn_dir = 'database'
print('Training...')
haar_cascade = cv2.CascadeClassifier(fn_haar)
# Create a list of images and a list of corresponding names
(images, lables, names, id) = ([], [], {}, 0)
for (subdirs, dirs, files) in os.walk(fn_dir):
	for subdir in dirs:
		names[id] = subdir
		subjectpath = os.path.join(fn_dir, subdir)
		for filename in os.listdir(subjectpath):
			path = subjectpath + '/' + filename
			lable = id
		images.append(cv2.imread(path, 0))
		lables.append(int(lable))
		id += 1
(im_width, im_height) = (112, 92)

# Create a Numpy array from the two lists above
(images, lables) = [numpy.array(lis) for lis in [images, lables]]

def draw_text(img, text, x, y):
	cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
def detect_face(im):
	im = cv2.flip(im, 1, 0)
	gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	mini = cv2.resize(gray, (int(gray.shape[1] / size),int(gray.shape[0] / size)))
	faces = haar_cascade.detectMultiScale(mini)
	faces = sorted(faces, key=lambda x: x[3])
	if faces:
		face_i = faces[0]
		(x, y, w, h) = [v * size for v in face_i]
		face = gray[y:y + h, x:x + w]
		return face,faces[0]
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(images, lables)
print("Training done")
'''
img=cv2.imread('face.jpg')
face, rect = detect_face(img)
label= recognizer.predict(face)
print(names[label[0]])
'''
cascadePath = "haarcascade_frontalface_default.xml"

# Create classifier from prebuilt model
faceCascade = cv2.CascadeClassifier(cascadePath);

# Set the font style
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize and start the video frame capture
cam = cv2.VideoCapture()
cam.open(0)
# Loop
while True:
    # Read the video frame
    ret, im =cam.read()

    # Convert the captured frame into grayscale
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    # Get all face from the video frame
    faces = faceCascade.detectMultiScale(gray, 1.2,5)

    # For each face in faces
    for(x,y,w,h) in faces:

        # Create rectangle around the face
        cv2.rectangle(im, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)

        # Recognize the face belongs to which ID
        label= recognizer.predict(gray[y:y+h,x:x+w])
        #print(recognizer.score)
        #print(names[label[0]])

        # Put text describe who is in the picture
        cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
        cv2.putText(im, str(names[label[0]]), (x,y-40), font, 2, (255,255,255), 3)

    # Display the video frame with the bounded rectangle
    cv2.imshow('FRAME',im) 

    # If 'q' is pressed, close program
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Stop the camera
cam.release()

# Close all windows
cv2.destroyAllWindows()







