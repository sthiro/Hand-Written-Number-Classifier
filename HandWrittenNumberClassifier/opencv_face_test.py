import cv2

# Load the pre-trained face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the video capture object for webcam
cap = cv2.VideoCapture(0)
width_of_object = 0
predicted_distance = 0
make_width_inverse = 1000

while True:
    # Read the current frame from the webcam
    ret, frame = cap.read()
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=20)
    
    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        width_of_object = w
    
    # Display the frame with detected faces
    cv2.imshow('Face Detection', frame)
    
    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

   # print(f"Width of the Object : {make_width_inverse - width_of_object}")
    predicted_distance = (make_width_inverse - width_of_object)*(30/398)
    print(f"predicted_distance  : {predicted_distance} cm")

# Release the video capture object and close any open windows
cap.release()
cv2.destroyAllWindows()
