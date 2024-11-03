from flask import Flask, render_template, Response
import cv2
import face_recognition
import numpy as np

app = Flask(__name__)
camera = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
try:
    krish_image = face_recognition.load_image_file(r'C:\Users\Mr Hack\Downloads\Tutorial 8\Krish.jpg')
    krish_face_encoding = face_recognition.face_encodings(krish_image)[0]
except Exception as e:
    print(f"Error loading Krish image: {e}")
    krish_face_encoding = None

# Load a second sample picture and learn how to recognize it.
try:
    bradley_image = face_recognition.load_image_file(r'C:\Users\Mr Hack\Downloads\Tutorial 8\Bradley\Ahmed.jpg')
    bradley_face_encoding = face_recognition.face_encodings(bradley_image)[0]
except Exception as e:
    print(f"Error loading Bradley image: {e}")
    bradley_face_encoding = None

# Create arrays of known face encodings and their names
known_face_encodings = []
known_face_names = []

if krish_face_encoding is not None:
    known_face_encodings.append(krish_face_encoding)
    known_face_names.append("Krish")

if bradley_face_encoding is not None:
    known_face_encodings.append(bradley_face_encoding)
    known_face_names.append("Ahmed")

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []

def gen_frames():
    while True:
        success, frame = camera.read()  # Read the camera frame
        if not success:
            break
        else:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []

            # Process each face found in the frame
            for face_encoding in face_encodings:
                name = "Unknown"
                if known_face_encodings:  # Check if there are known encodings
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    if any(matches):  # Only proceed if there are matches
                        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                        if face_distances.size > 0:  # Check if face_distances is not empty
                            best_match_index = np.argmin(face_distances)
                            if matches[best_match_index]:
                                name = known_face_names[best_match_index]
                face_names.append(name)

            # Draw rectangles around detected faces
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
