from tkinter import *
from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import imutils
import dlib
import cv2

def run_d_dtcn1():
    mixer.init()
    mixer.music.load("sound files/music.wav")

    def eye_aspect_ratio(eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    thresh = 0.25
    frame_check = 20
    detect = dlib.get_frontal_face_detector()
    predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

    cap = cv2.VideoCapture(0)
    flag = 0

    while True:
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = detect(gray, 0)

        for subject in subjects:
            shape = predict(gray, subject)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            if ear < thresh:
                flag += 1
                print(flag)
                if flag >= frame_check:
                    cv2.putText(frame, "****************ALERT!****************", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, "****************ALERT!****************", (10, 325),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    mixer.music.play()
            else:
                flag = 0

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            cap.release()  # Release the camera before closing
            break

def d_dtcn():
    root = Tk()
    root.configure(background="white")
    root.title("DROWSINESS DETECTION")
    Label(root, text="DROWSINESS DETECTION", font=("times new roman", 20), fg="black", bg="aqua", height=2).grid(
        row=2, rowspan=2, columnspan=5, sticky=N + E + W + S, padx=5, pady=10)

    Button(root, text="Run using web cam", font=("times new roman", 20), bg="#0D47A1", fg='white',
           command=run_d_dtcn1).grid(
        row=5, columnspan=5, sticky=W + E + N + S, padx=5, pady=5)

    Button(root, text="Exit", font=("times new roman", 20), bg="#0D47A1", fg='white', command=root.quit).grid(
        row=9, columnspan=5, sticky=W + E + N + S, padx=5, pady=5)

    root.mainloop()

# Flask part
from flask import Flask, redirect, url_for, render_template, request

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if request.form.get('Continue') == 'Continue':
            return render_template("test1.html")
    else:
        return render_template("index.html")

@app.route("/start", methods=['POST'])
def start():
    if request.form.get('Start') == 'Start':
        d_dtcn()  # Call d_dtcn function only when the Start button is clicked
    return redirect(url_for('home'))

@app.route('/contact', methods=['GET', 'POST'])
def cool_form():
    if request.method == 'POST':
        # do stuff when the form is submitted
        return redirect(url_for('drowsiness_detection'))
    return render_template('contact.html')

if __name__ == "__main__":
    app.run()
