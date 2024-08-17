import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import webbrowser

font = cv2.FONT_HERSHEY_COMPLEX

mp_face_detection = mp.solutions.face_detection
face_detector =  mp_face_detection.FaceDetection( min_detection_confidence = 0.6)

emotion = ""

def open_cam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("The camera could not be opened.")
        return

    root = tk.Toplevel()
    root.title("Facial Emotion")

    def close():
        cap.release()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", close)

    label = ttk.Label(root)
    label.pack(padx=10, pady=10)

    def update_image():
        global emotion
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #cv2.putText(frame, "emotion", (10,70), font, 2, (0,255,0), 2)
            results = face_detector.process(frame)
            
            if results.detections:
                for face in results.detections:
                    confidence = face.score
                    bounding_box = face.location_data.relative_bounding_box
                     
                    x = int(bounding_box.xmin * frame.shape[1])
                    w = int(bounding_box.width * frame.shape[1])
                    y = int(bounding_box.ymin * frame.shape[0])
                    h = int(bounding_box.height * frame.shape[0])
                     
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), thickness = 2)
                    
                    cropped = frame[y:y+h,x:x+h]
                    
                    tested = facial_landmark(cropped)
                    if tested.shape != (0,3):
                        predicted_tested = svm_model.predict(tested)
                        if predicted_tested[0]==1:
                            emotion="happy"
                        elif predicted_tested[0]==2:
                            emotion="angry"
                        elif predicted_tested[0]==3:
                            emotion="sad"
                        elif predicted_tested[0]==4:
                            emotion="fear"
                        elif predicted_tested[0]==5:
                            emotion="neutral"
                        elif predicted_tested[0]==6:
                            emotion="surprised"
                        cv2.putText(frame, emotion, (x,y), font, 2, (0,255,0), 2)
            
            img = Image.fromarray(frame)
            img = ImageTk.PhotoImage(image=img)
            label.img = img
            label.configure(image=img)
            root.after(1000, update_image)
        else:
            close()

    update_image()

    def show_url():
        close()
        yeni_pencere = tk.Toplevel()
        yeni_pencere.title("Music Recommendation")

        url = get_youtube_video_url(emotion)
        
        # URL'i ekrana yazdır
        url_label = ttk.Label(yeni_pencere, text=url)
        url_label.pack(padx=10, pady=10)

        # URL'yi açan buton
        def open_url():
            webbrowser.open(url)

        ac_buton = ttk.Button(yeni_pencere, text="go on...", command=open_url)
        ac_buton.pack(pady=10)

    url_button = ttk.Button(root, text="Suggest a song", command=show_url)
    url_button.pack(pady=10)

    root.mainloop()


main_window = tk.Tk()
main_window.title("App")


buton = ttk.Button(main_window, text="Open the camera", command=open_cam)
buton.pack(pady=20)


main_window.mainloop()