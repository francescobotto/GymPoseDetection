import cv2
import time
import PoseModule
import mediapipe as mp
import numpy as npm

def main():
    cap = cv2.VideoCapture("./PoseVideos/Squat.mp4")
    if not cap.isOpened():
        print("Error: Video file could not be opened.")
        return

    pTime = 0
    detector = PoseModule.poseDetector()

    # Variables to track minimum and maximum angles
    min_angle = float('inf')  # Initialize to a very high value
    max_angle = float('-inf')  # Initialize to a very low value

    while True:
        success, img = cap.read()
        if not success or img is None:
            print("End of video or failed to read frame.")
            break

        img = cv2.resize(img, (960, 540))
        img = detector.findPose(img)
        lmList = detector.getPosition(img)

        if len(lmList) > 0:
            # Example: Right leg (hip, knee, ankle)
            hip = lmList[24][1:3]
            knee = lmList[26][1:3]
            ankle = lmList[28][1:3]

            # Calculate the angle at the knee
            angle = detector.calculateAngle(hip, knee, ankle)

            # Update the min and max angles
            min_angle = min(min_angle, angle)
            max_angle = max(max_angle, angle)

            # Display the angle on the video
            cv2.putText(img, f"Angle: {int(angle)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        # Display FPS on the video
        cTime = time.time()
        fps = 1 / (cTime - pTime) if cTime - pTime > 0 else 0
        pTime = cTime

        cv2.putText(img, f"FPS: {int(fps)}", (30, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

    # Print the learned angle range after the video finishes
    print(f"Learned Angle Range: {min_angle:.2f}° - {max_angle:.2f}°")

    cap = cv2.VideoCapture("./PoseVideos/Squat.mp4")
    pTime = 0
    detector = PoseModule.poseDetector()

    # Variabili per apprendere gli angoli
    min_angle = float('inf')  # Minimo angolo osservato
    max_angle = float('-inf')  # Massimo angolo osservato

    while True:
        success, img = cap.read()
        if not success:
            break  # Termina il ciclo quando il video è finito

        img = cv2.resize(img, (960, 540))
        img = detector.findPose(img)
        lmList = detector.getPosition(img)

        if len(lmList) > 0:
            # Esempio: Angolo per la gamba destra (hip, knee, ankle)
            hip = lmList[24][1:3]  # ID 24: Hip (anca destra)
            knee = lmList[26][1:3]  # ID 26: Knee (ginocchio destro)
            ankle = lmList[28][1:3]  # ID 28: Ankle (caviglia destra)

            # Calcola l'angolo al ginocchio
            angle = detector.calculateAngle(hip, knee, ankle)

            # Aggiorna i parametri di apprendimento
            min_angle = min(min_angle, angle)
            max_angle = max(max_angle, angle)

            # Mostra l'angolo sul video
            cv2.putText(img, f"Angle: {int(angle)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        # Mostra FPS sul video
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f"FPS: {int(fps)}", (30, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

        # Mostra il video
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Mostra i parametri appresi dopo il video
    print(f"Learned Angle Range: {min_angle:.2f}° - {max_angle:.2f}°")
