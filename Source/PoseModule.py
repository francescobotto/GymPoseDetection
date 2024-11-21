import math
import cv2
import time
import mediapipe as mp


class poseDetector:
    def __init__(self, static_image_mode=False, model_complexity=1, smooth_landmarks=True,
                 enable_segmentation=False, smooth_segmentation=True,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.static_image_mode, self.model_complexity,
                                     self.smooth_landmarks, self.enable_segmentation,
                                     self.smooth_segmentation, self.min_detection_confidence,
                                     self.min_tracking_confidence)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks and draw:
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def getPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            h, w, c = img.shape
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lmList

    def calculateAngle(self, p1, p2, p3):
        """
        Calculate the angle between three points.
        Args:
        - p1, p2, p3: (x, y) tuples for three landmarks.
        Returns:
        - Angle in degrees.
        """
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360
        return angle


def learn_parameters(video_path):
    """
    Learn the minimum and maximum angles from a reference video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None, None

    detector = poseDetector()
    min_angle = float('inf')
    max_angle = float('-inf')

    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.resize(img, (960, 540))
        img = detector.findPose(img)
        lmList = detector.getPosition(img)

        if len(lmList) > 0:
            # Example: Right leg (hip, knee, ankle)
            hip = lmList[24][1:3]
            knee = lmList[26][1:3]
            ankle = lmList[28][1:3]

            angle = detector.calculateAngle(hip, knee, ankle)
            min_angle = min(min_angle, angle)
            max_angle = max(max_angle, angle)

            # Display the angle on the video
            cv2.putText(img, f"Angle: {int(angle)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        # Show the video
        cv2.imshow("Learning Parameters", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"Learned Angle Range: {min_angle:.2f}° - {max_angle:.2f}°")
    return min_angle, max_angle


def validate_video(video_path, min_angle, max_angle):
    """
    Validate a video using the learned angle parameters.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    detector = poseDetector()

    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.resize(img, (960, 540))
        img = detector.findPose(img)
        lmList = detector.getPosition(img)

        if len(lmList) > 0:
            # Example: Right leg (hip, knee, ankle)
            hip = lmList[24][1:3]
            knee = lmList[26][1:3]
            ankle = lmList[28][1:3]

            angle = detector.calculateAngle(hip, knee, ankle)

            if min_angle <= angle <= max_angle:
                feedback = "Correct"
                color = (0, 255, 0)  # Green for correct
            else:
                feedback = "Incorrect"
                color = (0, 0, 255)  # Red for incorrect

            cv2.putText(img, f"Angle: {int(angle)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
            cv2.putText(img, feedback, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        # Show the video
        cv2.imshow("Validation", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    # Learn parameters from the reference video
    ref_video = "./PoseVideos/IMG_5161.MOV"
    min_angle, max_angle = learn_parameters(ref_video)

    if min_angle is None or max_angle is None:
        print("Failed to learn parameters.")
        return

    # Validate a new video using the learned parameters
    test_video = "./PoseVideos/Squat.mp4"
    validate_video(test_video, min_angle, max_angle)


if __name__ == "__main__":
    main()
