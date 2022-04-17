import cv2
import numpy as np
import mediapipe as mp


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
dic={'LEFTELBOW':[],
'RIGHTELBOW':[],
'LEFTSHOULDER':[],
'RIGHTSHOULDER':[],
'LEFTHIP':[],
'RIGHTHIP':[],
'LEFTKNEE':[],
'RIGHTKNEE':[]
}
DIC={'LEFTELBOW':[],
'RIGHTELBOW':[],
'LEFTSHOULDER':[],
'RIGHTSHOULDER':[],
'LEFTHIP':[],
'RIGHTHIP':[],
'LEFTKNEE':[],
'RIGHTKNEE':[]
}


list=[0]
def calculate_angle(a,b,c):
		a = np.array(a) # First
		b = np.array(b) # Mid
		c = np.array(c) # End
		
		radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
		angle = np.abs(radians*180.0/np.pi)
		
		if angle >180.0:
			angle = 360-angle
			
		return angle 



class VideoCamera(object):
	def __init__(self):
		self.video = cv2.VideoCapture(0)

	def __del__(self):
		self.video.release()

	def get_frame(self):
		
		# We are using Motion JPEG, but OpenCV defaults to capture raw images,
		# so we must encode it into JPEG in order to correctly display the
		# video stream.

		

		with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
			while self.video.isOpened():
					ret, frame = self.video.read()
					
					
					# Recolor image to RGB
					image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
					image.flags.writeable = False
				
					# Make detection
					results = pose.process(image)
				
					# Recolor back to BGR
					image.flags.writeable = True
					image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
					
					# Extract landmarks
					try:
						landmarks = results.pose_landmarks.landmark
						
						# Get coordinates
						shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
						elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
						wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
						
						# Calculate angle
						angle = calculate_angle(shoulder, elbow, wrist)
						
						# Visualize angle
						cv2.putText(image, str(angle), 
									tuple(np.multiply(elbow, [640, 480]).astype(int)), 
									cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
											)
								
					except:
						pass
					mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                   			 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                   			 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                  )               
          
					
					# Render detections
					
							
					ret, jpeg = cv2.imencode('.jpg', image)
					return jpeg.tobytes()


