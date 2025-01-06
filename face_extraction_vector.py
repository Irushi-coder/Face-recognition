import cv2
import mediapipe as mp
import numpy as np

def extract_features(image_path):
    # Initialize Mediapipe Face Mesh module
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        return None

    # Convert the image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect face landmarks
    results = face_mesh.process(rgb_image)
    if not results.multi_face_landmarks:
        print("No faces detected.")
        return None

    # Extract the first face's landmarks
    face_landmarks = results.multi_face_landmarks[0]

    # Convert landmarks to a numpy array (468 landmarks, each with x, y, z)
    feature_vector = np.array(
        [[landmark.x, landmark.y, landmark.z] for landmark in face_landmarks.landmark]
    ).flatten()

    return feature_vector

if __name__ == "__main__":
    # Path to the user's photo
    image_path = "user_photo.jpg"  # Replace with your image file
    feature_vector = extract_features(image_path)
    
    if feature_vector is not None:
        print("Feature Vector:")
        print(feature_vector)
        print("Vector Length:", len(feature_vector))
    else:
        print("Failed to extract feature vector.")
