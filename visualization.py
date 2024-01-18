import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import pickle
import numpy as np
def visualize_frames(result):
    # Convert frame keys to numerical values for sorting
    sorted_frames = sorted(result.keys(), key=lambda x: float(x.split('_')[1]))

    for frame_key in sorted_frames:
        frame_data = result[frame_key]

        # Plot the frame image
        frame_image = frame_data['frame_image']
        plt.figure(figsize=(10, 6))
        plt.imshow(cv2.cvtColor(frame_image, cv2.COLOR_BGR2RGB))
        plt.title(f"Frame {frame_key}")
        plt.axis('off')  # Remove axis bars
        for idx, (face_id, face_data) in enumerate(frame_data.items()):
            if face_id == 'frame_image':
                continue
            x0, y0, w, h = face_data['bounding_box']
            rect = patches.Rectangle((x0, y0), w, h, linewidth=2, edgecolor='r', facecolor='none')
            plt.gca().add_patch(rect)
        plt.show()

        # Plot faces in a subplot
        num_faces = len(frame_data) - 1  # Excluding the 'frame_image' key
        fig, axes = plt.subplots(1, num_faces, figsize=(15, 5))

        if num_faces == 1:
            axes = [axes]  # Convert single axes to a list for iteration

        for idx, (face_id, face_data) in enumerate(frame_data.items()):
            if face_id == 'frame_image':
                continue

            # Get face information
            expression = face_data['expression']
            valence = face_data['valence']
            arousal = face_data['arousal']
            face_image = face_data['image']
            landmarks = face_data['landmarks']

            # Draw landmarks on the face image
            for landmark in landmarks:
                cv2.circle(face_image, (int(landmark[0]), int(landmark[1])), 2, (0, 255, 0), -1)  # Green circle

            # Plot face image with landmarks
            axes[idx].imshow(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
            axes[idx].set_title(f"Face {face_id}\nExpression: {expression}\nValence: {valence:.2f}, Arousal: {arousal:.2f}")
            axes[idx].axis('off')  # Remove axis bars

        plt.show()

def calculate_intensity(valence, arousal):
    return np.sqrt(valence**2 + arousal**2)

def generate_annotated_video(video_path, output_path, pickle_path, annotated_frames=None, freeze_factor=10):
    # Load the pickle file
    with open(pickle_path, 'rb') as f:
        annotations = pickle.load(f)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec = cv2.VideoWriter_fourcc(*'mp4v')  # Use MP4V codec for MP4 format

    # Create VideoWriter object to save the output video with audio
    out = cv2.VideoWriter(output_path, codec, fps, (width, height))

    # Get the list of frames to be annotated
    if annotated_frames is None:
        annotated_frames = list(annotations.keys())

    current_annotation_index = 0

    # Loop through each frame in the video
    for frame_index in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        # Read the frame from the original video
        ret, frame = cap.read()
        if not ret:
            break

        # Check if the current frame is annotated
        frame_id = f'frame_{frame_index}'
        if frame_id in annotated_frames:
            # Get annotations for the current frame
            frame_annotations = annotations[frame_id]

            # Calculate average intensity of emotions for all faces
            face_intensities = [calculate_intensity(face['valence'], face['arousal'])
                                for k, face in frame_annotations.items() if k != 'frame_image']

            if face_intensities:
                avg_intensity = np.mean(face_intensities)

                # Draw average intensity on the top of the frame
                cv2.putText(frame, f'Avg Intensity: {avg_intensity:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 2)

            # Loop through each face in the frame and annotate
            for face_key, face_data in frame_annotations.items():
                if 'face' in face_key:
                    # Extract bounding box and emotion information
                    bounding_box = face_data['bounding_box']
                    valence = face_data['valence']
                    arousal = face_data['arousal']
                    expression = face_data['expression']

                    # Calculate intensity of emotions
                    intensity = calculate_intensity(valence, arousal)

                    # Draw bounding box on the face with unique color
                    color = tuple(np.random.randint(0, 255, 3).tolist())
                    x, y, w, h = bounding_box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                    # Label with valence, arousal, and emotion (shortened)
                    label = f'V:{valence:.2f}, A:{arousal:.2f}, I:{intensity:.2f}, E:{expression}'
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Repeat the annotated frame multiple times (freeze effect)
            for _ in range(freeze_factor):
                out.write(frame)
        else:
            # Write the original frame to the output video
            out.write(frame)

    # Release video capture and writer
    cap.release()
    out.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()