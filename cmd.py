import argparse
import cv2
import os
import pickle
from faceprocessor import FaceProcessor

class CommandLineProcessor:
    def __init__(self, path, fps=1, n=5):
        self.path = path
        self.fps = fps
        self.n = n

    def process_input(self):
        if self.path.endswith(('.jpg', '.jpeg', '.png')):
            # Image path provided
            self.process_image()
        elif self.path.endswith(('.mp4', '.avi', '.mov')):
            # Video path provided
            self.process_video()
        else:
            print("Unsupported file format. Please provide a valid image or video file.")

    def process_image(self):
        # Load the image
        image = cv2.imread(self.path)
        if image is None:
            print("Error loading the image.")
            return

        # Initialize the FaceProcessor and perform emotion analysis
        face_processor = FaceProcessor()
        result = face_processor.emotion_analysis(image)

        # Display or save the results as needed
        if result is not None:
            with open(f"{os.path.splitext(self.path)[0]}.pkl", 'wb') as f:
                pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

    def process_video(self):
        # Process video and generate a summary
        face_processor = FaceProcessor()  # You might want to pass a dummy frame for initialization
        summary_frames = face_processor.video_summary(self.path, fps=self.fps, n=self.n)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emotion Analysis or Video Summary from Image or Video")
    parser.add_argument("file_path", type=str, help="Path to the image or video file")
    parser.add_argument("--fps", type=int, default=1, help="Frames per second (for video)")
    parser.add_argument("--n", type=int, default=5, help="Number of summary frames to generate (for video)")

    args = parser.parse_args()

    # Create a CommandLineProcessor and process the input
    command_line_processor = CommandLineProcessor(args.file_path, fps=args.fps, n=args.n)
    command_line_processor.process_input()
