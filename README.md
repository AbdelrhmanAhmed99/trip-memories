# trip-memories

This repository contains code for EmoNet Face Analysis. Follow the instructions below to set up and run the code.

## Prerequisites

- Python 3.6 or later
- Git

## Setup

1. **Create a Virtual Environment:**

    ```bash
    python3 -m venv myenv
    source myenv/bin/activate  # On Windows, use .\myenv\Scripts\activate
    pip install -r requirements.txt
    ```

    This will create a virtual environment named `myenv` and install the required packages.

2. **Clone EmoNet Repository:**

    Navigate to the `myenv` directory and run:

    ```bash
    cd myenv
    git clone https://github.com/face-analysis/emonet.git
    ```

3. **Copy Code Files:**

    Copy the `face_processor.py` and `cmd.py` into the `emonet` directory:

    ```bash
    cp path/to/your/faceprocessor.py ./
    cp path/to/your/cmd.py ./
    ```

4. **Run the Code:**

    Navigate to the `emonet` directory:

    Run the `cmd.py`:

    ```bash
    python cmd.py /content/v1.mp4 --fps 1 --n 10
    ```

    This will execute the EmoNet Face Analysis code.

5. **Output:**

    The output will be a pickle file named `v1.pkl` containing the video summary dictionary with the most important frames.

## Notes

- Make sure to activate the virtual environment (`source myenv/bin/activate`) before running the code.
  
## Notebooks
