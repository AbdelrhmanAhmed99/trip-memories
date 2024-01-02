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

    Copy the `face_processor.py` into the `emonet` directory:

    ```bash
    cp /content/myenv/face_processor.py emonet/
    ```

4. **Run the Code:**

    Navigate to the `emonet` directory:

    ```bash
    cd emonet
    ```

    Run the `run_file.py`:

    ```bash
    python run_file.py
    ```

    This will execute the EmoNet Face Analysis code.

## Notes

- Make sure to activate the virtual environment (`source myenv/bin/activate`) before running the code.
