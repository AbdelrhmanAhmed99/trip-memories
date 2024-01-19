# Trip Memories

This repository contains the source code for the trip memories pipeline that takes advantage of deep learning models to obtain emotional moments from your car trip
![Output GIF](Tests/output.gif)


## Colab Notebook Demo
Explore the Colab Notebook Demo for a step-by-step demonstration.[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_PWUEjVedu6RdzzhVNu25RahvcJwRhLi?usp=sharing)

By running the notebook, you will obtain the complete output corresponding to the 20-minute trip video related to the GIF above. This demonstration allows you to experience the entire process and observe the generated results in detail.


## Getting Started

Follow the steps below to set up the project on your local machine:

1. Clone the repository:
    ```bash
    git clone https://github.com/AbdelrhmanAhmed99/trip-memories.git
    cd trip-memories
    ```

2. Initialize submodules:
    ```bash
    git submodule update --init --recursive
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the command-line tool for video:
    ```bash
    python cmd.py /path/to/v1.mp4 --fps 1 --n 10
    ```
    Replace `/path/to/v1.mp4` with the path to your video, adjust the `--fps` parameter, and set the desired number of highlighted frames with `--n`.

   For photos, simply provide the path to the photo:
    ```bash
    python cmd.py /path/to/photo.jpg
    ```
    Replace `/path/to/photo.jpg` with the path to your photo.


## Note

- The paths in the `face_processor.py` file need to be configured if you are running the project locally. The paths are initially configured for a Colab environment.

- The output is a `.pkl` file containing memorable frames. You can visualize these frames in any way you prefer and are not limited to using the `generate_annotated_video` function in the `visualization.py` file.

## Experiments

Explore the following Colab notebooks for detailed experiments:

1. Face Landmark Detection Training Experiment [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1TDVExKz4HvsQbBImZfY8ytCELj20dxpC?usp=sharing)
2. Control Net Fine Tuning Experiment [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1sDW1Flrl2-UbZq4hcRRSBGY2c0Bf0vIT?usp=sharing)
3. DMD Database Variant Creation Experiment [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1m1OCOFcKGCK36dTcqkxlA3Hw0XTrr7vL?usp=sharing)
4. Whole System Experiment [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1oWM5bDXgHzojxywK5Y4Ctzm1qtuJx62o?usp=sharing)

Enjoy reliving your memorable travel experiences with Trip Memories!
