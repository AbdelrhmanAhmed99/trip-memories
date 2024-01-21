# Trip Memories

**Overview:**

The **Trip Memories** repository is a novel project that redefines the driving experience by focusing on personalization and anticipation of driver needs. It leverages deep learning models to extract emotions from car trip footage, creating a unique and personalized journey. The project is divided into three main modules: image enhancement and restoration, cabin monitoring, and emotion classification. The image enhancement and restoration module has undergone numerous experiments to reach a robust agnostic model capable of handling multi-task degradations and would be added to pipeline soon. The cabin monitoring module, which uses SPIGA, has been extensively researched and is tested to be relevant to driving scenarios. The emotion classification module uses EmoFan that repersents the core of project's attention.

**Applications:**

The applications of this project extend beyond mere technical advancements. By extracting emotions from car trip footage, other systems could be built upon that can anticipate our needs and tailor the journey to our moods and preferences. It can recommend music playlists that will elevate your mood, fine-tune cabin temperature and lighting for utmost comfort, or propose scenic routes that align with your adventurous spirit. These features, powered by advanced recommender systems and emotion analysis, have the potential to enhance safer, more stress-free drives. The system can also be extended to include other modalities in the addition to visual modality, and car exterior camera footage, which could enhance the output and create deeper memories. This personalization of the driving experience opens up new possibilities for car travel, making each trip a unique and enjoyable journey.


<p align="center">
  <img width="100%" src="Tests/output.gif" />
</p>

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
## Known Limitations

The current version of the system has a few limitations that will be addressed in future work:

1. **Cabin Motoring**: The system can struggle with totally rotated faces for pose estimation and correction. This is a challenge that we plan to tackle in the upcoming iterations.

2. **Image Enhancement**: Until the image enhancement module is added to the pipeline, the system could face issues for parts of the trip footage that are highly degraded. We are aware of this limitation and are working on a solution.

3. **Emotion Classification**: Model results could be further improved.

## Future Work

We have several improvements planned for the future:

1. **SPIGA Training**: We plan to further experiment with training SPIGA on different datasets that are more relevant to the driving scenarios, such as DMD. This could address the bottleneck between the face detection model and SPIGA.

2. **EmoFan Model**: We plan to retrain the EmoFan model after modifying it by replacing its simple attention mechanism with a more complex one on a recently accessed data, AffectNet.

3. **Image Enhancement**: We will continue our experimentation on image enhancement using the DMD database. This will help improve the quality of the footage, especially those parts that are highly degraded.

## Experiments

Explore the following Colab notebooks for detailed experiments:

1. Face Landmark Detection Training Experiment [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1TDVExKz4HvsQbBImZfY8ytCELj20dxpC?usp=sharing)
2. Fine Tuning Stable Diffusion Using ControlNet on DMD Variant Experiment [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1sDW1Flrl2-UbZq4hcRRSBGY2c0Bf0vIT?usp=sharing)
3. DMD Database Variant Creation Experiment [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1m1OCOFcKGCK36dTcqkxlA3Hw0XTrr7vL?usp=sharing)
4. Whole System Experiment [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1oWM5bDXgHzojxywK5Y4Ctzm1qtuJx62o?usp=sharing)

Enjoy reliving your memorable travel experiences with Trip Memories!
