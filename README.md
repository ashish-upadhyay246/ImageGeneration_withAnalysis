# Image Generation, Analysis, and Segmentation

This project consists of a Flask backend and a Streamlit frontend that work together to generate an image, provide basic and advanced CLIP analysis on it based on the keywords provided by the user and by a local dataset respectively. The image is then segmented into masks and polygons of ROI are extracted and visualized on the streamlit frontend.<br/>
The test cases for core functionalities, endpoints, and errors are provided in the 'flask_app' file.<br/> <br/>
Functionalities implemented: <br/>
(Required) Make a pipeline to generate an image with stable diffusion, provide CLIP analysis on it, and then segment the image into ROI.<br/>
(Advanced) Advanced segementation visualization by plotting all the masks to show segmentation boundaries, plot each polygon on a different image, and provide the coordinates of all the polygons.<br/>

(NOTE: Docker implementation is underway and the repository will be updated once it is completed.)


## Project Structure

- `flask_app.py`: The Flask backend that processes requests generated by the frontend.
- `streamlit_app.py`: The Streamlit frontend that provides a user interface for input and displays results.
- `sd.py`: Generate an image from the prompt and return a base64 string format of the image.
- `clip_basic.py`: Provide basic analysis on the image based on the keywords given by the user.
- `clip_advanced.py`: Provide advanced analysis on the image based on the keywords available in the 'common_objects.txt' file in the 'utility' folder.
- `sam.py`: Generates masks for the given image and extracts polygons, returns the list of masks along with the coordinates of the polygons.
- `modules.py`: Contains functions for plotting the maps and polygons, check if the returned image is black (in case of NSFW content), and to print the clip scores.
- `test_core.py`: pytest test cases to test the core functionalities - image being generated by stable diffusion, image being segmented by SAM, and basic and advanced CLIP analysis.
- `test_endpoints.py`: pytest test cases to test the endpoints '/generate' and '/analyze'.
- `test_err.py`: pytest test cases to test if the errors are being handled as needed.
- `util.py`: a utility script containing code to give a pop up of the generated and the annotated image after mask generation, and to provide a pop up showing all the polygons extracted from the masks.
- `common_objects.py`: contains a list of common objects to provide a dataset for extensive CLIP analysis.

## Prerequisites

- Python 3.12 or above
- The models detect for CUDA capability, if available they utilise the GPU otherwise the CPU is used. <br/>The code was developed using an Nvidia GPU.

## Model Configuration
- Stable diffusion: stable-diffusion-v1-5 https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5
- CLIP model: ViT-B/32
- SAM checkpoint: sam_vit_l_0b3195.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth

## Sample Results

![Screenshot 2024-10-21 024809](https://github.com/user-attachments/assets/825e7c30-0fb1-4c8e-acbb-b156f983ac36)<br/>
![Screenshot 2024-10-21 024855](https://github.com/user-attachments/assets/1f55ca6a-751c-49cf-8b63-f74c00824266)<br/>
![Screenshot 2024-10-21 025050](https://github.com/user-attachments/assets/4d141b11-80aa-4851-ad52-aefc80cb9569)

A few polygons along with their coordinates:<br/>
![Screenshot 2024-10-21 025126](https://github.com/user-attachments/assets/5ae3203a-6bcd-46cb-bb8e-ee8a60d307fa)
![Screenshot 2024-10-21 025137](https://github.com/user-attachments/assets/198dae32-960e-4a95-a452-138333e646aa)
![Screenshot 2024-10-21 025144](https://github.com/user-attachments/assets/f3ee4647-1b1b-4e18-81f2-05f7e2d818a8)

## Testing Output
Below screenshot depicts the successful implementation of the testing suite<br/>
![Screenshot 2024-10-21 024511](https://github.com/user-attachments/assets/3efa38d9-eaa6-4ec6-bba8-2acb6e61904f)

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/ashish-upadhyay246/ImageGeneration_withAnalysis.git
    cd Image_analysis
    ```
    or download it.

2. **Create a virtual environment:**

    Using `venv`

    ```bash
    python -m venv env
    source env/bin/activate  # On Windows, use `env\Scripts\activate`
    ```

    Using `conda`

    ```bash
    conda create --name project_env
    conda activate project_env
    ```

3. **Install the required packages:**

    Navigate to the directory where the requirements.txt file is located and run the below code in the project environment.
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

1. **Setup your SAM checkpoint in the 'SAMcheckpoints' folder:**

    Go to https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints <br/>
    Download a checkpoint of choice and place it inside the 'SAMcheckpoints' folder.<br/>
    Rename the checkpoint file to "model.pth".


2. **Start the Flask backend:**

    Go into the 'flask_app' directory and run the Flask app first.

    ```bash
    python flask_app.py
    ```

    The Flask server will run at `http://localhost:5000`.

3. **Start the Streamlit frontend:**

    Go into the 'streamlit_app' directory and run the Streamlit app.

    ```bash
    streamlit run streamlit_app.py
    ```

    The Streamlit app will open in your default web browser.

4. **For testing the functionalities:**

    Navigate to the 'flask_app' folder and rutn the 3 test scripts by the following code:

    ```bash
    pytest <test_case_filename>.py
    ```
    Note: before running the 'test_core.py' file make the below changes in the 'sd.py' and 'sam.py' files:<br/>
        - sd.py : at line 33, remove "flask_app/" from the file path and save.<br/>
        - sam.py: at line 20, remove "flask_app/" form the file path and save.<br/>
        Then only the 'test_core.py' file will work otherwise it will show an error.
