# RetinAI: Diabetic Retinopathy Image Screening

This project utilizes Vision Language Models (VLMs) to create an accessible and accurate screening tool for diabetic retinopathy. By analyzing retinal images, the system will assist healthcare providers in underserved areas with early detection of the disease, helping to prevent irreversible vision loss.

***

### Repository Structure
```bash
retinai/
├── data/              # Raw and processed retinal image datasets
├── notebooks/         # Jupyter notebooks for data exploration and prototyping
│   └── setup.ipynb
├── src/               # Source code for data loading, model training, and inference
├── ui/                # Code for the user interface
├── results/           # Early outputs and visualizations
├── docs/              # Project documentation, diagrams, and reports
├── .gitignore         # Specifies files to be ignored by Git
├── README.md          # Project overview and setup instructions
└── requirements.txt   # List of all project dependencies
```
***

### Installation and Setup

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/saririans/RetinAI.git
    cd retinai
    ```

2.  **Create and Activate a Virtual Environment**:

    ```bash
    # Conda
    conda create -n <VENV> python=3.10
    conda activate <VENV>

    # Or venv
    python -m venv <VENV>
    source <VENV>/bin/activate  
    ```

3.  **Install Required Libraries**:
    Once environment is active, install all necessary libraries from `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

***

### Dataset Information
This project uses the APTOS 2019 Blindness Detection dataset from Kaggle along with the Indian Diabetic Retinopathy Image Dataset (IDRiD). In order to download the APTOS data, you must use the Kaggle API and register for the competition.


```bash

pip install kaggle
```

Download the Data:
```bash

kaggle competitions download -c aptos2019-blindness-detection -p data/raw/
```

For the IDRiD dataset, you must download from this website [IDRiD Datasets](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid) with an IEEE account.

* **APTOS 2019 Blindness Detection:**
    * **Source:** A large scale dataset from Kaggle competition.
    * **Size:** 3,662 training images and 1,928 testing images.
    * **Labels:** Each image is assigned a single severity grade from 0-4 for diabetic retinopathy, making it ideal for a standard classification tasks.

* **IDRiD (Indian Diabetic Retinopathy Image Dataset):**
    * **Source:** An expert annotated dataset from a Grand Challenge.
    * **Size:** A smaller dataset with 413 training images.
    * **Labels:** More detailed data with image level grades for diabetic retinopathy, plus detailed annotations for segmenting specific retinal lesions. 

***

### How to Run

After setting up the environment and downloading the data, you can run the exploratory notebook.

1.  **Launch Jupyter Notebook**:
    ```bash
    jupyter notebook
    ```
2.  **Open and Run `setup.ipynb`**:
    In the Jupyter interface that opens in your browser, navigate to the `notebooks/` directory and open the `setup.ipynb` file. You can run the cells to see the data loading and initial analysis.

***

***

### Hipergator SLURM

For computationally intensive tasks, such as training the models on the full dataset, this project is configured to run on Hipergator.

The `slurm/` directory contains batch scripts for submitting jobs via the SLURM workload manager.

#### How to Run on Hipergator

1.  Log in to Hipergator and navigate to the project directory.
2.  Ensure you have created the Conda environment using the **environment.yml** file, NOT THE REQUIREMENT.TXT file.
3.  Load the Conda module and activate your environment:
    ```bash
    module load conda
    conda activate <VENV>
    ```
4.  Submit the desired job using `sbatch`. The scripts are set up to run the corresponding Python files from the `src/` directory.

    ```bash
    # To submit the data setup/preprocessing job
    sbatch slurm/idrid_setup.slurm

    # To submit the main training job
    sbatch slurm/idrid_train.slurm

    # To submit the testing job
    sbatch slurm/idrid_test.slurm
    ```

> **Note:** Before submitting, you may need to edit the `.slurm` files to specify your HiPerGator account information (e.g., `--account=...` or `--mail-user=...`) and adjust any resource requests as needed.
> These requests are for training and testing purposes via the LLAVA model.

***

### User Interface

This project includes an interactive web interface built with Gradio (`interface.py`) for easy, visual screening of retinal images.

#### How to Run the Interface

You can run the interface in an interactive session on HiPerGator.

1.  From the HiPerGator login node, request an interactive GPU session:
    ```bash
    srun --partition=hpg-turin --gpus=l4:1 --mem=32gb --time=02:00:00 --pty bash
    ```
2.  Once you are in the new session, load Conda and activate your environment:
    ```bash
    module load conda
    conda activate retinai_env
    ```
3.  Run the interface script:
    ```bash
    python interface.py
    ```
4.  The app will output a URL to view the interface.

#### Interface Preview
![Interface Preview 1](https://github.com/saririans/RetinAI/blob/main/media/Screenshot%202025-11-09%20at%2019.39.26.png?raw=true)
![Interface Preview2](https://github.com/saririans/RetinAI/blob/main/media/Screenshot%202025-11-09%20at%2019.48.16.png?raw=true)

### Interface Demo
![Interface Demo](https://github.com/saririans/RetinAI/blob/main/media/interfacetest.gif?raw=true)

### Author Information
* **Name**: Maheeka Pandita
* **Contact**: maheekapandita39@gmail.com


