# bird-identification

A bird audio identification tool designed to analyze recordings and generate a list of bird species detected using an AI model. Built with TensorFlow and Python, it operates within a Docker environment. The project is a work in progress and currently in a very preliminary stage.

## Setup

- `git clone`
- `cd bird-identification`
- `mkdir input && mkdir output && mkdir models`
- Place models to the `models` folder: BirdNET and Muuttolintujen kevät
- `docker compose up --build; docker compose down;`

### Usage

- Place audio files to a subfolder of the `input` folder. The script will search files here in this order: ./data, ./Data root.
- Place metadata.yaml file in the subfolder. The file should contain the following fields:
  - `lat`: Latitude in decimal degrees
  - `lon`: Longitude in decimal degrees
  - `day_of_year`: Day of the year, 1-365/6
- Run the script with `python main.py --dir <subfolder> --`
- Optional parameters:
    - `--thr`: Detection threshold as a decimal number between 0...1, default 0.5
    - `--noise`: Include noise in the output, default False
    - `--sdm`: Use species distribution model, default False
    - `--skip`: Skip files that already have a result, default False

## Todo

- Add type hints?
- Get date from file name
- Add baim
- Unit testing?
