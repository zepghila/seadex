# SeaDex

SeaDex is an open-source project aimed at developing a proof-of-concept tool for identifying marine species in images. The tool uses state-of-the-art image processing and machine learning techniques to segment underwater images, detect objects, and classify them into species.

The live document with the project's rationale, breakdown and brainstorming/research is this Miro board: https://miro.com/app/board/uXjVM5tbPEE=/

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- `Python 3.x`
- `pipenv`

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/seadex.git
```
2. Navigate to the project directory
```bash
cd seadex
```
3. Install project dependencies
```bash
pipenv install
```
4. Download the Segment Anything Model (SAM) from [SAM's github README](https://github.com/facebookresearch/segment-anything#readme). 
Download the `vit_h` model directly by clicking [here](https://github.com/facebookresearch/segment-anything#readme:~:text=or%20vit_h%3A-,ViT%2DH%20SAM%20model.,-vit_l%3A%20ViT).
This is included in `.gitignore`, being a large file (> 2Gb).

5. Run the project: two possible ways to do this, either 4a or 4b

    4a. **Activate the Pipenv shell**: This allows you to work within the project's virtual environment. Any Python scripts you run or packages you install will be confined to this environment.

    ```bash
    pipenv shell
    ```

    Once the shell is activated, you can run Python scripts as you normally would. For example:

    ```bash
    python main.py
    ```

    4b. **Run a specific command**: If you want to run a specific Python script without activating the Pipenv shell, you can use the `pipenv run` command. This is particularly useful for running single commands or for executing scripts in a production environment.

    ```bash
    pipenv run python main.py
    ```

### Usage
Detailed usage instructions will be provided as the project progresses.

### Contributing
Please read CONTRIBUTING.md for details on our code of conduct, and the process for submitting pull requests (TBD).

### License
This project is licensed under the Apache License 2.0, see LICENSE file for details.


