# LASY by Oraculumfidelity

(This is a Student project that has no legal right to this name)

## Setup and Installation of packages

#### Creation of a python environment (optional)
```bash
pyenv install 3.12.11
~/.pyenv/versions/3.12.11/bin/python -m venv <venv_name>
```

On Linux / Mac:
```bash
source ./<venv_name>/bin/activate
```
On Windows:
```Powershell
.\<venv_name>\bin\Activate.ps1
```

#### Install the required packages
```bash
pip install -r requirements.txt
```
> Note: This Code was tested on python 3.12.11 - other versions might work.

### Running the code
This application was made to be run as a streamlit app. 
```zsh
cd app
streamlit run Homepage.py
```
This should open a Webbrowser to the Homepage where you can then enjoy our app from.
