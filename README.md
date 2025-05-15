# Hill of Towie Wind Turbine Power Prediction
_Repository for the [Hill of Towie Wind Turbine Power Prediction](https://www.kaggle.com/competitions/hill-of-towie-wind-turbine-power-prediction) Kaggle competition_

[![CI](https://github.com/davidbossanyi/hill-of-towie-wind-turbine-power-prediction/actions/workflows/ci.yaml/badge.svg)](https://github.com/davidbossanyi/hill-of-towie-wind-turbine-power-prediction/actions/workflows/ci.yaml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/downloads/)

## Usage

### Accessing the dataset
To access the dataset, you need to be authenticated with your [Kaggle](https://www.kaggle.com/) account.
1. Create a Kaggle account if you don't have one.
2. Accept the [competition rules](https://www.kaggle.com/competitions/hill-of-towie-wind-turbine-power-prediction/rules) and join the competition.
3. Go to your Kaggle account settings and create a new API token. This will download a file called `kaggle.json` to your computer.
4. Save the `kaggle.json` file to `~/.kaggle/kaggle.json`

The code uses [`kagglehub`](https://github.com/Kaggle/kagglehub) to download and cache the dataset.

## Development
The development environment should be created and managed using [uv](https://docs.astral.sh/uv/). To create the environment:
```commandline
uv sync
```
To run the formatting, linting and testing:
```commandline
uv run poe all
```
Or simply
```commandline
poe all
```
if you have activated the virtual environment (VSCode will do this automatically for you). For example, to activate the environment from a PowerShell prompt:
```powershell
. ".venv\Scripts\activate.ps1"
```
