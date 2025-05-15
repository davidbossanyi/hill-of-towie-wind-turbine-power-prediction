# Hill of Towie Wind Turbine Power Prediction
_Repository for the [Hill of Towie Wind Turbine Power Prediction](https://www.kaggle.com/competitions/hill-of-towie-wind-turbine-power-prediction) Kaggle competition_

[![CI](https://github.com/davidbossanyi/hill-of-towie-wind-turbine-power-prediction/actions/workflows/ci.yaml/badge.svg)](https://github.com/davidbossanyi/hill-of-towie-wind-turbine-power-prediction/actions/workflows/ci.yaml)

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
