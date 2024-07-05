![PyPI - Python](https://img.shields.io/badge/python-%3E%3D3.12-blue)
[![License](https://img.shields.io/badge/License-MIT--Clause-green.svg)](https://github.com/Wilsven/healthhub-content-optimization/blob/main/LICENSE)

# Content Optimization for HealthHub

This repository contains the content optimization for HealthHub.

## Install Dependencies <a name="installation"></a>

### Anaconda (Recommended)

You can download the Anaconda Distribution for your respective operating system [here](https://docs.anaconda.com/anaconda/install/). You may also find out how to get started with Anaconda Distribution [here](https://docs.anaconda.com/anaconda/getting-started/). To verfiy your installation, you can head to the Command Line Interface (CLI) and run the following command:

```bash
conda list
```

You should see a list of packages installed in your active environment and their versions displayed. For more information, refer [here](https://docs.anaconda.com/anaconda/install/verify-install/).

---

Once set up, create a virtual environment using `conda` and install dependencies:

```bash
# Create a virtual environment
conda create -n <VENV_NAME> python=3.12 -y
conda activate <VENV_NAME>

# Install dependencies
pip install -r requirements.txt
```

### Poetry

Refer to the documentation [here](https://python-poetry.org/docs/#installing-with-the-official-installer) (recommended) on how to install Poetry based on your operating system.

> [!IMPORTANT]
> **For Mac users**, if encountering issues with `poetry command not found`, add `export PATH="$HOME/.local/bin:$PATH"` in your `.zshrc` file in your home folder and run `source ~/.zshrc`.

---

First create a virtual environment by running the following commands:

```bash
poetry shell
```

> [!TIP]
> If you see the following error; `The currently activated Python version 3.11.7 is not supported by the project (^3.12). Trying to find and use a compatible version.`, run:

```bash
poetry env use 3.12.3  # Python version used in the project
```

To install the defined dependencies for your project, just run the `install` command. The `install` command reads the [`pyproject.toml`](pyproject.toml) file from the current project, resolves the dependencies, and installs them.

```bash
poetry install
```

> [!WARNING]
> If you face an error installing `gensim` with `poetry`, run this command:

```bash
poetry run python -m pip install gensim --disable-pip-version-check --no-deps --no-cache-dir --no-binary gensim
```

If there is a [`poetry.lock`](poetry.lock) file in the current directory, it will use the exact versions from there instead of resolving them. This ensures that everyone using the library will get the same versions of the dependencies.

If there is no [`poetry.lock`](poetry.lock) file, Poetry will create one after dependency resolution.

> [!TIP]
> It is best practice to commit the `poetry.lock` to version control for more reproducible builds. For more information, refer [here](https://python-poetry.org/docs/basic-usage/#:~:text=changes%20in%20dependencies.-,Committing%20your%20poetry.lock%20file%20to%20version%20control,-As%20an%20application).

---

### venv

You can use Python's native virtual environment `venv` to setup the project

```bash
   # Create a virtual environment
   python3 -m venv <VENV_NAME>
```

You can then activate the environment and install the dependencies using the following commands -

For UNIX-based systems (macOS / Linux):
```bash
  # Activate virtual environment
  source <VENV_NAME>/bin/activate

  # Install dependencies
  pip install -r requirements.txt
```

For Windows:
```powershell
  # Activate virtual environment
  .\<VENV_NAME>\Scripts\activate

  # Install dependencies
  pip install -r requirements.txt
```


---

## File Structure

The exploratory/experimental code for content optimization is stored in the [`notebooks/`](notebooks) folder.

- [`artifacts/`](artifacts): contains the output of the exploratory/experimental code

    * [`notebooks/`](artifacts/notebooks): contains experiments generated by [`papermill`](https://papermill.readthedocs.io/en/latest/)

    * [`outputs/`](artifacts/outputs): contains the experiment outputs (i.e. confusion matrices) generated by [`papermill`](https://papermill.readthedocs.io/en/latest/)

        * [`statistical_vector_based_embeddings_similarity_scores.xlsx`](artifacts/outputs/statistical_vector_based_embeddings_similarity_scores.xlsx): contains the similarity scores of experiments generated from Statistical Vector-based Embedding techniques.

- [`content-optimization/`](content-optimization): contains the [Kedro](https://kedro.org/) pipeline for data preprocessing, engineering and clustering

    - For more information about the pipeline, refer to the [`README.md`](content-optimization/README.md).

- [`data/`](data): contains the data used in the exploratory/experimental code

    * [`healthhub_small/`](data/healthhub_small): contains a small subset of Health Hub raw data

    * [`healthhub_small_clean/`](data/healthhub_small_clean): contains the small subset of Health Hub cleaned data; also stores the embeddings generated from Sentence Transformers in a `parquet` format.

- [`notebooks/`](notebooks): contains the exploratory/experimental code where bulk of the logic is implemented

    * [`logger.py`](notebooks/logger.py): contains the code for logging

    * [`preprocess.ipynb`](notebooks/preprocess.ipynb): contains the code for preprocessing the raw data; cleaned output will be stored in [`healthhub_small_clean/`](data/healthhub_small_clean); only needed to run once

    * [`embeddings.ipynb`](notebooks/embeddings.ipynb): contains the code for generating embeddings; embeddings will be stored in [`healthhub_small_clean/`](data/healthhub_small_clean)

    * [`similarity.ipynb`](notebooks/similarity.ipynb): contains the code for calculating similarity between embeddings

    * [`runner.ipynb`](notebooks/runner.ipynb): contains the code for running the notebooks — [`embeddings.ipynb`](notebooks/embeddings.ipynb) and [`similarity.ipynb`](notebooks/similarity.ipynb); parameterize by [`papermill`](https://papermill.readthedocs.io/en/latest/); this notebook helps you run your experiments for different models and pooling strategies and evaluate the results in the `artifacts/` folder

    * [`emb_sim_statistical.ipynb`](notebooks/emb_sim_statistical.ipynb): contains the code for generating embeddings from Statistical Vector-based Embeddings (SVE) techniques and calculating the similarity between embeddings

    * [`runner_statistical.ipynb`](notebooks/runner_statistical.ipynb): contains the code for running the notebook — [`emb_sim_statistical.ipynb`](notebooks/emb_sim_statistical.ipynb); parameterize by [`papermill`](https://papermill.readthedocs.io/en/latest/); this notebook helps you run your experiments for different SVE techniques and similarity metrics and evaluate the results in the `artifacts/` folder

## Usage

To run the notebooks, you can use the [`runner.ipynb`](notebooks/runner.ipynb) or [`runner_statistical.ipynb`](notebooks/runner_statistical.ipynb):

```python
# runner.ipynb

import papermill as pm
from logger import logger

pm.inspect_notebook("<INPUT_NOTEBOOK>")  # inspects and outputs the notebook's parameters

pm.execute_notebook(
    input_path="<INPUT_NOTEBOOK>",  # input notebook path
    output_path="<OUTPUT_NOTEBOOK>",  # output notebook path
    parameters={...},  # parameters to be passed to the notebook in a dictionary
```

## Pushing to GitHub

> [!WARNING]
> Refrain from pushing into `main` branch directly — it is bad practice. Always create a new branch and make your changes on your new branch.

Every time you complete a feature or change on a branch and want to push it to GitHub to make a pull request, you need to ensure you lint your code.

You can simply run the command `pre-commit run --all-files` to lint your code. For more information, refer to the [pre-commit docs](https://pre-commit.com/). To see what linters are used, refer to the [`.pre-commit-config.yaml`](.pre-commit-config.yaml) YAML file.

Alternatively, there is a [`Makefile`](Makefile) that can also lint your code base when you run the simpler command `make lint`.

You should ensure that all cases are satisfied before you push to GitHub (you should see that all has passed). If not, please debug accordingly or your pull request may be rejected and closed.

The [`lint.yml`](.github/workflows/lint.yml) is a GitHub workflow that kicks off several GitHub Actions when a pull request is made. These GitHub Actions check that your code have been properly linted before it is passed for review. Once all actions have passed and the PR approved, your changes will be merged to the `main` branch.

> [!NOTE]
> The `pre-commit` will run regardless if you forget to explicitly call it. Nonetheless, it is recommended to call it explicitly so you can make any necessary changes in advanced.

## Running Kedro from Root Directory

It will be very common for you to run the pipeline every time a new update or feature gets added. This will ensure that you are working with the latest intermediate and primary data at all times.

It can then get quite cumbersome to manually delete every sub-directory containing the intermediate or primary data generated from the pipeline every time there is an update.

Therefore, we have provided a simple `make` command to automatically remove all intemediate and primary data before re-running the latest Kedro pipeline:

> [!TIP]
> To ensure your data directories in your Kedro project are clean, run:
> ```
> make clean
> ```
> If you want to run Kedro from the root directory, you can run the following command:
> ```
> make run
> ```
> This will populate all the new intermediate and primary data in the data directories in your Kedro project.

> [!CAUTION]
> Before running `make clean`, you should review the folders that would be deleted by `make clean`. To do so, run:
> ```
> make clean-dry-run
> ```
