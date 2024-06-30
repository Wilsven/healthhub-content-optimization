# Content Optimization

## Overview <a id="kedro-pipeline"></a>

> This Kedro project with Kedro-Viz setup was generated using `kedro 0.19.6`.

This visualization shows the current (latest) Kedro pipeline. This will be updated as the pipeline progresses.

<img src="docs/images/kedro-pipeline.png" height="1000">

## Rules and Guidelines

- Don't remove any lines from the `.gitignore` file provided (although you may modify or add to it)
- Make sure any results can be reproduced by following a [data engineering convention](https://docs.kedro.org/en/stable/faq/faq.html#what-is-data-engineering-convention)
- Don't commit data to the repository
- Don't commit any credentials or local configuration to the repository. Keep all credentials and local configuration in [`conf/local/`](conf/local)

## Install Dependencies

<a id="note"></a>

> [!NOTE]
> This assumes that you have already created and activated your virtual environment. For more information on how to get set up, refer [here](../README.md#installation). From this section onwards, do also take note that you are in the `content-optimization` directory. Simply check by running `pwd` in your terminal and `cd content-optimization` if you're not already there.

### Anaconda (Recommended)

Declare any dependencies in `requirements.txt` for `pip` installation. To install them, run:

```bash
pip install -r requirements.txt
```

### Poetry

If you're using Poetry instead, run:

```bash
cat requirements.txt | xargs poetry add
```

## File Structure

- [`conf/`](conf): contains all configurations for the project

    * [`base/`](conf/base): contains all configurations for the parameters used in the pipelines

    * [`local/`](conf/local): contains all local configurations for the project like secrets and credentials (not to be checked into version control)

> [!IMPORTANT]
> If you find any discrepancies in the extracted or merged data, please [open an issue](https://github.com/Wilsven/healthhub-content-optimization/issues).

- [`data/`](data): contains all data for the project at every stages; there are many sub-directories but here are the notable ones (will be updated as the pipeline progresses)

    * [`01_raw/all_contents/`](data/01_raw/all_contents): contains all raw data

    * [`02_intermediate/`](data/02_intermediate): contains all intermediate data

        * [`all_contents_standardized/`](data/02_intermediate/all_contents_standardized): contains all standardized data; kept only relevant columns and renamed the columns across all content categories to the same columns names

        * [`all_contents_extracted/`](data/02_intermediate/all_contents_extracted): contains all extracted data; stored in columns named `related_sections`, `extracted_content_body`, `extracted_links` and `extracted_headers`; below is a brief description what each column represents:

            * `related_sections`: related sections from the HTML content body; includes both "Related" as well as "Read these next"
            * `extracted_content_body`: extracted content body from the HTML content body
            * `extracted_links`: extracted links from the HTML content body; for example, links from the "Related" and "Read these next" sections
            * `extracted_headers`: extracted headers from the HTML content body; headers include all `<h>` tags

        * [`all_extracted_text/`](data/02_intermediate/all_extracted_text): contains all the extracted HTML content body; saved as `.txt` files; for validation and sanity checks

    * [`03_primary`](data/03_primary): contains the primary data; all processes (i.e. modeling) after data processing should only ingest the primary data

        * [`merged_data.parquet`](data/03_primary/merged_data.parquet): contains the merged data across all content categories

- [`notebooks/`](notebooks): contains all notebooks for the project; for preliminary and exploratory analysis; code to be refactored into nodes and pipelines

> [!TIP]
> It is a good to do some exploratory work in this directory to understand how the data flows and get transformed through the pipeline. Simply run `catalog.list()` to see all available data and parameters. Simply run `catalog.load("<DATA_NAME | PARAMETER>")` to load the data or parameter. For more information, simply refer to one of the existing notebooks. Happy exploring!

- [`src/content_optimization/`](src/content_optimization): contains all code for the project; contains the code for respective pipelines

## Run the Kedro Project

Similarly, ensure you're in the correct directory. Refer [here](#note) for more information. You can simply run the Kedro project with:

```bash
kedro run
```

This will run the entire project for all pipelines.

## Run Pipelines

### Data Processing

> [!IMPORTANT]
> Before running the `data_processing` [pipeline](src/content_optimization/pipelines/data_processing/pipeline.py)), ensure that you have the raw data in the [`data/01_raw/all_contents`](../content-optimization/data/01_raw/all_contents) directory.

You can run the entire `data_processing` pipeline by running:

```bash
kedro run --pipeline=data_processing
```

If for any reason, you would like to run specific nodes in the `data_processing` pipeline, you can run:

```bash
# Running only the `standardize_columns_node`
kedro run --nodes="standardize_columns_node"
```

The pipeline is a [Directed Acyclic Graph (DAG)](https://en.wikipedia.org/wiki/Directed_acyclic_graph). You can view the visualization [here](#kedro-pipeline). This means that if it's your first time running the pipeline, you should ensure that the nodes are ran in order.

> [!NOTE]
> For example in the `data_processing` pipeline, you should run the `standardize_columns_node` first, followed by the `extract_data_node` then `merge_data_node`. After this, you may run the nodes in any order for subsequent runs. This is because there may be intermediate outputs that are required in subsequent nodes.

### Data Science
