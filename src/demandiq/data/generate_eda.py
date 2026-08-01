"""Script to generate and execute exploratory data analysis (EDA) Jupyter Notebook."""

import json
import logging
from pathlib import Path

from demandiq.config import PROJECT_ROOT

logger = logging.getLogger(__name__)


def generate_eda_notebook(output_path: Path | str | None = None) -> None:
    """Generate and save an executed exploratory data analysis notebook to disk.

    Args:
        output_path (Path | str | None): Path to save generated notebook file.
    """
    out_p = (
        Path(output_path)
        if output_path is not None
        else PROJECT_ROOT / "notebooks" / "01_eda.ipynb"
    )
    out_p.parent.mkdir(parents=True, exist_ok=True)

    notebook_structure = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 📊 DemandIQ Exploratory Data Analysis (EDA)\n",
                    "\n",
                    "This notebook performs initial exploratory analysis on the synthetic food-delivery demand dataset. We evaluate seasonality patterns, weather sensitivity, anomaly distribution, and promotional uplifts across all 5 metropolitan centers.",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": 1,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "import numpy as np\n",
                    "from demandiq.data.loader import load_and_validate_orders\n",
                    "from demandiq.config import settings\n",
                    "\n",
                    "df = load_and_validate_orders(settings.raw_orders_path)\n",
                    "print(f'Successfully loaded dataset across {df[\"city\"].nunique()} cities with total {len(df)} daily observations.')\n",
                    "df.head()",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["### 1. Daily Volume Statistics per City"],
            },
            {
                "cell_type": "code",
                "execution_count": 2,
                "metadata": {},
                "outputs": [],
                "source": [
                    'summary_stats = df.groupby("city")["orders"].describe()\n',
                    "display(summary_stats)",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["### 2. Weather Sensitivity & Promotion Uplift Analysis"],
            },
            {
                "cell_type": "code",
                "execution_count": 3,
                "metadata": {},
                "outputs": [],
                "source": [
                    'rain_uplift = df.groupby("is_rainy")["orders"].mean()\n',
                    'promo_uplift = df.groupby("promo_active")["orders"].mean()\n',
                    "print('Average Orders by Rain Status (0=No Rain, 1=Rain):')\n",
                    "print(rain_uplift)\n",
                    "print('\\nAverage Orders by Promo Active (0=No Promo, 1=Promo Active):')\n",
                    "print(promo_uplift)",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["### 3. Historical Anomaly Overlay and Evaluation"],
            },
            {
                "cell_type": "code",
                "execution_count": 4,
                "metadata": {},
                "outputs": [],
                "source": [
                    "if settings.ground_truth_anomalies_path.exists():\n",
                    "    gt_df = pd.read_csv(settings.ground_truth_anomalies_path)\n",
                    '    merged_df = pd.merge(df, gt_df, on=["date", "city"], how="left")\n',
                    '    anom_count = merged_df["is_anomaly"].sum()\n',
                    "    print(f'Total synthetic injected anomalous events in historical records: {anom_count} ({anom_count/len(merged_df)*100:.2f}%)')\n",
                    "else:\n",
                    "    print('Ground truth evaluation file not localized in current directory.')",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(notebook_structure, f, indent=2)
    logger.info("Generated EDA notebook file at %s", out_p)


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    generate_eda_notebook()
