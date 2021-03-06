# AUTOGENERATED! DO NOT EDIT! File to edit: 35_charts.ipynb (unless otherwise specified).

__all__ = ['plot_examples_chart']

# Cell
import numpy as np, matplotlib.pyplot as plt, wandb
from matplotlib.lines import Line2D

# Cell


# Cell
def plot_examples_chart(split, table, metric):
    spec = "uts_nlp/line_chart_v2"
    fields = {"x": "epoch",'groupKeys': 'idx'}
    fields['y'] = f"{metric}"
    string_fields = dict()
    string_fields['title'] = f"{split}_{metric} vs epoch (examples)"
    chart = wandb.plot_table(vega_spec_name=spec, data_table=table,
                            fields=fields, string_fields=string_fields)
    return chart
