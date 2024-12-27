# portfolio_analysis/scripts/visualization/panel.py
from typing import Text, Dict
from bokeh.models.widgets import Panel, Tabs
from bokeh.plotting import Figure

def tab_figures(figures_dict: Dict[Text, Figure]):
    panels = []
    for key, fig in figures_dict.items():
        panels.append(Panel(child=fig, title=key))

    tabs = Tabs(tabs=panels)
    return tabs
