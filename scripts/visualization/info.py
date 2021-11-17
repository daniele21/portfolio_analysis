from typing import Dict, List
from bokeh.layouts import column
from bokeh.models import TextInput, TableColumn, DataTable, ColumnDataSource


def plot_info(text_inputs: List[Dict]):
    figures = {}

    for text_input in text_inputs:
        title, value = text_input['title'], text_input['value']
        text_fig = TextInput(title=title,
                             value=str(value))
        figures[title] = text_fig

    fig = column(children=list(figures.values()),
                 sizing_mode='stretch_height')  # "stretch_width", "stretch_height", "stretch_both", "scale_width", "scale_height", "scale_both", "fixed"

    return fig


def plot_info_table(text_inputs: Dict):
    source = ColumnDataSource(text_inputs)
    table_cols = [TableColumn(field='info', title="Info"),
                  TableColumn(field='value', title="Value")]
    data_table = DataTable(source=source,
                           columns=table_cols,
                           height=600)
    data_table.autosize_mode = 'fit_viewport'

    return data_table
