import pandas as pd
import plotly.graph_objects as go # or plotly.express as px
import dash
import datashader as ds
from colorcet import fire
import datashader.transfer_functions as tf
import plotly.express as px

from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import chart_studio.plotly as py
import plotly.graph_objs as go
import numpy as np
import json
import copy
import xarray as xr
from collections import OrderedDict


df2 = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/uber-rides-data1.csv')
dff = df2.query('Lat < 40.82').query('Lat > 40.70').query('Lon > -74.02').query('Lon < -73.91')


cvs2 = ds.Canvas(plot_width=1000, plot_height=1000)
agg = cvs2.points(dff, x='Lon', y='Lat')
# agg is an xarray object, see http://xarray.pydata.org/en/stable/ for more details
coords_lat, coords_lon = agg.coords['Lat'].values, agg.coords['Lon'].values
# Corners of the image, which need to be passed to mapbox
coordinates = [[coords_lon[0], coords_lat[0]],
               [coords_lon[-1], coords_lat[0]],
               [coords_lon[-1], coords_lat[-1]],
               [coords_lon[0], coords_lat[-1]]]


img2 = tf.shade(agg, cmap=fire)[::-1].to_pil()

import plotly.express as px
# Trick to create rapidly a figure with mapbox axes
fig = px.scatter_mapbox(dff[:1], lat='Lat', lon='Lon', zoom=12)
# Add the datashader image as a mapbox layer image
fig.update_layout(mapbox_style="carto-darkmatter",
                 mapbox_layers = [
                {
                    "sourcetype": "image",
                    "source": img2,
                    "coordinates": coordinates
                }]
)


########
########

n = 1000000
max_points = 100000

np.random.seed(2)
cols = ['Signal']  # Column name of signal
start = 1456297053  # Start time
end = start + n  # End time

# Generate a fake signal
time = np.linspace(start, end, n)
signal = np.random.normal(0, 0.3, size=n).cumsum() + 50

# Generate many noisy samples from the signal
noise = lambda var, bias, n: np.random.normal(bias, var, n)
data = {c: signal + noise(1, 10 * (np.random.random() - 0.5), n) for c in cols}

# # Pick a few samples and really blow them out
locs = np.random.choice(n, 10)

# print locs
data['Signal'][locs] *= 2

# # Default plot ranges:
x_range = (start, end)
y_range = (1.2 * signal.min(), 1.2 * signal.max())

# Create a dataframe
data['Time'] = np.linspace(start, end, n)
df = pd.DataFrame(data)

time_start = df['Time'].values[0]
time_end = df['Time'].values[-1]

cvs = ds.Canvas(x_range=x_range, y_range=y_range)

aggs = OrderedDict((c, cvs.line(df, 'Time', c)) for c in cols)
img = tf.shade(aggs['Signal'])

arr = np.array(img)
z = arr.tolist()

# axes
dims = len(z[0]), len(z)

x = np.linspace(x_range[0], x_range[1], dims[0])
y = np.linspace(y_range[0], y_range[1], dims[0])


####
####






external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css', '/assets/style.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


fig1 = {
    'data': [{
        'x': x,
        'y': y,
        'z': z,
        'type': 'heatmap',
        'showscale': False,
        'colorscale': [[0, 'rgba(255, 255, 255,0)'], [1, '#a3a7b0']]}],
    'layout': {
        'margin': {'t': 50, 'b': 20},
        'height': 250,
        'xaxis': {
            'showline': True,
            'zeroline': False,
            'showgrid': False,
            'showticklabels': True,
            'color': '#a3a7b0'
        },
        'yaxis': {
            'fixedrange': True,
            'showline': False,
            'zeroline': False,
            'showgrid': False,
            'showticklabels': False,
            'ticks': '',
            'color': '#a3a7b0'
        },
        'plot_bgcolor': '#23272c',
        'paper_bgcolor': '#23272c'}
}

fig2 = {
    'data': [
        {
            'x': x,
            'y': y,
            'z': z,
            'type': 'heatmap',
            'showscale': False,
            'colorscale': [[0, 'rgba(255, 255, 255,0)'], [1, '#75baf2']]
        }
    ],
    'layout': {
        'margin': {'t': 50, 'b': 20},
        'height': 250,
        'xaxis': {
            'fixedrange': True,
            'showline': True,
            'zeroline': False,
            'showgrid': False,
            'showticklabels': True,
            'color': '#a3a7b0'
        },
        'yaxis': {
            'fixedrange': True,
            'showline': False,
            'zeroline': False,
            'showgrid': False,
            'showticklabels': False,
            'ticks': '',
            'color': '#a3a7b0'
        },
        'plot_bgcolor': '#23272c',
        'paper_bgcolor': '#23272c'}
}


app.layout = html.Div([
    html.Div([
        html.Div([
            html.H1('Crime Density in NYC by Crime', id='header-0'),
            dcc.Graph(figure=fig)
        ], className='twelve columns')
    ], className='row'),

    html.Div([
        html.Div([
            html.P('Click and drag on the plot for high-res view of\
             selected data', id='header-1'),
            dcc.Graph(
                id='graph-1',
                figure=fig1,
                config={
                    'doubleClick': 'reset'
                }
            )
        ], className='twelve columns')
    ], className='row'),

    html.Div([
        html.Div([
            html.Div(
                children=[
                    html.Strong(
                        children=['0'],
                        id='header-2-strong'
                    ),
                    html.P(
                        children=[' points selected'],
                        id='header-2-p'
                    ),
                ],
                id='header-2'
            ),
            dcc.Graph(
                id='graph-2',
                figure=fig2
            )
        ], className='twelve columns')
    ], className='row')
    
       
])

@app.callback(
    [Output('header-2-strong', 'children'),
     Output('header-2-p', 'children')],
    [Input('graph-1', 'relayoutData')]
)
def selectionRange(selection):
    if selection is not None and 'xaxis.range[0]' in selection and \
            'xaxis.range[1]' in selection:
        x0 = selection['xaxis.range[0]']
        x1 = selection['xaxis.range[1]']
        sub_df = df[(df.Time >= x0) & (df.Time <= x1)]
        num_pts = len(sub_df)
        if num_pts < max_points:
            number = "{:,}".format(abs(int(selection['xaxis.range[1]']) - int(selection['xaxis.range[0]'])))
            number_print = " points selected between {0:,.4} and {1:,.4}". \
                format(selection['xaxis.range[0]'], selection['xaxis.range[1]'])
        else:
            number = "{:,}".format(abs(int(selection['xaxis.range[1]']) - int(selection['xaxis.range[0]'])))
            number_print = " points selected. Select less than {0:}k \
            points to invoke high-res scattergl trace".format(max_points / 1000)
    else:
        number = "0"
        number_print = " points selected"
    return [number, number_print]


@app.callback(
    Output('graph-2', 'figure'),
    [Input('graph-1', 'relayoutData')])
def selectionHighlight(selection):
    new_fig2 = fig2.copy()
    if selection is not None and 'xaxis.range[0]' in selection and \
            'xaxis.range[1]' in selection:
        x0 = selection['xaxis.range[0]']
        x1 = selection['xaxis.range[1]']
        sub_df = df[(df.Time >= x0) & (df.Time <= x1)]
        num_pts = len(sub_df)
        if num_pts < max_points:
            shape = dict(
                type='rect',
                xref='x',
                yref='paper',
                y0=0,
                y1=1,
                x0=x0,
                x1=x1,
                line={
                    'width': 0,
                },
                fillcolor='rgba(165, 131, 226, 0.10)'
            )

            new_fig2['layout']['shapes'] = [shape]
        else:
            new_fig2['layout']['shapes'] = []
    else:
        new_fig2['layout']['shapes'] = []
    return new_fig2


@app.callback(
    Output('graph-1', 'figure'),
    [Input('graph-1', 'relayoutData')])
def draw_undecimated_data(selection):
    new_fig1 = fig1.copy()
    if selection is not None and 'xaxis.range[0]' in selection and \
            'xaxis.range[1]' in selection:
        x0 = selection['xaxis.range[0]']
        x1 = selection['xaxis.range[1]']
        sub_df = df[(df.Time >= x0) & (df.Time <= x1)]
        num_pts = len(sub_df)
        if num_pts < max_points:
            high_res_data = [
                dict(
                    x=sub_df['Time'],
                    y=sub_df['Signal'],
                    type='scattergl',
                    marker=dict(
                        sizemin=1,
                        sizemax=30,
                        color='#a3a7b0'
                    )
                )]
            high_res_layout = new_fig1['layout']
            high_res = dict(data=high_res_data, layout=high_res_layout)
        else:
            high_res = fig1.copy()
    else:
        high_res = fig1.copy()
    return high_res

app.run_server(debug=True, use_reloader=False)  # Turn off reloader if inside Jupyter