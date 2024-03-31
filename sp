import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import random
import plotly.express as px
from sklearn.datasets import load_iris

# Create the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

# Define the layout
app.layout = dbc.Container([
    # Title at the top
    dbc.Row(dbc.Col(html.H1("Structure Pricer", className="text-center mb-4"))),

    # Main content row
    dbc.Row([
        # Main content column
        dbc.Col([
            # First row of content
            dbc.Row([
                dbc.Col([
                    html.Label("Currency"),
                    dcc.Dropdown(
                        id="currency",
                        options=[
                            {'label': 'USD', 'value': 'USD'},
                            {'label': 'EUR', 'value': 'EUR'}
                        ],
                        value='USD'
                    )
                ]),
                dbc.Col([
                    html.Label("Structure"),
                    dcc.Dropdown(
                        id="structure",
                        options=[
                            {'label': 'Option 1', 'value': 'opt1'},
                            {'label': 'Option 2', 'value': 'opt2'}
                        ],
                        value='opt1'
                    )
                ]),
                dbc.Col([
                    html.Label("Exp - Tenor"),
                    dcc.Input(id="exp-tenor", type="text", value='1')
                ]),
                dbc.Col([
                    html.Label("P or R"),
                    dcc.Dropdown(
                        id="p-or-r",
                        options=[
                            {'label': 'Option A', 'value': 'optA'},
                            {'label': 'Option B', 'value': 'optB'}
                        ],
                        value='optA'
                    )
                ]),
                dbc.Col([
                    html.Label("Long/Short"),
                    dcc.Dropdown(
                        id="long-short",
                        options=[
                            {'label': 'Long', 'value': 'long'},
                            {'label': 'Short', 'value': 'short'}
                        ],
                        value='long'
                    )
                ]),
                dbc.Col([
                    html.Label("Pmt Freq"),
                    dcc.Dropdown(
                        id="pmt-freq",
                        options=[
                            {'label': 'Monthly', 'value': 'monthly'},
                            {'label': 'Quarterly', 'value': 'quarterly'}
                        ],
                        value='monthly'
                    )
                ]),
                dbc.Col([
                    html.Label("FP/Spot"),
                    dcc.Dropdown(
                        id="fp-spot",
                        options=[
                            {'label': 'FP', 'value': 'fp'},
                            {'label': 'Spot', 'value': 'spot'}
                        ],
                        value='fp'
                    )
                ])
            ], className="mb-4"),
            
            # Second row of content
            dbc.Row([
                dbc.Col([
                    html.Label("1st Strike"),
                    dcc.Input(id="1st-strike", type="number", value=1, debounce=True, className="form-control")
                ], width=3),
                dbc.Col([
                    html.Label("1st Strike"),
                    dcc.Slider(
                        id='slider1',
                        min=0,
                        max=10,
                        step=0.1,
                        value=5,
                        marks={i: str(i) if i % 1 == 0 else '' for i in range(11)},
                        className="w-100"
                    )
                ], width=9)
            ], className="mb-4"),
            
            # Third row of content
            dbc.Row([
                dbc.Col([
                    html.Label("Width"),
                    dcc.Input(id="width", type="number", value=1, debounce=True, className="form-control")
                ], width=3),
                dbc.Col([
                    html.Label("Width"),
                    dcc.Slider(
                        id='slider2',
                        min=0,
                        max=2,
                        step=0.05,
                        value=0.5,
                        marks={i: str(i) if i % 0.1 == 0 else '' for i in range(11)},
                        className="w-100"
                    )
                ], width=9)
            ], className="mb-4"),
            
            # Fourth row of content
            dbc.Row([
                dbc.Col([
                    html.Label("Risks"),
                    dbc.Table(
                        id="risks",
                        children=[
                            html.Thead(html.Tr([html.Th("RiskMetric"), html.Th("Value")])),
                            html.Tbody([
                                html.Tr([html.Td("Delta"), html.Td("Data 2")]),
                                html.Tr([html.Td("Gamma"), html.Td("Data 4")]),
                                html.Tr([html.Td("Vega"), html.Td("Data 6")]),
                                html.Tr([html.Td("Theta"), html.Td("Data 8")]),
                                html.Tr([html.Td("Rho"), html.Td("Data 10")])
                            ])
                        ],
                        style={'width': '100%'}
                    ),
                    
                    html.Label("Legs Details"),
                    dbc.Table(
                        id="legs-details",
                        children=[
                            html.Thead(html.Tr([html.Th("Legs"), html.Th("ATMVol"), html.Th("PV")])),
                            html.Tbody([
                                html.Tr([html.Td("Leg 1"), html.Td("ATMVol 1"), html.Td("PV 1")]),
                                html.Tr([html.Td("Leg 2"), html.Td("ATMVol 2"), html.Td("PV 2")]),
                                html.Tr([html.Td("Leg 3"), html.Td("ATMVol 3"), html.Td("PV 3")]),
                                html.Tr([html.Td("Leg 4"), html.Td("ATMVol 4"), html.Td("PV 4")]),
                            ])
                        ],
                        style={'width': '100%'}
                    )
                ], width=3),
                dbc.Col([
                    html.Label("Payoff"),
                    dcc.Graph(id="payoff", className="w-100"),
                ], width=9)
            ], className="mb-4"),
            
            # Fifth row of content
            dbc.Row([
                dbc.Col([
                    html.Label("Carry"),
                    dbc.Table(
                            id="carry_table",
                            children=[
                                html.Thead(html.Tr([html.Th("Metric"), html.Th("Value")])),
                                html.Tbody([
                                    html.Tr([html.Td("Carry"), html.Td("carry")]),
                                    html.Tr([html.Td("Theta"), html.Td("theta")]),
                                    html.Tr([html.Td("Vol Roll"), html.Td("vol_roll")]),
                                    html.Tr([html.Td("Fwd Roll"), html.Td("fwd_roll")])
                                ])
                            ],
                        style={'width': '100%'}
                    ),
                ]),
                dbc.Col([
                    html.Button('Generate Historic PnL', id='generate-chart-btn', n_clicks=0, className="btn btn-primary mt-3"),
#                     html.Label("  Historic PnL"),
                    dcc.Graph(id="historic-pnl", className="w-100")
                ], width=9)
            ])
        ], className="p-4")
    ])
])

# Define a function to get the plotly figure for the payoff chart
def get_structure_payoff_dummy(currency, structure, exp_tenor, p_or_r, long_short, pmt_freq, fp_spot, first_strike, width):
    # Load the Iris dataset
    iris = load_iris()
    features = iris.data
    labels = iris.target_names[iris.target]

    # Randomly select two features
    feature1 = np.random.randint(0, 4)
    feature2 = np.random.randint(0, 4)
    while feature2 == feature1:
        feature2 = np.random.randint(0, 4)

    # Create a DataFrame with selected features
    data = {
        'Rate Level': features[:, feature1],
        'Payoff': features[:, feature2],
        'Species': labels
    }
    df = pd.DataFrame(data)

    # Create scatter plot
    fig = px.scatter(df, x='Rate Level', y='Payoff', color='Species', title='Structure Payoff')
    fig.update_layout(template='plotly_dark')

    return fig


# Define a function to get the plotly figure for the historic pnl
def get_historic_pnl_dummy(currency, structure, exp_tenor, p_or_r, long_short, pmt_freq, fp_spot, first_strike, width):
    # Load the Iris dataset
    iris = load_iris()
    features = iris.data

    # Create a DataFrame with the features
    feature = str(random.randint(1, 4))
    colname=f"Feature {feature}"
    df = pd.DataFrame(features, columns=['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4'])


    # Create a line chart using Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df[colname], mode='lines', name='Feature 1'))

    # Update layout
    fig.update_layout(title='Historic PnL Dummy Chart', xaxis_title='Time', yaxis_title='PV')
    fig.update_layout(template='plotly_dark')

    return fig



# Define callback to update the input box value based on the slider value
@app.callback(
    Output('1st-strike', 'value'),
    [Input('slider1', 'value')]
)
def update_input_from_slider(slider_value):
    return slider_value

# Define callback to update the slider value based on the input box value
@app.callback(
    Output('slider1', 'value'),
    [Input('1st-strike', 'value')]
)
def update_slider_from_input(input_value):
    return input_value

# Define callback to update the slider marks based on the input box value
@app.callback(
    Output('slider1', 'marks'),
    [Input('1st-strike', 'value')]
)
def update_slider_marks(input_value):
    marks1={i: str(i) if i % 0.1 == 0 else '' for i in range(11)}
    return marks1

# Define callback to update the input box value based on the slider value
@app.callback(
    Output('width', 'value'),
    [Input('slider2', 'value')]
)
def update_input_from_slider(slider_value):
    return slider_value

# Define callback to update the slider value based on the input box value
@app.callback(
    Output('slider2', 'value'),
    [Input('width', 'value')]
)
def update_slider_from_input(input_value):
    return input_value

# Define callback to update the slider marks based on the input box value
@app.callback(
    Output('slider2', 'marks'),
    [Input('width', 'value')]
)
def update_slider_marks(input_value):
    marks = {i/10: str(i/10) if i % 1 == 0 else '' for i in range(21)}
    return marks

# Define callback to update the payoff chart
@app.callback(
    Output('payoff', 'figure'),
    [Input('currency', 'value'),
     Input('structure', 'value'),
     Input('exp-tenor', 'value'),
     Input('p-or-r', 'value'),
     Input('long-short', 'value'),
     Input('pmt-freq', 'value'),
     Input('fp-spot', 'value'),
     Input('1st-strike', 'value'),
     Input('width', 'value')]
)
def update_payoff_chart(currency, structure, exp_tenor, p_or_r, long_short, pmt_freq, fp_spot, first_strike, width):
    if all([currency, structure, exp_tenor, p_or_r, long_short, pmt_freq, fp_spot, first_strike, width]):
        # Call the function to get the payoff chart figure
        return get_structure_payoff_dummy(currency, structure, exp_tenor, p_or_r, long_short, pmt_freq, fp_spot, first_strike, width)
    else:
        # If any input is missing, return an empty figure
        return go.Figure()

# Define callback to update the historic pnl chart on button click
@app.callback(
    Output('historic-pnl', 'figure'),
    [Input('generate-chart-btn', 'n_clicks')],
    [State('currency', 'value'),
     State('structure', 'value'),
     State('exp-tenor', 'value'),
     State('p-or-r', 'value'),
     State('long-short', 'value'),
     State('pmt-freq', 'value'),
     State('fp-spot', 'value'),
     State('1st-strike', 'value'),
     State('width', 'value')]
)
def update_historic_pnl_chart(n_clicks, currency, structure, exp_tenor, p_or_r, long_short, pmt_freq, fp_spot, first_strike, width):
    if n_clicks > 0 and all([currency, structure, exp_tenor, p_or_r, long_short, pmt_freq, fp_spot, first_strike, width]):
        # Call the function to get the historic pnl chart figure
        return get_historic_pnl_dummy(currency, structure, exp_tenor, p_or_r, long_short, pmt_freq, fp_spot, first_strike, width)
    else:
        # If any input is missing or button is not clicked, return an empty figure
        return go.Figure()

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=8051)
