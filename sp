# !pip install dash_bootstrap_components

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
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.VAPOR])

# Define the layout
app.layout = dbc.Container([
    # Title at the top
    dbc.Row([]),
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
                        id="currency_id",
                        options=[
                            {'label': 'GBP', 'value': 'GBP'},
                            {'label': 'EUR', 'value': 'EUR'},
                            {'label': 'USD', 'value': 'USD'},
                        ],
                        value='GBP'
                    )
                ]),
                dbc.Col([
                    html.Label("Exp - Tenor"),
                    dcc.Input(id="exp_tenor_id", type="text", debounce=True, value='1y1y')
                ]),
                dbc.Col([
                    html.Label("Structure"),
                    dcc.Dropdown(
                        id="structure_id",
                        options=[
                            {'label': 'Payer', 'value': 'payer'},
                            {'label': 'Receiver ', 'value': 'receiver'},
                            {'label': 'Payer Spread', 'value': 'p_spread'},
                            {'label': 'Receiver Spread', 'value': 'r_spread'},
                            {'label': 'Payer fly', 'value': 'p_fly'},
                            {'label': 'Receiver fly', 'value': 'r_fly'},
                            {'label': 'Payer Condor', 'value': 'p_condor'},
                            {'label': 'Receiver Condor', 'value': 'r_condor'},
                        ],
                        value='payer'
                    )
                ]),
                
                dbc.Col([
                    html.Label("Payer or Receiver"),
                    dcc.Dropdown(
                        id="p_r_id",
                        options=[
                            {'label': 'Payer', 'value': 'payer'},
                            {'label': 'Receiver', 'value': 'receiver'}
                        ],
                        value='payer'
                    )
                ]),
                
                dbc.Col([
                    html.Label("CapFloor or Swaption"),
                    dcc.Dropdown(
                        id="capfloor_swaption_id",
                        options=[
                            {'label': 'Cap/Floor', 'value': 'cap_floor'},
                            {'label': 'Swaption', 'value': 'swaption'}
                        ],
                        value='swaption'
                    )
                ]),
                
                dbc.Col([
                    html.Label("Long/Short"),
                    dcc.Dropdown(
                        id="long_short_id",
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
                        id="pmt_freq_id",
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
                        id="fp_spot_id",
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
                    dcc.Input(id="1st_strike_id", type="number", value=3, debounce=True, className="form-control")
                ], width=3),
                dbc.Col([
                    html.Label("1st Strike"),
                    dcc.Slider(
                        id='1st_strike_slider_id',
                        min=0,
                        max=10,
                        step=0.25,
                        value=3,
                        marks={i: f"{i}%" if i % 1 == 0 else '' for i in range(11)},
                        tooltip={"placement": "bottom", "always_visible": True},
                        className="w-100"
                    )
                ], width=9)
            ], className="mb-4"),
            
            # Third row of content
            dbc.Row([
                dbc.Col([
                    html.Label("Width"),
                    dcc.Input(id="width_id", type="number", value=1, debounce=True, className="form-control")
                ], width=3),
                dbc.Col([
                    html.Label("Width"),
                    dcc.Slider(
                        id='width_slider_id',
                        min=0,
                        max=2,
                        step=0.05,
                        value=0.5,
                        marks={i: f"{i}%" for i in range(21)},
                        tooltip={"placement": "bottom", "always_visible": True},
                        className="w-100"
                    )
                ], width=9)
            ], className="mb-4"),
            
            # Fourth row of content
            dbc.Row([
                dbc.Col([
                    html.Label("Risks"),
                    dbc.Table(
                        id="risks_table_id",
                        children=[
                            html.Thead(html.Tr([html.Th("RiskMetric"), html.Th("Value")])),
                            html.Tbody([
                                html.Tr([html.Td("Delta"), html.Td("delta", id='delta_id')]),
                                html.Tr([html.Td("Gamma"), html.Td("gamma", id='gamma_id')]),
                                html.Tr([html.Td("Vega"), html.Td("vega", id='vega_id')]),
                                html.Tr([html.Td("Theta"), html.Td("theta", id='theta_id')]),
                                html.Tr([html.Td("Rho"), html.Td("rho", id='rho_id')])
                            ])
                        ],
                        style={'width': '100%'}
                    ),
                    
                    html.Label("Legs Details"),
                    dbc.Table(
                        id="legs_details_table_id",
                        children=[
                            html.Thead(html.Tr([html.Th("Leg"), html.Th("ATMVol"), html.Th("PV")])),
                            html.Tbody([
                                html.Tr([html.Td("Leg 1"), html.Td("ATMVol 1", id='ATMVol_1_id'), html.Td("PV 1", id='PV_1_id')]),
                                html.Tr([html.Td("Leg 2"), html.Td("ATMVol 2", id='ATMVol_2_id'), html.Td("PV 2", id='PV_2_id')]),
                                html.Tr([html.Td("Leg 3"), html.Td("ATMVol 3", id='ATMVol_3_id'), html.Td("PV 3", id='PV_3_id')]),
                                html.Tr([html.Td("Leg 4"), html.Td("ATMVol 4", id='ATMVol_4_id'), html.Td("PV 4", id='PV_4_id')]),
                            ])
                        ],
                        style={'width': '100%'}
                    )
                ], width=3),
                dbc.Col([
                    html.Label("Payoff"),
                    dcc.Graph(id="payoff_graph_id", className="w-100"),
                ], width=9)
            ], className="mb-4"),
            
            # Fifth row of content
            dbc.Row([
                dbc.Col([
                    html.Label("Carry"),
                    dbc.Table(
                            id="carry_table_id",
                            children=[
                                html.Thead(html.Tr([html.Th("Metric"), html.Th("Value")])),
                                html.Tbody([
                                    html.Tr([html.Td("Carry"), html.Td("carry", id='carry_id')]),
                                    html.Tr([html.Td("Theta"), html.Td("theta", id='theta_carry_id')]),
                                    html.Tr([html.Td("Vol Roll"), html.Td("vol_roll", id='vol_roll_id')]),
                                    html.Tr([html.Td("Fwd Roll"), html.Td("fwd_roll", id='fwd_roll_id')])
                                ])
                            ],
                        style={'width': '100%'}
                    ),
                ]),
                dbc.Col([
                    html.Button('Generate Historic PnL', id='generate_chart_btn', n_clicks=0, className="btn btn-primary mt-3"),
#                     html.Label("  Historic PnL"),
                    dcc.Graph(id="historic_pnl_graph_id", className="w-100")
                ], width=9)
            ])
        ], className="p-4")
    ])
], style={"max-width": "1920px"})



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


# Random function to test applying a function with a callback and putting the result into a table cell
def mult(value):
    random_int = random.randint(1, 10)
    return value * random_int



# Define a callback so that Payer or Receiver is payer or receiver if structure is outright
@app.callback(
    Output('p_r_id', 'value'),
    [Input('structure_id', 'value')]
)
def p_or_r(structure):
    if structure=='payer':
        return 'payer'
    elif structure=='receiver':
        return 'receiver'



# Define callback to update the input box value based on the slider value
@app.callback(
    Output('1st_strike_id', 'value'),
    [Input('1st_strike_slider_id', 'value')]
)
def update_input_from_slider(slider_value):
    return slider_value

# Define callback to update the slider value based on the input box value
@app.callback(
    Output('1st_strike_slider_id', 'value'),
    [Input('1st_strike_id', 'value')]
)
def update_slider_from_input(input_value):
    return input_value

# Define callback to update the input box value based on the slider value
@app.callback(
    Output('width_id', 'value'),
    [Input('width_slider_id', 'value')]
)
def update_input_from_slider(slider_value):
    return slider_value

# Define callback to update the slider value based on the input box value
@app.callback(
    Output('width_slider_id', 'value'),
    [Input('width_id', 'value')]
)
def update_slider_from_input(input_value):
    return input_value


# Define callback to update the payoff chart
@app.callback(
    Output('payoff_graph_id', 'figure'),
    [Input('currency_id', 'value'),
     Input('structure_id', 'value'),
     Input('exp_tenor_id', 'value'),
     Input('p_r_id', 'value'),
     Input('long_short_id', 'value'),
     Input('pmt_freq_id', 'value'),
     Input('fp_spot_id', 'value'),
     Input('1st_strike_id', 'value'),
     Input('width_id', 'value')]
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
    Output('historic_pnl_graph_id', 'figure'),
    [Input('generate_chart_btn', 'n_clicks')],
    [State('currency_id', 'value'),
     State('structure_id', 'value'),
     State('exp_tenor_id', 'value'),
     State('p_r_id', 'value'),
     State('long_short_id', 'value'),
     State('pmt_freq_id', 'value'),
     State('fp_spot_id', 'value'),
     State('1st_strike_id', 'value'),
     State('width_id', 'value')]
)
def update_historic_pnl_chart(n_clicks, currency, structure, exp_tenor, p_or_r, long_short, pmt_freq, fp_spot, first_strike, width):
    if n_clicks > 0 and all([currency, structure, exp_tenor, p_or_r, long_short, pmt_freq, fp_spot, first_strike, width]):
        # Call the function to get the historic pnl chart figure
        return get_historic_pnl_dummy(currency, structure, exp_tenor, p_or_r, long_short, pmt_freq, fp_spot, first_strike, width)
    else:
        # If any input is missing or button is not clicked, return an empty figure
        return go.Figure()

# Callback to modify the value of delta when modifying the input first strike
@app.callback(
    Output('delta_id', 'children'),
    [Input('1st_strike_id', 'value')]
)
def update_delta(first_strike):
    return mult(first_strike)
    
    
# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=8051)
