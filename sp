import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.datasets import load_iris

# Create the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

# Define the layout
app.layout = dbc.Container([
    # Empty row below the title
    dbc.Row(html.Br(), style={"margin-bottom": "50px"}),

    # Title at the top
    dbc.Row(dbc.Col(html.H1("Dashboard Title", className="text-center"))),
    dbc.Row(html.Br(), style={"margin-bottom": "50px"}),

    # Main content row with a margin column on the left
    dbc.Row([
        # Margin column on the left
        dbc.Col(width=2),
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
            ]),
            
            html.Br(),
            
            # Second row of content
            dbc.Row([
                dbc.Col([
                    html.Label("1st Strike"),
                    dcc.Input(id="1st-strike", type="text", value='1')
                ], width=6),
                dbc.Col([
                    html.Label("Slider 1"),
                    dcc.Slider(
                        id='slider1',
                        min=0,
                        max=10,
                        step=0.1,
                        value=5
                    )
                ], width=6)
            ]),
            
            html.Br(),
            
            # Third row of content
            dbc.Row([
                dbc.Col([
                    html.Label("Width"),
                    dcc.Input(id="width", type="text", value='1')
                ], width=6),
                dbc.Col([
                    html.Label("Slider 2"),
                    dcc.Slider(
                        id='slider2',
                        min=0,
                        max=10,
                        step=0.1,
                        value=5
                    )
                ], width=6)
            ]),
            
            html.Br(),
            
            # Fourth row of content
            dbc.Row([
                dbc.Col([
                    html.Label("Risks"),
                    dbc.Table(
                        id="risks",
                        children=[
                            html.Thead(html.Tr([html.Th("Header 1"), html.Th("Header 2")])),
                            html.Tbody([
                                html.Tr([html.Td("Data 1"), html.Td("Data 2")]),
                                html.Tr([html.Td("Data 3"), html.Td("Data 4")]),
                                html.Tr([html.Td("Data 5"), html.Td("Data 6")]),
                                html.Tr([html.Td("Data 7"), html.Td("Data 8")]),
                                html.Tr([html.Td("Data 9"), html.Td("Data 10")])
                            ])
                        ],
                        style={'width': '100%'}
                    )
                ], width=6),
                dbc.Col([
                    html.Label("Payoff"),
                    dcc.Graph(id="payoff")
                ], width=6)
            ]),
            
            html.Br(),
            
            # Fifth row of content
            dbc.Row([
                dbc.Col([
                    html.Label("Legs Details"),
                    dbc.Table(
                        id="legs-details",
                        children=[
                            html.Thead(html.Tr([html.Th("Legs"), html.Th("ATMVol"), html.Th("PV")])),
                            html.Tbody([
                                html.Tr([html.Td("Leg 1"), html.Td("ATMVol 1"), html.Td("PV 1")]),
                                html.Tr([html.Td("Leg 2"), html.Td("ATMVol 2"), html.Td("PV 2")]),
                                html.Tr([html.Td("Leg 3"), html.Td("ATMVol 3"), html.Td("PV 3")])
                            ])
                        ],
                        style={'width': '100%'}
                    )
                ], width=6),
                dbc.Col([
                    html.Label("Historic PnL"),
                    dcc.Graph(id="historic-pnl"),
                    html.Button('Generate Chart', id='generate-chart-btn', n_clicks=0)
                ], width=6)
            ])
        ])
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
        'Feature 1': features[:, feature1],
        'Feature 2': features[:, feature2],
        'Species': labels
    }
    df = pd.DataFrame(data)

    # Create scatter plot
    fig = px.scatter(df, x='Feature 1', y='Feature 2', color='Species', title='Dummy Payoff Chart')

    return fig

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
        return get_structure_payoff_dummy(currency, structure, exp_tenor, p_or_r, long_short, pmt_freq, fp_spot, first_strike, width)
    else:
        # If any input is missing or button is not clicked, return an empty figure
        return go.Figure()

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=8051)

