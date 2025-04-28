import dash
from dash import dcc, html
from dash.dependencies import Output, Input
import plotly.graph_objs as go
import datetime
import numpy as np
import pandas as pd

# Load historical data
df = pd.read_csv('TOTALSL.csv', parse_dates=['observation_date'])
df.rename(columns={'observation_date': 'Date', 'TOTALSL': 'Debt'}, inplace=True)

historical_X = list(df['Date'])
historical_Y = list(df['Debt'])

# Initialize simulation
sim_X = [historical_X[-1]]
sim_Y = [historical_Y[-1]]

# Target simulation end date (1 year ahead)
target_date = historical_X[-1] + datetime.timedelta(days=365*10)

# Fixed y-axis range prediction
initial_max_debt = max(historical_Y)
expected_max_simulated_debt = initial_max_debt + 365 * 30  # assume ~30 increase per day for drama
y_axis_lower = min(historical_Y) * 0.95
y_axis_upper = expected_max_simulated_debt * 1.10  # a little extra breathing room

# How many days to simulate per interval
DAYS_PER_UPDATE = 50

app = dash.Dash(__name__)

app.layout = html.Div(style={'backgroundColor': 'lightgray', 'padding': '20px'}, children=[
    html.H1("Simulated Debt Clock", style={'color': 'black'}),
    dcc.Graph(id='live-graph', animate=True),
    dcc.Interval(
        id='interval-component',
        interval=100,  # Update every 0.1 second
        n_intervals=0
    )
])

@app.callback(Output('live-graph', 'figure'),
              Input('interval-component', 'n_intervals'))
def update_graph(n):
    global sim_X, sim_Y

    # Simulate multiple days per interval
    for _ in range(DAYS_PER_UPDATE):
        if sim_X[-1] < target_date:
            new_time = sim_X[-1] + datetime.timedelta(days=1)
            new_value = sim_Y[-1] + np.random.normal(loc=30, scale=15)  # More dramatic daily increase
            sim_X.append(new_time)
            sim_Y.append(new_value)

    fig = go.Figure()

    # Plot historical data
    fig.add_trace(go.Scatter(
        x=historical_X,
        y=historical_Y,
        mode='lines',
        name='Historical',
        line=dict(color='steelblue', width=2)
    ))

    # Plot simulated data
    fig.add_trace(go.Scatter(
        x=sim_X,
        y=sim_Y,
        mode='lines',
        name='Simulated',
        line=dict(color='crimson', width=2)
    ))

    fig.update_layout(
        title='Student Debt Over Time (Historical + Simulated)',
        xaxis_title='Time',
        yaxis_title='Debt (Millions USD)',
        plot_bgcolor='lightgray',
        paper_bgcolor='lightgray',
        font=dict(color='black'),
        margin=dict(l=40, r=40, t=60, b=40),
        height=500,
        xaxis_range=[historical_X[0], target_date],
        yaxis_range=[y_axis_lower, y_axis_upper],
        transition={'duration': 0}  # make updates snappy without transition lag
    )

    return fig


if __name__ == '__main__':
    app.run(debug=True)
