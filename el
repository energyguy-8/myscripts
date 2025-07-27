import pandas as pd
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt


def parse_opt(opt_str):
    parts = opt_str.split()
    notional = int(parts[1])
    tenor = parts[2].upper()  # Normalize tenor to uppercase internally
    strike = float(parts[4])
    hedge_ratio = float(int(parts[5][:-1]) / 100)
    full_delta = notional * int(tenor[:-1]) / 10
    return {
        'side': parts[0],
        'notional': notional,
        'tenor': tenor,
        'type': parts[3],
        'strike': strike,
        'hedge_ratio': hedge_ratio,
        'full_delta': full_delta
    }

def compute_delta(opt, rate):
    sign = 1 if opt['side'] == '+' else -1
    full_delta = opt['full_delta']
    strike = opt['strike']
    typ = opt['type']

    if typ == 'p':
        delta = full_delta if rate > strike else 0
    elif typ == 'r':
        delta = -full_delta if rate < strike else 0
    elif typ == 'str':
        payer_delta = full_delta if rate > strike else 0
        receiver_delta = -full_delta if rate < strike else 0
        delta = payer_delta + receiver_delta
    else:
        delta = 0
    return sign * delta

def get_base_delta(opt_str, rate):
    opt = parse_opt(opt_str)
    full_delta = opt['full_delta']
    hedge_ratio = opt['hedge_ratio']
    opt_type = opt['type']
    sign = 1 if opt['side'] == '+' else -1

    if hedge_ratio == 0:
        return 0
    elif hedge_ratio == 1:
        return compute_delta(opt, rate)
    elif hedge_ratio == 0.5:
        # For payer 'p' half hedge ratio is positive, else negative
        return full_delta * hedge_ratio * sign if opt_type == 'p' else full_delta * hedge_ratio * sign * -1

def delta_change_table(opt_str, base_rate, bp_range=(-10, 10)):
    opt = parse_opt(opt_str)
    base_delta = get_base_delta(opt_str, base_rate * 100)
    results = []
    for bp in range(bp_range[0], bp_range[1] + 1):
        rate = base_rate + bp / 10000
        current_delta = compute_delta(opt, rate * 100)
        delta_change = current_delta - base_delta
        results.append({
            "Shift (bp)": bp,
            "Delta Change (k)": delta_change
        })
    return pd.DataFrame(results)

def multi_option_delta_change_table(opt_str_list, live_rates, bp_range=(-10, 10)):
    # Normalize tenor keys in live_rates to uppercase for consistency
    live_rates_norm = {tenor.upper(): rate for tenor, rate in live_rates.items()}
    
    shifts = list(range(bp_range[0], bp_range[1] + 1))
    combined_df = pd.DataFrame({"Shift (bp)": shifts})

    for opt_str in opt_str_list:
        opt = parse_opt(opt_str)
        tenor = opt['tenor'].upper()
        if tenor not in live_rates_norm:
            raise ValueError(f"Missing base rate for tenor {tenor}")
        base_rate = live_rates_norm[tenor] / 100  # convert percent to decimal
        df = delta_change_table(opt_str, base_rate, bp_range)
        combined_df[opt_str] = df["Delta Change (k)"]

    # Force 0 delta change at 0bp for all options
    combined_df.loc[combined_df["Shift (bp)"] == 0, opt_str_list] = 0.0
    return combined_df.set_index("Shift (bp)")

def sum_delta_changes_by_tenor(df, opt_str_list, live_rates):
    # Normalize tenors to uppercase
    live_tenors = [tenor.upper() for tenor in live_rates.keys()]
    option_to_tenor = {opt: parse_opt(opt)['tenor'].upper() for opt in opt_str_list}

    # Group options by tenor (include all tenors even if no options)
    tenor_groups = {
        tenor: [opt for opt in opt_str_list if option_to_tenor.get(opt) == tenor]
        for tenor in live_tenors
    }

    # Transpose df so options are rows
    transposed_df = df.T

    tenor_delta_dict = {}
    for tenor in live_tenors:
        opts = tenor_groups.get(tenor, [])
        if opts:
            tenor_delta_dict[tenor] = transposed_df.loc[opts].sum(axis=0)
        else:
            # No options for this tenor — zeroes matching columns
            tenor_delta_dict[tenor] = pd.Series(0, index=transposed_df.columns)

    tenor_delta_df = pd.DataFrame(tenor_delta_dict).T  # tenors as rows, shifts as columns
    return tenor_delta_df


def style_negative_red(val):
    return 'color: red' if val < 0 else ''

def parse_option_block(option_block):
    """
    Takes a multi-line string with one option per line and returns a list of cleaned option strings.
    """
    return [line.strip() for line in option_block.strip().splitlines() if line.strip()]



# -------------------------
# Example usage:
live_rates = { 
  "1Y": 1.881, 
  "2Y": 2.031, 
  "5Y": 2.121,
    "10y": 2.341
}

# Option list

# option_text = """
# + 200 1y p 1.89 0%
# + 100 1y r 1.93 100%
# + 400 1y str 1.8805 0%
# - 500 1y p 1.886 50%
# """


option_text = """
+ 200 1y p 1.89 0%
+ 100 1y r 1.93 100%
+ 200 1y str 1.8805 0%
- 500 1y p 1.886 50%

+ 200 2y p 2.035 50%
+ 43 2y r 2.05 100%
+ 70 2y str 2.0 100%
- 90 2y p 1.98 100%

+ 40 5y p 2.15 0%
+ 32 5y r 2.11 0%
- 50 5y str 2.12 0%
+ 90 5y str 2.00 100%
"""

opt_str_list = parse_option_block(option_text)

df = multi_option_delta_change_table(opt_str_list, live_rates)

# ##############################################
# #### OVERRIDE ################################
# df.loc[1, '+ 100 2y r 2.05 100%'] = 20
# ##############################################


dfT = df.T
dfT=dfT.style.applymap(style_negative_red).format("{:,.0f}")

tenor_delta_df = sum_delta_changes_by_tenor(df, opt_str_list, live_rates)





# Create figure with individual tenor lines
fig = go.Figure()

# Plot each tenor line with dotted style
for tenor in tenor_delta_df.index:
    fig.add_trace(go.Scatter(
        x=tenor_delta_df.columns,
        y=tenor_delta_df.loc[tenor],
        mode='lines',
        name=tenor,
        line=dict(dash='dot')  # dotted line
    ))

# Add total delta change line (sum of all tenors)
fig.add_trace(go.Scatter(
    x=tenor_delta_df.columns,
    y=tenor_delta_df.sum(axis=0),
    mode='lines',
    name='Total',
    line=dict(width=2, color='black')  # solid white line
))

# Add horizontal lines every 10k in range
y_min = tenor_delta_df.min().min()
y_max = tenor_delta_df.max().max()
hline_step = 10

for y in range(int(y_min // hline_step * hline_step), int(y_max + hline_step), hline_step):
    fig.add_hline(y=y, line=dict(color='lightgray', width=0.5))

# Customize layout
fig.update_layout(
    title='Delta Change for parallel shift (expbook only)',
    xaxis_title='Rate Shift (bp)',
    yaxis_title='Delta Change (k)',
    template='plotly_white',
    xaxis=dict(dtick=1)  # show every bp tick
)

fig.show()

display(dfT)

tenor_delta_df=tenor_delta_df.style.applymap(style_negative_red).format("{:,.0f}")
# tenor_delta_dfT = tenor_delta_df.T
display(tenor_delta_df)

