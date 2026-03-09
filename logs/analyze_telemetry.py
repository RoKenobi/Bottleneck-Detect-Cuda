import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# -----------------------------
# 1. Load the data safely
# -----------------------------

base_dir = os.path.dirname(__file__)
file_path = os.path.join(base_dir, "gpu_telemetry.csv")

df = pd.read_csv(file_path, skipinitialspace=True)

# Clean column names
df.columns = [c.strip() for c in df.columns]

# -----------------------------
# 2. Clean telemetry columns
# -----------------------------

for col in df.columns:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace('MiB', '', regex=False)
        .str.replace('%', '', regex=False)
        .str.strip()
    )

# Convert numeric columns safely
df = df.apply(pd.to_numeric, errors='ignore')

# Convert timestamp column
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

# -----------------------------
# 3. Create Plotly figure
# -----------------------------

fig = make_subplots(
    specs=[[{"secondary_y": True}]]
)

# GPU Utilization
fig.add_trace(
    go.Scatter(
        x=df['timestamp'],
        y=df['utilization.gpu [%]'],
        name="GPU Utilization (%)",
        line=dict(color='royalblue', width=2)
    ),
    secondary_y=False,
)

# GPU Temperature
fig.add_trace(
    go.Scatter(
        x=df['timestamp'],
        y=df['temperature.gpu'],
        name="Temperature (°C)",
        line=dict(color='firebrick', width=3, dash='dot')
    ),
    secondary_y=True,
)

# Memory Usage (convert MiB -> GiB)
fig.add_trace(
    go.Scatter(
        x=df['timestamp'],
        y=df['memory.used [MiB]'] / 1024,
        name="Memory Used (GiB)",
        line=dict(color='forestgreen', width=2)
    ),
    secondary_y=False,
)

# -----------------------------
# 4. Layout Formatting
# -----------------------------

fig.update_layout(
    title="<b>A100 Hardware Telemetry: Stress Test Analysis</b>",
    xaxis_title="Time",
    legend=dict(x=0.01, y=0.99),
    template="plotly_dark",
    hovermode="x unified"
)

fig.update_yaxes(
    title_text="<b>Utilization (%) / Memory (GiB)</b>",
    secondary_y=False
)

fig.update_yaxes(
    title_text="<b>Temperature (°C)</b>",
    secondary_y=True
)

# -----------------------------
# 5. Save interactive dashboard
# -----------------------------

output_dir = os.path.join(base_dir, "results")
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir, "hardware_analysis.html")

fig.write_html(output_file)

print(f"Analysis complete! View the results here:\n{output_file}")
