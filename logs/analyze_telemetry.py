import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. Load the data
df = pd.read_csv('gpu_telemetry.csv')

# Clean up column names (stripping whitespace)
df.columns = [c.strip() for c in df.columns]

# 2. Create a figure with a secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# 3. Add GPU Utilization (Left Axis)
fig.add_trace(
    go.Scatter(x=df['timestamp'], y=df['utilization.gpu [%]'], 
               name="GPU Utilization (%)", line=dict(color='royalblue', width=2)),
    secondary_y=False,
)

# 4. Add GPU Temperature (Right Axis)
fig.add_trace(
    go.Scatter(x=df['timestamp'], y=df['temperature.gpu'], 
               name="Temperature (°C)", line=dict(color='firebrick', width=3, dash='dot')),
    secondary_y=True,
)

# 5. Add Memory Usage (Optional - Left Axis)
fig.add_trace(
    go.Scatter(x=df['timestamp'], y=df['memory.used [MiB]'] / 1024, 
               name="Memory Used (GiB)", line=dict(color='forestgreen', width=1)),
    secondary_y=False,
)

# 6. Formatting the layout
fig.update_layout(
    title='<b>A100 Hardware Telemetry: Stress Test Analysis</b>',
    xaxis_title='Time',
    legend=dict(x=0.01, y=0.99),
    template="plotly_dark"
)

fig.update_yaxes(title_text="<b>Utilization / Memory</b>", secondary_y=False)
fig.update_yaxes(title_text="<b>Temperature (°C)</b>", secondary_y=True)

# 7. Save as an interactive HTML file
fig.write_html("results/hardware_analysis.html")
print("Analysis complete! View the results in results/hardware_analysis.html")
