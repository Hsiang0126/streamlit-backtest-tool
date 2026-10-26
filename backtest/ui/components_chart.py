# ui/components_chart.py 中的 render_parameter_heatmap

def render_parameter_heatmap(
    heatmap_df: pd.DataFrame,
    curr_symbol: str = "NT$",
    x_title: str = "參數 X",
    y_title: str = "參數 Y",
    reverse_y: bool = True
):
    vals = heatmap_df.values.flatten()
    valid_vals = vals[~np.isnan(vals)]
    max_val = float(np.max(valid_vals)) if len(valid_vals) > 0 else 10.0
    min_val = float(np.min(valid_vals)) if len(valid_vals) > 0 else -10.0

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_df.values,
        x=heatmap_df.columns,
        y=heatmap_df.index,
        zmid=0.0,
        zmin=min(min_val, -5.0),
        zmax=max(max_val, 15.0),
        colorscale=[
            [0.0, "#d73a49"],
            [0.4, "#f85149"],
            [0.5, "#21262d"],
            [0.6, "#2ea44f"],
            [1.0, "#3fb950"]
        ],
        colorbar=dict(title="總報酬率 (%)"),
        hovertemplate=f"{x_title}: %{{x}}<br>{y_title}: %{{y}}<br>總報酬率: %{{z:.2f}}%<extra></extra>"
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#161b22",
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(title=x_title, showgrid=False),
        yaxis=dict(title=y_title, showgrid=False, autorange="reversed" if reverse_y else True)
    )
    st.plotly_chart(fig, use_container_width=True)
