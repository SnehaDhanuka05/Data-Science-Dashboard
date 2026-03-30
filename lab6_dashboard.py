"""
LAB 6 — Interactive Visualization & Dashboard
===============================================
DSE3231 | Manipal University Jaipur

Covers:
  - Plotly Express interactive charts
  - Plotly Dash — multi-tab dashboard
  - Filters, dropdowns, callbacks

Run:
    python lab6_dashboard.py
Then open: http://127.0.0.1:8050
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Try to import Dash (optional for running dashboard) ────
try:
    from dash import Dash, dcc, html, Input, Output, callback
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False
    print("⚠ Dash not installed. Run: pip install dash")
    print("  Plotly charts will still be generated as HTML files.")

import os
os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("data/orders_featured.csv", parse_dates=["Order Date"])
rfm = pd.read_csv("data/rfm.csv")

print("="*55)
print("  LAB 6 — Interactive Visualization & Dashboard")
print("="*55)


# ══════════════════════════════════════════════════════════
# STANDALONE PLOTLY CHARTS (saved as HTML)
# ══════════════════════════════════════════════════════════

# Chart 1 — Animated Sales by Category over Months
monthly_cat = df.groupby(["Order Month", "Category"])["Sales"].sum().reset_index()
fig1 = px.bar(monthly_cat, x="Category", y="Sales", color="Category",
              animation_frame="Order Month",
              title="Monthly Sales by Category (Animated)",
              labels={"Sales": "Total Sales ($)"},
              color_discrete_sequence=px.colors.qualitative.Set2)
fig1.update_layout(showlegend=False)
fig1.write_html("outputs/lab6_animated_bar.html")
print("✅ Saved → outputs/lab6_animated_bar.html")

# Chart 2 — Treemap: Sales hierarchy Region → Category → Sub-Category
fig2 = px.treemap(df, path=["Region", "Category", "Sub-Category"],
                  values="Sales", color="Profit",
                  color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
                  title="Sales Treemap: Region → Category → Sub-Category")
fig2.update_traces(textinfo="label+value+percent parent")
fig2.write_html("outputs/lab6_treemap.html")
print("✅ Saved → outputs/lab6_treemap.html")

# Chart 3 — Scatter: Sales vs Profit with hover details
fig3 = px.scatter(df, x="Sales", y="Profit",
                  color="Category", size="Quantity",
                  hover_data=["Sub-Category", "Region", "Discount"],
                  title="Sales vs Profit (Interactive Scatter)",
                  opacity=0.6,
                  color_discrete_sequence=px.colors.qualitative.Bold)
fig3.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Break-even")
fig3.write_html("outputs/lab6_scatter.html")
print("✅ Saved → outputs/lab6_scatter.html")

# Chart 4 — RFM Segment bubble chart
fig4 = px.scatter(rfm, x="Recency", y="Frequency", size="Monetary",
                  color="Segment", hover_data=["Customer ID", "RFM_Total"],
                  title="RFM Customer Segmentation",
                  size_max=25,
                  color_discrete_sequence=px.colors.qualitative.Pastel)
fig4.write_html("outputs/lab6_rfm_bubble.html")
print("✅ Saved → outputs/lab6_rfm_bubble.html")


# ══════════════════════════════════════════════════════════
# PLOTLY DASH — Multi-Tab Dashboard
# ══════════════════════════════════════════════════════════

if not DASH_AVAILABLE:
    print("\n⚠ Install Dash to run the interactive dashboard:")
    print("    pip install dash")
    exit()

app = Dash(__name__, title="Superstore Analytics Dashboard")

# ── Colour palette ─────────────────────────────────────────
COLORS = {
    "bg":      "#F8F9FA",
    "card":    "#FFFFFF",
    "primary": "#4C72B0",
    "accent":  "#DD8452",
    "text":    "#2C3E50",
}

# ── KPI cards helper ───────────────────────────────────────
def kpi_card(title, value, color="#4C72B0"):
    return html.Div([
        html.P(title, style={"margin": 0, "fontSize": "13px", "color": "#666"}),
        html.H3(value, style={"margin": "4px 0 0", "color": color, "fontSize": "22px"}),
    ], style={
        "backgroundColor": COLORS["card"],
        "padding": "16px 20px",
        "borderRadius": "10px",
        "boxShadow": "0 2px 8px rgba(0,0,0,0.08)",
        "flex": "1",
        "minWidth": "160px",
        "textAlign": "center",
    })


# ── Layout ─────────────────────────────────────────────────
app.layout = html.Div(style={"backgroundColor": COLORS["bg"], "minHeight": "100vh",
                              "fontFamily": "Segoe UI, sans-serif"}, children=[

    # Header
    html.Div([
        html.H1("🛒 Superstore Analytics Dashboard",
                style={"color": COLORS["text"], "margin": 0, "fontSize": "26px"}),
        html.P("DSE3231 — Data Science Lab | Manipal University Jaipur",
               style={"color": "#888", "margin": "4px 0 0"}),
    ], style={"backgroundColor": COLORS["card"], "padding": "20px 32px",
              "borderBottom": "3px solid " + COLORS["primary"],
              "boxShadow": "0 2px 8px rgba(0,0,0,0.06)"}),

    # Global Filters
    html.Div([
        html.Div([
            html.Label("Region", style={"fontWeight": "600", "fontSize": "13px"}),
            dcc.Dropdown(
                id="filter-region",
                options=[{"label": "All Regions", "value": "All"}] +
                        [{"label": r, "value": r} for r in sorted(df["Region"].unique())],
                value="All", clearable=False,
            ),
        ], style={"flex": "1"}),

        html.Div([
            html.Label("Category", style={"fontWeight": "600", "fontSize": "13px"}),
            dcc.Dropdown(
                id="filter-category",
                options=[{"label": "All Categories", "value": "All"}] +
                        [{"label": c, "value": c} for c in sorted(df["Category"].unique())],
                value="All", clearable=False,
            ),
        ], style={"flex": "1"}),

        html.Div([
            html.Label("Segment", style={"fontWeight": "600", "fontSize": "13px"}),
            dcc.Dropdown(
                id="filter-segment",
                options=[{"label": "All Segments", "value": "All"}] +
                        [{"label": s, "value": s} for s in sorted(df["Segment"].unique())],
                value="All", clearable=False,
            ),
        ], style={"flex": "1"}),
    ], style={"display": "flex", "gap": "20px", "padding": "20px 32px",
              "backgroundColor": "#EEF2F7"}),

    # KPI Row (dynamic)
    html.Div(id="kpi-row",
             style={"display": "flex", "gap": "16px", "padding": "0 32px 16px"}),

    # Tabs
    dcc.Tabs(id="tabs", value="sales", children=[
        dcc.Tab(label="📊 Sales Analysis",    value="sales"),
        dcc.Tab(label="💰 Profit Analysis",   value="profit"),
        dcc.Tab(label="👥 Customer Segments", value="customers"),
    ], style={"margin": "0 32px"}),

    html.Div(id="tab-content", style={"padding": "20px 32px"}),
])


# ── Callbacks ──────────────────────────────────────────────

def filter_df(region, category, segment):
    fdf = df.copy()
    if region   != "All": fdf = fdf[fdf["Region"]   == region]
    if category != "All": fdf = fdf[fdf["Category"] == category]
    if segment  != "All": fdf = fdf[fdf["Segment"]  == segment]
    return fdf


@app.callback(
    Output("kpi-row", "children"),
    [Input("filter-region", "value"),
     Input("filter-category", "value"),
     Input("filter-segment", "value")]
)
def update_kpis(region, category, segment):
    fdf = filter_df(region, category, segment)
    total_sales   = fdf["Sales"].sum()
    total_profit  = fdf["Profit"].sum()
    total_orders  = len(fdf)
    avg_margin    = fdf["Profit Margin %"].mean()
    return_rate   = fdf["Returned"].mean() * 100

    return [
        kpi_card("💰 Total Sales",   f"${total_sales:,.0f}",   "#4C72B0"),
        kpi_card("📈 Total Profit",  f"${total_profit:,.0f}",
                 "#55A868" if total_profit >= 0 else "#C44E52"),
        kpi_card("📦 Total Orders",  f"{total_orders:,}",      "#DD8452"),
        kpi_card("📊 Avg Margin",    f"{avg_margin:.1f}%",     "#9B59B6"),
        kpi_card("🔄 Return Rate",   f"{return_rate:.1f}%",    "#E74C3C"),
    ]


@app.callback(
    Output("tab-content", "children"),
    [Input("tabs", "value"),
     Input("filter-region", "value"),
     Input("filter-category", "value"),
     Input("filter-segment", "value")]
)
def render_tab(tab, region, category, segment):
    fdf = filter_df(region, category, segment)

    if tab == "sales":
        monthly = fdf.groupby(["Order Month","Category"])["Sales"].sum().reset_index()
        fig_trend = px.line(monthly, x="Order Month", y="Sales", color="Category",
                            markers=True, title="Monthly Sales by Category",
                            color_discrete_sequence=px.colors.qualitative.Set2)
        fig_trend.update_xaxes(tickvals=list(range(1,13)),
                               ticktext=["Jan","Feb","Mar","Apr","May","Jun",
                                         "Jul","Aug","Sep","Oct","Nov","Dec"])

        sub_sales = fdf.groupby("Sub-Category")["Sales"].sum().sort_values(ascending=False).head(10)
        fig_sub = px.bar(sub_sales, orientation="h", title="Top 10 Sub-Categories by Sales",
                         color=sub_sales.values,
                         color_continuous_scale="Blues",
                         labels={"value": "Sales ($)", "index": "Sub-Category"})
        fig_sub.update_layout(showlegend=False, coloraxis_showscale=False)

        return html.Div([
            html.Div([dcc.Graph(figure=fig_trend)], style={"flex": "1"}),
            html.Div([dcc.Graph(figure=fig_sub)],   style={"flex": "1"}),
        ], style={"display": "flex", "gap": "16px"})

    elif tab == "profit":
        fig_scatter = px.scatter(fdf, x="Sales", y="Profit", color="Category",
                                 size="Quantity", opacity=0.6,
                                 hover_data=["Sub-Category","Region","Discount"],
                                 title="Sales vs Profit (coloured by Category)",
                                 color_discrete_sequence=px.colors.qualitative.Bold)
        fig_scatter.add_hline(y=0, line_dash="dash", line_color="red")

        ship_profit = fdf.groupby("Ship Mode")["Profit"].mean().reset_index()
        fig_ship = px.bar(ship_profit, x="Ship Mode", y="Profit", color="Profit",
                          color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
                          title="Avg Profit by Shipping Mode")

        return html.Div([
            html.Div([dcc.Graph(figure=fig_scatter)], style={"flex": "2"}),
            html.Div([dcc.Graph(figure=fig_ship)],    style={"flex": "1"}),
        ], style={"display": "flex", "gap": "16px"})

    elif tab == "customers":
        fig_rfm = px.scatter(rfm, x="Recency", y="Frequency", size="Monetary",
                             color="Segment",
                             hover_data=["Customer ID","RFM_Total"],
                             title="RFM Customer Segments",
                             size_max=25,
                             color_discrete_sequence=px.colors.qualitative.Pastel)

        seg_counts = rfm["Segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment","Count"]
        fig_pie = px.pie(seg_counts, names="Segment", values="Count",
                         title="Customer Segment Distribution",
                         color_discrete_sequence=px.colors.qualitative.Set3)

        return html.Div([
            html.Div([dcc.Graph(figure=fig_rfm)], style={"flex": "2"}),
            html.Div([dcc.Graph(figure=fig_pie)], style={"flex": "1"}),
        ], style={"display": "flex", "gap": "16px"})


if __name__ == "__main__":
    print("\n🚀 Starting Dash dashboard...")
    print("   Open http://127.0.0.1:8050 in your browser\n")
    app.run(debug=True)
