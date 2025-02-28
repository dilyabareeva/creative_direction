import pandas as pd
import plotly.express as px

# ----- Step 1: Read from Excel files -----
# Replace the filenames below with your actual file names.
df1 = pd.read_excel("reports/gemma_2_2b_it_before_labeled.xlsx").dropna(
    subset=["group"]
)
df2 = pd.read_excel(
    "./reports/gemma_2_2b_it_after_affine_alpha_0.2_labeled.xlsx"
).dropna(subset=["group"])
df3 = pd.read_excel("reports/gemma_2_2b_it_after_add_alpha_0.2_labeled.xlsx").dropna(
    subset=["group"]
)

# To avoid column name collisions when merging, rename the "Correct?" column in each dataframe.
df1 = df1.rename(columns={"Correct?": "Correct_df1"})
df2 = df2.rename(columns={"Correct?": "Correct_df2"})
df3 = df3.rename(columns={"Correct?": "Correct_df3"})

# ----- Step 2: Merge dataframes by the "group" column -----
# First, merge df1 and df2 on "group", then merge the result with df3.
merged_df = df1.merge(df2[["group", "Correct_df2"]], on="group", how="inner")
merged_df = merged_df.merge(df3[["group", "Correct_df3"]], on="group", how="inner")


# ----- Step 3: Create horizontal bubble plots for the "Correct?" values in each dataframe -----
def bubble_plot(df, col, title, filename):
    # Get counts per category
    counts = df[col].value_counts().reset_index()
    counts.columns = [col, "Count"]

    # Define custom order: IMPRECISE at bottom, False in the middle, True on top
    order_map = {"IMPRECISE": 0 / 4, "False": 1 / 4, "True": 2 / 4}
    counts["y"] = counts[col].astype(str).map(order_map)

    # Sorting so that bubbles are drawn in order, with True (y=2) drawn last (on top)
    counts = counts.sort_values(by="y")

    # x coordinate is fixed so bubbles align horizontally; y comes from the custom order.
    counts["x"] = 0

    counts["label"] = counts.apply(lambda row: f"{row[col]}\n{row['Count']}", axis=1)

    # Create scatter plot with increased bubble sizes using size_max.
    fig = px.scatter(
        counts,
        x="x",
        y="y",
        size="Count",
        color=col,
        text="label",
        title=title,
        color_discrete_sequence=px.colors.qualitative.Vivid,
        size_max=120,  # Increase bubble sizes
    )

    # Use filled circles, center the text inside, and remove outlines.
    fig.update_traces(
        marker=dict(symbol="circle", opacity=1, line_width=0),
        textposition="middle center",
        textfont=dict(color="white", size=14),
    )

    # Remove all axes, gridlines, and background.
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
    )

    fig.write_image(filename)
    return fig


fig1 = bubble_plot(
    df1, "Correct_df1", "Bubble Plot for df1", "./plots/before_evaluation.svg"
)
fig2 = bubble_plot(
    df2, "Correct_df2", "Bubble Plot for df2", "./plots/affine_evaluation.svg"
)
fig3 = bubble_plot(
    df3, "Correct_df3", "Bubble Plot for df3", "./plots/add_act_evaluation.svg"
)

# ----- Step 4: Calculate transitions for "Correct?" values -----
# For df1 -> df2 transitions:
transition_df1_df2 = pd.crosstab(merged_df["Correct_df1"], merged_df["Correct_df2"])
# For df1 -> df3 transitions:
transition_df1_df3 = pd.crosstab(merged_df["Correct_df1"], merged_df["Correct_df3"])

# Display the combined transitions dataframe.
print("Transition to Affine")
print(transition_df1_df2)

print("Transition to Add")
print(transition_df1_df3)
