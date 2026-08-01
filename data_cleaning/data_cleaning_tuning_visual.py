"""
Visualize tuning confidence threshold.
plots: 2 line plots (percentage of data to keep/drop for each confidence threshold).
       2 pictures (100randomly sampled images, to see FP rate).
"""

import pandas as pd
import plotly.express as px
import numpy as np
import wandb


def plot_confidence_thresh_curve(res_tuning_df, keep=True):
    """
    plots a line plot, x axis: confidence threshold, y axis: percentage of data kept/dropped.
    params:
    res_tuning_df (DataFrame): dataframe with columns: Percentage_data_to_keep, and Confidence_level.
    keep (boolean): True if keep, False if drop.
    returns: Plotly figure
    """
    # loading results
    # res_tuning_df = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/Vision-Language-Models/data_cleaning/res_data_cleaning_tuning.csv")
    # res_tuning_df = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/Vision-Language-Models/data/fer2013icml/data_clean/test_res_tuning.csv")

    # adding to df proportion of data dropped
    if not keep:
        # then show drop
        res_tuning_df['Percentage_data_to_drop'] = 100 - res_tuning_df['Percentage_data_to_keep']
    action = 'Keep' if keep else 'Drop'
    col_name = 'Percentage_data_to_keep' if keep else 'Percentage_data_to_drop'

    # round values
    res_tuning_df[col_name] = res_tuning_df[col_name].round(3)
    # add a percentage sign
    res_tuning_df['percentage_sign'] = res_tuning_df[col_name].astype(str) + "%"



    # plot results in line plot - data to keep
    fig = px.line(res_tuning_df, x='Confidence_level', y=col_name, text='percentage_sign',
                  template="plotly_white")

    # values in y axis
    y_vals = list(range(0,101,20))
    y_vals_str = [f"{val}%" for val in y_vals]

    # edit the labels and values
    fig.update_layout(
        title=dict(text=f"Proportion of Data to {action} by Confidence Level",
                   font=dict(size=22, weight="bold")),
        xaxis=dict(title="Confidence Level",
                   title_font=dict(size=18, weight="bold"),
                   tickfont=dict(size=16, weight="bold"),
                   tickmode="array",
                   tickvals=res_tuning_df["Confidence_level"].unique()
        ),
        yaxis=dict(title=action,
                   title_font=dict(size=18, weight="bold"),
                   tickfont=dict(size=16, weight="bold"),
                   tickvals=y_vals,
                   ticktext=y_vals_str
        ),
        title_x=0.5
    )

    # text position
    if keep:
        res_tuning_df["text_position"] = np.where(
            (res_tuning_df[col_name] > 85) | (res_tuning_df[col_name] < 20),
            "top center", "middle right")
        fig.update_traces(
            textposition=res_tuning_df["text_position"],
            textfont=dict(size=16)
        )
    else:
        res_tuning_df["text_position"] = np.where(
            (res_tuning_df[col_name] < 3) | (res_tuning_df[col_name] > 90),
            "top center", "middle left")
        fig.update_traces(
            textposition=res_tuning_df["text_position"],
            textfont=dict(size=16)
        )

    return fig


# wandb.login()
#
#
# wandb.init(project="fer_plus", name="Tuning confidence threshold",
#              notes="using RetinaFace model to filter out non-face images, and defining the confidence threshold for Retinaface",
#              tags=["data cleaning", "retinaface", "fer plus"])
#
# df = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/Vision-Language-Models/data/fer+/retinaface_tuning.csv")
# fig_keep = plot_confidence_thresh_curve(res_tuning_df=df, keep=True)
# fig_drop = plot_confidence_thresh_curve(res_tuning_df=df, keep=False)
# wandb.log({"Data to Keep by Confidence Threshold": fig_keep})
# wandb.log({"Data to Drop by Confidence Threshold": fig_drop})