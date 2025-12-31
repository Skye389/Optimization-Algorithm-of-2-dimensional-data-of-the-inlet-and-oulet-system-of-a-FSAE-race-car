import pandas as pd
from pyecharts.charts import Line, Page
from pyecharts import options as opts

df = pd.read_excel("Test data1.xlsx")

groups = []
row_count = df.shape[0]
for i in range(0, row_count-1, 2):
    model_torque = pd.to_numeric(df.iloc[i], errors='coerce').tolist()
    software_torque = pd.to_numeric(df.iloc[i+1], errors='coerce').tolist()
    groups.append({
        "model_torque": model_torque,
        "software_torque": software_torque
    })

x_labels = df.columns.tolist()

charts = []
for idx, group in enumerate(groups):
    line = (
        Line()
        .add_xaxis(x_labels)
        .add_yaxis(
            "Model Result",
            group["model_torque"],
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False)  # 关闭点标签
        )
        .add_yaxis(
            "Software Result",
            group["software_torque"],
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False)  # 关闭点标签
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title=f"Group {idx+1}"),
            xaxis_opts=opts.AxisOpts(name="Speed"),
            yaxis_opts=opts.AxisOpts(name="Torque", min_=45, max_=70),
            legend_opts=opts.LegendOpts(pos_top="10%")
        )
    )
    charts.append(line)

page = Page(layout=Page.DraggablePageLayout)
page.add(*charts)
page.render("all_groups.html")
print("All charts have been saved to all_groups.html")

