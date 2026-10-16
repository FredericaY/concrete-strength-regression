import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render_table(rows, columns, widths, path, figsize, center_columns=()):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax.axis("off")
    table = ax.table(
        cellText=rows, colLabels=columns, colWidths=widths,
        cellLoc="right", colLoc="center", bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    for (row, column), cell in table.get_celld().items():
        cell.set_facecolor("white")
        cell.get_text().set_color("black")
        cell.visible_edges = "B"
        cell.set_edgecolor("#cccccc")
        cell.set_linewidth(0.4)
        cell.PAD = 0.06
        if row == 0:
            cell.visible_edges = "BT"
            cell.set_edgecolor("black")
            cell.set_linewidth(0.8)
            cell.get_text().set_fontweight("bold")
        elif row == len(rows):
            cell.set_edgecolor("black")
            cell.set_linewidth(0.8)
        if column == 0:
            cell.get_text().set_ha("left")
        elif column in center_columns:
            cell.get_text().set_ha("center")
    fig.savefig(path, dpi=300)
    plt.close(fig)
