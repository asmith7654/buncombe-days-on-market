# --------------------------------------------------
# 0.  SETTINGS  ------------------------------------
# --------------------------------------------------
CSV_FILE       = "1m_plus.csv"
TARGET         = "days on market"            # y‑axis 
TARGET_TITLE   = "DOM"                       # for chart titles
PREDICTOR_CAP  = 6                           # max predictors per figure
FIG_WIDTH      = 14                          # total figure width (inches)
ROW_HEIGHT     = 2.75                        # inches per row

# --------------------------------------------------
# 1.  IMPORTS  -------------------------------------
# --------------------------------------------------
import math
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from matplotlib.backends.backend_pdf import PdfPages

# --------------------------------------------------
# 2.  LOAD + PREP DATA  ----------------------------
# --------------------------------------------------
df = pd.read_csv(CSV_FILE)
# Do not include rows where the middle commute is greater than 45 minutes (bug fix)
df = df[df["middle commute"] < 45]

# --------------------------------------------------
# 3.  PICK NUMERIC PREDICTORS  ---------------------
# --------------------------------------------------
numeric_cols = df.select_dtypes(include="number").columns.tolist()

if TARGET in numeric_cols:
    numeric_cols.remove(TARGET)

# --------------------------------------------------
# 4.  HELPER FUNCTIONS  ----------------------------
# --------------------------------------------------

def fit_line_get_resid(x: pd.Series, y: pd.Series):
    """Return ŷ and residuals of simple OLS (with intercept)."""
    X_ = sm.add_constant(x, has_constant="add")
    model = sm.OLS(y, X_, missing="drop").fit()
    y_hat = model.predict(X_)
    resid = y - y_hat
    return y_hat, resid


def chunks(lst, size):
    """Yield successive *size*-length chunks from *lst*."""
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def plot_predictor_block(pred_list, page_no):
    """
    Plot predictors in a grid.  Each predictor occupies two side‑by‑side plots:
      • left  – scatter + OLS fit line (predictor → TARGET)
      • right – residuals (TARGET − ŷ)
    Layout: 4 columns total (2 cols per predictor  ×  2 predictors per row).
    """
    n_pred = len(pred_list)
    ncols = 4
    nrows = math.ceil(n_pred / 2)
    figsize = (FIG_WIDTH, ROW_HEIGHT * nrows)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharey=False)

    # ensure axes is 2‑D for consistent indexing
    if nrows == 1:
        axes = axes.reshape(1, -1)

    # hide any unused axes (when #preds is odd)
    total_plots = nrows * ncols
    used_plots = n_pred * 2
    for idx in range(used_plots, total_plots):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    for i, col_name in enumerate(pred_list):
        r = i // 2
        c_base = (i % 2) * 2  # 0 or 2

        sub_df = df[[col_name, TARGET]].dropna()

        # Fit OLS model for stats and residuals
        X_ = sm.add_constant(sub_df[col_name])
        model = sm.OLS(sub_df[TARGET], X_).fit()
        r_squared = model.rsquared
        p_value = model.pvalues[col_name]

        # --- scatter + fit line
        ax_scatter = axes[r, c_base]
        sns.regplot(
            data=sub_df,
            x=col_name,
            y=TARGET,
            ax=ax_scatter,
            scatter_kws={"s": 18, "alpha": 0.7},
            line_kws={"color": "red"},
        )
        ax_scatter.set_title(f"{col_name} vs {TARGET_TITLE} ($1m+)")
        ax_scatter.set_xlabel(col_name)
        ax_scatter.set_ylabel(TARGET_TITLE)

        # Add R² and p-value annotation
        textstr = f"$R^2$ = {r_squared:.3f}\n$p$ = {p_value:.3g}"
        ax_scatter.text(
            0.05,
            0.95,
            textstr,
            transform=ax_scatter.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
        )

        # --- residual plot
        ax_resid = axes[r, c_base + 1]
        resid = model.resid
        ax_resid.scatter(sub_df[col_name], resid, s=18, alpha=0.7)
        ax_resid.axhline(0, color="red", lw=1, ls="--")
        ax_resid.set_title(f"Residuals ({col_name})")
        ax_resid.set_xlabel(col_name)
        ax_resid.set_ylabel("y − ŷ")

    fig.suptitle(f"Predictors page {page_no}", y=1.02, fontsize=14)
    fig.tight_layout(pad=1.4)
    return fig

# --------------------------------------------------
# 5.  DRIVER  — SAVE FIGURES LOCALLY  --------------
# --------------------------------------------------

def main():
    out_dir = Path("plots_1m_plus")
    out_dir.mkdir(exist_ok=True)

    pdf_path = out_dir / "all_predictor_plots.pdf"
    with PdfPages(pdf_path) as pdf:
        page = 1
        for block in chunks(numeric_cols, PREDICTOR_CAP):
            fig = plot_predictor_block(block, page_no=page)

            # individual PNG
            png_name = out_dir / f"predictor_page_{page:02d}.png"
            fig.savefig(png_name, dpi=300, bbox_inches="tight")
            print(f"✅  wrote {png_name}")

            # append to PDF
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            page += 1
    print(f"✅  multi‑page PDF saved to {pdf_path}")


if __name__ == "__main__":
    main()





