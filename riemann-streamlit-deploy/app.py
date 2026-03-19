import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad
import streamlit as st
from io import StringIO

# ----------------------
# 頁面設定
# ----------------------
st.set_page_config(
    page_title="微積分視覺實驗室：黎曼和教學版",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------
# 介面樣式：大字輸入框
# ----------------------
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1.2rem;
        max-width: 96rem;
    }

    div[data-testid="stTextInput"] label p {
        font-size: 28px !important;
        font-weight: 700 !important;
    }

    div[data-testid="stTextInput"] input {
        font-size: 30px !important;
        font-weight: 600 !important;
        height: 64px !important;
        padding: 10px 14px !important;
    }

    div[data-testid="stTextInput"] > div > div {
        border-radius: 12px !important;
    }

    .big-note {
        font-size: 1.15rem;
        font-weight: 600;
        color: #333333;
        padding: 0.4rem 0 0.8rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 18,
    "axes.labelsize": 15,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
})

# ----------------------
# 方法定義
# ----------------------
methods_dict = {
    "Left": lambda f, a, b, n: (
        np.linspace(a, b, n, endpoint=False),
        f(np.linspace(a, b, n, endpoint=False)),
        (b - a) / n,
    ),
    "Right": lambda f, a, b, n: (
        np.linspace(a, b, n, endpoint=False) + (b - a) / n,
        f(np.linspace(a, b, n, endpoint=False) + (b - a) / n),
        (b - a) / n,
    ),
    "Midpoint": lambda f, a, b, n: (
        np.linspace(a, b, n, endpoint=False) + (b - a) / (2 * n),
        f(np.linspace(a, b, n, endpoint=False) + (b - a) / (2 * n)),
        (b - a) / n,
    ),
    "Trapezoid": lambda f, a, b, n: (
        np.linspace(a, b, n + 1),
        f(np.linspace(a, b, n + 1)),
        None,
    ),
}

method_labels = {
    "Left": "左端點法",
    "Right": "右端點法",
    "Midpoint": "中點法",
    "Trapezoid": "梯形法",
}

# ----------------------
# 安全函數解析
# ----------------------
def parse_function(func_str: str):
    allowed_names = {
        "np": np,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "exp": np.exp,
        "log": np.log,
        "sqrt": np.sqrt,
        "abs": np.abs,
        "pi": np.pi,
        "e": np.e,
    }

    def f(x):
        return eval(func_str, {"__builtins__": {}}, {"x": x, **allowed_names})

    return f


# ----------------------
# 計算方法近似值
# ----------------------
def compute_method_value(f, method: str, a: float, b: float, n: int) -> float:
    x_vals, y_vals, dx = methods_dict[method](f, a, b, n)
    if method != "Trapezoid":
        return float(np.sum(y_vals * dx))

    total = 0.0
    for i in range(len(x_vals) - 1):
        xi, xi1 = x_vals[i], x_vals[i + 1]
        yi, yi1 = y_vals[i], y_vals[i + 1]
        total += (yi + yi1) / 2 * (xi1 - xi)
    return float(total)


# ----------------------
# 單一方法繪圖
# ----------------------
def draw_single_method(f, func_str: str, method: str, a: float, b: float, n: int, color_hex: str):
    x_plot = np.linspace(a, b, 600)
    y_plot = f(x_plot)
    x_vals, y_vals, dx = methods_dict[method](f, a, b, n)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(x_plot, y_plot, color="blue", linewidth=2.5, label=f"f(x) = {func_str}")

    if method != "Trapezoid":
        riemann_sum = np.sum(y_vals * dx)
        for i in range(len(x_vals)):
            if method == "Left":
                ax.bar(
                    x_vals[i], y_vals[i], width=dx, color=color_hex, alpha=0.55,
                    align="edge", edgecolor="gray", linewidth=1
                )
            elif method == "Right":
                ax.bar(
                    x_vals[i] - dx, y_vals[i], width=dx, color=color_hex, alpha=0.55,
                    align="edge", edgecolor="gray", linewidth=1
                )
            elif method == "Midpoint":
                ax.bar(
                    x_vals[i] - dx / 2, y_vals[i], width=dx, color=color_hex, alpha=0.55,
                    align="edge", edgecolor="gray", linewidth=1
                )
    else:
        riemann_sum = 0.0
        for i in range(len(x_vals) - 1):
            xi, xi1 = x_vals[i], x_vals[i + 1]
            yi, yi1 = y_vals[i], y_vals[i + 1]
            riemann_sum += (yi + yi1) / 2 * (xi1 - xi)
            ax.fill_between([xi, xi1], [yi, yi1], color=color_hex, alpha=0.55)
            ax.plot([xi, xi1], [yi, yi1], color="gray", linewidth=1.2)
            ax.plot([xi, xi], [0, yi], color="gray", linewidth=0.9)
            ax.plot([xi1, xi1], [0, yi1], color="gray", linewidth=0.9)

    exact, _ = quad(f, a, b)
    error = abs(exact - riemann_sum)

    ax.set_title(f"{method_labels[method]}（n = {n}）")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(alpha=0.25)
    ax.axhline(0, color="black", linewidth=1)
    ax.legend(loc="upper left")

    return fig, float(riemann_sum), float(exact), float(error)


# ----------------------
# 誤差曲線圖
# ----------------------
def draw_error_curve(f, a: float, b: float, max_n: int = 30):
    exact, _ = quad(f, a, b)
    n_values = list(range(1, max_n + 1))

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for method in methods_dict.keys():
        errors = []
        for n in n_values:
            approx = compute_method_value(f, method, a, b, n)
            errors.append(abs(exact - approx))
        ax.plot(n_values, errors, linewidth=2, label=method_labels[method])

    ax.set_title("誤差隨分割數 n 的變化")
    ax.set_xlabel("分割數 n")
    ax.set_ylabel("誤差")
    ax.grid(alpha=0.25)
    ax.legend()
    return fig


# ----------------------
# 標題與主輸入區
# ----------------------
st.title("微積分視覺實驗室：黎曼和教學比較版")
st.markdown("這一版支援大字函數輸入、四種方法同步比較，以及誤差隨 n 變化的教學圖。")

st.subheader("請輸入你要觀察的函數")
st.markdown('<div class="big-note">例如：x**2、x、np.sin(x)、np.exp(-x)、sqrt(x+1)</div>', unsafe_allow_html=True)

example_options = {
    "x**2": "x**2",
    "x": "x",
    "np.sin(x)": "np.sin(x)",
    "np.cos(x)": "np.cos(x)",
    "np.exp(-x)": "np.exp(-x)",
    "sqrt(x+1)": "sqrt(x+1)",
}

func_default = st.session_state.get("func_default", "x**2")
func_str = st.text_input("請輸入函數 f(x)", value=func_default)

quick_cols = st.columns(6)
for i, sample in enumerate(example_options.keys()):
    if quick_cols[i].button(sample, use_container_width=True):
        st.session_state["func_default"] = example_options[sample]
        st.rerun()

# ----------------------
# 側邊欄控制
# ----------------------
st.sidebar.header("操作面板")
color_hex = st.sidebar.color_picker("圖形顏色", "#ff6b6b")
a = st.sidebar.number_input("積分下限 a", value=0.0)
b = st.sidebar.number_input("積分上限 b", value=5.0)
n = st.sidebar.slider("目前展示的分割數 n", 1, 80, 6)
focus_method = st.sidebar.selectbox(
    "放大觀察的方法",
    ["Left", "Right", "Midpoint", "Trapezoid"],
    format_func=lambda m: method_labels[m],
)
show_error_curve = st.sidebar.checkbox("顯示誤差隨 n 變化圖", value=True)
error_curve_max_n = st.sidebar.slider("誤差圖的最大 n", 5, 60, 30)

if a >= b:
    st.error("請設定正確區間：必須滿足 a < b。")
    st.stop()

# ----------------------
# 解析函數
# ----------------------
try:
    f = parse_function(func_str)
    _ = f(np.linspace(a, b, 20))
except Exception:
    st.error("函數輸入錯誤。可輸入例如：x**2、np.sin(x)、sqrt(x+1)、exp(-x)")
    st.stop()

# ----------------------
# 四種方法同步比較
# ----------------------
st.subheader("四種方法同時比較")
methods = list(methods_dict.keys())
exact, _ = quad(f, a, b)
results = []
plot_cols = st.columns(2)

for i, method in enumerate(methods):
    fig, approx, _, err = draw_single_method(f, func_str, method, a, b, n, color_hex)
    with plot_cols[i % 2]:
        st.pyplot(fig, use_container_width=True)
    results.append({
        "方法": method_labels[method],
        "近似值": round(approx, 6),
        "精確值": round(float(exact), 6),
        "誤差": round(err, 6),
    })

st.subheader("四種方法數值比較表")
results_df = pd.DataFrame(results)
st.dataframe(results_df, use_container_width=True, hide_index=True)

csv_buffer = StringIO()
results_df.to_csv(csv_buffer, index=False)
st.download_button(
    label="下載目前比較表 CSV",
    data=csv_buffer.getvalue().encode("utf-8-sig"),
    file_name="riemann_method_comparison.csv",
    mime="text/csv",
)

# ----------------------
# 放大觀察區
# ----------------------
st.subheader("單一方法放大觀察")
fig_focus, approx_focus, exact_focus, error_focus = draw_single_method(
    f, func_str, focus_method, a, b, n, color_hex
)
st.pyplot(fig_focus, use_container_width=True)
metric_cols = st.columns(3)
metric_cols[0].metric("近似值", f"{approx_focus:.5f}")
metric_cols[1].metric("精確值", f"{exact_focus:.5f}")
metric_cols[2].metric("誤差", f"{error_focus:.5f}")

# ----------------------
# 誤差隨 n 變化
# ----------------------
if show_error_curve:
    st.subheader("誤差隨 n 變化的小圖")
    error_fig = draw_error_curve(f, a, b, max_n=error_curve_max_n)
    st.pyplot(error_fig, use_container_width=True)
    st.info("觀察重點：當 n 增加時，多數方法的誤差會下降；中點法與梯形法常常比左端點法、右端點法更準。")

# ----------------------
# 教學提示
# ----------------------
with st.expander("教學提示與觀察方向"):
    st.markdown(
        """
        1. 先輸入 `x**2`，觀察四種方法在相同 n 下的差異。  
        2. 再把 n 從 4 調高到 20，觀察誤差如何變小。  
        3. 嘗試 `np.sin(x)` 或 `np.exp(-x)`，比較不同函數的近似結果。  
        4. 比較哪一種方法通常更接近精確積分值。  
        """
    )
