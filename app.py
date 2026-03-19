import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad
import streamlit as st
from io import StringIO
from datetime import datetime

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

    .task-card {
        background: #f8fbff;
        border: 1px solid #d8e8ff;
        border-radius: 14px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
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

TASK_CARDS = {
    "任務 1：拋物線面積觀察": {
        "title": "任務 1：拋物線面積觀察",
        "description": "請用 f(x)=x**2、區間 [0,5]，比較四種方法在 n=4 與 n=20 時的誤差變化。",
        "goal": "找出哪一種方法在這個函數下通常較準確。",
    },
    "任務 2：正弦函數比較": {
        "title": "任務 2：正弦函數比較",
        "description": "請用 f(x)=np.sin(x)，觀察在不同 n 下，四種方法的近似值與誤差差異。",
        "goal": "說明為什麼 n 增加後，結果會更接近精確積分值。",
    },
    "任務 3：指數衰減函數": {
        "title": "任務 3：指數衰減函數",
        "description": "請用 f(x)=np.exp(-x)，比較左端點法與右端點法的高估或低估情形。",
        "goal": "判斷函數遞減時，哪一種方法容易偏大或偏小。",
    },
    "任務 4：自訂函數探究": {
        "title": "任務 4：自訂函數探究",
        "description": "請自訂一個函數，例如 sqrt(x+1) 或 x+2，並自行設定區間與 n 進行觀察。",
        "goal": "寫出你對不同近似方法表現的結論。",
    },
}

EXAMPLE_OPTIONS = {
    "x**2": "x**2",
    "x": "x",
    "np.sin(x)": "np.sin(x)",
    "np.cos(x)": "np.cos(x)",
    "np.exp(-x)": "np.exp(-x)",
    "sqrt(x+1)": "sqrt(x+1)",
}

# ----------------------
# Session State 初始化
# ----------------------
if "func_default" not in st.session_state:
    st.session_state["func_default"] = "x**2"
if "records" not in st.session_state:
    st.session_state["records"] = []
if "selected_task" not in st.session_state:
    st.session_state["selected_task"] = list(TASK_CARDS.keys())[0]


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


def make_summary_table(f, a: float, b: float, n: int) -> pd.DataFrame:
    exact, _ = quad(f, a, b)
    rows = []
    for method in methods_dict.keys():
        approx = compute_method_value(f, method, a, b, n)
        rows.append({
            "方法": method_labels[method],
            "近似值": round(approx, 6),
            "精確值": round(float(exact), 6),
            "誤差": round(abs(exact - approx), 6),
        })
    return pd.DataFrame(rows)


# ----------------------
# 標題與主輸入區
# ----------------------
st.title("微積分視覺實驗室：黎曼和教學比較版")
st.markdown("這一版支援大字函數輸入、四種方法同步比較、作答紀錄與全班資料匯出。")

st.subheader("請輸入你要觀察的函數")
st.markdown('<div class="big-note">例如：x**2、x、np.sin(x)、np.exp(-x)、sqrt(x+1)</div>', unsafe_allow_html=True)

func_str = st.text_input("請輸入函數 f(x)", value=st.session_state["func_default"])

quick_cols = st.columns(6)
for i, sample in enumerate(EXAMPLE_OPTIONS.keys()):
    if quick_cols[i].button(sample, use_container_width=True):
        st.session_state["func_default"] = EXAMPLE_OPTIONS[sample]
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

results_df = make_summary_table(f, a, b, n)
focus_row = results_df.loc[results_df["方法"] == method_labels[focus_method]].iloc[0]

# ----------------------
# 學生資料與任務卡
# ----------------------
st.subheader("學生資料與任務任務卡")
student_cols = st.columns(3)
student_name = student_cols[0].text_input("姓名", value="")
student_id = student_cols[1].text_input("學號", value="")
student_class = student_cols[2].text_input("班級", value="")

selected_task = st.selectbox("選擇題目任務卡", list(TASK_CARDS.keys()), index=list(TASK_CARDS.keys()).index(st.session_state["selected_task"]))
st.session_state["selected_task"] = selected_task
current_task = TASK_CARDS[selected_task]
st.markdown(
    f"""
    <div class="task-card">
        <div style="font-size:1.25rem;font-weight:700; margin-bottom:0.4rem;">{current_task['title']}</div>
        <div style="margin-bottom:0.4rem;"><strong>任務說明：</strong>{current_task['description']}</div>
        <div><strong>學習目標：</strong>{current_task['goal']}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------
# 學生作答區
# ----------------------
st.subheader("學生作答區")
answer_cols = st.columns(3)
prediction = answer_cols[0].text_area("預測", height=150, placeholder="請先寫下你預測哪一種方法會比較準，或 n 增加後會發生什麼事。")
observation = answer_cols[1].text_area("觀察", height=150, placeholder="請描述你從圖形與數值表中觀察到的現象。")
conclusion = answer_cols[2].text_area("結論", height=150, placeholder="請寫出你的學習結論。")

# ----------------------
# 四種方法同步比較
# ----------------------
st.subheader("四種方法同時比較")
methods = list(methods_dict.keys())
plot_cols = st.columns(2)

for i, method in enumerate(methods):
    fig, approx, exact, err = draw_single_method(f, func_str, method, a, b, n, color_hex)
    with plot_cols[i % 2]:
        st.pyplot(fig, use_container_width=True)

st.subheader("四種方法數值比較表")
st.dataframe(results_df, use_container_width=True, hide_index=True)

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
# 紀錄區
# ----------------------
st.subheader("操作紀錄管理")
record_cols = st.columns([1.2, 1, 1.2])

if record_cols[0].button("把本次操作存進系統", use_container_width=True):
    if not student_name.strip() or not student_id.strip() or not student_class.strip():
        st.warning("請先完整填寫姓名、學號與班級，再儲存紀錄。")
    else:
        record = {
            "時間": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "姓名": student_name.strip(),
            "學號": student_id.strip(),
            "班級": student_class.strip(),
            "任務卡": selected_task,
            "函數": func_str,
            "區間下限 a": a,
            "區間上限 b": b,
            "分割數 n": n,
            "放大方法": method_labels[focus_method],
            "左端點法誤差": float(results_df.loc[results_df["方法"] == "左端點法", "誤差"].iloc[0]),
            "右端點法誤差": float(results_df.loc[results_df["方法"] == "右端點法", "誤差"].iloc[0]),
            "中點法誤差": float(results_df.loc[results_df["方法"] == "中點法", "誤差"].iloc[0]),
            "梯形法誤差": float(results_df.loc[results_df["方法"] == "梯形法", "誤差"].iloc[0]),
            "目前放大方法近似值": float(approx_focus),
            "精確積分值": float(exact_focus),
            "目前放大方法誤差": float(error_focus),
            "預測": prediction.strip(),
            "觀察": observation.strip(),
            "結論": conclusion.strip(),
        }
        st.session_state["records"].append(record)
        st.success("已成功儲存本次操作紀錄。")

records_df = pd.DataFrame(st.session_state["records"])

if not records_df.empty:
    csv_buffer = StringIO()
    records_df.to_csv(csv_buffer, index=False)
    record_cols[1].download_button(
        label="下載全班操作紀錄 CSV",
        data=csv_buffer.getvalue().encode("utf-8-sig"),
        file_name="riemann_class_records.csv",
        mime="text/csv",
        use_container_width=True,
    )
else:
    record_cols[1].button("下載全班操作紀錄 CSV", disabled=True, use_container_width=True)

if record_cols[2].button("清空目前所有紀錄", use_container_width=True):
    st.session_state["records"] = []
    st.success("目前所有紀錄已清空。")
    st.rerun()

# ----------------------
# 目前比較表下載
# ----------------------
summary_csv = StringIO()
results_df.to_csv(summary_csv, index=False)
st.download_button(
    label="下載目前比較表 CSV",
    data=summary_csv.getvalue().encode("utf-8-sig"),
    file_name="riemann_method_comparison.csv",
    mime="text/csv",
)

# ----------------------
# 顯示已儲存紀錄
# ----------------------
if not records_df.empty:
    st.subheader("已儲存的全班操作紀錄")
    st.dataframe(records_df, use_container_width=True, hide_index=True)

# ----------------------
# 教學提示
# ----------------------
with st.expander("教學提示與觀察方向"):
    st.markdown(
        """
        1. 先輸入 `x**2`，觀察四種方法在相同 n 下的差異。  
        2. 再把 n 從 4 調高到 20，觀察誤差如何變小。  
        3. 嘗試 `np.sin(x)` 或 `np.exp(-x)`，比較不同函數的近似結果。  
        4. 在作答區寫下你的預測、觀察與結論，再把本次操作存進系統。  
        """
    )
