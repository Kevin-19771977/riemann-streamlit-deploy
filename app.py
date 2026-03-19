import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import streamlit as st
import pandas as pd
from io import StringIO

st.set_page_config(page_title="微積分視覺實驗室：黎曼和學生互動版", layout="wide")

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 20,
    "axes.labelsize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12
})

methods_dict = {
    "左端點法": lambda f, a, b, n: (
        np.linspace(a, b, n, endpoint=False),
        f(np.linspace(a, b, n, endpoint=False)),
        (b - a) / n
    ),
    "右端點法": lambda f, a, b, n: (
        np.linspace(a, b, n, endpoint=False) + (b - a) / n,
        f(np.linspace(a, b, n, endpoint=False) + (b - a) / n),
        (b - a) / n
    ),
    "中點法": lambda f, a, b, n: (
        np.linspace(a, b, n, endpoint=False) + (b - a) / (2 * n),
        f(np.linspace(a, b, n, endpoint=False) + (b - a) / (2 * n)),
        (b - a) / n
    ),
    "梯形法": lambda f, a, b, n: (
        np.linspace(a, b, n + 1),
        f(np.linspace(a, b, n + 1)),
        None
    )
}

TASK_CARDS = {
    "任務 1：比較 x**2 的四種方法": "令 f(x)=x**2，區間 [0,5]，比較四種方法在 n=4、10、30 時的誤差。",
    "任務 2：觀察 sin 函數": "令 f(x)=np.sin(x)，自行調整區間與 n，觀察哪種方法較穩定。",
    "任務 3：指數衰減函數": "令 f(x)=np.exp(-x)，比較中點法與梯形法的誤差。",
    "任務 4：自訂函數探索": "自行輸入函數，記錄當 n 增加時誤差是否變小，並寫出結論。"
}

if "records" not in st.session_state:
    st.session_state.records = []


def parse_function(func_str):
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


def compute_method_value(f, method, a, b, n):
    x_vals, y_vals, dx = methods_dict[method](f, a, b, n)
    if method != "梯形法":
        return np.sum(y_vals * dx)
    total = 0
    for i in range(len(x_vals) - 1):
        xi, xi1 = x_vals[i], x_vals[i + 1]
        yi, yi1 = y_vals[i], y_vals[i + 1]
        total += (yi + yi1) / 2 * (xi1 - xi)
    return total


def draw_single_method(f, func_str, method, a, b, n, color_hex):
    x_plot = np.linspace(a, b, 600)
    y_plot = f(x_plot)
    x_vals, y_vals, dx = methods_dict[method](f, a, b, n)

    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.plot(x_plot, y_plot, color="blue", linewidth=2.8, label=f"f(x) = {func_str}")

    if method != "梯形法":
        riemann_sum = np.sum(y_vals * dx)
        for i in range(len(x_vals)):
            if method == "左端點法":
                ax.bar(x_vals[i], y_vals[i], width=dx, color=color_hex, alpha=0.55,
                       align="edge", edgecolor="gray", linewidth=1)
            elif method == "右端點法":
                ax.bar(x_vals[i] - dx, y_vals[i], width=dx, color=color_hex, alpha=0.55,
                       align="edge", edgecolor="gray", linewidth=1)
            elif method == "中點法":
                ax.bar(x_vals[i] - dx / 2, y_vals[i], width=dx, color=color_hex, alpha=0.55,
                       align="edge", edgecolor="gray", linewidth=1)
    else:
        riemann_sum = 0
        for i in range(len(x_vals) - 1):
            xi, xi1 = x_vals[i], x_vals[i + 1]
            yi, yi1 = y_vals[i], y_vals[i + 1]
            riemann_sum += (yi + yi1) / 2 * (xi1 - xi)
            ax.fill_between([xi, xi1], [yi, yi1], color=color_hex, alpha=0.55)
            ax.plot([xi, xi1], [yi, yi1], color="gray", linewidth=1.2)
            ax.plot([xi, xi], [0, yi], color="gray", linewidth=1.0)
            ax.plot([xi1, xi1], [0, yi1], color="gray", linewidth=1.0)

    exact, _ = quad(f, a, b)
    error = abs(exact - riemann_sum)

    ax.set_title(f"{method}（n = {n}）")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(alpha=0.25)
    ax.axhline(0, color="black", linewidth=1)
    ax.legend()
    return fig, riemann_sum, exact, error


st.title("微積分視覺實驗室：黎曼和學生互動正式版")
st.markdown("可用於課堂操作、學生任務、作答紀錄與 CSV 匯出。")

st.sidebar.header("操作面板")
example = st.sidebar.selectbox(
    "選擇範例函數",
    ["自訂", "x", "x**2", "np.sin(x)", "np.cos(x)", "np.exp(-x)", "sqrt(x+1)"]
)
default_func = "x**2" if example == "自訂" else example
func_str = st.sidebar.text_input("輸入函數 f(x)", value=default_func)
view_mode = st.sidebar.radio("顯示模式", ["單一方法", "四種方法比較"])
method = st.sidebar.selectbox("選擇方法", list(methods_dict.keys()))
n = st.sidebar.slider("分割數 n", 1, 100, 6)
color_hex = st.sidebar.color_picker("圖形顏色", "#ff6b6b")
a = st.sidebar.number_input("積分下限 a", value=0.0)
b = st.sidebar.number_input("積分上限 b", value=5.0)

if a >= b:
    st.error("請設定正確區間：必須滿足 a < b。")
    st.stop()

try:
    f = parse_function(func_str)
    _ = f(np.linspace(a, b, 30))
except Exception:
    st.error("函數輸入錯誤。可輸入例如：x**2、np.sin(x)、sqrt(x+1)、exp(-x)")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["互動圖形", "方法比較", "學生作答區", "教學說明"])

with tab1:
    if view_mode == "單一方法":
        fig, riemann_sum, exact, error = draw_single_method(f, func_str, method, a, b, n, color_hex)
        st.pyplot(fig, use_container_width=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("黎曼和近似值", f"{riemann_sum:.5f}")
        c2.metric("精確積分值", f"{exact:.5f}")
        c3.metric("誤差", f"{error:.5f}")
    else:
        methods = list(methods_dict.keys())
        cols = st.columns(2)
        exact, _ = quad(f, a, b)
        results = []
        for i, m in enumerate(methods):
            fig, approx, _, err = draw_single_method(f, func_str, m, a, b, n, color_hex)
            with cols[i % 2]:
                st.pyplot(fig, use_container_width=True)
            results.append((m, approx, exact, err))
        st.subheader("四種方法數值比較")
        st.dataframe(pd.DataFrame({
            "方法": [r[0] for r in results],
            "近似值": [round(r[1], 5) for r in results],
            "精確值": [round(r[2], 5) for r in results],
            "誤差": [round(r[3], 5) for r in results],
        }), use_container_width=True)
    st.info("觀察重點：當分割數 n 增加時，黎曼和通常會更接近精確積分值。")

with tab2:
    exact, _ = quad(f, a, b)
    method_names = list(methods_dict.keys())
    approx_vals = []
    errors = []
    for m in method_names:
        approx = compute_method_value(f, m, a, b, n)
        approx_vals.append(approx)
        errors.append(abs(exact - approx))
    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots(figsize=(6, 4.5))
        ax1.bar(method_names, approx_vals)
        ax1.axhline(exact, linestyle="--", linewidth=1.5, label="精確值")
        ax1.set_title("各方法近似值比較")
        ax1.set_ylabel("近似值")
        ax1.legend()
        ax1.grid(alpha=0.2)
        st.pyplot(fig1, use_container_width=True)
    with col2:
        fig2, ax2 = plt.subplots(figsize=(6, 4.5))
        ax2.bar(method_names, errors)
        ax2.set_title("各方法誤差比較")
        ax2.set_ylabel("誤差")
        ax2.grid(alpha=0.2)
        st.pyplot(fig2, use_container_width=True)
    st.dataframe(pd.DataFrame({
        "方法": method_names,
        "近似值": [round(v, 5) for v in approx_vals],
        "精確值": [round(exact, 5)] * len(method_names),
        "誤差": [round(e, 5) for e in errors],
    }), use_container_width=True)

with tab3:
    st.subheader("學生資料與任務卡")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        student_name = st.text_input("姓名")
    with col_b:
        student_id = st.text_input("學號")
    with col_c:
        class_name = st.text_input("班級")

    task_title = st.selectbox("選擇任務卡", list(TASK_CARDS.keys()))
    st.info(TASK_CARDS[task_title])

    prediction = st.text_area("作答 1：你預測哪一種方法誤差會比較小？為什麼？", height=100)
    observation = st.text_area("作答 2：你實際觀察到什麼結果？", height=100)
    conclusion = st.text_area("作答 3：請寫下你的結論", height=100)

    approx = compute_method_value(f, method, a, b, n)
    exact, _ = quad(f, a, b)
    error = abs(exact - approx)

    if st.button("儲存本次紀錄"):
        st.session_state.records.append({
            "姓名": student_name,
            "學號": student_id,
            "班級": class_name,
            "任務卡": task_title,
            "函數": func_str,
            "方法": method,
            "a": a,
            "b": b,
            "n": n,
            "近似值": round(float(approx), 8),
            "精確值": round(float(exact), 8),
            "誤差": round(float(error), 8),
            "預測": prediction,
            "觀察": observation,
            "結論": conclusion,
        })
        st.success("已儲存本次紀錄。")

    if st.session_state.records:
        df = pd.DataFrame(st.session_state.records)
        st.subheader("目前紀錄")
        st.dataframe(df, use_container_width=True)

        csv_data = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "下載 CSV 紀錄",
            data=csv_data,
            file_name="riemann_student_records.csv",
            mime="text/csv"
        )

        if st.button("清空所有紀錄"):
            st.session_state.records = []
            st.success("已清空紀錄。請重新整理頁面查看。")

with tab4:
    st.markdown("""
### 1. 什麼是黎曼和？
黎曼和是把曲線下方的面積切成很多小塊，再用長方形或梯形去近似面積的方法。

### 2. 四種常見方法
- **左端點法**：每小區間取左端點高度
- **右端點法**：每小區間取右端點高度
- **中點法**：每小區間取中點高度
- **梯形法**：每小區間用梯形近似

### 3. 建議學生操作任務
1. 先輸入 `x**2`
2. 比較四種方法在 n=4、n=10、n=30 時的誤差
3. 再改成 `np.sin(x)` 或 `np.exp(-x)`
4. 記錄哪一種方法通常較準確
""")
