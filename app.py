import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import streamlit as st

# ----------------------
# 頁面設定
# ----------------------
st.set_page_config(
    page_title="微積分視覺實驗室：黎曼和",
    layout="wide"
)

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 20,
    "axes.labelsize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12
})

# ----------------------
# 方法定義
# ----------------------
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
    ),
    "隨機取點法": None
}

# ----------------------
# 安全函數解析
# ----------------------
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
        "e": np.e
    }

    def f(x):
        return eval(func_str, {"__builtins__": {}}, {"x": x, **allowed_names})

    return f

# ----------------------
# 隨機取點法資料
# ----------------------
def random_sample_method(f, a, b, n, seed=None):
    rng = np.random.default_rng(seed)
    dx = (b - a) / n
    left_edges = np.linspace(a, b, n, endpoint=False)
    random_x = left_edges + rng.random(n) * dx
    random_y = f(random_x)
    return left_edges, random_x, random_y, dx

# ----------------------
# 繪圖函數
# ----------------------
def draw_single_method(f, func_str, method, a, b, n, color_hex, seed=None):
    x_plot = np.linspace(a, b, 600)
    y_plot = f(x_plot)

    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.plot(x_plot, y_plot, color="blue", linewidth=2.8, label=f"f(x) = {func_str}")

    if method == "隨機取點法":
        left_edges, random_x, y_vals, dx = random_sample_method(f, a, b, n, seed=seed)
        riemann_sum = np.sum(y_vals * dx)

        for i in range(len(left_edges)):
            ax.bar(
                left_edges[i], y_vals[i], width=dx,
                color=color_hex, alpha=0.55,
                align="edge", edgecolor="gray", linewidth=1
            )
            ax.plot(random_x[i], y_vals[i], "ko", markersize=4)

    elif method != "梯形法":
        x_vals, y_vals, dx = methods_dict[method](f, a, b, n)
        riemann_sum = np.sum(y_vals * dx)

        for i in range(len(x_vals)):
            if method == "左端點法":
                ax.bar(
                    x_vals[i], y_vals[i], width=dx,
                    color=color_hex, alpha=0.55,
                    align="edge", edgecolor="gray", linewidth=1
                )
            elif method == "右端點法":
                ax.bar(
                    x_vals[i] - dx, y_vals[i], width=dx,
                    color=color_hex, alpha=0.55,
                    align="edge", edgecolor="gray", linewidth=1
                )
            elif method == "中點法":
                ax.bar(
                    x_vals[i] - dx / 2, y_vals[i], width=dx,
                    color=color_hex, alpha=0.55,
                    align="edge", edgecolor="gray", linewidth=1
                )

    else:
        x_vals, y_vals, dx = methods_dict[method](f, a, b, n)
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

def compute_method_value(f, method, a, b, n, seed=None):
    if method == "隨機取點法":
        left_edges, random_x, y_vals, dx = random_sample_method(f, a, b, n, seed=seed)
        return np.sum(y_vals * dx)

    x_vals, y_vals, dx = methods_dict[method](f, a, b, n)

    if method != "梯形法":
        return np.sum(y_vals * dx)
    else:
        total = 0
        for i in range(len(x_vals) - 1):
            xi, xi1 = x_vals[i], x_vals[i + 1]
            yi, yi1 = y_vals[i], y_vals[i + 1]
            total += (yi + yi1) / 2 * (xi1 - xi)
        return total

# ----------------------
# session state：管理隨機重抽
# ----------------------
if "random_seed" not in st.session_state:
    st.session_state.random_seed = 42

# ----------------------
# 標題
# ----------------------
st.title("微積分視覺實驗室：黎曼和學生互動版")
st.markdown("透過操作不同近似方法與分割數，觀察曲線下方面積的近似過程，理解黎曼和與定積分之間的關係。")

# ----------------------
# 側邊欄
# ----------------------
st.sidebar.header("操作面板")

example_options = {
    "自訂": "x**2",
    "x": "x",
    "x**2": "x**2",
    "np.sin(x)": "np.sin(x)",
    "np.cos(x)": "np.cos(x)",
    "np.exp(-x)": "np.exp(-x)",
    "sqrt(x+1)": "sqrt(x+1)"
}

example = st.sidebar.selectbox("選擇範例函數", list(example_options.keys()))
func_str = st.sidebar.text_input("輸入函數 f(x)", value=example_options[example])

view_mode = st.sidebar.radio("顯示模式", ["單一方法", "五種方法比較"])
method = st.sidebar.selectbox("選擇方法", list(methods_dict.keys()))
n = st.sidebar.slider("分割數 n", 1, 100, 6)
color_hex = st.sidebar.color_picker("圖形顏色", "#ff6b6b")

a = st.sidebar.number_input("積分下限 a", value=0.0)
b = st.sidebar.number_input("積分上限 b", value=5.0)

show_area_text = st.sidebar.checkbox("顯示學習提醒", value=True)

# 隨機設定
random_mode = st.sidebar.radio(
    "隨機取點模式",
    ["固定結果", "每次按按鈕重新抽樣"]
)

if random_mode == "固定結果":
    manual_seed = st.sidebar.number_input("亂數種子", value=42, step=1)
    seed = int(manual_seed)
else:
    if st.sidebar.button("重新隨機抽樣"):
        st.session_state.random_seed = int(np.random.randint(0, 10**9))
    seed = st.session_state.random_seed
    st.sidebar.caption(f"目前抽樣編號（seed）：{seed}")

if a >= b:
    st.error("請設定正確區間：必須滿足 a < b。")
    st.stop()

# ----------------------
# 解析函數
# ----------------------
try:
    f = parse_function(func_str)
    test_x = np.linspace(a, b, 30)
    test_y = f(test_x)

    if not np.all(np.isfinite(test_y)):
        st.error("函數在此區間內出現無效值，請調整函數或積分區間。")
        st.stop()

except Exception:
    st.error("函數輸入錯誤。可輸入例如：x**2、np.sin(x)、sqrt(x+1)、exp(-x)")
    st.stop()

# ----------------------
# 主區塊
# ----------------------
tab1, tab2, tab3 = st.tabs(["互動圖形", "方法比較", "教學說明"])

with tab1:
    if method == "隨機取點法" and random_mode == "每次按按鈕重新抽樣":
        st.button(
            "重新隨機抽樣",
            key="main_random_button",
            on_click=lambda: st.session_state.update(
                random_seed=int(np.random.randint(0, 10**9))
            )
        )
        seed = st.session_state.random_seed
        st.caption(f"目前抽樣編號（seed）：{seed}")

    if view_mode == "單一方法":
        fig, riemann_sum, exact, error = draw_single_method(
            f, func_str, method, a, b, n, color_hex, seed=seed
        )
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
            current_seed = seed if m == "隨機取點法" else None
            fig, approx, _, err = draw_single_method(
                f, func_str, m, a, b, n, color_hex, seed=current_seed
            )
            with cols[i % 2]:
                st.pyplot(fig, use_container_width=True)
            results.append((m, approx, exact, err))

        st.subheader("五種方法數值比較")
        st.dataframe(
            {
                "方法": [r[0] for r in results],
                "近似值": [round(r[1], 5) for r in results],
                "精確值": [round(r[2], 5) for r in results],
                "誤差": [round(r[3], 5) for r in results],
            },
            use_container_width=True
        )

    if show_area_text:
        st.info("觀察重點：當分割數 n 增加時，黎曼和通常會更接近精確積分值，誤差會逐漸變小。隨機取點法則會因抽樣位置不同而改變近似值。")

with tab2:
    st.subheader("方法誤差比較")

    exact, _ = quad(f, a, b)
    method_names = list(methods_dict.keys())
    approx_vals = []
    errors = []

    for m in method_names:
        current_seed = seed if m == "隨機取點法" else None
        approx = compute_method_value(f, m, a, b, n, seed=current_seed)
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
        plt.xticks(rotation=15)
        st.pyplot(fig1, use_container_width=True)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(6, 4.5))
        ax2.bar(method_names, errors)
        ax2.set_title("各方法誤差比較")
        ax2.set_ylabel("誤差")
        ax2.grid(alpha=0.2)
        plt.xticks(rotation=15)
        st.pyplot(fig2, use_container_width=True)

    st.dataframe(
        {
            "方法": method_names,
            "近似值": [round(v, 5) for v in approx_vals],
            "精確值": [round(exact, 5)] * len(method_names),
            "誤差": [round(e, 5) for e in errors]
        },
        use_container_width=True
    )

with tab3:
    st.subheader("學習說明")
    st.markdown("""
### 1. 什麼是黎曼和？
黎曼和是把曲線下方的面積切成很多小塊，再用長方形或梯形去近似面積的方法。

### 2. 五種常見方法
- **左端點法**：每小區間取左端點高度
- **右端點法**：每小區間取右端點高度
- **中點法**：每小區間取中點高度
- **梯形法**：每小區間用梯形近似
- **隨機取點法**：每小區間隨機取一個點，該點的函數值作為長方形高度

### 3. 如何觀察？
你可以改變：
- 函數
- 積分區間
- 分割數 n
- 近似方法

然後比較：
- 近似值是否接近精確值
- 哪一種方法誤差較小
- 當 n 增加時，誤差如何變化
- 隨機取點法在不同抽樣下是否會波動

### 4. 建議學生操作任務
1. 先輸入 `x**2`
2. 比較五種方法在 n=4、n=10、n=30 時的誤差
3. 選擇隨機取點法，按下「重新隨機抽樣」
4. 觀察相同 n 下近似值如何改變
5. 再改成 `np.sin(x)` 或 `np.exp(-x)`
6. 記錄哪一種方法通常較準確
""")
