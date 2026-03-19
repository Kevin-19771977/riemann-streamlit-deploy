# 微積分視覺實驗室：黎曼和教學版

這個專案是以 Streamlit 製作的互動式網頁教學工具，適合用於微積分課程中說明黎曼和、定積分近似、四種近似方法比較，以及誤差隨分割數變化的觀察。

## 功能特色

- 中文介面
- 函數輸入框大字版，適合學生直接輸入
- 四種方法同步比較：Left / Right / Midpoint / Trapezoid
- 單一方法放大觀察區
- 誤差隨 n 變化的小圖
- 可下載比較表 CSV

## 本機執行

1. 安裝套件：

   pip install -r requirements.txt

2. 啟動程式：

   streamlit run app.py

## 部署到 Streamlit Community Cloud

1. 將此資料夾內容上傳到 GitHub repository
2. 到 Streamlit Community Cloud 建立新 app
3. Repository 選你的 repo
4. Branch 選 `main`
5. Main file path 填 `app.py`
6. 按 Deploy

## 檔案結構

- `app.py`：主程式
- `requirements.txt`：相依套件
- `.streamlit/config.toml`：介面設定
- `README.md`：說明文件
