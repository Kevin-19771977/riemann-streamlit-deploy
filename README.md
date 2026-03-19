# Streamlit Riemann Sum Student App

這是可部署到 Streamlit Community Cloud 的黎曼和學生互動平台。

## 本機執行

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## 部署到 Streamlit Community Cloud

1. 將整個資料夾上傳到 GitHub repository
2. 到 Streamlit Community Cloud 建立新 app
3. Repository 選你的專案
4. Main file path 填 `app.py`
5. 按 Deploy

## 功能

- 中文互動介面
- 左端點法 / 右端點法 / 中點法 / 梯形法
- 四種方法比較
- 學生作答區
- 任務卡
- CSV 匯出
