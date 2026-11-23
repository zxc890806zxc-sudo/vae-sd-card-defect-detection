# 範例資料 / Sample data

`samples/` 放的是少量範例影像，資料夾結構跟程式讀取的路徑一致，切換到對應資料夾就能直接執行程式（執行方式見根目錄 README）。

```text
samples/
  single_card/
    model-image/              # 訓練用無瑕疵影像
    test-image/defect/        # 測試：有瑕疵
    test-image/defect-free/   # 測試：無瑕疵
  multi_region/
    model-image/D1 ~ D4/      # 每個區域的訓練影像
    test-image/defect/D1 ~ D4/
    test-image/defect-free/D1 ~ D4/
```

範例圖每類只有 3～4 張，只夠確認程式能正常執行，不能用來評估模型表現或重新訓練出可用的模型。完整資料集沒有公開。

The samples follow the folder layout the scripts expect. They are only enough to check that the scripts run; they are not a training or evaluation dataset. The full dataset is not published.
