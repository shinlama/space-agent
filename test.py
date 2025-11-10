import pandas as pd
df = pd.read_csv("서울시_상권_카페빵_표본.csv", encoding="utf-8-sig")
df.head(2).to_csv("sample_cafes.csv", index=False, encoding="utf-8-sig")