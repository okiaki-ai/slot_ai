import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import time

# ==========================================
# 画面の基本設定
# ==========================================
st.set_page_config(page_title="マイジャグラー5 予測AI", layout="wide")
st.title("🎰 マイジャグラー5 狙い目予測AI & 傾向分析")
st.write("過去データに基づき、明日の勝率予測と高設定投入パターンの分析を行います。")

# ==========================================
# 1. データの読み込み
# ==========================================
@st.cache_data(ttl=600) 
def load_data():
    cache_buster = int(time.time())
    try:
        base_url = st.secrets["spreadsheet_url"]
    except KeyError:
        st.error("❌ セキュリティ用の設定が見つかりません。")
        st.stop()
        
    url = f"{base_url}&_={cache_buster}"
    df = pd.read_csv(url)
    
    # 台番号が「平均」となっている行を除外
    df = df[df['台番号'].astype(str) != '平均']
    
    # 数値変換処理
    df['G数'] = pd.to_numeric(df['G数'].astype(str).str.replace(",", ""), errors='coerce').fillna(0).astype(int)
    df['差枚'] = pd.to_numeric(df['差枚'].astype(str).str.replace("+", "").str.replace(",", ""), errors='coerce').fillna(0).astype(int)
    df['BB'] = pd.to_numeric(df['BB'], errors='coerce').fillna(0).astype(int)
    df['RB'] = pd.to_numeric(df['RB'], errors='coerce').fillna(0).astype(int)
    df['合成確率_表示用'] = df['合成確率'].astype(str)
    
    df['BB確率'] = pd.to_numeric(df['BB確率'].astype(str).str.replace("1/", ""), errors='coerce').fillna(0.0)
    df['RB確率'] = pd.to_numeric(df['RB確率'].astype(str).str.replace("1/", ""), errors='coerce').fillna(0.0)
    df['合成確率'] = pd.to_numeric(df['合成確率'].astype(str).str.replace("1/", ""), errors='coerce').fillna(0.0)
    
    df['日付'] = pd.to_datetime(df['日付'])
    return df.sort_values(['台番号', '日付'])

with st.spinner('データを読み込み中...'):
    df = load_data()
    latest_date = df['日付'].max()

# ==========================================
# 2. AI学習（予測ロジック）
# ==========================================
# (既存の学習ロジックを維持)
df['日'] = df['日付'].dt.day
df['還元日'] = df['日'].apply(lambda x: 1 if x == 3 or (1 <= x <= 5) or (27 <= x <= 31) else 0)
df['警戒日'] = df['日'].apply(lambda x: 1 if 10 <= x <= 20 else 0)
corner_list = [521, 540, 541, 560]
df['角台'] = df['台番号'].isin(corner_list).astype(int)

for i in range(1, 8):
    df[f'{i}日前の差枚'] = df.groupby('台番号')['差枚'].shift(i).fillna(0)

df['1日前のRB確率'] = df.groupby('台番号')['RB確率'].shift(1).fillna(0.0)
df['7日間合計'] = df['差枚'] + df['1日前の差枚'] + df['2日前の差枚'] + df['3日前の差枚'] + df['4日前の差枚'] + df['5日前の差枚'] + df['6日前の差枚']
df['V字回復候補'] = df['1日前の差枚'].apply(lambda x: 1 if -4000 <= x <= -2500 else 0)
df['回収トラップ'] = df['1日前の差枚'].apply(lambda x: 1 if x > 3000 else 0)

df['翌日の差枚'] = df.groupby('台番号')['差枚'].shift(-1)
df['翌日勝つか'] = (df['翌日の差枚'] > 0).astype(int)

train_df = df.dropna(subset=['翌日の差枚'])
features = ['G数', '差枚', 'BB', 'RB', '合成確率', '還元日', '警戒日', '角台', 'V字回復候補', '回収トラップ',
            '1日前の差枚', '2日前の差枚', '3日前の差枚', '4日前の差枚', '5日前の差枚', '6日前の差枚', '7日前の差枚']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(train_df[features], train_df['翌日勝つか'])

# ==========================================
# 3. メイン画面：明日の予測
# ==========================================
latest_df = df[df['日付'] == latest_date].copy()
latest_df['明日勝つ確率(%)'] = model.predict_proba(latest_df[features])[:, 1] * 100
recommendations = latest_df.sort_values('明日勝つ確率(%)', ascending=False)

# 除外ルール適用
exclude_condition = (
    (recommendations['差枚'] >= 1000) | 
    (recommendations['1日前の差枚'] >= 1000) | 
    (recommendations['2日前の差枚'] >= 1000) | 
    (recommendations['7日間合計'] >= 2000) |
    ((recommendations['RB確率'] > 0) & (recommendations['RB確率'] <= 310.0)) |
    ((recommendations['1日前のRB確率'] > 0) & (recommendations['1日前のRB確率'] <= 310.0))
)
recommendations = recommendations[~exclude_condition]

st.subheader(f"📅 予測基準日: {latest_date.strftime('%Y-%m-%d')}")
st.info(f"💡 厳しい条件をクリアした **{len(recommendations)}台** をピックアップ。")

result_display = recommendations[['台番号', '明日勝つ確率(%)', '7日間合計', '差枚', 'RB確率', 'BB', 'RB']].copy()
st.dataframe(result_display.rename(columns={'明日勝つ確率(%)': '勝率%', '7日間合計': '7日計'}), use_container_width=True, hide_index=True)

# ==========================================
# 4. 🌟 新機能：高設定投入前の波形パターン分析
# ==========================================
st.markdown("---")
st.subheader("🔍 高設定投入前の「予兆」波形分析")
st.write("過去に「高設定挙動（RB 1/290以下 & 差枚+1000以上）」を見せた台の、その直前7日間のスランプグラフを分析します。")

# 高設定だった日を抽出
high_setting_days = df[
    (df['RB確率'] > 0) & (df['RB確率'] <= 290) & (df['差枚'] >= 1000)
].copy()

if high_setting_days.empty:
    st.warning("現在、分析対象となる過去の高設定データが不足しています。データが蓄積されるまでお待ちください。")
else:
    # ユーザーが分析する過去の事例を選択できるようにする
    high_setting_days = high_setting_days.sort_values('日付', ascending=False)
    case_list = high_setting_days.apply(lambda x: f"{x['日付'].strftime('%m/%d')} - 台番号:{x['台番号']} (差枚:{x['差枚']})", axis=1).tolist()
    
    selected_case = st.selectbox("分析する過去の事例を選択してください", case_list)
    
    # 選択された事例のデータを特定
    selected_idx = case_list.index(selected_case)
    target_row = high_setting_days.iloc[selected_idx]
    target_machine = target_row['台番号']
    target_date = target_row['日付']
    
    # その台の全履歴を取得し、当日までの7日間を切り出す
    machine_history = df[df['台番号'] == target_machine].sort_values('日付')
    pre_high_history = machine_history[machine_history['日付'] <= target_date].tail(8)
    
    if len(pre_high_history) >= 2:
        # スランプグラフ（累積差枚）の計算
        pre_high_history = pre_high_history.copy()
        # グラフ表示用に日付を調整（当日を「高設定投入日」とする）
        plot_labels = []
        for d in pre_high_history['日付']:
            diff_days = (d - target_date).days
            if diff_days == 0: plot_labels.append("★当日(高設定)")
            else: plot_labels.append(f"{diff_days}日前")
        
        pre_high_history.index = plot_labels
        
        # 累積差枚を計算（7日前を起点とする）
        # グラフを0から始めたいので、最初の日の前に0を追加
        daily_diffs = pre_high_history['差枚'].tolist()
        cumulative_diffs = [0]
        current_sum = 0
        for d in daily_diffs:
            current_sum += d
            cumulative_diffs.append(current_sum)
            
        # グラフ用データ
        chart_labels = ["起点"] + plot_labels
        plot_df = pd.DataFrame({'累積差枚': cumulative_diffs}, index=chart_labels)
        
        st.write(f"### 台番号 {target_machine}：{target_date.strftime('%Y-%m-%d')} の投入前波形")
        st.line_chart(plot_df)
        
        # 当時の詳細データ表示
        st.write("▼ 当日の詳細データ")
        st.table(pd.DataFrame([{
            "日付": target_date.strftime('%Y-%m-%d'),
            "台番号": target_machine,
            "差枚": f"+{target_row['差枚']}枚",
            "RB確率": f"1/{target_row['RB確率']:.1f}",
            "BB確率": f"1/{target_row['BB確率']:.1f}",
            "G数": f"{target_row['G数']}G"
        }]))
        
        st.info("💡 グラフの見方：このグラフは「高設定が投入された日（★）」に向かって、その台が過去7日間でどのように凹んでいたか、あるいは浮いていたかを示しています。複数の事例を切り替えて見ていくことで、『どんな凹み方の後に高設定が来やすいか』という共通点が見えてきます。")
    else:
        st.warning("この事例の直前データが不足しているため、グラフを表示できません。")

st.caption("※「高設定」の定義：RB確率 1/290以下 かつ 差枚+1000枚以上")