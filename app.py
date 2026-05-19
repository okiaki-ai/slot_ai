import streamlit as st
import pandas as pd
import time

# ==========================================
# ⚙️ ページ基本設定 (必ず一番上に書く)
# ==========================================
st.set_page_config(page_title="Juggler AI Dashboard", page_icon="🎰", layout="wide")

# ==========================================
# 📊 機種ごとの設定6基準値
# ==========================================
# ※以前 secrets に保存した基本URL（最後に &gid=XXXX がつく前の状態）を前提としています
MACHINE_DATA = {
    "🐯 マイジャグラーV": {
        "gid": "0",  # 🌟 スプレッドシートの「マイジャグ5」タブのGID番号（※もし変更していれば書き換えてください）
        "prob_6_total": 114.6,
        "prob_6_bb": 229.1,
        "prob_6_rb": 229.1
    },
    "🐶 ゴーゴージャグラー3": {
        "gid": "123456789",  # 🌟 スプレッドシートの「ゴージャグ」タブのGID番号をここに入力
        "prob_6_total": 117.4,
        "prob_6_bb": 234.9,
        "prob_6_rb": 234.9
    },
    "🐿️ ハッピージャグラーV III": {
        "gid": "987654321",  # 🌟 スプレッドシートの「ハッピー」タブのGID番号をここに入力
        "prob_6_total": 120.0,
        "prob_6_bb": 226.0,
        "prob_6_rb": 256.0
    }
}

# ==========================================
# 📱 サイドバー (左メニュー) のUI
# ==========================================
st.sidebar.title("🎰 分析メニュー")
st.sidebar.write("---")

# 機種選択プルダウン
selected_machine = st.sidebar.selectbox(
    "📊 分析する機種を選択",
    list(MACHINE_DATA.keys())
)

# 選択された機種の情報を取得
target_info = MACHINE_DATA[selected_machine]

st.sidebar.write("---")
st.sidebar.subheader("💡 設定6の目安")
st.sidebar.write(f"合算確率: **1/{target_info['prob_6_total']}**")
st.sidebar.write(f"BB確率: **1/{target_info['prob_6_bb']}**")
st.sidebar.write(f"RB確率: **1/{target_info['prob_6_rb']}**")

# ==========================================
# 🔑 データ取得関数 (過去のコードを完全移植)
# ==========================================
@st.cache_data(ttl=600) 
def load_data(gid_id):
    cache_buster = int(time.time())
    try:
        # secrets からベースの公開URLを取得
        base_url = st.secrets["spreadsheet_url"]
    except KeyError:
        st.error("❌ セキュリティ用の設定が見つかりません。")
        st.stop()
        
    # 選択されたタブのGIDとキャッシュバスターをURLに結合
    url = f"{base_url}&gid={gid_id}&_={cache_buster}"
    
    try:
        df = pd.read_csv(url)
        
        # 数値計算のためにデータ型を安全に変換
        numeric_cols = ["台番号", "G数", "差枚", "BB", "RB"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        return df
    except Exception as e:
        st.error(f"❌ データの読み込みに失敗しました。URLや公開設定を確認してください。\nエラー: {e}")
        st.stop()

# ==========================================
# 🖥️ メイン画面のUI
# ==========================================
st.title(f"{selected_machine} 分析ダッシュボード")

# データの読み込み
with st.spinner("📡 データを取得中..."):
    df = load_data(target_info["gid"])

if df.empty or len(df) == 0:
    st.warning("⚠️ 表示できるデータがありません。")
else:
    # --- 全体サマリー ---
    st.subheader("📈 全体サマリー (全期間)")
    
    total_g = df["G数"].sum()
    total_diff = df["差枚"].sum()
    total_bb = df["BB"].sum()
    total_rb = df["RB"].sum()
    
    # 0割りエラー防止
    avg_total_prob = round(total_g / (total_bb + total_rb), 1) if (total_bb + total_rb) > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("総ゲーム数", f"{total_g:,} G")
    col2.metric("総差枚数", f"{total_diff:,} 枚")
    col3.metric("平均合算確率", f"1/{avg_total_prob}" if avg_total_prob > 0 else "-")
    col4.metric("総データ件数", f"{len(df)} 件")

    st.write("---")

    # --- AI 優秀台ピックアップ ---
    st.subheader("✨ AIピックアップ (設定6基準クリア台)")
    
    # G数が2000G以上で、合算が設定6の基準より良い台を抽出
    excellent_df = df[
        (df["G数"] >= 2000) & 
        ((df["G数"] / (df["BB"] + df["RB"])) <= target_info["prob_6_total"])
    ].copy()
    
    if not excellent_df.empty:
        # 合算が良い順に並び替え
        excellent_df["計算_合算"] = excellent_df["G数"] / (excellent_df["BB"] + excellent_df["RB"])
        excellent_df = excellent_df.sort_values("計算_合算").drop(columns=["計算_合算"])
        
        st.success(f"🎯 優秀台を {len(excellent_df)} 件発見しました！")
        st.dataframe(excellent_df, use_container_width=True)
    else:
        st.info("ℹ️ 現在のデータ内に、G数2000以上で設定6基準をクリアしている台はありません。")

    st.write("---")

    # --- 全データ表示 (日付絞り込み機能付き) ---
    st.subheader("📅 詳細データ検索")
    
    if "日付" in df.columns:
        dates = df["日付"].unique().tolist()
        dates.sort(reverse=True) # 新しい日付順
        dates.insert(0, "すべて")
        
        selected_date = st.selectbox("日付で絞り込む", dates)
        
        if selected_date == "すべて":
            display_df = df
        else:
            display_df = df[df["日付"] == selected_date]
            
        st.dataframe(display_df, use_container_width=True)
    else:
        st.dataframe(df, use_container_width=True)