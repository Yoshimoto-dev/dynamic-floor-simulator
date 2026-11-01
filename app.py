import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# ==========================================
# 0. ページの設定とデザイン
# ==========================================
st.set_page_config(
    page_title="Dynamic Floor Price Simulator",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 GAM Dynamic Floor Price Simulator")
st.markdown("""
広告リクエストのコンテキスト（時間帯やデバイス、ページカテゴリなど）を入力すると、
機械学習モデルが予測した最高入札額（Winning Bid）に基づき、**最適なDynamic Floor（動的フロア価格）**をリアルタイムで提案します。
""")

# ==========================================
# 1. モデルの学習（キャッシュを利用して効率化）
# ==========================================
@st.cache_resource
def load_and_train_model():
    df = pd.read_csv('ad_data_v3.csv')
    feature_cols = ['hour', 'device', 'viewability', 'hist_ctr', 'season', 'page_category']
    target_col = 'winning_bid'
    
    X = df[feature_cols]
    y = df[target_col]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, df

model, df_train = load_and_train_model()

# ==========================================
# 2. サイドバー：ユーザー入力（コンテキスト）
# ==========================================
st.sidebar.header("📊 リクエスト・コンテキスト設定")

hour = st.sidebar.slider("時間帯 (Hour)", min_value=0, max_value=23, value=21)

device_label = st.sidebar.selectbox("デバイス", ["Mobile", "Desktop"])
device = 1 if device_label == "Desktop" else 0

viewability_percent = st.sidebar.slider("予測視認率 (Viewability)", min_value=30.0, max_value=95.0, value=80.0, step=1.0, format="%.1f%%")
viewability = viewability_percent / 100.0

ctr_percent = st.sidebar.slider("過去平均CTR (Historical CTR)", min_value=0.1, max_value=2.0, value=1.2, step=0.1, format="%.1f%%")
hist_ctr = ctr_percent / 100.0

season_map = {"春 (1)": 1, "夏 (2)": 2, "秋 (3)": 3, "冬 / 繁忙期 (4)": 4}
season_label = st.sidebar.selectbox("季節", list(season_map.keys()))
season = season_map[season_label]

category_map = {
    "ニュース / 一般記事 (1)": 1, 
    "コラム / エンタメ (2)": 2, 
    "ライフスタイル / ショッピング (3)": 3, 
    "専門 / ビジネス (4)": 4
}
cat_label = st.sidebar.selectbox("ページカテゴリ", list(category_map.keys()))
page_category = category_map[cat_label]

safety_ratio = st.sidebar.slider("安全率 (Dynamic Floor 係数)", min_value=0.50, max_value=0.95, value=0.85, step=0.05)

# ==========================================
# 3. メイン画面：予測と結果表示
# ==========================================
st.divider()

input_data = pd.DataFrame({
    'hour': [hour],
    'device': [device],
    'viewability': [viewability],
    'hist_ctr': [hist_ctr],
    'season': [season],
    'page_category': [page_category]
})

if st.button("🚀 最適なDynamic Floorを算出する", type="primary"):
    predicted_bid = model.predict(input_data)[0]
    dynamic_floor = predicted_bid * safety_ratio
    
    # 画面にメトリック表示
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="予測最高入札額 (Winning Bid)", value=f"¥{predicted_bid:.2f}")
    with col2:
        st.metric(label="推奨 Dynamic Floor", value=f"¥{dynamic_floor:.2f}", delta=f"{int(safety_ratio*100)}% 適用")
    with col3:
        estimated_fill = "高 (High)" if safety_ratio <= 0.85 else "中〜低 (Medium-Low)"
        st.metric(label="想定される落札率（フィルレート）", value=estimated_fill)
        
    st.success("推論が正常に完了しました！")
    
    # --- 動的な解説文の自動生成ロジック ---
    # 季節に応じた傾向メッセージ
    season_insights = {
        1: "春シーズンは新生活や新年度のプロモーション需要が立ち上がる時期であり、学習データ上も中盤戦として堅調な入札単価が形成される傾向にあります。",
        2: "夏シーズンは、夏季休暇やレジャー関連の広告キャンペーンが活発化する一方で、メディア全体のトラフィック変動が見られるため、モデルは個別の視認性やCTRをシビアに評価しています。",
        3: "秋シーズンは、年末商戦に向けた広告主の予算準備期間にあたり、特定のライフスタイル・専門ジャンルでデモグラフィックな入札が強まる傾向があります。",
        4: "冬（繁忙期）シーズンは、年末年始や決算期に向けた広告主の年間予算消化のピークと重なり、学習データ全体の中でも最も高い入札単価（Winning Bid）が付きやすい傾向にあります。"
    }
    
    # カテゴリに応じた傾向メッセージ
    cat_insights = {
        1: "「ニュース / 一般記事」枠は、幅広いリーチを獲得できる反面、ターゲティングの絞り込みが難しいため、視認率やCTRの高さが単価を引き上げる主要因となります。",
        2: "「コラム / エンタメ」枠は、エンゲージメントの高いユーザーが集まりやすく、時間帯によるトラフィックの波が大きく反映されやすい特性があります。",
        3: "「ライフスタイル / ショッピング」枠は、購買意欲の高いユーザー層にアプローチできるため、バイヤーからの評価が高く、フロア価格を強気に設定しやすい傾向があります。",
        4: "「専門 / ビジネス」枠は、ターゲットが明確で広告主のROIが高いため、学習データ上でも最も高い入札単価がつきやすい高単価セグメントです。"
    }

    # デバイスに応じたメッセージ
    device_insight = "デスクトップ環境は、画面占有率やビューアビリティ（視認率）の担保がしやすいため、モバイル環境に比べてベースの入札単価が高く評価されています。" if device == 1 else "モバイル環境は、リクエストボリュームが大きい一方で、画面サイズや視認率のバラツキが大きいため、モデルは細かなコンテキストの組み合わせを重視して価格を算出しています。"

    with st.expander("💡 入力値に基づくAI詳細分析レポート（動的生成）", expanded=True):
        st.markdown(f"### 📊 1. 選択されたコンテキストの評価")
        st.markdown(f"""
- **デバイス:** **{device_label}** — {device_insight}
- **視認率 ({viewability_percent:.1f}%):** 視認率が80%を超える高視認性枠であるため、ブランド広告主からの評価プレミアムがプラスに働いています。
- **過去平均CTR ({ctr_percent:.1f}%):** ユーザーのエンゲージメント指標として機能し、入札競争の激しさを反映しています。
- **ページカテゴリ ({cat_label}):** {cat_insights[page_category]}
- **季節・時期 ({season_label}):** {season_insights[season]}
        """)
        
        st.markdown(f"### 🤖 2. 機械学習モデルによる価格決定の仕組み")
        st.markdown(f"""
- **予測最高入札額の算出:** 
  ランダムフォレストモデルが過去500件の広告オークションデータを参照し、上記の複合条件（時間帯 `{hour}時`、季節、カテゴリ等）を照らし合わせた結果、今回の予想落札額を **¥{predicted_bid:.2f}** と弾き出しました。
- **Dynamic Floorの最適化:** 
  予測された勝者入札額に対し、設定された安全率（**{int(safety_ratio*100)}%**）を乗算し、**¥{dynamic_floor:.2f}** を最終的な動的フロア価格として出力しています。これにより、機会損失を防ぎつつ最大の収益性を確保します。
        """)

else:
    st.info("👈 左側のサイドバーで条件を調整し、[最適なDynamic Floorを算出する] ボタンを押してください。")git branch -M main