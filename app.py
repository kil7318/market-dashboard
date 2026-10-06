import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from datetime import datetime

# ------------------------------------------------------------
# 기본 설정
# ------------------------------------------------------------
st.set_page_config(page_title="나만의 마켓 대시보드", page_icon="📊", layout="wide")

NOTICE = "※ 하나의 지표만으로 매수·매도를 결정하기보다, 서로 다른 데이터가 같은 방향을 가리키는지 확인하는 점검용 화면입니다."

# 한국 업종 ETF (종목코드는 야후 파이낸스 기준, 안 불러와지면 화면에 알려줍니다)
KR_ETF = {
    "KODEX 반도체": "091160.KS",
    "KODEX 은행": "091170.KS",
    "KODEX 보험": "140700.KS",
    "KODEX 증권": "102970.KS",
    "KODEX 자동차": "091180.KS",
    "KODEX 2차전지산업": "305720.KS",
    "KODEX 철강": "117680.KS",
    "KODEX 건설": "117700.KS",
    "KODEX 에너지화학": "117460.KS",
    "TIGER 화장품": "228790.KS",
    "TIGER 헬스케어": "143860.KS",
    "KODEX 바이오": "244580.KS",
}

# 미국 업종 ETF
US_ETF = {
    "XLK 기술": "XLK", "XLF 금융": "XLF", "XLE 에너지": "XLE", "XLV 헬스케어": "XLV",
    "XLY 경기소비재": "XLY", "XLP 필수소비재": "XLP", "XLI 산업재": "XLI",
    "XLB 소재": "XLB", "XLU 유틸리티": "XLU", "XLRE 부동산": "XLRE",
    "XLC 커뮤니케이션": "XLC", "SMH 반도체": "SMH", "IGV 소프트웨어": "IGV",
    "ITA 방산": "ITA", "XBI 바이오": "XBI",
}

# ------------------------------------------------------------
# 데이터 불러오기 (1시간 동안 저장해두고 재사용 → 빠르고 차단 위험 적음)
# ------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="데이터 불러오는 중...")
def load_prices(tickers: tuple, period: str = "2y") -> pd.DataFrame:
    df = yf.download(list(tickers), period=period, auto_adjust=True, progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame(tickers[0])
    return df.dropna(how="all")


# ------------------------------------------------------------
# 계산 함수
# ------------------------------------------------------------
def momentum_avg(s: pd.Series) -> float:
    """1·3·6·12개월 수익률의 평균 (%)"""
    s = s.dropna()
    rets = [s.iloc[-1] / s.iloc[-n - 1] - 1 for n in (21, 63, 126, 252) if len(s) > n]
    return float(np.mean(rets) * 100) if rets else np.nan


def period_return(s: pd.Series, n: int) -> float:
    s = s.dropna()
    return float((s.iloc[-1] / s.iloc[-n - 1] - 1) * 100) if len(s) > n else np.nan


def sortino(s: pd.Series, n: int = 126) -> float:
    """하락 위험 대비 수익 효율 (클수록 좋음)"""
    r = s.dropna().pct_change().dropna().tail(n)
    downside = np.sqrt((np.minimum(r, 0) ** 2).mean())
    return float(r.mean() / downside * np.sqrt(252)) if downside > 0 else np.nan


def relative_strength_table(names: dict, bench: str, bench_name: str) -> pd.DataFrame | None:
    tickers = tuple(names.values()) + (bench,)
    px = load_prices(tickers)
    if bench not in px.columns:
        st.error(f"기준지수({bench_name}) 데이터를 불러오지 못했습니다. 잠시 후 새로고침 해주세요.")
        return None

    bench_ret = period_return(px[bench], 126)
    rows, failed = [], []
    for name, tk in names.items():
        if tk not in px.columns or px[tk].dropna().shape[0] < 130:
            failed.append(name)
            continue
        s = px[tk].dropna()
        rows.append({
            "ETF": name,
            "6개월 수익률(%)": period_return(s, 126),
            f"{bench_name} 대비(%p)": period_return(s, 126) - bench_ret,
            "20일 수익률(%)": period_return(s, 20),
            "소라티노(6M)": sortino(s),
            "60일선 위": "✅" if s.iloc[-1] > s.tail(60).mean() else "❌",
        })
    if failed:
        st.caption(f"⚠️ 데이터를 불러오지 못한 ETF: {', '.join(failed)}")
    if not rows:
        return None

    df = pd.DataFrame(rows)
    rs_col = f"{bench_name} 대비(%p)"
    # 종합점수 = 상대강도 순위 + 소라티노 순위 (작을수록 좋음)
    df["종합순위"] = (df[rs_col].rank(ascending=False) + df["소라티노(6M)"].rank(ascending=False)).rank(method="min").astype(int)
    return df.sort_values("종합순위").round(2).reset_index(drop=True)


def pct_rank_last(x: np.ndarray) -> float:
    """최근 값이 과거 값들 중 상위 몇 %에 있는지 (0~100)"""
    return float((x <= x[-1]).mean() * 100)


def rolling_pct(s: pd.Series, window: int = 750, minp: int = 250) -> pd.Series:
    """시점마다 '과거 약 3년 대비 지금 위치'를 0~100 점수로 환산"""
    return s.rolling(window, min_periods=minp).apply(pct_rank_last, raw=True)


def calc_rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / down)


def fear_greed_label(score: float) -> str:
    if score < 20:
        return "극단적 공포 🔴"
    if score < 40:
        return "공포 🟠"
    if score < 60:
        return "중립 ⚪"
    if score < 80:
        return "탐욕 🟢"
    return "극단적 탐욕 🔥"


def header(title: str, desc: str):
    st.title(title)
    st.info(desc)
    st.caption(NOTICE)
    st.caption(f"마지막 업데이트: {datetime.now():%Y-%m-%d %H:%M}  (데이터는 1시간마다 갱신)")


def price_chart(s: pd.Series, title: str, ma_list=(60, 120)):
    s = s.dropna().tail(300)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s.values, name=title, line=dict(width=2)))
    full = s
    for m in ma_list:
        fig.add_trace(go.Scatter(x=full.index, y=full.rolling(m).mean(), name=f"{m}일선", line=dict(width=1, dash="dot")))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), title=title, legend=dict(orientation="h"))
    return fig


# ------------------------------------------------------------
# 왼쪽 메뉴
# ------------------------------------------------------------
st.sidebar.title("나만의 마켓 대시보드")
st.sidebar.caption("v0.2 · 개인 투자 점검용")
page = st.sidebar.radio(
    "메뉴",
    ["📌 시장 요약", "🚨 미국 위험신호", "😱 한국 공포·탐욕 지수", "🇰🇷 한국 ETF 상대강도", "🇺🇸 미국 ETF 상대강도"],
)
st.sidebar.divider()
st.sidebar.caption("본 화면은 투자 참고용이며 투자 판단과 책임은 본인에게 있습니다.")

# ------------------------------------------------------------
# 1) 시장 요약
# ------------------------------------------------------------
if page == "📌 시장 요약":
    header("시장 요약", "한국·미국 주요 지수와 환율, 변동성, 원자재를 한눈에 보는 화면입니다. 매일 아침 5분 점검용으로 쓰세요.")

    items = {
        "KOSPI": "^KS11", "KOSDAQ": "^KQ11", "S&P500": "^GSPC", "나스닥": "^IXIC",
        "원/달러": "KRW=X", "VIX(공포지수)": "^VIX", "금": "GC=F", "비트코인": "BTC-USD",
    }
    px = load_prices(tuple(items.values()), period="1y")

    cols = st.columns(4)
    for i, (name, tk) in enumerate(items.items()):
        with cols[i % 4]:
            if tk in px.columns and px[tk].dropna().shape[0] > 2:
                s = px[tk].dropna()
                chg = (s.iloc[-1] / s.iloc[-2] - 1) * 100
                st.metric(name, f"{s.iloc[-1]:,.2f}", f"{chg:+.2f}%")
            else:
                st.metric(name, "N/A")

    st.subheader("KOSPI 추세")
    if "^KS11" in px.columns:
        st.plotly_chart(price_chart(px["^KS11"], "KOSPI"))
        s = px["^KS11"].dropna()
        ma60, ma120 = s.tail(60).mean(), s.tail(120).mean()
        state = "상승 추세 (60일선 > 120일선, 가격이 60일선 위)" if (s.iloc[-1] > ma60 > ma120) else "추세 점검 필요"
        st.write(f"현재 판정: **{state}**")

# ------------------------------------------------------------
# 2) 미국 위험신호 (QQQ + TIP 카나리아)
# ------------------------------------------------------------
elif page == "🚨 미국 위험신호":
    header(
        "미국 위험신호 · QQQ & TIP 카나리아",
        "나스닥(QQQ)은 성장주 모멘텀, TIP(물가연동채)은 금리·인플레이션 환경을 반영합니다. "
        "두 모멘텀(1·3·6·12개월 평균)이 모두 양수면 공격, 하나라도 음수면 방어입니다.",
    )
    px = load_prices(("QQQ", "TIP"))
    if {"QQQ", "TIP"} <= set(px.columns):
        q, t = momentum_avg(px["QQQ"]), momentum_avg(px["TIP"])
        attack = (q > 0) and (t > 0)

        c1, c2, c3 = st.columns(3)
        c1.metric("현재 신호", "공격 모드 🟢" if attack else "방어 모드 🔴")
        c2.metric("QQQ 모멘텀", f"{q:+.2f}%", help="1·3·6·12개월 수익률 평균")
        c3.metric("TIP 모멘텀", f"{t:+.2f}%", help="둘 다 양수면 공격")

        st.plotly_chart(price_chart(px["QQQ"], "QQQ", ma_list=(50, 200)))
        st.caption("QQQ가 200일선 아래로 내려가는지도 함께 확인해보세요.")
    else:
        st.error("QQQ/TIP 데이터를 불러오지 못했습니다. 잠시 후 새로고침 해주세요.")

# ------------------------------------------------------------
# 2-2) 한국 공포·탐욕 지수 (피어앤그리드 오실레이터)
# ------------------------------------------------------------
elif page == "😱 한국 공포·탐욕 지수":
    header(
        "한국 공포·탐욕 지수",
        "KOSPI의 이격도(120일선)·52주 위치·RSI·변동성·원/달러 환율 5가지를 각각 '최근 3년 대비 현재 위치'(0~100점)로 바꿔 평균낸 점수입니다. "
        "0에 가까우면 공포, 100에 가까우면 탐욕입니다. 시장 과열·침체의 온도계로 쓰세요.",
    )
    px = load_prices(("^KS11", "KRW=X"), period="5y")

    if {"^KS11", "KRW=X"} <= set(px.columns):
        ks = px["^KS11"].dropna()
        fx = px["KRW=X"].reindex(ks.index).ffill()

        # 지표 5개 (원래 값)
        gap = ks / ks.rolling(120).mean() - 1
        pos52 = (ks - ks.rolling(252).min()) / (ks.rolling(252).max() - ks.rolling(252).min())
        rsi = calc_rsi(ks)
        vol = ks.pct_change().rolling(20).std() * np.sqrt(252)
        fxgap = fx / fx.rolling(120).mean() - 1

        # 지표별 점수 (높을수록 탐욕)
        scores = pd.DataFrame({
            "이격도(120일)": rolling_pct(gap),
            "52주 위치": pos52 * 100,
            "RSI(14일)": rsi,
            "변동성(20일)": 100 - rolling_pct(vol),   # 변동성 클수록 공포
            "원/달러 환율": 100 - rolling_pct(fxgap),  # 원화 약세일수록 공포
        }).dropna()
        composite = scores.mean(axis=1)

        now = float(composite.iloc[-1])
        prev = float(composite.iloc[-6]) if len(composite) > 6 else now

        ma120 = ks.rolling(120).mean()
        above = ks.iloc[-1] > ma120.iloc[-1]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("종합 점수", f"{now:.0f} / 100", f"{now - prev:+.1f} (5일 전 대비)")
        c2.metric("현재 판정", fear_greed_label(now))
        c3.metric(
            "KOSPI vs 120일선",
            "120일선 위 ✅" if above else "120일선 아래 ❌",
            f"{gap.iloc[-1] * 100:+.1f}%",
        )
        c4.metric("기준일", f"{composite.index[-1]:%Y-%m-%d}")

        # 게이지
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=now,
            gauge=dict(
                axis=dict(range=[0, 100]),
                bar=dict(color="#333"),
                steps=[
                    dict(range=[0, 20], color="#e74c3c"), dict(range=[20, 40], color="#f39c12"),
                    dict(range=[40, 60], color="#bdc3c7"), dict(range=[60, 80], color="#82e0aa"),
                    dict(range=[80, 100], color="#27ae60"),
                ],
            ),
        ))
        gauge.update_layout(height=280, margin=dict(l=20, r=20, t=20, b=10))
        st.plotly_chart(gauge)

        # 지표별 상세표
        st.subheader("지표별 상세")
        raw_now = {
            "이격도(120일)": f"{gap.iloc[-1] * 100:+.1f}%",
            "52주 위치": f"{pos52.iloc[-1] * 100:.0f}%",
            "RSI(14일)": f"{rsi.iloc[-1]:.1f}",
            "변동성(20일)": f"{vol.iloc[-1] * 100:.1f}%",
            "원/달러 환율": f"{fx.iloc[-1]:,.0f}원 (120일선 대비 {fxgap.iloc[-1] * 100:+.1f}%)",
        }
        table = pd.DataFrame({
            "지표": list(raw_now.keys()),
            "현재 값": list(raw_now.values()),
            "점수(0공포~100탐욕)": [round(float(scores[k].iloc[-1]), 1) for k in raw_now],
            "해석": [fear_greed_label(float(scores[k].iloc[-1])) for k in raw_now],
        })
        st.dataframe(table, hide_index=True)

        # 추이 차트
        st.subheader("종합 점수 추이")
        hist = composite.tail(500)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="종합 점수", line=dict(width=2)))
        fig.add_hline(y=20, line_dash="dot", line_color="#e74c3c", annotation_text="공포 20")
        fig.add_hline(y=80, line_dash="dot", line_color="#27ae60", annotation_text="탐욕 80")
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10), yaxis=dict(range=[0, 100]))
        st.plotly_chart(fig)

        st.caption(
            "※ 이 지수는 야후 파이낸스 데이터로 직접 계산한 단순 모델입니다. "
            "점수가 낮다고 바로 매수, 높다고 바로 매도하는 신호가 아니며, 극단 구간에서 분할 접근·비중 조절을 점검하는 용도로 쓰세요."
        )
    else:
        st.error("KOSPI/환율 데이터를 불러오지 못했습니다. 잠시 후 새로고침 해주세요.")

# ------------------------------------------------------------
# 3) 한국 ETF 상대강도
# ------------------------------------------------------------
elif page == "🇰🇷 한국 ETF 상대강도":
    header(
        "한국 ETF 소라티노 및 상대강도",
        "KOSPI보다 강하면서 하락 위험 대비 성과가 좋은 업종 ETF를 찾는 화면입니다. "
        "상대강도(6개월, KOSPI 대비)와 소라티노를 합산해 종합순위를 매깁니다.",
    )
    df = relative_strength_table(KR_ETF, "^KS11", "KOSPI")
    if df is not None:
        st.dataframe(df, hide_index=True)

# ------------------------------------------------------------
# 4) 미국 ETF 상대강도
# ------------------------------------------------------------
elif page == "🇺🇸 미국 ETF 상대강도":
    header(
        "미국 ETF 소라티노 및 상대강도",
        "수익률만 높은 업종보다 '하락 위험 대비 수익이 좋고 SPY보다 강한 업종'을 찾는 화면입니다.",
    )
    df = relative_strength_table(US_ETF, "SPY", "SPY")
    if df is not None:
        st.dataframe(df, hide_index=True)
