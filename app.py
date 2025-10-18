# app.py — Live Portfolio-Volatilitäts-Dashboard (Pro)
# Start:  python -m streamlit run app.py

import io
import math
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import streamlit as st

# Optional: arch für GARCH(1,1)
try:
    from arch import arch_model
    HAS_ARCH = True
except Exception:
    HAS_ARCH = False

# PDF
from fpdf import FPDF

# ========= Streamlit Setup =========
st.set_page_config(page_title="Live Portfolio-Volatilität (Pro)", layout="wide")
st.title("⚡ Live Portfolio-Volatilitäts-Dashboard (Pro)")
st.caption("RV vs. IV, Beta, Korrelationen, Stresstests, Greeks, GARCH, Faktoren, PDF-Export — live.")

# ========= Zuordnungen =========
WKN_TO_TICKER = {"A1W3F9": "XUFN.DE", "A1106A": "XDEW.DE", "A1W6DH": "STHE.L"}
ISIN_TO_TICKER = {"IE00BCHWNT26": "XUFN.DE", "IE00BF8HV600": "STHE.L", "IE00BLNMYC90": "XDEW.DE"}
IV_PROXIES = {"XDEW.DE": "RSP", "XUFN.DE": "XLF", "STHE.L": "HYG"}  # US-Proxies für IV

# ========= Helper =========
def resolve_symbol(code: str) -> str:
    code = code.strip().upper()
    return WKN_TO_TICKER.get(code) or ISIN_TO_TICKER.get(code) or code

def as_series(x, name=None) -> pd.Series:
    if isinstance(x, pd.Series): s = x
    elif isinstance(x, pd.DataFrame): s = x.iloc[:, 0]
    else: s = pd.Series(x)
    if name: s = s.copy(); s.name = name
    return s

def to_scalar(x):
    if isinstance(x, (pd.Series, pd.Index, np.ndarray)):
        return float(np.asarray(x).reshape(-1)[0])
    return float(x)

def to_utc(dt):
    return dt.replace(tzinfo=timezone.utc) if getattr(dt, "tzinfo", None) is None else dt.astimezone(timezone.utc)

def yearfrac(t0, t1):
    t0, t1 = to_utc(t0), to_utc(t1)
    return max((t1 - t0).days / 365.0, 1e-6)

def interval_defaults(interval: str) -> str:
    return {"1m":"5d","2m":"10d","5m":"30d","15m":"60d","30m":"60d","60m":"730d","1d":"5y"}.get(interval,"5y")

def interval_minutes(interval: str) -> float:
    if interval.endswith("m"): return float(interval[:-1])
    if interval.endswith("h"): return float(interval[:-1])*60.0
    if interval.endswith("d"): return float(interval[:-1])*60.0*24.0
    return 0.0

def annualize_from_interval(std_i: float, interval: str) -> float:
    mins = interval_minutes(interval)
    if mins <= 0: return np.nan
    if mins < 60*20:  # intraday
        steps_per_day = 390.0 / mins  # 6.5h US-Handel
        steps_per_year = steps_per_day * 252.0
    else:
        steps_per_year = 252.0
    return std_i * np.sqrt(steps_per_year)

def drawdown(series: pd.Series) -> pd.Series:
    if series is None or series.empty: return pd.Series(dtype=float)
    cummax = series.cummax()
    dd = series / cummax - 1.0
    dd.name = "Drawdown"
    return dd

# ========= Black–Scholes, IV & Greeks =========
def _d1(S,K,r,q,sigma,T): return (np.log(S/K)+(r-q+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
def _d2(d1,sigma,T): return d1 - sigma*np.sqrt(T)

def bs_price(S,K,r,q,sigma,T,typ="call"):
    if T<=0 or sigma<=0:
        dq,dr=np.exp(-q*T),np.exp(-r*T)
        return max(0.0,S*dq-K*dr) if typ=="call" else max(0.0,K*dr-S*dq)
    d1=_d1(S,K,r,q,sigma,T);d2=_d2(d1,sigma,T)
    dq,dr=np.exp(-q*T),np.exp(-r*T)
    return S*dq*norm.cdf(d1)-K*dr*norm.cdf(d2) if typ=="call" else K*dr*norm.cdf(-d2)-S*dq*norm.cdf(-d1)

def implied_vol(price,S,K,r,q,T,typ="call"):
    intrinsic=max(0.0,(S-K) if typ=="call" else (K-S))
    if price<=0 or price<intrinsic:return None
    f=lambda sig:bs_price(S,K,r,q,sig,T,typ)-price
    try:return brentq(f,1e-6,5.0,maxiter=200)
    except: return None

def greeks(S,K,r,q,sigma,T,typ="call"):
    if T<=0 or sigma<=0:
        return {"Delta":0.0,"Gamma":0.0,"Vega":0.0,"Theta":0.0,"Rho":0.0}
    d1=_d1(S,K,r,q,sigma,T); d2=_d2(d1,sigma,T)
    dq,dr=np.exp(-q*T),np.exp(-r*T)
    if typ=="call":
        delta=dq*norm.cdf(d1)
        theta=(- (S*dq*norm.pdf(d1)*sigma)/(2*np.sqrt(T)) - r*K*dr*norm.cdf(d2) + q*S*dq*norm.cdf(d1))
        rho=K*T*dr*norm.cdf(d2)
    else:
        delta=-dq*norm.cdf(-d1)
        theta=(- (S*dq*norm.pdf(d1)*sigma)/(2*np.sqrt(T)) + r*K*dr*norm.cdf(-d2) - q*S*dq*norm.cdf(-d1))
        rho=-K*T*dr*norm.cdf(-d2)
    gamma=(dq*norm.pdf(d1))/(S*sigma*np.sqrt(T))
    vega=S*dq*norm.pdf(d1)*np.sqrt(T)
    return {"Delta":float(delta),"Gamma":float(gamma),"Vega":float(vega)/100.0,"Theta":float(theta)/365.0,"Rho":float(rho)/100.0}

def pick_nearest_expiry(ticker,target_days=30):
    t=yf.Ticker(ticker);exps=t.options
    if not exps:return None
    today=datetime.now(timezone.utc).date()
    def dd(e): d=datetime.strptime(e,"%Y-%m-%d").date(); return abs((d-today).days-target_days)
    return sorted(exps,key=dd)[0]

def atm_iv_30d(ticker,r=0.02,q=0.00,prefer_proxy=True):
    tried=[]
    for tk in [ticker]+([IV_PROXIES.get(ticker)] if prefer_proxy and IV_PROXIES.get(ticker) else []):
        if not tk:continue
        tried.append(tk)
        exp=pick_nearest_expiry(tk,30)
        if not exp:continue
        hist=yf.download(tk,period="3mo",auto_adjust=True,progress=False)
        if hist.empty:continue
        S=to_scalar(as_series(hist["Close"]).iloc[-1])
        chain=yf.Ticker(tk).option_chain(exp);calls=chain.calls.copy()
        if calls.empty:continue
        calls["mid"]=(calls["bid"].fillna(0)+calls["ask"].fillna(0))/2
        calls["mid"]=calls["mid"].where(calls["mid"]>0,calls["lastPrice"].fillna(0))
        calls=calls[calls["mid"]>0]
        if calls.empty:continue
        k_row=calls.iloc[(calls["strike"]-S).abs().argmin()]
        K,mid=to_scalar(k_row["strike"]),to_scalar(k_row["mid"])
        T=yearfrac(datetime.now(timezone.utc),datetime.strptime(exp,"%Y-%m-%d"))
        iv=implied_vol(mid,S,K,r,q,T,"call")
        if iv is not None:return float(iv),tk,exp,S,K
    return np.nan,(tried[-1] if tried else None),None,np.nan,np.nan

# ========= Data Fetch =========
@st.cache_data(ttl=300,show_spinner=False)
def fetch_close_series_live(ticker:str,period:str,interval:str)->pd.Series:
    try:
        df=yf.download(ticker,period=period,interval=interval,auto_adjust=True,progress=False)
        if df is not None and not df.empty and "Close" in df.columns:
            s=df["Close"]; 
            if isinstance(s,pd.DataFrame): s=s.iloc[:,0]
            return as_series(s.dropna(),name=ticker)
    except Exception: pass
    df=yf.download(ticker,period="5y",interval="1d",auto_adjust=True,progress=False)
    if df is None or df.empty or "Close" not in df.columns:
        return pd.Series(name=ticker,dtype=float)
    s=df["Close"]
    if isinstance(s,pd.DataFrame): s=s.iloc[:,0]
    return as_series(s.dropna(),name=ticker)

# ========= Sidebar =========
st.sidebar.header("Portfolio-Einstellungen")
codes_text=st.sidebar.text_area("WKN/ISIN/Ticker (kommagetrennt)","IE00BCHWNT26, IE00BF8HV600, IE00BLNMYC90")
weights_text=st.sidebar.text_input("Gewichte in % (kommagetrennt)","32.08, 34.66, 23.76")
interval=st.sidebar.selectbox("Intervall (max. live)",["1m","2m","5m","15m","30m","60m","1d"],index=0)
period=st.sidebar.text_input("Periode",interval_defaults(interval))
risk_free=st.sidebar.number_input("Risikofreier Jahreszins (für IV)",0.0,0.2,0.02,0.005)

st.sidebar.markdown("---")
st.sidebar.header("Risiko & Szenarien")
bench_choice = st.sidebar.selectbox("Benchmark für Tracking Error", ["SPY", "Eigener Ticker"], index=0)
bench_ticker = "SPY" if bench_choice=="SPY" else st.sidebar.text_input("Eigener Benchmark-Ticker", "SPY").strip().upper()
var_conf = st.sidebar.selectbox("VaR/ES Konfidenz", [0.95, 0.99], index=0)
var_window = st.sidebar.number_input("VaR/ES Fenster (Punkte)", min_value=50, max_value=5000, value=400, step=50)
shock1 = st.sidebar.number_input("Markt-Schock 1 (%)", value=-5.0, step=0.5)
shock2 = st.sidebar.number_input("Markt-Schock 2 (%)", value=-10.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.header("Charts & Refresh")
auto_refresh=st.sidebar.checkbox("Auto-Refresh",True)
refresh_sec=st.sidebar.slider("Refresh-Intervall (Sekunden)",10,300,60,10)
lookback_points = st.sidebar.number_input("Lookback (Punkte)", min_value=50, max_value=5000, value=400, step=50)
sma_on = st.sidebar.checkbox("SMA20/SMA50 anzeigen", True)
roll_win = st.sidebar.number_input("Rolling-Fenster (Vol/Korr)", min_value=10, max_value=300, value=60, step=10)

run_btn=st.sidebar.button("Berechnen / Aktualisieren")

if auto_refresh:
    try:
        if hasattr(st,"autorefresh") and callable(st.autorefresh):
            st.autorefresh(interval=refresh_sec*1000,key="auto_refresh")
        else:
            st.caption(f"🔁 Auto-Refresh alle {refresh_sec} Sekunden aktiviert.")
    except Exception:
        st.caption(f"🔁 Auto-Refresh alle {refresh_sec} Sekunden aktiviert.")

st.markdown(f"**Letzte Aktualisierung:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | **Intervall:** {interval} | **Periode:** {period}")
st.markdown("**Hinweis:** 1-Minuten-Daten nur während Handelszeiten und wenige Tage rückwirkend. Fallback auf Daily möglich.")

# ========= Portfolio-Berechnung =========
def compute_portfolio(codes,weights_pct,period,interval,rf,bench_ticker):
    symbols=[resolve_symbol(c) for c in codes]
    series=[fetch_close_series_live(tk,period,interval) for tk in symbols]
    series=[s for s in series if s is not None and not s.empty]
    if not series: 
        st.error("Keine Kursdaten geladen."); 
        return None
    prices=pd.concat(series,axis=1,join="outer").dropna(how="all")
    symbols=list(prices.columns)

    w=pd.Series(weights_pct,index=symbols)
    w=w/(w.sum() if w.sum()!=0 else 1.0)

    rets=prices.pct_change().dropna()
    for c in rets.columns: rets[c]=as_series(rets[c],name=c)

    # RV annualisiert
    vols_ann = rets.std().apply(lambda x: annualize_from_interval(float(x), interval)).reindex(symbols)

    # IV (30T)
    iv_rows={}
    for tk in symbols:
        iv,src,exp,S,K=atm_iv_30d(tk,r=rf)
        iv_rows[tk]={"IV":iv,"Proxy":(src if src and src!=tk else "")}
    iv_df=pd.DataFrame(iv_rows).T.reindex(symbols)

    # Beta vs SPY
    spy=fetch_close_series_live("SPY",period,interval).pct_change().dropna()
    spy=as_series(spy,name="spy")
    betas={}
    for tk in rets.columns:
        asset=as_series(rets[tk],name="asset")
        xy=pd.concat([asset,spy],axis=1,join="inner").dropna()
        betas[tk]=float(xy.cov().loc["asset","spy"]/xy["spy"].var()) if len(xy)>50 and xy["spy"].var()>0 else np.nan
    beta=pd.Series(betas).reindex(symbols)
    port_beta=float((w*beta).sum(skipna=True))

    # Portfolio-Vol (Intervall → annualisiert), VaR/ES, TE
    Sigma_i=rets.tail(2000).cov()
    port_var_i=float(w.values@Sigma_i.values@w.values.T)
    port_vol_i=math.sqrt(port_var_i) if port_var_i>0 else np.nan
    port_vol_ann=annualize_from_interval(port_vol_i,interval)

    port_ret = (rets @ w).dropna()

    # VaR/ES (historisch)
    if len(port_ret) >= var_window:
        window_rets = port_ret.tail(var_window)
        alpha = 1 - float(var_conf)
        var_hist = -np.nanquantile(window_rets, alpha)
        tail = window_rets[window_rets <= np.nanquantile(window_rets, alpha)]
        es_hist = -float(tail.mean()) if len(tail) else np.nan
    else:
        var_hist = np.nan; es_hist = np.nan

    # Benchmark + TE (annualisiert)
    bench = fetch_close_series_live(bench_ticker, period, interval)
    bench_ret = as_series(bench.pct_change().dropna(), name="bench")
    aligned = pd.concat([port_ret, bench_ret], axis=1, join="inner").dropna()
    aligned.columns = ["port", "bench"]
    diff = aligned["port"] - aligned["bench"]
    te_ann = annualize_from_interval(float(diff.std()), interval) if len(diff)>5 else np.nan

    # Risikobeiträge (Std.-Anteil)
    if np.isfinite(port_var_i) and port_var_i > 0:
        m = Sigma_i.values @ w.values
        rc_std = (w.values * m) / (port_vol_i if port_vol_i>0 else np.nan)
        rc = pd.Series(rc_std, index=symbols, name="Risikobeitrag (Std.-Anteil)")
    else:
        rc = pd.Series([np.nan]*len(symbols), index=symbols, name="Risikobeitrag (Std.-Anteil)")

    corr = rets.corr()
    last_px = prices.ffill().iloc[-1]
    table = pd.DataFrame({
        "Ticker":symbols,
        "Aktueller Kurs":[float(last_px[tk]) if tk in last_px else np.nan for tk in symbols],
        "Beobachtete Schwankung (annualisiert)":vols_ann.values,
        "Erwartete Schwankung 30 Tage (annualisiert)":iv_df["IV"].values,
        "Volatilitäts-Risikoprämie (IV − RV)":(iv_df["IV"]-vols_ann).values,
        "IV-Proxy":iv_df["Proxy"].values,
        "Gewicht":w.values,
        "Beta (kurzfristig)":beta.values
    })

    return (prices, rets, table, w, beta, port_beta, port_vol_ann,
            var_hist, es_hist, corr, rc, port_ret, bench_ret, te_ann)

# ========= Stresstests =========
def scenarios_parametric(w, beta, shock_pct_list):
    out = []
    port_beta = float((w * beta).sum(skipna=True))
    for s in shock_pct_list:
        port_ret = port_beta * (s/100.0)
        out.append({"Szenario": f"Markt-Schock {s:.1f}%", "Portfolio-Rendite": port_ret})
    return pd.DataFrame(out)

def scenarios_historical(port_ret: pd.Series):
    rows = []
    if port_ret is None or port_ret.empty:
        return pd.DataFrame(columns=["Szenario","Portfolio-Rendite"])
    worst_day = float(port_ret.min()); rows.append({"Szenario":"Schlimmster Tag (Historie)", "Portfolio-Rendite": worst_day})
    if len(port_ret) >= 5:
        wr = port_ret.rolling(5).sum().dropna()
        if not wr.empty:
            rows.append({"Szenario":"Schlimmste Woche (Historie)", "Portfolio-Rendite": float(wr.min())})
    # Corona-Fenster (falls im Ausschnitt)
    try:
        daily_port = port_ret.copy()
        daily_port.index = pd.to_datetime(daily_port.index)
        slice_2020 = daily_port[(daily_port.index >= "2020-02-15") & (daily_port.index <= "2020-03-31")]
        if not slice_2020.empty:
            rows.append({"Szenario":"Corona-Crash (Feb–Mär 2020)", "Portfolio-Rendite": float(slice_2020.sum())})
    except Exception:
        pass
    return pd.DataFrame(rows)

# ========= Faktoren (Proxy-basiert) =========
def factor_proxies_df(period, interval):
    # Markt: SPY, Small: IWM, Value: VLUE (proxy)
    tickers = {"MKT":"SPY", "SMB":"IWM", "HML":"VLUE"}
    data = {}
    for k,t in tickers.items():
        s = fetch_close_series_live(t, period, interval)
        data[k] = s.pct_change()
    return pd.DataFrame(data).dropna(how="any")

def factor_attribution(port_ret: pd.Series, rf: float, period: str, interval: str):
    # Proxy: Excess-Returns ggü. rf ~ 0 pro Intervall (vereinfachend)
    fac = factor_proxies_df(period, interval)
    df = pd.concat([port_ret, fac], axis=1, join="inner").dropna()
    df.columns = ["PORT","MKT","SMB","HML"]
    if len(df)<60:
        return None, None
    X = df[["MKT","SMB","HML"]].values
    y = df["PORT"].values
    X_ = np.column_stack([np.ones(len(X)), X])  # +Intercept
    beta_hat, *_ = np.linalg.lstsq(X_, y, rcond=None)
    coeffs = pd.Series(beta_hat, index=["Alpha","MKT","SMB","HML"])
    # Erklärte Varianz
    yhat = X_ @ beta_hat
    r2 = 1 - np.var(y - yhat)/np.var(y)
    return coeffs, float(r2)

# ========= Greeks / Black–Scholes Seite =========
def greeks_page():
    st.subheader("Greeks & Black–Scholes (Einzel-Option)")
    tk = st.text_input("Underlying (Ticker)", "SPY").strip().upper()
    s_hist = yf.download(tk, period="6mo", auto_adjust=True, progress=False)
    if s_hist.empty:
        st.warning("Keine Kursdaten.")
        return
    S = float(as_series(s_hist["Close"]).iloc[-1])
    col1,col2,col3,col4,col5 = st.columns(5)
    with col1:
        typ = st.selectbox("Optionstyp", ["call","put"], index=0)
    with col2:
        K = st.number_input("Strike", value=round(S,2), step=1.0, format="%.2f")
    with col3:
        days = st.number_input("Restlaufzeit (Kalendertage)", min_value=1, value=30, step=1)
    with col4:
        r = st.number_input("Zinssatz p.a.", min_value=0.0, max_value=0.2, value=0.02, step=0.005)
    with col5:
        q = st.number_input("Div.-Rendite p.a.", min_value=0.0, max_value=0.2, value=0.00, step=0.005)
    T = days/365.0

    # IV-Schätzung at-the-money (optional)
    iv_est = np.nan
    try:
        exp = pick_nearest_expiry(tk, days)
        if exp:
            chain = yf.Ticker(tk).option_chain(exp).calls
            if not chain.empty:
                mid = (chain["bid"].fillna(0)+chain["ask"].fillna(0))/2
                chain = chain.assign(mid=mid.where(mid>0, chain["lastPrice"].fillna(0)))
                chain = chain[chain["mid"]>0]
                row = chain.iloc[(chain["strike"]-S).abs().argmin()]
                iv_est = implied_vol(to_scalar(row["mid"]), S, to_scalar(row["strike"]), r, q, T, "call")
    except Exception:
        pass
    sigma = st.number_input("Volatilität p.a. (σ)", min_value=0.01, max_value=2.0, value=float(iv_est) if np.isfinite(iv_est) else 0.2, step=0.01, format="%.3f")

    price = bs_price(S, K, r, q, sigma, T, typ=typ)
    g = greeks(S, K, r, q, sigma, T, typ=typ)

    kpi1,kpi2,kpi3,kpi4,kpi5 = st.columns(5)
    kpi1.metric("Preis", f"{price:.2f}")
    kpi2.metric("Delta", f"{g['Delta']:.3f}")
    kpi3.metric("Gamma", f"{g['Gamma']:.5f}")
    kpi4.metric("Vega (per 1%)", f"{g['Vega']:.3f}")
    kpi5.metric("Theta (pro Tag)", f"{g['Theta']:.3f}")

    # Kurven über S
    gridS = np.linspace(0.7*S, 1.3*S, 60)
    prices = [bs_price(s, K, r, q, sigma, T, typ) for s in gridS]
    deltas = [greeks(s, K, r, q, sigma, T, typ)["Delta"] for s in gridS]
    gammas = [greeks(s, K, r, q, sigma, T, typ)["Gamma"] for s in gridS]
    vegas  = [greeks(s, K, r, q, sigma, T, typ)["Vega"] for s in gridS]
    thetas = [greeks(s, K, r, q, sigma, T, typ)["Theta"] for s in gridS]

    c1,c2 = st.columns(2)
    with c1:
        fig,ax=plt.subplots(figsize=(6,3)); ax.plot(gridS, prices); ax.set_title("Optionspreis vs. Underlying"); ax.grid(True,alpha=0.3)
        st.pyplot(fig)
    with c2:
        fig,ax=plt.subplots(figsize=(6,3)); ax.plot(gridS, deltas,label="Delta"); ax.plot(gridS, gammas,label="Gamma")
        ax.legend(); ax.set_title("Delta & Gamma vs. Underlying"); ax.grid(True,alpha=0.3); st.pyplot(fig)
    c3,c4 = st.columns(2)
    with c3:
        fig,ax=plt.subplots(figsize=(6,3)); ax.plot(gridS, vegas); ax.set_title("Vega vs. Underlying"); ax.grid(True,alpha=0.3)
        st.pyplot(fig)
    with c4:
        fig,ax=plt.subplots(figsize=(6,3)); ax.plot(gridS, thetas); ax.set_title("Theta vs. Underlying"); ax.grid(True,alpha=0.3)
        st.pyplot(fig)

# ========= GARCH Forecast =========
def garch_or_ewma_vol(ret_series: pd.Series, horizon_days=1):
    """Gibt (annualisierte) 1-Tages-Vol-Forecast zurück."""
    r = ret_series.dropna()
    if len(r)<200: return np.nan, "Zu wenig Daten"
    if HAS_ARCH:
        try:
            # ARCH arbeitet i.d.R. mit Prozent * 100; wir lassen hier in Dezimal, arch kann beide
            am = arch_model(r*100.0, vol="Garch", p=1, q=1, mean="Zero", dist="normal")
            res = am.fit(disp="off")
            f = res.forecast(horizon=horizon_days)
            # Varianz in Prozent^2; zurück zu Dezimal:
            var_daily = float(f.variance.values[-1,0]) / (100.0**2)
            vol_daily = math.sqrt(var_daily)
            vol_ann = vol_daily * math.sqrt(252)
            return vol_ann, "GARCH(1,1)"
        except Exception:
            pass
    # Fallback EWMA
    lam = 0.94
    v = 0.0
    for x in r[::-1]:
        v = lam*v + (1-lam)*(x**2)
    vol_daily = math.sqrt(v)
    return vol_daily*math.sqrt(252), "EWMA(λ=0.94)"

# ========= PDF-Export (mit Charts) =========
def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

def build_pdf(kpi_text: str, tables: dict, images: list, filename="portfolio_report.pdf") -> bytes:
    pdf = FPDF(unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)

    # Titel
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Portfolio-Report", ln=1, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Erstellt am {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=1, align="C")
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 11)
    for line in kpi_text.split("\n"):
        pdf.multi_cell(0, 5, line)

    # Tabellen (als simple Textblöcke)
    for title, df in tables.items():
        if df is None or df.empty: continue
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, title, ln=1)
        pdf.set_font("Courier", "", 9)
        # Begrenze Zeilen
        head = " | ".join(list(df.columns))
        pdf.multi_cell(0, 5, head)
        pdf.ln(1)
        for i,(_, row) in enumerate(df.iterrows()):
            line = " | ".join([str(row[c]) for c in df.columns])
            pdf.multi_cell(0, 5, line)
            if i>40:
                pdf.multi_cell(0,5,"... (gekürzt)")
                break

    # Bilder
    for title, png_bytes in images:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, title, ln=1)
        # Bild einfügen (max Breite 180mm)
        img_path = io.BytesIO(png_bytes.getvalue())
        pdf.image(img_path, w=180)

    return pdf.output(dest="S").encode("latin-1", errors="ignore")

# ========= UI: Tabs =========
main_tab, greeks_tab = st.tabs(["📈 Portfolio", "🧮 Greeks & Black-Scholes"])

with greeks_tab:
    greeks_page()

with main_tab:
    if st.button("Berechnen / Aktualisieren", use_container_width=True):
        run_btn = True

    if 'run_btn' not in locals():
        run_btn = False

    if run_btn:
        try:
            codes=[c.strip() for c in codes_text.split(",") if c.strip()]
            weights=[float(x.strip().replace(",", ".")) for x in weights_text.split(",")]
            if len(weights)!=len(codes): st.error("Anzahl Gewichte passt nicht."); st.stop()

            out = compute_portfolio(codes, weights, period, interval, risk_free, bench_ticker)
            if out is None: st.stop()
            (prices, rets, table, w, beta_series, port_beta, port_vol_ann,
             var_hist, es_hist, corr, rc, port_ret, bench_ret, te_ann) = out

            # KPIs
            avg_corr = float(np.nanmean(corr.values[np.triu_indices_from(corr,1)])) if corr.shape[0]>1 else 0.0
            hhi = float(np.sum(w.values**2))
            # Einfaches Rating
            v=port_vol_ann or 0;b=abs(port_beta or 0);c=(avg_corr+1)/2;h=hhi or 0
            v_norm=min(v/0.25,1.0);b_norm=min(b/1.5,1.0);c_norm=c;h_norm=float(np.clip((h-0.2)/(0.8-0.2),0,1))
            score=100*(0.4*v_norm+0.25*b_norm+0.2*c_norm+0.15*h_norm)
            label="niedrig" if score<33 else "mittel" if score<66 else "hoch"

            c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
            c1.metric("Rating",label.upper(),f"Score {score:.0f}/100")
            c2.metric("Volatilität p.a.",f"{(port_vol_ann*100):.2f}%")
            c3.metric("Beta (gewichtet)",f"{port_beta:.2f}")
            c4.metric("Durchschn. Korr.",f"{avg_corr:.2f}")
            c5.metric("Tracking Error p.a.",f"{(te_ann*100):.2f}%")
            c6.metric(f"VaR {int(var_conf*100)}% (hist.)", f"{(var_hist*100):.2f}%")
            c7.metric(f"ES {int(var_conf*100)}% (hist.)", f"{(es_hist*100):.2f}%")

            st.divider()

            # Equity & Drawdown
            st.subheader("Portfolio-Ansicht")
            eq = (1 + port_ret.fillna(0)).add(1).cumprod()
            dd = drawdown(eq)
            colA, colB = st.columns(2)
            with colA:
                fig, ax = plt.subplots(figsize=(7,3))
                eq.tail(lookback_points).plot(ax=ax); ax.set_title("Portfolio-Equity (normalisiert)")
                ax.grid(True, alpha=0.3); buf_eq = fig_to_png_bytes(fig); st.pyplot(fig)
            with colB:
                fig, ax = plt.subplots(figsize=(7,3))
                dd.tail(lookback_points).plot(ax=ax, color="tab:red")
                ax.set_title(f"Portfolio-Drawdown (max: {dd.min():.1%})")
                ax.grid(True, alpha=0.3); buf_dd = fig_to_png_bytes(fig); st.pyplot(fig)

            # Risikobeiträge
            st.subheader("Risikobeiträge (Std.-Anteil)")
            fig, ax = plt.subplots(figsize=(8,3))
            (rc / np.nansum(rc)).plot(kind="bar", ax=ax); ax.set_ylim(0,1)
            ax.grid(axis="y", alpha=0.2); buf_rc = fig_to_png_bytes(fig); st.pyplot(fig)

            st.divider()

            # Stresstests
            st.subheader("Szenario-Stresstests")
            scen_param = scenarios_parametric(w, beta_series, [shock1, shock2])
            scen_hist = scenarios_historical(port_ret)
            scen_all = pd.concat([scen_param, scen_hist], ignore_index=True)

            colS1, colS2 = st.columns([1,1])
            with colS1:
                st.markdown("**Parametrische Szenarien (über Portfolio-Beta):**")
                sp = scen_param.assign(**{"Portfolio-Veränderung (%)": scen_param["Portfolio-Rendite"]*100}).drop(columns=["Portfolio-Rendite"])
                st.dataframe(sp,use_container_width=True)
            with colS2:
                st.markdown("**Historische Stresstests:**")
                if not scen_hist.empty:
                    sh = scen_hist.assign(**{"Portfolio-Veränderung (%)": scen_hist["Portfolio-Rendite"]*100}).drop(columns=["Portfolio-Rendite"])
                    st.dataframe(sh,use_container_width=True)
                else:
                    st.info("Nicht genug Historie für historische Stresstests.")

            if not scen_all.empty:
                fig, ax = plt.subplots(figsize=(8,3))
                (scen_all.set_index("Szenario")["Portfolio-Rendite"]*100).plot(kind="bar", ax=ax, color=["tab:orange"]*len(scen_all))
                ax.set_ylabel("Portfolio-Veränderung (%)"); ax.grid(axis="y", alpha=0.2)
                buf_scen = fig_to_png_bytes(fig); st.pyplot(fig)
            else:
                buf_scen = io.BytesIO()

            st.divider()

            # Kennzahlen je ETF
            st.subheader("Kennzahlen je ETF (live)")
            t=table.copy()
            t["Gewicht"]=t["Gewicht"].map(lambda v:f"{v*100:.2f}%")
            t["Beta (kurzfristig)"]=t["Beta (kurzfristig)"].map(lambda v: f"{v:.2f}" if np.isfinite(v) else "—")
            for c in ["Beobachtete Schwankung (annualisiert)","Erwartete Schwankung 30 Tage (annualisiert)","Volatilitäts-Risikoprämie (IV − RV)"]:
                t[c]=t[c].map(lambda v:f"{v*100:.2f}%" if np.isfinite(v) else "—")
            t["Aktueller Kurs"]=t["Aktueller Kurs"].map(lambda v:f"{v:.2f}" if np.isfinite(v) else "—")
            st.dataframe(t,use_container_width=True)

            st.divider()

            # Korrelationen & Beta
            st.subheader("Korrelationen & Beta")
            cA, cB = st.columns(2)
            with cA:
                fig2,ax2=plt.subplots(figsize=(6,4))
                im=ax2.imshow(corr.values,aspect="auto")
                ax2.set_xticks(np.arange(len(corr.columns)));ax2.set_yticks(np.arange(len(corr.index)))
                ax2.set_xticklabels(corr.columns,rotation=45,ha="right");ax2.set_yticklabels(corr.index)
                cb=fig2.colorbar(im);cb.set_label("Korrelationskoeffizient")
                ax2.set_title("Korrelationen der Renditen")
                buf_corr = fig_to_png_bytes(fig2); st.pyplot(fig2)
            with cB:
                fig3,ax3=plt.subplots(figsize=(6,4))
                beta_series.plot(kind="bar", ax=ax3)
                ax3.set_ylabel("Beta (kurzfristig, vs. SPY)"); ax3.set_title("Marktrisiko (Beta) je ETF")
                ax3.grid(axis="y", alpha=0.3); buf_beta = fig_to_png_bytes(fig3); st.pyplot(fig3)

            st.divider()

            # Faktor-Attribution (Proxy)
            st.subheader("Faktor-Attribution (Proxy)")
            coeffs, r2 = factor_attribution(port_ret, risk_free, period, interval)
            if coeffs is not None:
                st.write("Regressionskoeffizienten (PORT ~ α + MKT + SMB + HML):")
                st.dataframe(coeffs.to_frame("Koeffizient").T)
                st.write(f"Erklärte Varianz R²: {r2:.2f}")
            else:
                st.info("Zu wenig Daten für stabile Faktoren-Attribution (mind. ~60 Punkte empfohlen).")

            st.divider()

            # GARCH/EWMA Forecast
            st.subheader("Volatilitäts-Forecast (1-Tag)")
            vol_fc, model_name = garch_or_ewma_vol(port_ret)
            st.metric("Forecast-Volatilität p.a.", f"{(vol_fc*100):.2f}%", model_name)

            # Vol-Chart (RV vs. IV)
            st.subheader("Schwankung: Beobachtet vs. vom Markt erwartet (IV)")
            fig1,ax1=plt.subplots(figsize=(8,4))
            x=np.arange(len(table))
            ax1.bar(x-0.2,table["Beobachtete Schwankung (annualisiert)"]*100,width=0.4,label="Beobachtet")
            ax1.bar(x+0.2,table["Erwartete Schwankung 30 Tage (annualisiert)"]*100,width=0.4,label="Erwartet (IV)")
            ax1.set_xticks(x);ax1.set_xticklabels(table["Ticker"])
            ax1.set_ylabel("%");ax1.legend()
            buf_vol = fig_to_png_bytes(fig1); st.pyplot(fig1)

            # ========= PDF Export =========
            st.subheader("📄 PDF-Export")
            kpi_text = (
                f"Rating: {label.upper()} (Score {score:.0f}/100)\n"
                f"Volatilität p.a.: {(port_vol_ann*100):.2f}%\n"
                f"Beta (gewichtet): {port_beta:.2f}\n"
                f"Durchschnittskorrelation: {avg_corr:.2f}\n"
                f"Tracking Error p.a.: {(te_ann*100):.2f}%\n"
                f"VaR {int(var_conf*100)}% (hist.): {(var_hist*100):.2f}% | ES: {(es_hist*100):.2f}%\n"
                f"Vol-Forecast (1T): {(vol_fc*100):.2f}% via {model_name}"
            )
            # Tabelle für PDF (kompakt)
            table_pdf = table.copy()
            table_pdf["Gewicht"] = (table_pdf["Gewicht"]*100).map(lambda v:f"{v:.2f}%")
            for c in ["Beobachtete Schwankung (annualisiert)","Erwartete Schwankung 30 Tage (annualisiert)","Volatilitäts-Risikoprämie (IV − RV)"]:
                table_pdf[c] = (table_pdf[c]*100).map(lambda v:f"{v:.2f}%" if np.isfinite(v) else "—")
            table_pdf["Aktueller Kurs"] = table_pdf["Aktueller Kurs"].map(lambda v:f"{v:.2f}" if np.isfinite(v) else "—")
            # Szenario-Tabelle
            scen_pdf = pd.DataFrame()
            if 'scen_all' in locals() and not scen_all.empty:
                scen_pdf = scen_all.copy()
                scen_pdf["Portfolio-Veränderung (%)"] = (scen_pdf["Portfolio-Rendite"]*100).map(lambda v:f"{v:.2f}%")
                scen_pdf = scen_pdf[["Szenario","Portfolio-Veränderung (%)"]]

            images = [
                ("Portfolio-Equity", buf_eq),
                ("Portfolio-Drawdown", buf_dd),
                ("Risikobeiträge", buf_rc),
                ("Szenarien", buf_scen),
                ("Korrelationen", buf_corr),
                ("Beta je ETF", buf_beta),
                ("RV vs. IV", buf_vol),
            ]

            if st.button("PDF generieren"):
                pdf_bytes = build_pdf(
                    kpi_text=kpi_text,
                    tables={
                        "Kennzahlen je ETF": table_pdf,
                        "Szenario-Ergebnisse": scen_pdf
                    },
                    images=images,
                    filename="portfolio_report.pdf"
                )
                st.download_button("Download PDF", data=pdf_bytes, file_name="portfolio_report.pdf", mime="application/pdf")

            st.subheader("Download CSV")
            st.download_button("📥 CSV-Export (Kennzahlen)", data=table.to_csv(index=False).encode("utf-8"),
                               file_name="portfolio_metrics_live.csv", mime="text/csv")

        except Exception as e:
            st.exception(e)
    else:
        st.info("Gib links deine Codes & Gewichte ein und klicke **Berechnen / Aktualisieren**. "
                "Aktiviere Auto-Refresh für Live-Updates.")
