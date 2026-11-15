"""
Earth System Trend Detective
NASA Space Apps Challenge 2026 - Bandung

Data: (a) NASA Giovanni time-series CSV (bundled "dataset NASA.csv" or uploaded)
      https://giovanni.gsfc.nasa.gov/
      (b) NASA POWER monthly point API (automatic)
      https://power.larc.nasa.gov/docs/services/api/temporal/monthly/
Stats: Mann-Kendall trend test + Sen's slope (standard for climate time series)

Run:  python -m pip install streamlit requests pandas numpy scipy
      python -m streamlit run app.py
"""
import io
import re
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import requests
import streamlit as st
from scipy.stats import norm

POWER_URL = "https://power.larc.nasa.gov/api/temporal/monthly/point"
BUNDLED_CSV = Path(__file__).parent / "dataset NASA.csv"

VARIABLES = {
    "Air temperature at 2 m (T2M)": ("T2M", "°C"),
    "Precipitation (PRECTOTCORR)": ("PRECTOTCORR", "mm/day"),
    "Surface solar radiation (ALLSKY_SFC_SW_DWN)": ("ALLSKY_SFC_SW_DWN", "kWh/m²/day"),
    "Relative humidity at 2 m (RH2M)": ("RH2M", "%"),
    "Wind speed at 10 m (WS10M)": ("WS10M", "m/s"),
}

LOCATIONS = {
    "Bandung": (-6.9175, 107.6191),
    "Jakarta": (-6.2088, 106.8456),
    "Surabaya": (-7.2575, 112.7521),
    "Medan": (3.5952, 98.6722),
    "Pontianak": (-0.0263, 109.3425),
    "Makassar": (-5.1477, 119.4327),
    "Jayapura": (-2.5337, 140.7181),
}

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

MODE_COMBINED = "Giovanni CSV + NASA POWER (recommended)"
MODE_UPLOAD = "Upload NASA Giovanni CSV"
MODE_POWER = "NASA POWER only"

CSV_NAME = "Bandung (MERRA-2, CSV)"
POWER_BANDUNG_NAME = "Bandung (POWER)"


# ---------------------------------------------------------------- data ----
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_power(param: str, lat: float, lon: float, start: int, end: int):
    """Download monthly data from NASA POWER. Returns (monthly_series, annual_series)."""
    params = {
        "parameters": param,
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": start,
        "end": end,
        "format": "JSON",
    }
    r = requests.get(POWER_URL, params=params, timeout=60)
    r.raise_for_status()
    raw = r.json()["properties"]["parameter"][param]
    return monthly_from_power(raw), annual_from_monthly(raw)


def annual_from_monthly(raw: dict) -> pd.Series:
    """Keys look like 'YYYYMM'. MM == 13 is NASA's own annual value. -999 = missing."""
    out = {}
    for key, val in raw.items():
        year, month = int(key[:4]), int(key[4:])
        if month == 13 and val != -999:
            out[year] = float(val)
    return pd.Series(out, name="value").sort_index()


def monthly_from_power(raw: dict) -> pd.Series:
    out = {}
    for key, val in raw.items():
        year, month = int(key[:4]), int(key[4:])
        if 1 <= month <= 12 and val != -999:
            out[pd.Timestamp(year=year, month=month, day=1)] = float(val)
    return pd.Series(out, name="value").sort_index()


def read_giovanni(text: str):
    """Parse a Giovanni (or any 'time,value') CSV. Returns (time-indexed series, unit)."""
    lines = text.splitlines()
    unit, hdr = "", None
    for i, ln in enumerate(lines):
        low = ln.strip().lower()
        if not unit:
            m = re.match(r"\W*units\W*[=,:]\s*(.+)", ln.strip(), flags=re.I)
            if m:
                unit = m.group(1).strip().strip('"')
        if low.startswith("time") and "," in low:
            hdr = i
            break
    if hdr is None:
        hdr = 0
    df = pd.read_csv(io.StringIO("\n".join(lines[hdr:])))
    t = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    v = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    d = pd.DataFrame({"t": t, "v": v}).dropna()
    d = d[d["v"] > -999]  # drop common fill values (-999, -9999, ...)
    if d.empty:
        raise ValueError("No valid rows found (expected columns: time,value).")
    d = d.sort_values("t")
    return pd.Series(d["v"].values, index=pd.DatetimeIndex(d["t"]), name="value"), unit


def to_annual(s: pd.Series) -> pd.Series:
    """Annual mean of a time-indexed series; years with <80% coverage are dropped."""
    step = s.index.to_series().diff().dt.days.median()
    expected = 365.0 / step if step and step > 0 else 1
    g = s.groupby(s.index.year).agg(["mean", "count"])
    g = g[g["count"] >= 0.8 * expected]
    return g["mean"].rename("value")


def is_monthly(s: pd.Series) -> bool:
    step = s.index.to_series().diff().dt.days.median()
    return bool(step and 25 <= step <= 35)


def make_dataset(raw: pd.Series, coords=None, source=""):
    """Bundle a raw time series with its annual means (and monthly values if monthly)."""
    return {
        "annual": to_annual(raw),
        "monthly": raw if is_monthly(raw) else None,
        "coords": coords,
        "source": source,
    }


# --------------------------------------------------------------- stats ----
def mann_kendall(y: np.ndarray):
    """Mann-Kendall test (with tie correction). Returns (S, z, p)."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    s = 0
    for i in range(n - 1):
        s += np.sign(y[i + 1:] - y[i]).sum()
    _, counts = np.unique(y, return_counts=True)
    var_s = (n * (n - 1) * (2 * n + 5) - np.sum(counts * (counts - 1) * (2 * counts + 5))) / 18
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0
    p = 2 * (1 - norm.cdf(abs(z)))
    return s, z, p


def sens_slope(x: np.ndarray, y: np.ndarray):
    """Sen's slope = median of all pairwise slopes. Returns (slope, intercept)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    slopes = [
        (y[j] - y[i]) / (x[j] - x[i])
        for i in range(len(x) - 1)
        for j in range(i + 1, len(x))
        if x[j] != x[i]
    ]
    slope = float(np.median(slopes))
    intercept = float(np.median(y - slope * x))
    return slope, intercept


def analyse(series: pd.Series, alpha: float = 0.05) -> dict:
    x, y = series.index.values, series.values
    s, z, p = mann_kendall(y)
    slope, intercept = sens_slope(x, y)
    if p < alpha:
        direction = "increasing" if slope > 0 else "decreasing"
    else:
        direction = "no significant trend"
    return {
        "slope_per_year": slope,
        "slope_per_decade": slope * 10,
        "total_change": slope * (x[-1] - x[0]),
        "p_value": p,
        "significant": p < alpha,
        "direction": direction,
        "intercept": intercept,
    }


# -------------------------------------------------------------- charts ----
def palette(is_temp: bool):
    """(high colour, low colour) for anomaly charts."""
    return ("#d1495b", "#2e86ab") if is_temp else ("#2a9d8f", "#e9a03b")


def trend_chart(series: pd.Series, res: dict, unit: str, color: str):
    df = pd.DataFrame(
        {
            "Year": series.index.values,
            "Observed": series.values,
            "Trend": res["intercept"] + res["slope_per_year"] * series.index.values,
        }
    )
    long = df.melt("Year", var_name="Series", value_name="Value")
    base = alt.Chart(long).encode(
        x=alt.X("Year:Q", axis=alt.Axis(format="d", title=None)),
        y=alt.Y("Value:Q", scale=alt.Scale(zero=False), title=unit),
        tooltip=[alt.Tooltip("Year:Q", format="d"), "Series:N", alt.Tooltip("Value:Q", format=".3f")],
    )
    obs = base.transform_filter(alt.datum.Series == "Observed").mark_line(point=True, color=color)
    trend = base.transform_filter(alt.datum.Series == "Trend").mark_line(
        color="#888888", strokeDash=[6, 4], strokeWidth=2
    )
    return (obs + trend).properties(height=220)


def anomaly_bars(series: pd.Series, unit: str, is_temp: bool):
    base_mean = series.mean()
    df = pd.DataFrame({"Year": series.index.values, "Anomaly": series.values - base_mean})
    hi, lo = palette(is_temp)
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Year:O", axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("Anomaly:Q", title=f"Anomaly vs {series.index.min()}-{series.index.max()} mean ({unit})"),
            color=alt.condition(alt.datum.Anomaly > 0, alt.value(hi), alt.value(lo)),
            tooltip=["Year:O", alt.Tooltip("Anomaly:Q", format="+.3f")],
        )
        .properties(height=280)
    )


def month_year_heatmap(monthly: pd.Series, unit: str, is_temp: bool):
    """Month x year heatmap of anomalies relative to each calendar month's average."""
    m = monthly.copy()
    anom = m - m.groupby(m.index.month).transform("mean")
    df = pd.DataFrame({"Year": m.index.year, "Month": [MONTHS[i - 1] for i in m.index.month], "Anomaly": anom.values})
    lim = float(np.nanmax(np.abs(df["Anomaly"]))) or 1.0
    scheme = "redblue" if is_temp else "brownbluegreen"
    return (
        alt.Chart(df)
        .mark_rect()
        .encode(
            x=alt.X("Year:O", axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("Month:O", sort=MONTHS),
            color=alt.Color(
                "Anomaly:Q",
                scale=alt.Scale(scheme=scheme, reverse=is_temp, domain=[-lim, lim]),
                legend=alt.Legend(title=f"Anomaly ({unit})"),
            ),
            tooltip=["Year:O", "Month:O", alt.Tooltip("Anomaly:Q", format="+.2f")],
        )
        .properties(height=300)
    )


def comparison_bars(rows, unit: str, is_temp: bool):
    hi, _ = palette(is_temp)
    df = pd.DataFrame(
        {
            "Location": [r["Location"] for r in rows],
            "Change": [r[f"Change per decade ({unit})"] for r in rows],
            "Significant": [r["Significant?"] for r in rows],
        }
    )
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            y=alt.Y("Location:N", sort="-x", title=None),
            x=alt.X("Change:Q", title=f"Change per decade ({unit})"),
            color=alt.Color(
                "Significant:N",
                scale=alt.Scale(domain=["Yes", "No"], range=[hi, "#b0b0b0"]),
                legend=alt.Legend(title="Significant?"),
            ),
            tooltip=["Location:N", alt.Tooltip("Change:Q", format="+.3f"), "Significant:N"],
        )
        .properties(height=max(120, 42 * len(df)))
    )


def overlay_anomalies(datasets: dict, unit: str):
    frames = [
        pd.DataFrame({"Year": d["annual"].index.values, "Anomaly": d["annual"].values - d["annual"].mean(), "Location": n})
        for n, d in datasets.items()
    ]
    df = pd.concat(frames)
    return (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("Year:Q", axis=alt.Axis(format="d", title=None)),
            y=alt.Y("Anomaly:Q", title=f"Anomaly vs own mean ({unit})"),
            color=alt.Color("Location:N", legend=alt.Legend(orient="bottom")),
            tooltip=[alt.Tooltip("Year:Q", format="d"), "Location:N", alt.Tooltip("Anomaly:Q", format="+.3f")],
        )
        .properties(height=320)
    )


def validation_chart(a: pd.Series, b: pd.Series, name_a: str, name_b: str, unit: str):
    df = pd.concat([a.rename(name_a), b.rename(name_b)], axis=1).dropna().rename_axis("Year").reset_index()
    long = df.melt("Year", var_name="Source", value_name="Value")
    return (
        alt.Chart(long)
        .mark_line(point=True)
        .encode(
            x=alt.X("Year:Q", axis=alt.Axis(format="d", title=None)),
            y=alt.Y("Value:Q", scale=alt.Scale(zero=False), title=unit),
            color=alt.Color("Source:N", legend=alt.Legend(orient="bottom")),
            tooltip=[alt.Tooltip("Year:Q", format="d"), "Source:N", alt.Tooltip("Value:Q", format=".3f")],
        )
        .properties(height=320)
    )


def trend_map(rows, datasets: dict, unit: str):
    """One dot per coordinate: red = significant increase, blue = significant decrease, grey = no trend."""
    points = {}
    for r in rows:
        coords = datasets[r["Location"]]["coords"]
        if not coords:
            continue
        d = r[f"Change per decade ({unit})"]
        if r["Significant?"] == "Yes":
            color = [209, 73, 91, 200] if d > 0 else [46, 134, 171, 200]
        else:
            color = [150, 150, 150, 200]
        trend_word = r["What's happening"]
        line = f"{r['Location']}: {d:+.3g} {unit}/decade ({trend_word})"
        if coords in points:
            points[coords]["label"] += "\n" + line
        else:
            points[coords] = {"lat": coords[0], "lon": coords[1], "color": color, "label": line}
    if not points:
        st.info("No coordinates available for a map.")
        return
    df = pd.DataFrame(points.values())
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position="[lon, lat]",
        get_fill_color="color",
        get_radius=30000,
        radius_min_pixels=10,
        radius_max_pixels=30,
        pickable=True,
    )
    zoom = 9 if len(df) == 1 else 4
    view = pdk.ViewState(latitude=float(df["lat"].mean()), longitude=float(df["lon"].mean()), zoom=zoom)
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, tooltip={"text": "{label}"}), height=480)
    st.caption("Red = significant increase · Blue = significant decrease · Grey = no significant trend. Hover for details.")


# ------------------------------------------------------------------ UI ----
def load_bundled_csv():
    raw, _ = read_giovanni(BUNDLED_CSV.read_text(encoding="utf-8", errors="ignore"))
    if raw.mean() > 200:  # Kelvin -> degrees C
        raw = raw - 273.15
    return raw


def collect_combined():
    """Bundled Giovanni CSV (Bandung, T2M) + NASA POWER points for validation and comparison."""
    if not BUNDLED_CSV.exists():
        st.error(f"File not found: {BUNDLED_CSV}")
        return None
    raw = load_bundled_csv()
    ds = {CSV_NAME: make_dataset(raw, LOCATIONS["Bandung"], "Giovanni CSV")}
    y0, y1 = int(raw.index.year.min()), int(raw.index.year.max())

    cities = st.sidebar.multiselect(
        "Compare with other cities (NASA POWER)",
        [c for c in LOCATIONS if c != "Bandung"],
        default=["Jakarta", "Pontianak", "Medan"],
    )
    validate = st.sidebar.checkbox("Validate Bandung against NASA POWER", value=True)
    st.info(
        f"**Bandung** comes from the bundled Giovanni file *dataset NASA.csv* (MERRA-2 monthly 2 m air temperature, "
        f"{y0}-{y1}, converted K → °C). Other places and the validation series come from the NASA POWER API."
    )
    wanted = (["Bandung"] if validate else []) + cities
    for name in wanted:
        lat, lon = LOCATIONS[name]
        label = POWER_BANDUNG_NAME if name == "Bandung" else f"{name} (POWER)"
        try:
            with st.spinner(f"Fetching T2M for {name} from NASA POWER..."):
                monthly, annual = fetch_power("T2M", lat, lon, y0, y1)
            ds[label] = {"annual": annual, "monthly": monthly, "coords": (lat, lon), "source": "POWER"}
        except Exception as e:
            st.warning(f"Could not fetch NASA POWER data for {name} ({e}). Showing the CSV data only for it.")
    return ds, "Air temperature at 2 m", "°C"


def collect_upload():
    st.markdown(
        "**How to get data:** open [NASA Giovanni](https://giovanni.gsfc.nasa.gov/), choose "
        "*Time Series, Area-Averaged*, pick a variable (e.g. GPM precipitation, MERRA-2 temperature) "
        "and a region, then download the **CSV**. Upload one file per region or variable."
    )
    var_label = st.text_input("Variable name", "Variable from Giovanni")
    files = st.file_uploader("Giovanni time-series CSV file(s)", type=["csv", "txt"], accept_multiple_files=True)
    raws, unit_guess = {}, ""
    for f in files or []:
        try:
            raw, u = read_giovanni(f.getvalue().decode("utf-8", errors="ignore"))
        except Exception as e:
            st.error(f"{f.name}: {e}")
            continue
        unit_guess = unit_guess or u
        raws[f.name.rsplit(".", 1)[0]] = raw
    unit = st.text_input("Unit", unit_guess)
    c1, c2 = st.columns(2)
    factor = c1.number_input(
        "Multiply values by", value=1.0, format="%.6g",
        help="Precipitation in kg m-2 s-1: use 86400 to get mm/day.",
    )
    offset = c2.number_input(
        "Then add", value=0.0, format="%.6g",
        help="Temperature in K: use -273.15 to get degrees C.",
    )
    if not raws:
        st.info("Upload at least one CSV to start.")
        return None
    return {n: make_dataset(r * factor + offset, None, "Giovanni CSV") for n, r in raws.items()}, var_label, unit


def collect_power():
    var_label = st.sidebar.selectbox("Variable", list(VARIABLES))
    param, unit = VARIABLES[var_label]
    places = st.sidebar.multiselect("Locations", list(LOCATIONS), default=["Bandung", "Jakarta", "Pontianak"])
    start, end = st.sidebar.slider("Years", 1985, 2024, (1995, 2024))
    if not places:
        st.info("Choose at least one location in the sidebar.")
        return None
    ds = {}
    for name in places:
        lat, lon = LOCATIONS[name]
        try:
            with st.spinner(f"Fetching {param} for {name} from NASA POWER..."):
                monthly, annual = fetch_power(param, lat, lon, start, end)
            ds[name] = {"annual": annual, "monthly": monthly, "coords": (lat, lon), "source": "POWER"}
        except Exception as e:
            st.error(f"Could not fetch data for {name}: {e}. Try the bundled Giovanni CSV option instead.")
    return ds, var_label, unit


def main():
    st.set_page_config(page_title="Earth System Trend Detective", page_icon="🛰️", layout="wide")
    st.title("🛰️ Earth System Trend Detective")
    st.caption(
        "Pick an Earth-science variable and region. The tool shows **what** is changing, "
        "**where**, **how much**, and whether the change is **statistically significant**."
    )

    with st.sidebar:
        st.header("Settings")
        source = st.radio("Data source", [MODE_COMBINED, MODE_UPLOAD, MODE_POWER])
        alpha = st.selectbox("Significance level (α)", [0.01, 0.05, 0.10], index=1)
        st.markdown(
            "**Method:** Mann-Kendall test for the presence of a trend and Sen's slope for its size, "
            "applied to annual means."
        )

    if source == MODE_COMBINED:
        got = collect_combined()
    elif source == MODE_UPLOAD:
        got = collect_upload()
    else:
        got = collect_power()
    if not got:
        return
    datasets, var_label, unit = got
    is_temp = unit.strip() in ("°C", "C", "degC", "°F")

    rows, results = [], {}
    for name, d in list(datasets.items()):
        series = d["annual"]
        if len(series) < 10:
            st.warning(f"{name}: only {len(series)} valid years; at least 10 are needed for a trend test.")
            datasets.pop(name)
            continue
        res = analyse(series, alpha)
        results[name] = res
        y0, y1 = int(series.index.min()), int(series.index.max())
        rows.append(
            {
                "Location": name,
                "Years": f"{y0}-{y1}",
                "What's happening": res["direction"],
                f"Change per decade ({unit})": float(f"{res['slope_per_decade']:.4g}"),
                f"Total change ({unit})": float(f"{res['total_change']:.4g}"),
                "p-value": round(res["p_value"], 4),
                "Significant?": "Yes" if res["significant"] else "No",
            }
        )
    if not rows:
        return

    # ---- headline numbers for the first (primary) series
    main_name = rows[0]["Location"]
    mres = results[main_name]
    st.subheader(f"Headline: {main_name}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Change per decade", f"{mres['slope_per_decade']:+.3g} {unit}")
    c2.metric("Total change", f"{mres['total_change']:+.3g} {unit}", help=rows[0]["Years"])
    c3.metric("p-value", f"{mres['p_value']:.4f}")
    c4.metric("Trend", mres["direction"].capitalize())

    # ---- summary table + plain-language findings
    st.subheader("Summary")
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    for r in rows:
        d = r[f"Change per decade ({unit})"]
        if r["Significant?"] == "Yes":
            word = "rose" if d > 0 else "fell"
            st.success(
                f"**{r['Location']}** ({r['Years']}): {var_label} {word} by about "
                f"{abs(d):.3g} {unit} per decade, a statistically significant trend (p = {r['p-value']})."
            )
        else:
            st.info(
                f"**{r['Location']}** ({r['Years']}): no statistically significant trend at "
                f"α = {alpha} (p = {r['p-value']}). Year-to-year variation dominates; "
                "a longer record might change this."
            )

    # ---- visual tabs
    has_validation = CSV_NAME in datasets and POWER_BANDUNG_NAME in datasets
    tab_names = ["📈 Trends", "⚖️ Compare places", "🔥 Anomalies & heatmap"]
    if has_validation:
        tab_names.append("✅ Validation")
    tab_names.append("🗺️ Map")
    tabs = dict(zip(tab_names, st.tabs(tab_names)))

    hi, _ = palette(is_temp)
    with tabs["📈 Trends"]:
        st.caption("Solid line = observed annual mean · dashed grey = Sen's slope trend.")
        cols = st.columns(min(len(datasets), 3))
        for i, (name, d) in enumerate(datasets.items()):
            with cols[i % len(cols)]:
                res = results[name]
                st.markdown(f"**{name}**")
                st.altair_chart(trend_chart(d["annual"], res, unit, hi), width="stretch")
                st.caption(f"{res['slope_per_decade']:+.3g} {unit}/decade · p = {res['p_value']:.3f}")

    with tabs["⚖️ Compare places"]:
        if len(datasets) < 2:
            st.info("Add more locations or files to compare them.")
        else:
            left, right = st.columns(2)
            with left:
                st.markdown("**Rate of change per decade**")
                st.altair_chart(comparison_bars(rows, unit, is_temp), width="stretch")
                st.caption("Coloured bars are statistically significant; grey bars are not.")
            with right:
                st.markdown("**Annual anomalies (each place vs its own mean)**")
                st.altair_chart(overlay_anomalies(datasets, unit), width="stretch")
                st.caption("Anomalies put places with different average levels on one scale.")

    with tabs["🔥 Anomalies & heatmap"]:
        pick = st.selectbox("Series", list(datasets), key="anom_pick")
        st.markdown("**Annual anomaly**")
        st.altair_chart(anomaly_bars(datasets[pick]["annual"], unit, is_temp), width="stretch")
        monthly = datasets[pick]["monthly"]
        if monthly is not None and len(monthly) >= 24:
            st.markdown("**Month × year heatmap**")
            st.altair_chart(month_year_heatmap(monthly, unit, is_temp), width="stretch")
            st.caption(
                "Each cell is the deviation of that month from the average of the same calendar month, "
                "so the seasonal cycle is removed and long-term change stands out."
            )
        else:
            st.info("A heatmap needs monthly data; this series is not monthly.")

    if has_validation:
        with tabs["✅ Validation"]:
            a, b = datasets[CSV_NAME]["annual"], datasets[POWER_BANDUNG_NAME]["annual"]
            common = a.index.intersection(b.index)
            if len(common) < 5:
                st.info("Not enough overlapping years to compare the two sources.")
            else:
                a, b = a.loc[common], b.loc[common]
                r = float(np.corrcoef(a, b)[0, 1])
                bias = float((a - b).mean())
                rmse = float(np.sqrt(((a - b) ** 2).mean()))
                sa, sb = results[CSV_NAME]["slope_per_decade"], results[POWER_BANDUNG_NAME]["slope_per_decade"]
                st.markdown("**Cross-check: Giovanni CSV vs NASA POWER API (same place)**")
                st.caption(
                    "NASA POWER's meteorological data are derived from MERRA-2, the same reanalysis as the Giovanni file. "
                    "This is therefore a consistency check of the data pipeline, not an independent confirmation."
                )
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Correlation (r)", f"{r:.2f}")
                m2.metric("Mean offset (CSV − POWER)", f"{bias:+.2f} {unit}")
                m3.metric("RMSE", f"{rmse:.2f} {unit}")
                m4.metric("Trend CSV vs POWER", f"{sa:+.2f} vs {sb:+.2f}", help=f"{unit}/decade")
                st.altair_chart(validation_chart(a, b, "MERRA-2 (Giovanni CSV)", "NASA POWER", unit), width="stretch")
                if (sa > 0) == (sb > 0) and r >= 0.8:
                    st.success(
                        f"Both routes give the same result (r = {r:.2f}, trend {sa:+.2f} vs {sb:+.2f} {unit}/decade), "
                        "so the CSV was read and processed correctly. Small differences, if any, come from "
                        "grid-cell selection or rounding."
                    )
                else:
                    st.warning(
                        f"The two routes differ (r = {r:.2f}, trend {sa:+.2f} vs {sb:+.2f} {unit}/decade). "
                        "Check that the Giovanni area matches the POWER point."
                    )

    with tabs["🗺️ Map"]:
        trend_map(rows, datasets, unit)

    with st.expander("About the data and method"):
        st.markdown(
            "- **Data:** NASA Giovanni time series (bundled or uploaded) and NASA POWER monthly point data; "
            "daily/monthly values are averaged to annual means (years with <80% coverage are dropped).\n"
            "- **Mann-Kendall** is a non-parametric test; it does not assume normally distributed data.\n"
            "- **Sen's slope** is the median of all pairwise slopes and is robust to outliers.\n"
            "- A trend is called *significant* when p < α; non-significant trends are still reported.\n"
            "- **Anomaly** = value minus the long-term mean; the heatmap removes each calendar month's average.\n"
            "- **Validation** compares the Giovanni CSV with the NASA POWER API for Bandung. POWER's "
            "meteorology is derived from MERRA-2, so this checks consistency, not independence.\n"
            "- Trends at one location can differ in sign from another, even when the same process drives both.\n"
            "- Only linear trends are tested; the seasonal cycle is removed by using annual means."
        )


if __name__ == "__main__":
    main()
