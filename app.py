import io
import math
import re
from datetime import date
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import sympy as sp
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Hispanic Heritage Historical Data Explorer", page_icon="📈", layout="wide")

COUNTRIES = {
    "Mexico (MX)": "MEX", "Brazil (BR)": "BRA", "Argentina (AR)": "ARG",
    "Colombia (CO)": "COL", "Chile (CL)": "CHL", "Peru (PE)": "PER",
    "Venezuela (VE)": "VEN", "Ecuador (EC)": "ECU", "Uruguay (UY)": "URY",
}
INDICATORS = {
    "Life expectancy at birth": ("SP.DYN.LE00.IN", "years", "Average number of years a newborn is expected to live if current mortality patterns persist.", "nonnegative"),
    "Unemployment, national estimate": ("SL.UEM.TOTL.NE.ZS", "% of labor force", "Share of the labor force that is unemployed, using national estimates.", "percent"),
    "GDP per capita": ("NY.GDP.PCAP.CD", "current US$ per person", "Gross domestic product divided by midyear population; not average wealth or household income.", "nonnegative"),
    "Crude birth rate": ("SP.DYN.CBRT.IN", "births per 1,000 people per year", "Annual live births per 1,000 people; this is a rate, not a birth count.", "nonnegative"),
    "Net migration": ("SM.POP.NETM", "people (net)", "Net number of migrants over the period; not a count of emigration alone.", "any"),
    "Intentional homicide rate": ("VC.IHR.PSRC.P5", "per 100,000 people", "Intentional homicides per 100,000 people.", "nonnegative"),
    "Consumer price inflation": ("FP.CPI.TOTL.ZG", "% annual", "Annual percentage change in the consumer price index.", "any"),
    "Central government debt": ("GC.DOD.TOTL.GD.ZS", "% of GDP", "Central government debt as a percentage of GDP.", "any"),
    "Total reserves including gold": ("FI.RES.TOTL.CD", "current US$", "Total reserves including gold, in current US dollars.", "nonnegative"),
    "External balance on goods and services": ("NE.RSB.GNFS.ZS", "% of GDP", "External balance on goods and services as a percentage of GDP.", "any"),
}
COURSES = ["Honors Algebra 2", "Precalculus", "Calculus"]
CONTEXTS = ["Business", "Engineering", "Medical", "Sports science", "Biology", "Bioengineering", "Economic forecasting", "Weather", "Custom"]
SUGGESTIONS = {
    "Business": ("Revenue over time", "time", "weeks", "revenue", "USD", "f(t) = a t² + b t + c can model a short-term revenue trend."),
    "Engineering": ("Object position", "time", "seconds", "position", "meters", "A polynomial can approximate position over a limited interval; derivatives describe velocity and acceleration."),
    "Medical": ("Drug concentration", "time", "hours", "concentration", "mg/L", "A decaying exponential is often a useful conceptual model; real medication behavior needs clinical data."),
    "Sports science": ("Running distance", "time", "seconds", "distance", "meters", "Position derivatives can represent speed and acceleration."),
    "Biology": ("Population", "time", "days", "population", "organisms", "Population models are simplified and should not be extrapolated far without evidence."),
    "Bioengineering": ("Cell growth", "time", "hours", "cell count", "cells", "Growth curves depend on conditions and may change outside the observed interval."),
    "Economic forecasting": ("Economic indicator", "time", "years", "indicator", "index units", "Economic forecasts are uncertain and a polynomial is not a causal explanation."),
    "Weather": ("Temperature", "time", "hours", "temperature", "°C", "A local curve may describe a short interval, not long-term weather."),
    "Custom": ("Custom function", "x", "units", "f(x)", "units", "Describe what your variables mean."),
}
WORLD_BANK = "https://api.worldbank.org/v2"

st.title("Hispanic Heritage Historical Data Explorer")
st.caption("HISPANIC HERITAGE DATA LAB • historical data, modeling, and calculus")
st.info("Brazil is included as a Latin American comparison country. It is not a Spanish-speaking country, but it provides a useful regional comparison.")

with st.sidebar:
    st.header("Data selection")
    country_name = st.selectbox("Featured country", list(COUNTRIES.keys()))
    indicator_name = st.selectbox("Indicator", list(INDICATORS.keys()))
    comparison_names = st.multiselect("Comparison countries (same indicator)", [x for x in COUNTRIES if x != country_name], default=[])
    degree = st.slider("Polynomial degree", 3, 8, 3)
    sample_step = st.slider("Graph sample increment (years)", 1, 10, 5)
    horizon = st.slider("Forward display horizon (years)", 0, 50, 10)
    printer_mode = st.checkbox("Printer-friendly view", value=False)
    st.divider()
    st.header("Function lab")
    course = st.selectbox("Mathematics pathway", COURSES)
    context = st.selectbox("Context", CONTEXTS)
    mode = st.radio("Function source", ["Historical curve of best fit", "Suggested function", "Build polynomial", "Enter my own function"], index=0)
    st.caption("Use x for the independent variable. Custom expressions support polynomial, rational, exponential, logarithmic, and trigonometric functions.")

code, units, definition, constraint = INDICATORS[indicator_name]
country_code = COUNTRIES[country_name]
current_year = date.today().year
start_year, end_year = current_year - 70, current_year - 1
retrieved_at = date.today().isoformat()

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_indicator(iso3: str, indicator: str, start: int, end: int):
    url = f"{WORLD_BANK}/country/{iso3}/indicator/{indicator}"
    params = {"format": "json", "date": f"{start}:{end}", "per_page": 20000}
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
        return pd.DataFrame(columns=["year", "value"]), url
    records = []
    for item in payload[1]:
        if item and item.get("date") and item.get("value") is not None:
            try:
                records.append({"year": int(item["date"]), "value": float(item["value"])})
            except (TypeError, ValueError):
                continue
    df = pd.DataFrame(records, columns=["year", "value"]).sort_values("year").reset_index(drop=True)
    return df, url

def validate_data(df: pd.DataFrame, rule: str):
    errors = []
    if df.empty:
        return ["There are no usable observations."]
    if not {"year", "value"}.issubset(df.columns):
        return ["Table must have year and value columns."]
    if df["year"].isna().any() or not np.all(np.equal(df["year"].dropna(), np.floor(df["year"].dropna()))):
        errors.append("Years must be whole numbers.")
    if df["year"].duplicated().any():
        errors.append("Each year must be unique.")
    if df["value"].isna().any() or not np.isfinite(pd.to_numeric(df["value"], errors="coerce").dropna()).all():
        errors.append("Values must be finite numbers; blanks are not allowed in the working table.")
    vals = pd.to_numeric(df["value"], errors="coerce").dropna()
    if rule == "nonnegative" and (vals < 0).any():
        errors.append("This indicator cannot be negative in the working table.")
    if rule == "percent" and ((vals < 0) | (vals > 100)).any():
        errors.append("Unemployment percentage must be between 0 and 100.")
    if rule == "nonnegative" and indicator_name == "Life expectancy at birth" and (vals > 125).any():
        errors.append("Life expectancy values above 125 years need review.")
    return errors

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_comparison(iso3: str, indicator: str, start: int, end: int):
    return fetch_indicator(iso3, indicator, start, end)

with st.spinner("Retrieving historical data from the World Bank API..."):
    try:
        source_df, source_url = fetch_indicator(country_code, code, start_year, end_year)
        api_error = None
    except (requests.RequestException, ValueError, KeyError) as exc:
        source_df = pd.DataFrame(columns=["year", "value"])
        source_url = f"{WORLD_BANK}/country/{country_code}/indicator/{code}"
        api_error = f"{type(exc).__name__}: {exc}"

if api_error:
    st.error(f"The World Bank API request failed. No substitute data were generated. Details: {api_error}")
    st.markdown(f"[Open API endpoint]({source_url})")
    st.stop()
if source_df.empty:
    st.warning("The API returned no non-null observations for this selection. Choose another country or indicator.")
    st.markdown(f"[Open API endpoint]({source_url})")
    st.stop()

st.subheader("Historical data")
st.write(f"**{indicator_name} — {country_name}**")
st.write(f"**Definition:** {definition}")
st.write(f"**Units:** {units} · **Requested period:** {start_year}–{end_year} · **Retrieved:** {retrieved_at}")
st.markdown(f"**Source:** [World Bank API]({source_url}) · [Indicator metadata](https://api.worldbank.org/v2/indicator/{code}?format=json)")
st.caption(f"Actual coverage: {int(source_df.year.min())}–{int(source_df.year.max())} · {len(source_df)} non-null observations. The requested period does not guarantee 70 observations.")

if "working_df" not in st.session_state or st.session_state.get("data_key") != (country_code, code):
    st.session_state.working_df = source_df.copy()
    st.session_state.data_key = (country_code, code)
    st.session_state.edited = False
if st.button("Reset to source data"):
    st.session_state.working_df = source_df.copy()
    st.session_state.edited = False
    st.session_state.pop("single_result", None)
    st.session_state.pop("batch_result", None)

st.markdown("**Editable working table**")
st.caption("Edit or remove rows below. Keep years unique and values numeric. Editing marks this as a user-modified scenario; the original source data remains unchanged.")
edited = st.data_editor(st.session_state.working_df, num_rows="dynamic", use_container_width=True, key=f"editor_{country_code}_{code}")
try:
    edited = edited[["year", "value"]].copy()
    edited["year"] = pd.to_numeric(edited["year"], errors="coerce")
    edited["value"] = pd.to_numeric(edited["value"], errors="coerce")
except Exception:
    edited = pd.DataFrame(columns=["year", "value"])
validation_errors = validate_data(edited, constraint)
if not edited.equals(st.session_state.working_df):
    st.session_state.edited = True
st.session_state.working_df = edited
if st.session_state.edited:
    st.warning("User-modified scenario: the working table differs from the source data.")
if validation_errors:
    for err in validation_errors:
        st.error(err)
    st.stop()
edited = edited.sort_values("year").reset_index(drop=True)
if len(edited) < degree + 1 or edited["year"].nunique() < degree + 1:
    st.error(f"Degree {degree} requires at least {degree + 1} distinct valid years. Current usable rows: {len(edited)}.")
    st.stop()

years = edited["year"].astype(int).to_numpy()
values = edited["value"].astype(float).to_numpy()
base_year = int(years.min())
t = years - base_year
try:
    coeff = np.polyfit(t, values, degree)
    poly = np.poly1d(coeff)
    fitted = poly(t)
    residuals = values - fitted
    r2 = r2_score(values, fitted) if len(values) > 1 else float("nan")
    mae = mean_absolute_error(values, fitted)
except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
    st.error(f"Could not fit the selected polynomial: {exc}")
    st.stop()

def polynomial_equation(coefs, base, variable="t"):
    terms = []
    d = len(coefs) - 1
    for i, c in enumerate(coefs):
        power = d - i
        if abs(c) < 1e-12:
            continue
        if power == 0:
            term = f"{c:.8g}"
        elif power == 1:
            term = f"{c:.8g}({variable})"
        else:
            term = f"{c:.8g}({variable})^{power}"
        terms.append(term)
    return " + ".join(terms).replace("+ -", "- ")

equation = "y = " + polynomial_equation(coeff, base_year)
st.subheader("Polynomial model")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Degree", degree)
m2.metric("In-sample R²", f"{r2:.4f}")
m3.metric("In-sample MAE", f"{mae:.4g} {units}")
m4.metric("Observations", len(values))
st.code(f"t = year − {base_year}\n{equation}")
st.caption(f"Fitted range: {years.min()}–{years.max()}. Mean observed value: {np.mean(values):.6g} {units}. R² and MAE are in-sample metrics, not guarantees of future accuracy.")
with st.expander("Residuals and chronological holdout"):
    residual_table = pd.DataFrame({"year": years, "actual": values, "fitted": fitted, "residual (actual - fitted)": residuals})
    st.dataframe(residual_table, use_container_width=True)
    if len(values) >= max(degree + 3, 8):
        split = max(degree + 1, int(len(values) * 0.8))
        if split < len(values):
            try:
                hold_coeff = np.polyfit(t[:split], values[:split], degree)
                hold_pred = np.polyval(hold_coeff, t[split:])
                st.write(f"Chronological holdout MAE on last {len(values)-split} observations: **{mean_absolute_error(values[split:], hold_pred):.5g} {units}**. This is a separate holdout score; the fit uses earlier observations only.")
            except Exception as exc:
                st.caption(f"Holdout comparison unavailable: {exc}")
    else:
        st.caption("Not enough observations for a useful chronological holdout at this degree.")

# Function selection and context customization
suggested = SUGGESTIONS[context]
context_title = st.text_input("Context description", value=suggested[0])
x_name = st.text_input("Independent variable name", value=suggested[1])
x_units = st.text_input("Independent variable units", value=suggested[2])
y_name = st.text_input("Dependent variable name", value=suggested[3])
y_units = st.text_input("Dependent variable units", value=suggested[4])
st.caption(suggested[5])
param_col1, param_col2, param_col3 = st.columns(3)
with param_col1:
    a_param = st.number_input("Parameter a", value=1.0, step=0.5)
with param_col2:
    b_param = st.number_input("Parameter b", value=0.0, step=0.5)
with param_col3:
    c_param = st.number_input("Parameter c", value=0.0, step=0.5)
custom_expr = st.text_input("Your function f(x)", value="x**2" if mode == "Enter my own function" else "a*x**2 + b*x + c")
x = sp.Symbol("x", real=True)
function_error = None
try:
    if mode == "Historical curve of best fit":
        expr = sum(sp.Float(float(coef)) * (x - base_year) ** (degree-i) for i, coef in enumerate(coeff))
        # Here x is the calendar year; centered transformation is applied once.
    elif mode == "Suggested function":
        if context in ("Medical", "Bioengineering"):
            expr = sp.Float(a_param) * sp.exp(-sp.Float(abs(b_param) if b_param else 0.1) * x) + sp.Float(c_param)
        elif context == "Weather":
            expr = sp.Float(a_param) * sp.sin(x) + sp.Float(b_param) * x + sp.Float(c_param)
        elif context == "Biology":
            expr = sp.Float(a_param) * sp.exp(sp.Float(b_param if b_param else 0.1) * x) + sp.Float(c_param)
        else:
            expr = sp.Float(a_param)*x**2 + sp.Float(b_param)*x + sp.Float(c_param)
    elif mode == "Build polynomial":
        expr = sp.Float(a_param)*x**2 + sp.Float(b_param)*x + sp.Float(c_param)
    else:
        # Whitelist a small, familiar SymPy expression language; never eval Python code.
        expr = sp.sympify(custom_expr, locals={"x": x, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
                                                "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt,
                                                "pi": sp.pi, "E": sp.E})
        if expr.free_symbols - {x}:
            raise ValueError("Use x as the only variable and numeric constants.")
except Exception as exc:
    function_error = str(exc)
    expr = None

if function_error:
    st.error(f"Function could not be parsed: {function_error}")
    st.stop()

f = sp.lambdify(x, expr, modules=["numpy"])
d1_expr = sp.diff(expr, x)
d2_expr = sp.diff(d1_expr, x)
d1 = sp.lambdify(x, d1_expr, modules=["numpy"])
d2 = sp.lambdify(x, d2_expr, modules=["numpy"])
st.subheader("Function analysis")
st.write(f"**Context:** {context_title}")
st.latex(sp.latex(sp.Eq(sp.Function("f")(x), expr)))
st.write(f"**First derivative:**")
st.latex(sp.latex(d1_expr))
st.write("**Second derivative:**")
st.latex(sp.latex(d2_expr))
if mode == "Historical curve of best fit":
    st.caption(f"For this mode, x is calendar year and the polynomial is centered internally at base year {base_year}. Other modes use x as the context's independent variable.")

if mode == "Historical curve of best fit":
    default_lo, default_hi = float(years.min()), float(years.max())
else:
    default_lo, default_hi = 0.0, 10.0
if default_hi <= default_lo:
    default_hi = default_lo + 1
interval_col1, interval_col2, interval_col3 = st.columns(3)
with interval_col1:
    analysis_lo = st.number_input("Analysis interval start", value=float(default_lo))
with interval_col2:
    analysis_hi = st.number_input("Analysis interval end", value=float(default_hi))
with interval_col3:
    current_x = st.number_input("Current x value", value=float((default_lo + default_hi)/2))
if analysis_hi <= analysis_lo:
    st.error("Analysis interval end must be greater than the start.")
    st.stop()

def safe_scalar(fun, val):
    try:
        out = float(np.asarray(fun(val)))
        return out if math.isfinite(out) else float("nan")
    except Exception:
        return float("nan")

fx = safe_scalar(f, current_x)
d1x = safe_scalar(d1, current_x)
d2x = safe_scalar(d2, current_x)
avg_rate = (safe_scalar(f, analysis_hi)-safe_scalar(f, analysis_lo))/(analysis_hi-analysis_lo)
secant_slope = avg_rate
secant_intercept = safe_scalar(f, analysis_lo) - secant_slope*analysis_lo
tangent_intercept = fx - d1x*current_x if math.isfinite(fx) and math.isfinite(d1x) else float("nan")
v1, v2, v3 = st.columns(3)
v1.metric(f"f({current_x:g})", f"{fx:.6g}" if math.isfinite(fx) else "undefined", y_units)
v2.metric("Instantaneous rate f′(x)", f"{d1x:.6g}" if math.isfinite(d1x) else "undefined", f"{y_units} per {x_units}")
v3.metric("Second derivative f″(x)", f"{d2x:.6g}" if math.isfinite(d2x) else "undefined", f"{y_units} per {x_units}²")
st.write(f"**Average rate of change** on [{analysis_lo:g}, {analysis_hi:g}]: {avg_rate:.6g} {y_units} per {x_units}" if math.isfinite(avg_rate) else "Average rate of change is undefined on this interval.")
if math.isfinite(d1x):
    st.write(f"At x = {current_x:g}, the tangent-line approximation is y = {d1x:.6g}(x − {current_x:g}) + {fx:.6g}.")
if math.isfinite(avg_rate):
    st.write(f"Secant-line equation: y = {secant_slope:.6g}x + {secant_intercept:.6g}.")

# Course-specific explanations
if course == "Honors Algebra 2":
    st.markdown("**Algebra 2 focus:** Compare domain and range, graph trends, end behavior, and average rates of change. Polynomial extrapolation can grow rapidly and should not be treated as a reliable forecast.")
elif course == "Precalculus":
    st.markdown("**Precalculus focus:** Degree and coefficients affect end behavior and turning points. Concavity can be understood by checking whether average rates are increasing or decreasing.")
else:
    st.markdown("**Calculus focus:** Derivatives describe local change. Critical points require domain checks and derivative sign changes; a zero of f′ is not automatically an extremum, and a zero of f″ is not automatically an inflection point.")

# Numerical analysis across an interval
grid = np.linspace(analysis_lo, analysis_hi, 1500)
with np.errstate(all="ignore"):
    gy = np.asarray(f(grid), dtype=float)
    gd1 = np.asarray(d1(grid), dtype=float)
    gd2 = np.asarray(d2(grid), dtype=float)
    if gy.ndim == 0: gy = np.full_like(grid, gy)
    if gd1.ndim == 0: gd1 = np.full_like(grid, gd1)
    if gd2.ndim == 0: gd2 = np.full_like(grid, gd2)
    gy[~np.isfinite(gy)] = np.nan
    gd1[~np.isfinite(gd1)] = np.nan
    gd2[~np.isfinite(gd2)] = np.nan

def sign_changes(arr):
    arr = np.asarray(arr)
    finite = np.isfinite(arr)
    idx = np.where(finite[:-1] & finite[1:] & (np.sign(arr[:-1]) != np.sign(arr[1:])))[0]
    return [float((grid[i]+grid[i+1])/2) for i in idx]

crit = sign_changes(gd1)
infl_candidates = sign_changes(gd2)
inflections = []
for point in infl_candidates:
    left, right = safe_scalar(d2, point - (analysis_hi-analysis_lo)/3000), safe_scalar(d2, point + (analysis_hi-analysis_lo)/3000)
    if math.isfinite(left) and math.isfinite(right) and left*right < 0:
        inflections.append(point)

if course == "Calculus":
    st.markdown("#### Numerical critical-point and concavity analysis")
    if crit:
        rows = []
        for point in crit:
            leftv, rightv = safe_scalar(d1, point - (analysis_hi-analysis_lo)/3000), safe_scalar(d1, point + (analysis_hi-analysis_lo)/3000)
            kind = "local minimum (sign changes − to +)" if leftv < 0 < rightv else "local maximum (sign changes + to −)" if leftv > 0 > rightv else "not classified as an extremum"
            rows.append({"x (approx.)": point, "f(x)": safe_scalar(f, point), "classification": kind})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.write("No sign-changing derivative roots were detected numerically in this interval.")
    st.write("Increasing/decreasing intervals are approximated from sampled derivative signs; endpoints and derivative discontinuities should be checked.")
    if inflections:
        st.write("Verified numerical inflection candidates (f″ changes sign): " + ", ".join(f"{p:.5g}" for p in inflections))
    else:
        st.write("No concavity-changing inflection points were detected numerically in this interval.")
    finite_y = gy[np.isfinite(gy)]
    if len(finite_y):
        st.write(f"Approximate absolute extrema on the sampled closed interval: min {np.min(finite_y):.6g}, max {np.max(finite_y):.6g} (numerical grid, not symbolic proof).")
    try:
        domain_restrictions = []
        for den in sp.denom(sp.together(expr)).as_ordered_factors():
            roots = sp.solve(sp.Eq(den, 0), x)
            domain_restrictions.extend([sp.sstr(root) for root in roots])
        if domain_restrictions:
            st.write("Potential domain exclusions from denominator: x ≠ " + ", ".join(domain_restrictions))
        if expr.has(sp.log):
            st.write("Logarithms require positive arguments.")
        if expr.has(sp.tan):
            st.write("Tangent is undefined where its cosine denominator is zero.")
    except Exception:
        st.caption("Some domain restrictions may not be found automatically; inspect the expression and graph.")

# Graph: separate valid runs to avoid connecting across undefined values
show_f = st.checkbox("Show function", True)
show_d1 = st.checkbox("Show first derivative", course == "Calculus")
show_d2 = st.checkbox("Show second derivative", False)
show_tangent = st.checkbox("Show tangent line", True)
show_secant = st.checkbox("Show secant line", True)
fig, ax = plt.subplots(figsize=(10, 5))
def plot_segmented(axis, xs, ys, label, linestyle="-"):
    valid = np.isfinite(ys)
    starts = np.where(valid & np.r_[True, ~valid[:-1]])[0]
    ends = np.where(valid & np.r_[~valid[1:], True])[0]
    first = True
    for s, e in zip(starts, ends):
        if e-s < 1:
            continue
        axis.plot(xs[s:e+1], ys[s:e+1], linestyle=linestyle, label=label if first else None)
        first = False

if show_f:
    plot_segmented(ax, grid, gy, f"f(x) ({y_units})")
if show_d1:
    plot_segmented(ax, grid, gd1, "f′(x)")
if show_d2:
    plot_segmented(ax, grid, gd2, "f″(x)")
if show_tangent and math.isfinite(tangent_intercept) and math.isfinite(d1x):
    ax.plot(grid, d1x*grid+tangent_intercept, "--", label="Tangent line")
if show_secant and math.isfinite(secant_intercept) and math.isfinite(secant_slope):
    ax.plot(grid, secant_slope*grid+secant_intercept, ":", label="Secant line")
if math.isfinite(fx):
    ax.scatter([current_x], [fx], marker="o", s=55, label="Current point")
ax.axvline(analysis_lo, linestyle="--", alpha=0.4, label="Interval boundary")
ax.axvline(analysis_hi, linestyle="--", alpha=0.4)
ax.set_xlabel(f"{x_name} ({x_units})")
ax.set_ylabel(f"{y_name} ({y_units})")
ax.set_title(context_title)
ax.grid(True, alpha=0.25)
ax.legend(loc="best", fontsize="small")
fig.tight_layout()
st.pyplot(fig, use_container_width=True)

# Historical curve / predictions, with explicit distinction between observed and fitted
if mode == "Historical curve of best fit":
    st.subheader("Historical model and estimates")
    year_input = st.number_input("Single year (1950–2100)", min_value=1950, max_value=2100, value=int(np.clip(base_year, 1950, 2100)), step=1)
    if st.button("Calculate Value"):
        value = float(poly(int(year_input)-base_year))
        st.session_state.single_result = {"year": int(year_input), "estimate": value, "key": (country_code, code, degree, tuple(years), tuple(values))}
    result_key = (country_code, code, degree, tuple(years), tuple(values))
    if st.session_state.get("single_result", {}).get("key") == result_key:
        res = st.session_state.single_result
        observed = res["year"] in set(years.tolist())
        in_range = years.min() <= res["year"] <= years.max()
        st.success(f"{res['year']}: fitted estimate = {res['estimate']:.7g} {units}. " +
                   (f"This year is inside the fitted range ({years.min()}–{years.max()})." if in_range else f"This year is outside the fitted range ({years.min()}–{years.max()}); extrapolation is uncertain.") +
                   (" A source observation also exists for this year; the estimate is still the model value, not the actual observation." if observed else " No source observation exists for this exact year in the working table."))
    st.markdown("**Batch estimates**")
    batch_mode = st.radio("Batch input", ["Year range", "Custom years"], horizontal=True)
    batch_years = []
    if batch_mode == "Year range":
        c1, c2, c3 = st.columns(3)
        with c1: batch_start = st.number_input("Start year", 1950, 2100, int(np.clip(years.min(), 1950, 2100)), key="batch_start")
        with c2: batch_end = st.number_input("End year", 1950, 2100, int(np.clip(years.max(), 1950, 2100)), key="batch_end")
        with c3: batch_step = st.slider("Range step", 1, 10, 1)
        if batch_end >= batch_start:
            batch_years = list(range(int(batch_start), int(batch_end)+1, int(batch_step)))
        else:
            st.error("End year must be greater than or equal to start year.")
    else:
        custom_year_text = st.text_input("Comma-separated years", value=f"{int(years.min())},{int(years.max())}")
        for part in custom_year_text.split(","):
            part = part.strip()
            if not part: continue
            try:
                yy = int(part)
                if str(yy) != part and not re.fullmatch(r"[+-]?\d+", part):
                    raise ValueError()
                if 1950 <= yy <= 2100: batch_years.append(yy)
                else: st.warning(f"Ignored {part}: year must be from 1950 to 2100.")
            except ValueError:
                st.warning(f"Ignored '{part}': enter a whole-number year.")
    batch_years = list(dict.fromkeys(batch_years))
    if batch_years:
        batch_df = pd.DataFrame({"year": batch_years})
        batch_df["estimate"] = [float(poly(y-base_year)) for y in batch_years]
        batch_df["within_fitted_range"] = batch_df.year.between(int(years.min()), int(years.max()))
        batch_df["source_observation"] = batch_df.year.isin(years)
        st.session_state.batch_result = {"df": batch_df, "key": result_key}
    batch_state = st.session_state.get("batch_result")
    if batch_state and batch_state.get("key") == result_key:
        st.dataframe(batch_state["df"], use_container_width=True)
    else:
        st.caption("Enter valid batch years to calculate estimates.")
    x_hist_min, x_hist_max = int(years.min()), int(years.max())
    plot_min = min(x_hist_min, int(years.min()))
    plot_max = max(x_hist_max, min(2100, x_hist_max + int(horizon)))
    plot_x = np.linspace(plot_min, plot_max, 800)
    plot_y = poly(plot_x-base_year)
    hist_fig, hist_ax = plt.subplots(figsize=(10, 4.5))
    hist_ax.scatter(years, values, label="Observed source / working data", zorder=3)
    hist_ax.plot(plot_x[plot_x <= x_hist_max], plot_y[plot_x <= x_hist_max], label="Fitted curve (historical range)")
    if plot_max > x_hist_max:
        hist_ax.plot(plot_x[plot_x > x_hist_max], plot_y[plot_x > x_hist_max], "--", label="Extrapolated segment")
    hist_ax.axvline(x_hist_max, linestyle=":", label="Historical boundary")
    hist_ax.set_xlabel("Year"); hist_ax.set_ylabel(units); hist_ax.grid(True, alpha=0.25); hist_ax.legend()
    hist_fig.tight_layout()
    st.pyplot(hist_fig, use_container_width=True)
    st.caption(f"The sample increment is {sample_step} year(s). Markers below sample the fitted curve; the smooth line is shown separately.")
    sampled_years = np.arange(plot_min, plot_max+1, sample_step)
    st.dataframe(pd.DataFrame({"sample year": sampled_years, "fitted model value": [float(poly(y-base_year)) for y in sampled_years]}), use_container_width=True)
    st.warning("Polynomial extrapolations can become implausible quickly. Values are not clipped, and no 95% prediction interval is claimed. Residuals and MAE describe in-sample error, not all sources of uncertainty.")

    # Same-indicator country comparison
    if comparison_names:
        comp_fig, comp_ax = plt.subplots(figsize=(10, 4.5))
        comp_ax.plot(years, values, marker="o", label=f"{country_name} (working data)")
        comp_notes = []
        for comp_name in comparison_names:
            try:
                comp_df, _ = fetch_comparison(COUNTRIES[comp_name], code, start_year, end_year)
                if not comp_df.empty:
                    comp_ax.plot(comp_df.year, comp_df.value, marker=".", label=comp_name)
                    comp_notes.append(f"{comp_name}: {comp_df.year.min()}–{comp_df.year.max()} ({len(comp_df)} observations)")
                else:
                    comp_notes.append(f"{comp_name}: no non-null observations")
            except Exception as exc:
                comp_notes.append(f"{comp_name}: retrieval failed ({type(exc).__name__})")
        comp_ax.set_xlabel("Year"); comp_ax.set_ylabel(units); comp_ax.set_title(f"{indicator_name} comparison")
        comp_ax.grid(True, alpha=0.25); comp_ax.legend(); comp_fig.tight_layout()
        st.pyplot(comp_fig, use_container_width=True)
        st.caption("Coverage can differ by country, so lines may start or stop in different years. " + " · ".join(comp_notes))

# Contextual interpretation with units
st.subheader("Interpretation in context")
if context == "Engineering":
    st.write(f"At the selected point, f(x) is {y_name.lower()} ({y_units}); f′(x) is its rate of change ({y_units} per {x_units}); f″(x) tells how that rate changes ({y_units} per {x_units}²).")
elif context in ("Business", "Economic forecasting"):
    st.write(f"f(x) represents the selected economic or business measure ({y_units}). A positive f′ means it is increasing locally, a negative f′ means decreasing; f″ indicates whether the rate is increasing or decreasing.")
elif context in ("Medical", "Bioengineering", "Biology"):
    st.write(f"f(x) is the modeled {y_name.lower()} ({y_units}). The first derivative measures local change per {x_units}; the second derivative measures how that change rate shifts. This educational model is not medical advice or a validated biological forecast.")
elif context == "Sports science":
    st.write(f"f(x) is {y_name.lower()} ({y_units}). Its first derivative gives the local rate per {x_units}; its second derivative describes the change in that rate.")
elif context == "Weather":
    st.write(f"f(x) is {y_name.lower()} ({y_units}). The first derivative describes how quickly it changes per {x_units}, and the second derivative describes how that rate itself changes.")
else:
    st.write(f"Interpret f(x) as {y_name} ({y_units}); f′(x) has units {y_units} per {x_units}, and f″(x) has units {y_units} per {x_units}². Interpret signs and size using your custom context.")

# Export current selection and calculated results
st.subheader("Export")
summary = pd.DataFrame([{
    "country": country_name, "country_code": country_code, "indicator": indicator_name, "indicator_code": code,
    "definition": definition, "units": units, "retrieval_date": retrieved_at,
    "source_url": source_url, "edited_scenario": bool(st.session_state.edited),
    "degree": degree, "base_year": base_year, "equation": equation, "in_sample_r2": r2,
    "in_sample_mae": mae, "observation_count": len(values), "historical_start": int(years.min()),
    "historical_end": int(years.max()), "mean_observed": float(np.mean(values))
}])
batch_state = st.session_state.get("batch_result", {})
export_batch = batch_state.get("df", pd.DataFrame()) if batch_state.get("key") == (country_code, code, degree, tuple(years), tuple(values)) else pd.DataFrame(columns=["year", "estimate", "within_fitted_range", "source_observation"])
csv_bytes = edited.to_csv(index=False).encode("utf-8")
st.download_button("Download working data CSV", data=csv_bytes, file_name="working_data.csv", mime="text/csv")

xlsx_buffer = io.BytesIO()
with pd.ExcelWriter(xlsx_buffer, engine="xlsxwriter") as writer:
    summary.to_excel(writer, sheet_name="Summary", index=False)
    edited.to_excel(writer, sheet_name="Working data", index=False)
    export_batch.to_excel(writer, sheet_name="Calculated years", index=False)
st.download_button("Download Excel workbook (.xlsx)", data=xlsx_buffer.getvalue(), file_name="historical_data_explorer.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

pdf_buffer = io.BytesIO()
pdf_doc = SimpleDocTemplate(pdf_buffer, pagesize=landscape(letter), rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
styles = getSampleStyleSheet()
story = [Paragraph("Hispanic Heritage Historical Data Explorer", styles["Title"]),
         Paragraph(f"{country_name} · {indicator_name}", styles["Heading2"]),
         Paragraph(f"Definition: {definition}", styles["BodyText"]),
         Paragraph(f"Units: {units} · Retrieved: {retrieved_at} · Source: {source_url}", styles["BodyText"]),
         Paragraph(f"Scenario edited: {bool(st.session_state.edited)} · Coverage: {years.min()}–{years.max()} · Observations: {len(values)}", styles["BodyText"]),
         Spacer(1, 8), Paragraph("Polynomial model", styles["Heading2"]),
         Paragraph(f"Centered at base year {base_year}; degree {degree}", styles["BodyText"]),
         Paragraph(equation.replace("&", "&amp;").replace("<", "&lt;"), styles["BodyText"]),
         Paragraph(f"In-sample R²: {r2:.6g} · In-sample MAE: {mae:.6g} {units} · Mean: {np.mean(values):.6g} {units}", styles["BodyText"]),
         Paragraph("Limitations: polynomial extrapolations are uncertain; fit metrics are in-sample and are not prediction intervals. No causal explanation is inferred.", styles["BodyText"]),
         Spacer(1, 8)]
img_buf = io.BytesIO()
hist_fig_for_pdf = None
if mode == "Historical curve of best fit":
    hist_fig_for_pdf, hist_ax_for_pdf = plt.subplots(figsize=(8, 3))
    hist_ax_for_pdf.scatter(years, values, label="Observed")
    hist_ax_for_pdf.plot(plot_x, plot_y, label="Fitted / extrapolated")
    hist_ax_for_pdf.axvline(x_hist_max, linestyle=":", label="Historical boundary")
    hist_ax_for_pdf.set_xlabel("Year"); hist_ax_for_pdf.set_ylabel(units); hist_ax_for_pdf.legend(fontsize="small")
    hist_fig_for_pdf.tight_layout()
    hist_fig_for_pdf.savefig(img_buf, format="png", dpi=120)
    plt.close(hist_fig_for_pdf)
    img_buf.seek(0)
    story.append(Image(img_buf, width=500, height=190))
preview = edited.head(12).round(6)
preview_data = [list(preview.columns)] + preview.astype(str).values.tolist()
preview_table = Table(preview_data, repeatRows=1)
preview_table.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.lightgrey), ("GRID", (0,0), (-1,-1), 0.4, colors.grey), ("FONTSIZE", (0,0), (-1,-1), 7), ("VALIGN", (0,0), (-1,-1), "TOP")]))
story += [Paragraph("Working data preview (first 12 rows)", styles["Heading2"]), preview_table]
if not export_batch.empty:
    story += [Spacer(1, 8), Paragraph("Calculated-year results", styles["Heading2"])]
    batch_preview = export_batch.head(25).round(6)
    batch_data = [list(batch_preview.columns)] + batch_preview.astype(str).values.tolist()
    bt = Table(batch_data, repeatRows=1)
    bt.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.lightgrey), ("GRID", (0,0), (-1,-1), 0.4, colors.grey), ("FONTSIZE", (0,0), (-1,-1), 7)]))
    story.append(bt)
pdf_doc.build(story)
st.download_button("Download PDF report", data=pdf_buffer.getvalue(), file_name="historical_data_explorer_report.pdf", mime="application/pdf")

if printer_mode:
    st.markdown("---")
    st.subheader("Printer-friendly report summary")
    st.write(f"{country_name} — {indicator_name} ({units})")
    st.write(f"Coverage: {years.min()}–{years.max()} | n = {len(values)} | degree = {degree}")
    st.write(f"Equation: {equation}")
    st.write(f"In-sample R² = {r2:.4f}; in-sample MAE = {mae:.5g} {units}")
    st.write("Limitations: polynomial models summarize patterns and do not prove causes. Extrapolations are uncertain.")

st.caption("Data source: World Bank API. This classroom tool is for exploration and is not a substitute for official statistical interpretation.")
