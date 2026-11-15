# Earth System Trend Detective

NASA Space Apps Challenge 2026 - Bandung.

Detects and visualises long-term trends in Earth-science variables (Mann-Kendall test + Sen's slope).
Data: NASA Giovanni CSV (bundled `dataset NASA.csv`, MERRA-2 T2M, Bandung 2001-2024) and the NASA POWER API.

```
python -m pip install -r requirements.txt
python -m streamlit run app.py
```
