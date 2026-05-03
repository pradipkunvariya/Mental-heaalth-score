# 🧠 Student Mental Health Dashboard

An interactive data analytics dashboard built with **Streamlit** and **Plotly** to explore and visualize student mental health data.

---

## 📸 Features

- 📊 **Overview** — Score distribution, box plots, scatter charts with trendlines, correlation heatmap
- 👥 **Demographics** — Gender, education, diet breakdowns with pie charts and bar graphs
- 😰 **Stress & Anxiety** — Frequency analysis and impact on mental health score
- 💤 **Lifestyle** — Exercise, diet, sleep, study hours analysis
- 📋 **Data Explorer** — Sortable table, summary stats, download as CSV or Excel

---

## 🗂️ Project Structure

```
your-repo/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── student_mental_health_v3.csv    # Dataset
└── README.md                       # This file
```

---

## 📦 Dataset

The dataset contains **300 student records** with the following features:

| Column | Description |
|---|---|
| `age` | Student age |
| `gender` | Male / Female / Prefer not to say |
| `education` | School / Undergraduate / Postgraduate |
| `sleep_hours` | Average daily sleep hours |
| `screen_time` | Average daily screen time (hrs) |
| `study_hours` | Average daily study hours |
| `exercise` | Yes / No |
| `diet` | Healthy / Average / Unhealthy |
| `stress_frequency` | Never / Rarely / Sometimes / Often / Always |
| `anxiety_frequency` | Never / Rarely / Sometimes / Often / Always |
| `mental_health_score` | Score from 1 to 10 |

---

## 🚀 Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

**4. Open in browser**
```
http://localhost:8501
```

---

## ☁️ Deploy on Streamlit Cloud

1. Push all files to your GitHub repository
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Click **New app**
4. Select your repository, branch, and set `app.py` as the main file
5. Click **Deploy**

---

## 🛠️ Built With

- [Streamlit](https://streamlit.io/) — App framework
- [Plotly](https://plotly.com/python/) — Interactive charts
- [Pandas](https://pandas.pydata.org/) — Data manipulation
- [NumPy](https://numpy.org/) — Numerical computing
- [OpenPyXL](https://openpyxl.readthedocs.io/) — Excel export

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
