# 🚨 Fraud Detection in Financial Transactions

Anomaly detection + class imbalance handling + alert dashboard, using synthetic anonymized transaction data.

## Project mein kya hai

```
fraud-detection-project/
├── data/                       -> yaha dataset generate hoga (transactions.csv)
├── src/
│   ├── generate_data.py        -> synthetic anonymized dataset banata hai
│   ├── preprocess.py           -> scaling helper functions
│   ├── train_isolation_forest.py  -> MAIN training script (Isolation Forest + SMOTE)
│   └── train_autoencoder.py    -> OPTIONAL deep learning model (AutoEncoder)
├── dashboard/
│   └── app.py                  -> Streamlit alert dashboard
├── notebooks/
│   └── EDA.ipynb               -> exploratory data analysis notebook
├── models/                     -> trained models yaha save honge
├── outputs/                    -> metrics + predictions yaha save honge
└── requirements.txt
```

## Project kya karta hai (guidelines coverage)

| Requirement | Kaise implement kiya |
|---|---|
| Anonymized transaction dataset | `src/generate_data.py` — real Kaggle-style anonymized (V1...V10, Time, Amount) dataset generate karta hai, sensitive data ki zaroorat nahi |
| Anomaly detection (Isolation Forest) | `src/train_isolation_forest.py` |
| Anomaly detection (AutoEncoders) | `src/train_autoencoder.py` (optional, TensorFlow chahiye) |
| Class imbalance techniques | SMOTE (oversampling) + contamination-aware Isolation Forest |
| Alert-based dashboard | `dashboard/app.py` — Streamlit app jisme flagged/suspicious transactions dikhte hain |

---

## VS Code mein step-by-step kaise chalayein

### Step 1: Zip file extract karo
Zip file ko kisi bhi folder me extract kar lo (right-click → Extract All), jaise `D:\Projects\fraud-detection-project`.

### Step 2: VS Code me folder open karo
- VS Code kholo
- `File > Open Folder` pe click karo
- `fraud-detection-project` folder select karo

### Step 3: Python extension check karo
- Left sidebar me Extensions icon (squares wala) pe click karo
- "Python" search karo (by Microsoft) — agar installed nahi hai to Install kar lo

### Step 4: Terminal kholo
- Top menu se `Terminal > New Terminal` (ya shortcut: `` Ctrl+ ` ``)
- Terminal neeche khul jayega, make sure ye project ke root folder me hi ho

### Step 5: Virtual environment banao (recommended, optional bhi kar sakte ho)
```bash
python -m venv venv
```
Activate karo:
- **Windows:**
  ```bash
  venv\Scripts\activate
  ```
- **Mac/Linux:**
  ```bash
  source venv/bin/activate
  ```
Activate hone ke baad terminal me `(venv)` dikhega.

> VS Code neeche-right corner me interpreter select karne ka prompt bhi dega — usme apna `venv` select kar lena (`Python: Select Interpreter`).

### Step 6: Dependencies install karo
```bash
pip install -r requirements.txt
```
(Ye ek baar hi karna hai, 2-3 minute lag sakte hain)

### Step 7: Dataset generate karo
```bash
python src/generate_data.py
```
Isse `data/transactions.csv` ban jayegi (20,000 transactions, ~0.6% fraud — realistic imbalance).

### Step 8: Model train karo
```bash
python src/train_isolation_forest.py
```
Ye chalayega:
- Isolation Forest (unsupervised anomaly detection)
- SMOTE se class imbalance fix karke Random Forest (supervised comparison model)

Output terminal me precision/recall/ROC-AUC dikhega, aur ye files save hongi:
- `models/isolation_forest.joblib`
- `models/rf_smote.joblib`
- `outputs/metrics.json`
- `outputs/predictions.csv` (dashboard isi file ko use karta hai)

### Step 9: Dashboard chalao 🎉
```bash
streamlit run dashboard/app.py
```
Ye automatically browser me khul jayega (`http://localhost:8501`). Agar nahi khule to link ko manually browser me paste kar do.

Dashboard me tum dekh sakte ho:
- Total transactions, fraud count, model ne kitne flag kiye
- Anomaly score distribution chart
- Flagged (suspicious) transactions ki table — filter/sort/download bhi kar sakte ho
- Isolation Forest vs Random Forest+SMOTE — dono model switch kar sakte ho sidebar se

Band karne ke liye terminal me `Ctrl+C` dabao.

### Step 10 (Optional): AutoEncoder try karo
Agar deep learning wala model bhi chalana hai:
```bash
pip install tensorflow
python src/train_autoencoder.py
```

### Step 11 (Optional): EDA notebook
`notebooks/EDA.ipynb` file VS Code me open karo, "Jupyter" extension install karega prompt aayega — install kar lo, phir "Run All" kar sakte ho.

---

## Apna real dataset use karna ho to

`data/transactions.csv` ko apni khud ki anonymized dataset se replace kar do — bas columns match hone chahiye:
- Anonymized features (kisi bhi naam se, jaise V1, V2...)
- `Time` (optional)
- `Amount`
- `Class` (0 = normal, 1 = fraud)

Fir seedha Step 8 se shuru kar do.

## Common issues

| Problem | Solution |
|---|---|
| `python` command not found | Python install karo python.org se, "Add to PATH" checkbox zaroor tick karo |
| `pip install` fail ho raha | `pip install --upgrade pip` chalao pehle, phir retry |
| Streamlit browser me nahi khul raha | terminal me di gayi `Local URL` ko manually browser me copy-paste karo |
| Dashboard me "predictions.csv not found" error | Pehle Step 7 aur Step 8 complete karo |
