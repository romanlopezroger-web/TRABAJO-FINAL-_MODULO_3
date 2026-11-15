"""
Script maestro de modelado, validación y generación de entregables para:
MODELADO_APRENDIZAJE_AUTOMATICO.ipynb
"""

import sys
import time
from pathlib import Path

# Autoflush para logging inmediato
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)

# -------------------------------------------------------------
# 1. Configuración de Directorios y Estilo Visual
# -------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DIR_FIG = BASE_DIR / "figuras"
DIR_TAB = BASE_DIR / "tablas"
DIR_FIG.mkdir(exist_ok=True)
DIR_TAB.mkdir(exist_ok=True)

# Paleta obligatoria
PALETTE = {
    "lr": "#8B5CF6",       # Violeta
    "dt": "#F59E0B",       # Naranja
    "rf": "#10B981",       # Verde esmeralda
    "dummy": "#94A3B8",    # Gris neutro
    "pos": "#06B6D4",      # Turquesa (clase 1)
    "neg": "#F97373",      # Coral (clase 0)
    "accent1": "#3B82F6",  # Azul eléctrico de apoyo
    "accent2": "#EC4899",  # Magenta de apoyo
    "dark": "#1E293B",     # Carbón para textos y ejes
}

# Configuración tipográfica y visual
plt.rcParams.update({
    "font.sans-serif": "DejaVu Sans",
    "axes.edgecolor": "#CBD5E1",
    "axes.linewidth": 1.1,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.titlecolor": PALETTE["dark"],
    "axes.labelsize": 11,
    "axes.labelcolor": PALETTE["dark"],
    "xtick.color": PALETTE["dark"],
    "ytick.color": PALETTE["dark"],
    "grid.color": "#E2E8F0",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.facecolor": "#FFFFFF",
    "legend.edgecolor": "#CBD5E1",
    "figure.titlesize": 15,
    "figure.titleweight": "bold",
    "savefig.dpi": 300,
})

def main():
    print("=" * 70)
    print("PROYECTO FINAL: MODELADO PREDICTIVO CON APRENDIZAJE AUTOMATICO")
    print("Dataset: Bank Marketing (UCI) | Clasificación Supervisada")
    print("=" * 70)

    # ---------------------------------------------------------
    # 2. Carga y Verificación de Datos
    # ---------------------------------------------------------
    path_mod = BASE_DIR / "bank_full_modelo.csv"
    path_prep = BASE_DIR / "bank_full_preparado.csv"

    if not path_mod.exists():
        raise FileNotFoundError(f"No se encontró {path_mod}")
    
    df_mod = pd.read_csv(path_mod)
    df_prep = pd.read_csv(path_prep)

    print(f"\n[1/7] Datos cargados exitosamente:")
    print(f"  - bank_full_modelo:    {df_mod.shape[0]:,} filas, {df_mod.shape[1]} columnas")
    print(f"  - bank_full_preparado: {df_prep.shape[0]:,} filas, {df_prep.shape[1]} columnas")
    print(f"  - Valores nulos en df_mod: {df_mod.isna().sum().sum()}")
    print(f"  - Variable 'duration' presente en predictoras: {'duration' in df_mod.columns}")
    print(f"  - Variable objetivo 'yb' presente: {'yb' in df_mod.columns}")
    
    conteo_yb = df_mod["yb"].value_counts()
    prop_yb = df_mod["yb"].value_counts(normalize=True) * 100
    print(f"  - Distribución 'yb': Clase 0={conteo_yb[0]:,} ({prop_yb[0]:.2f}%), Clase 1={conteo_yb[1]:,} ({prop_yb[1]:.2f}%)")

    # Separar X e y
    X = df_mod.drop(columns=["yb"])
    y = df_mod["yb"].astype(int)
    feature_names = list(X.columns)
    print(f"  - Predictoras ({len(feature_names)}): {feature_names[:4]} ... {feature_names[-3:]}")

    # ---------------------------------------------------------
    # 3. Partición Estratificada y Protocolo de Validación
    # ---------------------------------------------------------
    print("\n[2/7] Generando partición Train/Test (80/20) y StratifiedKFold (5 splits)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print(f"  - Train: {X_train.shape[0]:,} registros | Positivos: {y_train.sum():,} ({y_train.mean()*100:.2f}%)")
    print(f"  - Test:  {X_test.shape[0]:,} registros | Positivos: {y_test.sum():,} ({y_test.mean()*100:.2f}%)")

    # Guardar tabla de particiones
    df_part = pd.DataFrame({
        "Conjunto": ["Entrenamiento (Train)", "Prueba (Test)", "Total"],
        "Total_Registros": [len(y_train), len(y_test), len(y)],
        "Negativos (0)": [(y_train==0).sum(), (y_test==0).sum(), (y==0).sum()],
        "Positivos (1)": [(y_train==1).sum(), (y_test==1).sum(), (y==1).sum()],
        "Tasa_Positivos_%": [y_train.mean()*100, y_test.mean()*100, y.mean()*100]
    })
    df_part.to_csv(DIR_TAB / "09_resumen_particiones_train_test.csv", index=False)

    # ---------------------------------------------------------
    # 4. Entrenamiento y Optimización de Hiperparámetros
    # ---------------------------------------------------------
    print("\n[3/7] Entrenando y optimizando modelos candidatos...")

    # A) Dummy Classifier
    print("  -> Evaluando Baseline (DummyClassifier)...")
    dummy = DummyClassifier(strategy="stratified", random_state=42)
    dummy.fit(X_train, y_train)

    # B) Regresión Logística con StandardScaler en Pipeline
    print("  -> Optimizando Regresión Logística (GridSearchCV)...")
    pipe_lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs"))
    ])
    param_lr = {
        "clf__C": [0.1, 1.0, 10.0],
        "clf__class_weight": ["balanced", None]
    }
    gs_lr = GridSearchCV(pipe_lr, param_lr, cv=cv, scoring="roc_auc", n_jobs=1, return_train_score=True)
    t0 = time.time()
    gs_lr.fit(X_train, y_train)
    t_lr = time.time() - t0
    best_lr = gs_lr.best_estimator_
    print(f"     Mejor LR: {gs_lr.best_params_} | ROC-AUC CV={gs_lr.best_score_:.4f} ({t_lr:.1f}s)")

    # C) Árbol de Decisión
    print("  -> Optimizando Árbol de Decisión (GridSearchCV)...")
    param_dt = {
        "max_depth": [4, 6, 8],
        "min_samples_leaf": [20, 50],
        "class_weight": ["balanced", None]
    }
    gs_dt = GridSearchCV(DecisionTreeClassifier(random_state=42), param_dt, cv=cv, scoring="roc_auc", n_jobs=1, return_train_score=True)
    t0 = time.time()
    gs_dt.fit(X_train, y_train)
    t_dt = time.time() - t0
    best_dt = gs_dt.best_estimator_
    print(f"     Mejor DT: {gs_dt.best_params_} | ROC-AUC CV={gs_dt.best_score_:.4f} ({t_dt:.1f}s)")

    # D) Random Forest Classifier (con n_jobs=-1 en el estimador)
    print("  -> Optimizando Random Forest Classifier (GridSearchCV)...")
    param_rf = {
        "max_depth": [10, 14],
        "min_samples_leaf": [10, 25],
        "class_weight": ["balanced"]
    }
    rf_base = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    gs_rf = GridSearchCV(rf_base, param_rf, cv=cv, scoring="roc_auc", n_jobs=1, return_train_score=True)
    t0 = time.time()
    gs_rf.fit(X_train, y_train)
    t_rf = time.time() - t0
    best_rf = gs_rf.best_estimator_
    print(f"     Mejor RF: {gs_rf.best_params_} | ROC-AUC CV={gs_rf.best_score_:.4f} ({t_rf:.1f}s)")

    # ---------------------------------------------------------
    # 5. Evaluación Cruzada Rigurosa y Métricas Detalladas
    # ---------------------------------------------------------
    print("\n[4/7] Calculando métricas en Validación Cruzada (5 folds) para cada modelo...")

    model_dict = {
        "Baseline (Dummy)": dummy,
        "Regresion Logistica": best_lr,
        "Arbol de Decision": best_dt,
        "Random Forest": best_rf
    }

    cv_results_list = []
    oof_preds = {name: np.zeros(len(y_train)) for name in model_dict}
    oof_probs = {name: np.zeros(len(y_train)) for name in model_dict}

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
        X_tr_f, X_val_f = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr_f, y_val_f = y_train.iloc[train_idx], y_train.iloc[val_idx]

        for name, model in model_dict.items():
            m = clone(model)
            m.fit(X_tr_f, y_tr_f)
            y_pred = m.predict(X_val_f)
            y_prob = m.predict_proba(X_val_f)[:, 1]

            oof_preds[name][val_idx] = y_pred
            oof_probs[name][val_idx] = y_prob

            acc = accuracy_score(y_val_f, y_pred)
            prec = precision_score(y_val_f, y_pred, zero_division=0)
            rec = recall_score(y_val_f, y_pred)
            f1 = f1_score(y_val_f, y_pred)
            roc = roc_auc_score(y_val_f, y_prob)
            ap = average_precision_score(y_val_f, y_prob)

            cv_results_list.append({
                "Modelo": name,
                "Fold": fold_idx,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1_Score": f1,
                "ROC_AUC": roc,
                "Average_Precision": ap
            })

    df_cv_all = pd.DataFrame(cv_results_list)

    # Resumen CV con media y std
    df_cv_summary = df_cv_all.groupby("Modelo").agg(
        Accuracy_Mean=("Accuracy", "mean"),
        Accuracy_Std=("Accuracy", "std"),
        Precision_Mean=("Precision", "mean"),
        Precision_Std=("Precision", "std"),
        Recall_Mean=("Recall", "mean"),
        Recall_Std=("Recall", "std"),
        F1_Mean=("F1_Score", "mean"),
        F1_Std=("F1_Score", "std"),
        ROC_AUC_Mean=("ROC_AUC", "mean"),
        ROC_AUC_Std=("ROC_AUC", "std"),
        AP_Mean=("Average_Precision", "mean"),
        AP_Std=("Average_Precision", "std")
    ).reset_index()

    # Ordenar por ROC_AUC_Mean descendente
    df_cv_summary = df_cv_summary.sort_values(by="ROC_AUC_Mean", ascending=False)
    print("\nResumen Validación Cruzada (5-Fold CV):")
    print(df_cv_summary[["Modelo", "ROC_AUC_Mean", "AP_Mean", "F1_Mean", "Recall_Mean", "Precision_Mean", "Accuracy_Mean"]].to_string(index=False))

    df_cv_summary.to_csv(DIR_TAB / "10_comparativa_modelos_cv.csv", index=False)

    # ---------------------------------------------------------
    # 6. Evaluación en Conjunto de Prueba Independiente (Test)
    # ---------------------------------------------------------
    print("\n[5/7] Evaluando modelos finales en el conjunto de prueba (Test)...")
    test_metrics = []
    test_probs = {}
    test_preds = {}

    for name, model in model_dict.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        test_probs[name] = y_prob
        test_preds[name] = y_pred

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)

        test_metrics.append({
            "Modelo": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1_Score": f1,
            "ROC_AUC": roc,
            "Average_Precision": ap
        })

    df_test_summary = pd.DataFrame(test_metrics).sort_values(by="ROC_AUC", ascending=False)
    print("\nDesempeño en Test Set Independiente (20%):")
    print(df_test_summary.to_string(index=False))
    df_test_summary.to_csv(DIR_TAB / "11_evaluacion_test_set.csv", index=False)

    # Classification Reports
    for name in ["Regresion Logistica", "Arbol de Decision", "Random Forest"]:
        rep = classification_report(y_test, test_preds[name], digits=4, output_dict=True)
        df_rep = pd.DataFrame(rep).transpose()
        df_rep.to_csv(DIR_TAB / f"12_classification_report_{name.lower().replace(' ', '_')}.csv")

    # ---------------------------------------------------------
    # 7. Evaluación Temporal Complementaria (2008-2009 vs 2010)
    # ---------------------------------------------------------
    print("\n[6/7] Ejecutando evaluación temporal complementaria (2008-2009 vs 2010)...")
    anio = df_prep["anio"].values
    mask_train_time = (anio == 2008) | (anio == 2009)
    mask_test_time = (anio == 2010)

    X_train_time, y_train_time = X[mask_train_time], y[mask_train_time]
    X_test_time, y_test_time = X[mask_test_time], y[mask_test_time]

    time_results = []
    for name in ["Regresion Logistica", "Arbol de Decision", "Random Forest"]:
        m = clone(model_dict[name])
        m.fit(X_train_time, y_train_time)
        prob = m.predict_proba(X_test_time)[:, 1]
        pred = m.predict(X_test_time)

        time_results.append({
            "Modelo": name,
            "Train_Periodo": "2008-2009 (42,588 obs)",
            "Test_Periodo": "2010 (2,620 obs)",
            "Tasa_Test_Real_%": y_test_time.mean() * 100,
            "Accuracy": accuracy_score(y_test_time, pred),
            "Recall": recall_score(y_test_time, pred),
            "Precision": precision_score(y_test_time, pred),
            "F1_Score": f1_score(y_test_time, pred),
            "ROC_AUC": roc_auc_score(y_test_time, prob),
            "Average_Precision": average_precision_score(y_test_time, prob)
        })

    df_time = pd.DataFrame(time_results)
    print(df_time[["Modelo", "ROC_AUC", "Average_Precision", "Recall", "Precision", "F1_Score"]].to_string(index=False))
    df_time.to_csv(DIR_TAB / "13_evaluacion_temporal_2008_2009_vs_2010.csv", index=False)

    # ---------------------------------------------------------
    # 8. Generación de las 12 Visualizaciones Profesionales
    # ---------------------------------------------------------
    print("\n[7/7] Generando las 12 visualizaciones de nivel profesional (300 DPI)...")

    # FIGURA 1: Distribución de la variable objetivo
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(["No Contrata (0)", "Contrata Depósito (1)"], [conteo_yb[0], conteo_yb[1]],
                  color=[PALETTE["neg"], PALETTE["pos"]], width=0.55, edgecolor="#334155", linewidth=1.2)
    ax.set_title("Figura 1: Distribución de la Variable Objetivo (yb)\nEvidencia de Desbalance de Clases Severo", pad=15)
    ax.set_ylabel("Frecuencia Absoluta (Clientes)")
    ax.set_ylim(0, 45000)
    for bar, pct in zip(bars, [prop_yb[0], prop_yb[1]]):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 1000, f"{yval:,}\n({pct:.2f}%)",
                ha="center", va="bottom", fontsize=11, fontweight="bold", color=PALETTE["dark"])
    ax.axhline(conteo_yb.mean(), color=PALETTE["accent1"], linestyle=":", linewidth=1.5, label=f"Media teórica balanceada ({int(conteo_yb.mean()):,})")
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig1_path = DIR_FIG / "fig01_distribucion_variable_objetivo.png"
    plt.savefig(fig1_path)
    plt.close()
    print("  -> Generada:", fig1_path.name)

    # FIGURA 2: Comparación de proporciones Train vs Test
    fig, ax = plt.subplots(figsize=(9, 5))
    x_indices = np.arange(2)
    width = 0.35
    b1 = ax.bar(x_indices - width/2, [y_train.mean()*100, (1-y_train.mean())*100], width,
                label=f"Train (N={len(y_train):,})", color=PALETTE["accent1"], edgecolor="#1E293B", linewidth=1)
    b2 = ax.bar(x_indices + width/2, [y_test.mean()*100, (1-y_test.mean())*100], width,
                label=f"Test (N={len(y_test):,})", color=PALETTE["accent2"], edgecolor="#1E293B", linewidth=1)
    ax.set_title("Figura 2: Verificación de Estratificación entre Particiones\nConsistencia de la Tasa de Conversión en Train (80%) y Test (20%)", pad=15)
    ax.set_xticks(x_indices)
    ax.set_xticklabels(["Contratan (Clase 1)", "No Contratan (Clase 0)"])
    ax.set_ylabel("Porcentaje dentro del conjunto (%)")
    ax.set_ylim(0, 100)
    for bar in list(b1) + list(b2):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 2, f"{yval:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig2_path = DIR_FIG / "fig02_proporciones_particiones_train_test.png"
    plt.savefig(fig2_path)
    plt.close()
    print("  -> Generada:", fig2_path.name)

    # FIGURA 3: Comparación de métricas de los modelos
    fig, ax = plt.subplots(figsize=(12, 6))
    metrics_to_plot = ["ROC_AUC_Mean", "AP_Mean", "F1_Mean", "Recall_Mean", "Precision_Mean"]
    labels_metrics = ["ROC-AUC", "Average Prec.", "F1-Score", "Recall", "Precision"]
    model_colors = [PALETTE["dummy"], PALETTE["lr"], PALETTE["dt"], PALETTE["rf"]]
    model_names_order = ["Baseline (Dummy)", "Regresion Logistica", "Arbol de Decision", "Random Forest"]

    x_pos = np.arange(len(metrics_to_plot))
    bar_w = 0.18

    for i, name in enumerate(model_names_order):
        row = df_cv_summary[df_cv_summary["Modelo"] == name].iloc[0]
        vals = [row[m] for m in metrics_to_plot]
        offset = (i - 1.5) * bar_w
        bars = ax.bar(x_pos + offset, vals, bar_w, label=name, color=model_colors[i], edgecolor="#1E293B", linewidth=1)
        for b in bars:
            height = b.get_height()
            if height > 0.05:
                ax.text(b.get_x() + b.get_width()/2, height + 0.015, f"{height:.2f}",
                        ha="center", va="bottom", fontsize=8, rotation=90, fontweight="bold")

    ax.set_title("Figura 3: Comparación Multidimensional de Métricas en Validación Cruzada (5-Fold CV)\nDesempeño bajo Desbalance de Clases", pad=15)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels_metrics, fontweight="bold")
    ax.set_ylabel("Puntuación Media CV")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", ncol=2)
    plt.tight_layout()
    fig3_path = DIR_FIG / "fig03_comparacion_metricas_modelos.png"
    plt.savefig(fig3_path)
    plt.close()
    print("  -> Generada:", fig3_path.name)

    # FIGURA 4: Curvas ROC
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    plot_models = [
        ("Baseline (Dummy)", test_probs["Baseline (Dummy)"], PALETTE["dummy"], ":"),
        ("Regresion Logistica", test_probs["Regresion Logistica"], PALETTE["lr"], "-"),
        ("Arbol de Decision", test_probs["Arbol de Decision"], PALETTE["dt"], "--"),
        ("Random Forest", test_probs["Random Forest"], PALETTE["rf"], "-")
    ]
    for name, probs, color, lstyle in plot_models:
        fpr, tpr, _ = roc_curve(y_test, probs)
        score = roc_auc_score(y_test, probs)
        lwidth = 2.5 if name == "Random Forest" else 1.8
        ax.plot(fpr, tpr, label=f"{name} (AUC = {score:.4f})", color=color, linestyle=lstyle, linewidth=lwidth)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Clasificador Aleatorio (AUC = 0.5000)")
    ax.set_title("Figura 4: Curvas ROC en Conjunto de Prueba Independiente\nCapacidad Discriminativa Global", pad=15)
    ax.set_xlabel("Tasa de Falsos Positivos (1 - Especificidad)")
    ax.set_ylabel("Tasa de Verdaderos Positivos (Sensibilidad / Recall)")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.legend(loc="lower right", framealpha=0.95)
    plt.tight_layout()
    fig4_path = DIR_FIG / "fig04_curvas_roc_comparativas.png"
    plt.savefig(fig4_path)
    plt.close()
    print("  -> Generada:", fig4_path.name)

    # FIGURA 5: Curvas Precision-Recall
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for name, probs, color, lstyle in plot_models:
        prec, rec, _ = precision_recall_curve(y_test, probs)
        ap = average_precision_score(y_test, probs)
        lwidth = 2.5 if name == "Random Forest" else 1.8
        ax.plot(rec, prec, label=f"{name} (AP = {ap:.4f})", color=color, linestyle=lstyle, linewidth=lwidth)

    baseline_pr = y_test.mean()
    ax.axhline(baseline_pr, color="#64748B", linestyle=":", label=f"Línea Base Positiva ({baseline_pr*100:.2f}%)")
    ax.set_title("Figura 5: Curvas Precision-Recall en Conjunto de Prueba\nEvaluación Crítica para Detección de la Clase Minoritaria", pad=15)
    ax.set_xlabel("Recall (Sensibilidad)")
    ax.set_ylabel("Precision (Valor Predictivo Positivo)")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.legend(loc="upper right", framealpha=0.95)
    plt.tight_layout()
    fig5_path = DIR_FIG / "fig05_curvas_precision_recall.png"
    plt.savefig(fig5_path)
    plt.close()
    print("  -> Generada:", fig5_path.name)

    # FIGURA 6: Matrices de Confusión comparadas
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    cm_models = [
        ("Regresión Logística", test_preds["Regresion Logistica"], axes[0], "Purples"),
        ("Árbol de Decisión", test_preds["Arbol de Decision"], axes[1], "Oranges"),
        ("Random Forest", test_preds["Random Forest"], axes[2], "Greens")
    ]

    for title, preds, ax_sub, cmap_name in cm_models:
        cm = confusion_matrix(y_test, preds)
        cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
        annot = np.array([
            [f"{cm[0,0]:,}\n({cm_norm[0,0]*100:.1f}%)", f"{cm[0,1]:,}\n({cm_norm[0,1]*100:.1f}%)"],
            [f"{cm[1,0]:,}\n({cm_norm[1,0]*100:.1f}%)", f"{cm[1,1]:,}\n({cm_norm[1,1]*100:.1f}%)"]
        ])
        sns.heatmap(cm, annot=annot, fmt="", cmap=cmap_name, cbar=False, ax=ax_sub,
                    linewidths=1.5, linecolor="#FFFFFF", annot_kws={"size": 11, "fontweight": "bold"})
        ax_sub.set_title(title, pad=12, fontsize=12, fontweight="bold")
        ax_sub.set_xlabel("Predicción del Modelo")
        ax_sub.set_ylabel("Valor Real Observado")
        ax_sub.set_xticklabels(["No (0)", "Sí (1)"])
        ax_sub.set_yticklabels(["No (0)", "Sí (1)"])

    fig.suptitle("Figura 6: Matrices de Confusión en Conjunto de Prueba Independiente (N = 9,042)\nCompromiso entre Captura de Clientes (Recall) y Eficiencia (Precision)", y=1.03)
    plt.tight_layout()
    fig6_path = DIR_FIG / "fig06_matrices_confusion_comparadas.png"
    plt.savefig(fig6_path, bbox_inches="tight")
    plt.close()
    print("  -> Generada:", fig6_path.name)

    # FIGURA 7: Importancia de variables Random Forest
    rf_est = best_rf
    importances = rf_est.feature_importances_
    indices = np.argsort(importances)[::-1][:15] # Top 15
    top_feats = [feature_names[i] for i in indices]
    top_imps = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 6.5))
    y_pos = np.arange(len(top_feats))[::-1]
    bars = ax.barh(y_pos, top_imps, color=PALETTE["rf"], edgecolor="#1E293B", height=0.65)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_feats, fontsize=10)
    ax.set_xlabel("Importancia de la Característica (Mean Decrease in Impurity - Gini)")
    ax.set_title("Figura 7: Top 15 Variables más Importantes en Random Forest\nFactores Determinantes en la Decisión de Suscripción", pad=15)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.003, bar.get_y() + bar.get_height()/2, f"{w*100:.2f}%",
                va="center", ha="left", fontsize=9, fontweight="bold", color=PALETTE["dark"])
    ax.set_xlim(0, max(top_imps) * 1.15)
    plt.tight_layout()
    fig7_path = DIR_FIG / "fig07_importancia_variables_random_forest.png"
    plt.savefig(fig7_path)
    plt.close()
    print("  -> Generada:", fig7_path.name)

    # FIGURA 8: Coeficientes de Regresión Logística
    lr_est = best_lr.named_steps["clf"]
    coefs = lr_est.coef_[0]
    df_coef = pd.DataFrame({"Feature": feature_names, "Coef": coefs})
    df_coef["AbsCoef"] = df_coef["Coef"].abs()
    top_pos = df_coef.sort_values(by="Coef", ascending=False).head(8)
    top_neg = df_coef.sort_values(by="Coef", ascending=True).head(8)
    top_combined = pd.concat([top_pos, top_neg]).sort_values(by="Coef", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    y_pos = np.arange(len(top_combined))
    colors_coef = [PALETTE["neg"] if c < 0 else PALETTE["pos"] for c in top_combined["Coef"]]
    bars = ax.barh(y_pos, top_combined["Coef"], color=colors_coef, edgecolor="#1E293B", height=0.65)
    ax.axvline(0, color="#1E293B", linewidth=1.2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_combined["Feature"], fontsize=10)
    ax.set_xlabel("Valor del Coeficiente Estandarizado (Log-Odds)")
    ax.set_title("Figura 8: Coeficientes más Influyentes de la Regresión Logística\nImpacto Direccional Positivo (Turquesa) y Negativo (Coral)", pad=15)
    for bar, val in zip(bars, top_combined["Coef"]):
        ha = "left" if val > 0 else "right"
        offset = 0.03 if val > 0 else -0.03
        ax.text(val + offset, bar.get_y() + bar.get_height()/2, f"{val:.3f}",
                va="center", ha=ha, fontsize=9, fontweight="bold", color=PALETTE["dark"])
    plt.tight_layout()
    fig8_path = DIR_FIG / "fig08_coeficientes_regresion_logistica.png"
    plt.savefig(fig8_path)
    plt.close()
    print("  -> Generada:", fig8_path.name)

    # FIGURA 9: Árbol de Decisión visualizado
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(best_dt, max_depth=3, feature_names=feature_names, class_names=["No", "Sí"],
              filled=True, rounded=True, precision=2, ax=ax, fontsize=9)
    ax.set_title("Figura 9: Estructura de Decisión Jerárquica del Árbol (Profundidad = 3)\nReglas de Negocio Explícitas para Segmentación y Asignación", pad=15, fontsize=14)
    plt.tight_layout()
    fig9_path = DIR_FIG / "fig09_arbol_decision_visualizado.png"
    plt.savefig(fig9_path, dpi=300)
    plt.close()
    print("  -> Generada:", fig9_path.name)

    # FIGURA 10: Curvas de aprendizaje / Sobreajuste Train vs Val
    print("  -> Calculando curvas de sobreajuste...")
    depths = [2, 4, 6, 8, 10, 14]
    train_scores_dt = []
    val_scores_dt = []
    train_scores_rf = []
    val_scores_rf = []

    for d in depths:
        dt_t = DecisionTreeClassifier(max_depth=d, min_samples_leaf=20, class_weight="balanced", random_state=42)
        dt_t.fit(X_train, y_train)
        train_scores_dt.append(roc_auc_score(y_train, dt_t.predict_proba(X_train)[:, 1]))
        val_scores_dt.append(roc_auc_score(y_test, dt_t.predict_proba(X_test)[:, 1]))

        rf_t = RandomForestClassifier(n_estimators=100, max_depth=d, min_samples_leaf=10, class_weight="balanced", random_state=42, n_jobs=-1)
        rf_t.fit(X_train, y_train)
        train_scores_rf.append(roc_auc_score(y_train, rf_t.predict_proba(X_train)[:, 1]))
        val_scores_rf.append(roc_auc_score(y_test, rf_t.predict_proba(X_test)[:, 1]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    ax1.plot(depths, train_scores_dt, "o-", color=PALETTE["neg"], label="Train ROC-AUC", linewidth=2)
    ax1.plot(depths, val_scores_dt, "s-", color=PALETTE["dt"], label="Test ROC-AUC", linewidth=2)
    ax1.set_title("Árbol de Decisión: Variación de Profundidad Máxima", pad=12)
    ax1.set_xlabel("Profundidad Máxima (max_depth)")
    ax1.set_ylabel("ROC-AUC")
    ax1.set_ylim(0.65, 1.0)
    ax1.legend(loc="lower right")

    ax2.plot(depths, train_scores_rf, "o-", color=PALETTE["neg"], label="Train ROC-AUC", linewidth=2)
    ax2.plot(depths, val_scores_rf, "s-", color=PALETTE["rf"], label="Test ROC-AUC", linewidth=2)
    ax2.set_title("Random Forest: Variación de Profundidad Máxima", pad=12)
    ax2.set_xlabel("Profundidad Máxima (max_depth)")
    ax2.set_ylabel("ROC-AUC")
    ax2.set_ylim(0.65, 1.0)
    ax2.legend(loc="lower right")

    fig.suptitle("Figura 10: Diagnóstico de Sobreajuste (Overfitting) según Complejidad del Modelo\nBrecha entre Desempeño en Entrenamiento y Prueba", y=1.03)
    plt.tight_layout()
    fig10_path = DIR_FIG / "fig10_curvas_aprendizaje_sobreajuste.png"
    plt.savefig(fig10_path, bbox_inches="tight")
    plt.close()
    print("  -> Generada:", fig10_path.name)

    # ---------------------------------------------------------
    # 9. Aplicación Comercial: Tabla de Deciles, Lift y Ganancia
    # ---------------------------------------------------------
    rf_test_probs = test_probs["Random Forest"]
    df_test_business = pd.DataFrame({
        "y_true": y_test.values,
        "prob_rf": rf_test_probs
    }).sort_values(by="prob_rf", ascending=False).reset_index(drop=True)

    df_test_business["decil"] = pd.qcut(df_test_business.index, q=10, labels=[f"D{i}" for i in range(1, 11)])

    tabla_deciles = df_test_business.groupby("decil", observed=True).agg(
        Total_Clientes=("y_true", "count"),
        Contrataciones=("y_true", "sum"),
        Tasa_Conversion=("y_true", "mean")
    ).reset_index()

    tasa_base = y_test.mean()
    total_positivos = y_test.sum()

    tabla_deciles["Tasa_Conversion_%"] = tabla_deciles["Tasa_Conversion"] * 100
    tabla_deciles["Contrataciones_Acum"] = tabla_deciles["Contrataciones"].cumsum()
    tabla_deciles["Clientes_Acum"] = tabla_deciles["Total_Clientes"].cumsum()
    tabla_deciles["Pct_Poblacion_%"] = (tabla_deciles["Clientes_Acum"] / len(y_test)) * 100
    tabla_deciles["Ganancia_Acumulada_%"] = (tabla_deciles["Contrataciones_Acum"] / total_positivos) * 100
    tabla_deciles["Lift"] = tabla_deciles["Tasa_Conversion"] / tasa_base
    tabla_deciles["Lift_Acumulado"] = (tabla_deciles["Contrataciones_Acum"] / tabla_deciles["Clientes_Acum"]) / tasa_base

    print("\nTabla de Deciles y Lift Comercial (Random Forest en Test):")
    print(tabla_deciles[["decil", "Total_Clientes", "Contrataciones", "Tasa_Conversion_%", "Ganancia_Acumulada_%", "Lift", "Lift_Acumulado"]].to_string(index=False))
    tabla_deciles.to_csv(DIR_TAB / "14_tabla_deciles_lift_comercial.csv", index=False)

    # FIGURA 11: Curvas de Ganancia Acumulada y Lift
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    pop_pct = np.insert(tabla_deciles["Pct_Poblacion_%"].values, 0, 0)
    gain_pct = np.insert(tabla_deciles["Ganancia_Acumulada_%"].values, 0, 0)

    ax1.plot(pop_pct, gain_pct, "o-", color=PALETTE["rf"], linewidth=2.5, label="Random Forest")
    ax1.plot([0, 100], [0, 100], "k--", alpha=0.6, label="Línea Base Aleatoria (Sin Modelo)")
    ax1.axvline(20, color=PALETTE["accent2"], linestyle=":", linewidth=1.5, label="Corte Top 20% Población")
    gain_20 = tabla_deciles.loc[tabla_deciles["decil"]=="D2", "Ganancia_Acumulada_%"].values[0]
    ax1.plot(20, gain_20, "ro", markersize=8)
    ax1.annotate(f"Top 20% captura el {gain_20:.1f}%\nde todos los depósitos",
                 (20, gain_20), textcoords="offset points", xytext=(15, -15),
                 arrowprops=dict(arrowstyle="->", color=PALETTE["dark"]), fontweight="bold")

    ax1.set_title("Curva de Ganancia Acumulada (Cumulative Gains)", pad=12)
    ax1.set_xlabel("% de Clientes Contactados (Ordenados por Probabilidad)")
    ax1.set_ylabel("% de Contrataciones Totales Capturadas")
    ax1.set_xlim(0, 102)
    ax1.set_ylim(0, 105)
    ax1.legend(loc="lower right")

    decils_x = np.arange(1, 11)
    ax2.plot(decils_x, tabla_deciles["Lift"].values, "s-", color=PALETTE["accent1"], linewidth=2.5, label="Lift por Decil")
    ax2.plot(decils_x, tabla_deciles["Lift_Acumulado"].values, "o-", color=PALETTE["rf"], linewidth=2.5, label="Lift Acumulado")
    ax2.axhline(1.0, color="k", linestyle="--", alpha=0.6, label="Línea Base (Lift = 1.0)")
    ax2.set_title("Curva de Lift por Decil y Acumulado", pad=12)
    ax2.set_xlabel("Decil de Clientes (D1 = Mayor Probabilidad)")
    ax2.set_ylabel("Factor de Multiplicación (Lift vs Tasa Base)")
    ax2.set_xticks(decils_x)
    ax2.set_xticklabels([f"D{i}" for i in range(1, 11)])
    ax2.set_ylim(0, max(tabla_deciles["Lift"]) * 1.15)
    for x, val in zip(decils_x[:3], tabla_deciles["Lift"].values[:3]):
        ax2.text(x, val + 0.15, f"{val:.2f}x", ha="center", fontsize=9, fontweight="bold", color=PALETTE["dark"])
    ax2.legend(loc="upper right")

    fig.suptitle("Figura 11: Eficiencia Comercial y Curvas de Ganancia del Modelo Predictivo", y=1.02)
    plt.tight_layout()
    fig11_path = DIR_FIG / "fig11_curva_lift_ganancia_acumulada.png"
    plt.savefig(fig11_path, bbox_inches="tight")
    plt.close()
    print("  -> Generada:", fig11_path.name)

    # FIGURA 12: Priorización comercial Top 10% y Top 20%
    d1 = tabla_deciles.iloc[0]
    d2_cum = tabla_deciles.iloc[1]

    top_data = pd.DataFrame({
        "Estrategia": ["Base Aleatoria (10%)", "Top 10% Modelo", "Base Aleatoria (20%)", "Top 20% Modelo"],
        "Clientes_Contactados": [int(len(y_test)*0.1), d1["Clientes_Acum"], int(len(y_test)*0.2), d2_cum["Clientes_Acum"]],
        "Contrataciones_Capturadas": [int(total_positivos*0.1), d1["Contrataciones_Acum"], int(total_positivos*0.2), d2_cum["Contrataciones_Acum"]],
        "Tasa_Conversion_%": [tasa_base*100, d1["Tasa_Conversion_%"], tasa_base*100, (d2_cum["Contrataciones_Acum"]/d2_cum["Clientes_Acum"])*100],
        "Color": [PALETTE["dummy"], PALETTE["pos"], PALETTE["dummy"], PALETTE["rf"]]
    })

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(top_data["Estrategia"], top_data["Contrataciones_Capturadas"],
                  color=top_data["Color"], edgecolor="#1E293B", width=0.55)
    ax.set_title("Figura 12: Simulación de Captura Comercial de Clientes\nEstrategia Aleatoria vs Focalización Predictiva (Top 10% y Top 20%)", pad=15)
    ax.set_ylabel("Depósitos Contratados Capturados (en N = 9,042 de Test)")
    ax.set_ylim(0, total_positivos * 0.8)

    for bar, (_, row) in zip(bars, top_data.iterrows()):
        h = bar.get_height()
        pct = (h / total_positivos) * 100
        ax.text(bar.get_x() + bar.get_width()/2, h + 15,
                f"{h} depósitos\n({pct:.1f}% del total)\nTasa: {row['Tasa_Conversion_%']:.1f}%",
                ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=PALETTE["dark"])

    ax.axhline(total_positivos, color=PALETTE["accent2"], linestyle=":", linewidth=1.5, label=f"Total Depósitos en Test ({total_positivos})")
    ax.legend(loc="upper left")
    plt.tight_layout()
    fig12_path = DIR_FIG / "fig12_priorizacion_top_clientes.png"
    plt.savefig(fig12_path)
    plt.close()
    print("  -> Generada:", fig12_path.name)

    print("\n" + "=" * 70)
    print("EJECUCION COMPLETADA EXITOSAMENTE. TODOS LOS ENTREGABLES GUARDADOS.")
    print("=" * 70)

if __name__ == "__main__":
    main()
