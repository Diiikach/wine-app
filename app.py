import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics

APP_DIR = Path(__file__).parent

FEATURES = [
    'fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
    'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
    'pH', 'sulphates', 'alcohol', 'is_red',
]

st.set_page_config(page_title='Wine Quality', layout='wide')


@st.cache_resource
def load_model():
    return joblib.load(APP_DIR / 'model.joblib')


@st.cache_data
def load_config():
    return json.loads((APP_DIR / 'config.json').read_text())


@st.cache_data
def load_test_data():
    return pd.read_csv(APP_DIR / 'test.csv')


@st.cache_data
def test_proba(_model, X):
    return _model.predict_proba(X)[:, 1]


model = load_model()
config = load_config()
test_df = load_test_data()

X_test = test_df.drop(columns='target')
y_test = test_df['target']
proba_test = test_proba(model, X_test)

ranges = X_test.describe().T[['min', 'max', '50%']]

st.sidebar.header('Порог решения')
threshold = st.sidebar.slider(
    'Порог t', min_value=0.0, max_value=1.0,
    value=float(config['threshold']), step=0.01,
)
st.session_state['threshold'] = threshold
st.sidebar.caption(f'Порог по умолчанию из config.json: {config["threshold"]:.2f}')

tab1, tab2, tab3, tab4 = st.tabs([
    'Оценка одного вина', 'Модельные метрики', 'Пакетное предсказание', 'Что выучила модель',
])

with tab1:
    st.subheader('Оценка одного вина')

    cols = st.columns(3)
    values = {}
    numeric_features = [f for f in FEATURES if f != 'is_red']
    for i, feat in enumerate(numeric_features):
        col = cols[i % 3]
        lo, hi, med = ranges.loc[feat, ['min', 'max', '50%']]
        values[feat] = col.number_input(
            feat, min_value=float(lo), max_value=float(hi), value=float(med),
        )

    is_red = st.toggle('Красное вино', value=True)
    values['is_red'] = 1 if is_red else 0

    sample = pd.DataFrame([values])[FEATURES]
    proba_one = model.predict_proba(sample)[0, 1]
    decision = 'хорошее' if proba_one >= threshold else 'не хорошее'

    st.metric('Вероятность "хорошего" вина', f'{proba_one:.2%}')
    st.write(f'Решение модели при пороге {threshold:.2f}: **{decision}**')

with tab2:
    st.subheader('Модельные метрики')

    pred_test = (proba_test >= threshold).astype(int)

    col1, col2 = st.columns(2)

    with col1:
        cm = metrics.confusion_matrix(y_test, pred_test)
        disp = metrics.ConfusionMatrixDisplay(cm, display_labels=['не хорошее', 'хорошее'])
        fig_cm, ax_cm = disp.plot().figure_, disp.ax_
        st.pyplot(fig_cm)

    with col2:
        precision = metrics.precision_score(y_test, pred_test, zero_division=0)
        recall = metrics.recall_score(y_test, pred_test, zero_division=0)
        f1 = metrics.f1_score(y_test, pred_test, zero_division=0)
        st.metric('Precision', f'{precision:.3f}')
        st.metric('Recall', f'{recall:.3f}')
        st.metric('F1', f'{f1:.3f}')

    col3, col4 = st.columns(2)

    with col3:
        fpr, tpr, _ = metrics.roc_curve(y_test, proba_test)
        cur_fpr = np.sum((proba_test >= threshold) & (y_test == 0)) / np.sum(y_test == 0)
        cur_tpr = recall
        import matplotlib.pyplot as plt
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, label='ROC')
        ax_roc.plot([0, 1], [0, 1], linestyle='--', color='gray')
        ax_roc.scatter([cur_fpr], [cur_tpr], color='red', zorder=5, label=f't={threshold:.2f}')
        ax_roc.set_xlabel('FPR'); ax_roc.set_ylabel('TPR'); ax_roc.legend()
        st.pyplot(fig_roc)

    with col4:
        prec_curve, rec_curve, _ = metrics.precision_recall_curve(y_test, proba_test)
        fig_pr, ax_pr = plt.subplots()
        ax_pr.plot(rec_curve, prec_curve, label='PR')
        ax_pr.scatter([recall], [precision], color='red', zorder=5, label=f't={threshold:.2f}')
        ax_pr.set_xlabel('Recall'); ax_pr.set_ylabel('Precision'); ax_pr.legend()
        st.pyplot(fig_pr)

    st.info(
        'Если порог увеличить — модель станет строже: подборка будет содержать меньше вин, '
        'но среди рекомендованных будет выше доля действительно хороших (растёт precision, падает recall). '
        'Если порог уменьшить — подборка станет шире, в неё попадёт больше хороших вин, но вместе с ними '
        'и больше посредственных (растёт recall, падает precision).'
    )

with tab3:
    st.subheader('Пакетное предсказание')
    st.caption(f'Используется порог со вкладки "Модельные метрики": {st.session_state["threshold"]:.2f}')

    uploaded = st.file_uploader('Загрузите CSV с признаками вин (без столбца target)', type='csv')

    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
            missing = [f for f in FEATURES if f not in batch_df.columns]
            if missing:
                st.error(f'В файле не хватает столбцов: {missing}')
            else:
                batch_proba = model.predict_proba(batch_df[FEATURES])[:, 1]
                result_df = batch_df.copy()
                result_df['proba'] = batch_proba
                result_df['decision'] = np.where(
                    batch_proba >= st.session_state['threshold'], 'хорошее', 'не хорошее')
                st.dataframe(result_df)
                st.download_button(
                    'Скачать результат',
                    result_df.to_csv(index=False).encode('utf-8'),
                    file_name='predictions.csv',
                    mime='text/csv',
                )
        except Exception as e:
            st.error(f'Не удалось обработать файл: {e}')

with tab4:
    st.subheader('Что выучила модель')

    logreg = model.named_steps['logisticregression']
    coefs = pd.Series(logreg.coef_[0], index=FEATURES).sort_values()

    import matplotlib.pyplot as plt
    fig_coef, ax_coef = plt.subplots(figsize=(6, 5))
    colors = ['tab:red' if c < 0 else 'tab:green' for c in coefs.values]
    ax_coef.barh(coefs.index, coefs.values, color=colors)
    ax_coef.set_xlabel('Коэффициент (после StandardScaler)')
    st.pyplot(fig_coef)

    st.caption(
        'Зелёные столбцы — признаки, которые повышают вероятность "хорошего" вина при росте значения, '
        'красные — понижают. Чем длиннее столбец, тем сильнее признак влияет на решение модели.'
    )
