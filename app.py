import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

@st.cache_data
def load_data():
    df = pd.read_excel("preprocessing2.xlsx")  # Ganti dengan file data Tokopedia Anda
    df['rating'] = df['rating'].astype(str).str.extract('(\d+)')
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(0).astype(int)
    df['sentimen'] = df['rating'].apply(lambda x: 'Positif' if x >= 4 else 'Negatif')
    df = df.dropna(subset=['ulasan_stemmed'])
    return df

data = load_data()

@st.cache_resource
def prepare_models(df):
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_all = tfidf.fit_transform(df['ulasan_stemmed'])
    y_all = df['sentimen'].values

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_all, y_all, df.index, test_size=0.2, random_state=42, stratify=y_all
    )

    train_texts = df.loc[idx_train]

    # Model Naive Bayes
    nb = MultinomialNB()
    nb.fit(X_train, y_train)

    # Model Decision Tree
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)

    # Model Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    return tfidf, nb, dt, rf, X_train, y_train, train_texts

tfidf, nb_model, dt_model, rf_model, X_train, y_train, train_texts = prepare_models(data)

def preprocess_text(text):
    text = text.lower()
    return ''.join(c for c in text if c.isalnum() or c.isspace())

def predict_sentiment(text, model):
    clean = preprocess_text(text)
    tfidf_input = tfidf.transform([clean])
    label = model.predict(tfidf_input)[0]
    prob = model.predict_proba(tfidf_input)[0]
    return label, prob

def evaluate_model(model, X, y_true):
    y_pred = model.predict(X)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label='Positif')
    rec = recall_score(y_true, y_pred, pos_label='Positif')
    f1 = f1_score(y_true, y_pred, pos_label='Positif')
    cm = confusion_matrix(y_true, y_pred, labels=['Positif', 'Negatif'])
    return acc, prec, rec, f1, cm, y_pred

st.set_page_config("Dashboard Tokopedia CS Predictor", layout="wide", page_icon="logo.png")
st.sidebar.image("logo.png", use_column_width=True)  # Ganti dengan logo Tokopedia
st.sidebar.title("Tokopedia CS Predictor")
menu = st.sidebar.radio("Dashboard Tokopedia CS Predictor", ["Dashboard", "Confusion Matrix", "Wordcloud", "Perbandingan Model", "Prediksi Kepuasan Pelanggan"])

if menu == "Dashboard":
    st.title("Dashboard Ulasan Aplikasi Tokopedia")
    st.write("Jumlah data setelah preprocessing:", len(data))
    count_sentiment = data['sentimen'].value_counts()
    st.subheader("Distribusi Sentimen")
    st.bar_chart(count_sentiment)
    st.subheader("Contoh Data")
    st.dataframe(data[['rating', 'ulasan', 'like', 'ulasan_stemmed','sentimen']].sample(10))

elif menu == "Confusion Matrix":
    st.title("Hasil Evaluasi Confusion Matrix")
    acc_nb, _, _, _, cm_nb, _ = evaluate_model(nb_model, X_train, y_train)
    acc_dt, _, _, _, cm_dt, _ = evaluate_model(dt_model, X_train, y_train)
    acc_rf, _, _, _, cm_rf, _ = evaluate_model(rf_model, X_train, y_train)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Naive Bayes")
        st.write(f"Akurasi: {acc_nb:.2f}")
        fig, ax = plt.subplots()
        sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Greens', xticklabels=['Positif', 'Negatif'], yticklabels=['Positif', 'Negatif'], ax=ax)
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Aktual")
        st.pyplot(fig)
    with col2:
        st.subheader("Decision Tree")
        st.write(f"Akurasi: {acc_dt:.2f}")
        fig, ax = plt.subplots()
        sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Oranges', xticklabels=['Positif', 'Negatif'], yticklabels=['Positif', 'Negatif'], ax=ax)
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Aktual")
        st.pyplot(fig)
    with col3:
        st.subheader("Random Forest")
        st.write(f"Akurasi: {acc_rf:.2f}")
        fig, ax = plt.subplots()
        sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', xticklabels=['Positif', 'Negatif'], yticklabels=['Positif', 'Negatif'], ax=ax)
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Aktual")
        st.pyplot(fig)

elif menu == "Wordcloud":
    st.title("Wordcloud Analisa Sentimen")
    _, _, _, _, _, nb_pred = evaluate_model(nb_model, X_train, y_train)
    _, _, _, _, _, dt_pred = evaluate_model(dt_model, X_train, y_train)
    _, _, _, _, _, rf_pred = evaluate_model(rf_model, X_train, y_train)

    def plot_wordcloud(predictions, label, model_name, color):
        filtered = train_texts[predictions == label]
        text = ' '.join(filtered['ulasan_stemmed'])
        if text.strip():
            wc = WordCloud(width=800, height=400, background_color='white', colormap=color).generate(text)
            fig, ax = plt.subplots()
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.write(f"Tidak ada ulasan untuk label {label}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Naive Bayes - Positif")
        plot_wordcloud(nb_pred, 'Positif', "Naive Bayes", 'Greens')
        st.subheader("Naive Bayes - Negatif")
        plot_wordcloud(nb_pred, 'Negatif', "Naive Bayes", 'Reds')
    with col2:
        st.subheader("Decision Tree - Positif")
        plot_wordcloud(dt_pred, 'Positif', "Decision Tree", 'Greens')
        st.subheader("Decision Tree - Negatif")
        plot_wordcloud(dt_pred, 'Negatif', "Decision Tree", 'Reds')
    with col3:
        st.subheader("Random Forest - Positif")
        plot_wordcloud(rf_pred, 'Positif', "Random Forest", 'Greens')
        st.subheader("Random Forest - Negatif")
        plot_wordcloud(rf_pred, 'Negatif', "Random Forest", 'Reds')

elif menu == "Perbandingan Model":
    st.title("Perbandingan Model Confusion Matrix")
    acc_nb, prec_nb, rec_nb, f1_nb, _, _ = evaluate_model(nb_model, X_train, y_train)
    acc_dt, prec_dt, rec_dt, f1_dt, _, _ = evaluate_model(dt_model, X_train, y_train)
    acc_rf, prec_rf, rec_rf, f1_rf, _, _ = evaluate_model(rf_model, X_train, y_train)
    metrics = ["Akurasi", "Presisi", "Recall", "F1-Score"]
    nb_scores = [acc_nb, prec_nb, rec_nb, f1_nb]
    dt_scores = [acc_dt, prec_dt, rec_dt, f1_dt]
    rf_scores = [acc_rf, prec_rf, rec_rf, f1_rf]
    x = np.arange(len(metrics))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.barh(x - width, nb_scores, height=0.25, label="Naive Bayes", color="#2e8b57")
    bars2 = ax.barh(x, dt_scores, height=0.25, label="Decision Tree", color="#ff8c00")
    bars3 = ax.barh(x + width, rf_scores, height=0.25, label="Random Forest", color="#1e90ff")
    ax.set_yticks(x)
    ax.set_yticklabels(metrics)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Skor")
    ax.set_title("Perbandingan Model")
    ax.legend()
    for bar in bars1:
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f"{bar.get_width():.2f}", va='center')
    for bar in bars2:
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f"{bar.get_width():.2f}", va='center')
    for bar in bars3:
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f"{bar.get_width():.2f}", va='center')
    st.pyplot(fig)

elif menu == "Prediksi Kepuasan Pelanggan":
    st.title("Prediksi Ulasan Kepuasan Pelanggan Baru")

    uploaded_file = st.file_uploader("Unggah file CSV atau Excel yang berisi kolom ulasan", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_new = pd.read_csv(uploaded_file)
            else:
                df_new = pd.read_excel(uploaded_file)

            st.write("Kolom ditemukan:", df_new.columns.tolist())
            st.dataframe(df_new.head())

            # Jika ada kolom rating, buat label otomatis
            if 'rating' in df_new.columns:
                df_new = df_new[df_new['rating'] != 3]  # Hilangkan rating netral
                df_new['label'] = df_new['rating'].apply(lambda x: 'Positif' if x > 3 else 'Negatif')

            if 'ulasan' not in df_new.columns:
                st.error("❌ Kolom 'ulasan' tidak ditemukan. Harap pastikan nama kolom tepat.")
                st.stop()

            df_new = df_new.dropna(subset=['ulasan'])
            if df_new['ulasan'].dropna().empty:
                st.error("❌ Kolom 'ulasan' kosong. Harap isi dengan data yang valid.")
                st.stop()

            # Preprocessing
            df_new['clean'] = df_new['ulasan'].apply(preprocess_text)
            tfidf_new = tfidf.transform(df_new['clean'])

            # Prediksi
            df_new['Prediksi_NB'] = nb_model.predict(tfidf_new)
            df_new['Prediksi_DT'] = dt_model.predict(tfidf_new)
            df_new['Prediksi_RF'] = rf_model.predict(tfidf_new)

            # Grafik batang distribusi prediksi
            st.subheader("Grafik Hasil Prediksi Kepuasan Pelanggan")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("*Naive Bayes*")
                nb_counts = df_new['Prediksi_NB'].value_counts()
                fig_nb = px.bar(
                    x=nb_counts.index,
                    y=nb_counts.values,
                    labels={'x': 'Sentimen', 'y': 'Jumlah'},
                    color=nb_counts.index,
                    color_discrete_map={'Positif': '#2ecc71', 'Negatif': '#e74c3c'},
                    title='Distribusi Sentimen - Naive Bayes'
                )
                st.plotly_chart(fig_nb, use_container_width=True)

            with col2:
                st.markdown("*Decision Tree*")
                dt_counts = df_new['Prediksi_DT'].value_counts()
                fig_dt = px.bar(
                    x=dt_counts.index,
                    y=dt_counts.values,
                    labels={'x': 'Sentimen', 'y': 'Jumlah'},
                    color=dt_counts.index,
                    color_discrete_map={'Positif': '#2ecc71', 'Negatif': '#e74c3c'},
                    title='Distribusi Sentimen - Decision Tree'
                )
                st.plotly_chart(fig_dt, use_container_width=True)

            with col3:
                st.markdown("*Random Forest*")
                rf_counts = df_new['Prediksi_RF'].value_counts()
                fig_rf = px.bar(
                    x=rf_counts.index,
                    y=rf_counts.values,
                    labels={'x': 'Sentimen', 'y': 'Jumlah'},
                    color=rf_counts.index,
                    color_discrete_map={'Positif': '#2ecc71', 'Negatif': '#e74c3c'},
                    title='Distribusi Sentimen - Random Forest'
                )
                st.plotly_chart(fig_rf, use_container_width=True)

            # Confusion Matrix jika ada label asli
            if 'label' in df_new.columns:
                st.subheader("Confusion Matrix")

                cm_nb = confusion_matrix(df_new['label'], df_new['Prediksi_NB'], labels=['Positif', 'Negatif'])
                cm_dt = confusion_matrix(df_new['label'], df_new['Prediksi_DT'], labels=['Positif', 'Negatif'])
                cm_rf = confusion_matrix(df_new['label'], df_new['Prediksi_RF'], labels=['Positif', 'Negatif'])

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("*Naive Bayes*")
                    fig1, ax1 = plt.subplots()
                    sns.heatmap(cm_nb, annot=True, fmt='.0f', cmap='Blues',
                                xticklabels=['Positif', 'Negatif'], yticklabels=['Positif', 'Negatif'], ax=ax1)
                    ax1.set_xlabel("Prediksi")
                    ax1.set_ylabel("Aktual")
                    st.pyplot(fig1)

                with col2:
                    st.markdown("*Decision Tree*")
                    fig2, ax2 = plt.subplots()
                    sns.heatmap(cm_dt, annot=True, fmt='.0f', cmap='Oranges',
                                xticklabels=['Positif', 'Negatif'], yticklabels=['Positif', 'Negatif'], ax=ax2)
                    ax2.set_xlabel("Prediksi")
                    ax2.set_ylabel("Aktual")
                    st.pyplot(fig2)

                with col3:
                    st.markdown("*Random Forest*")
                    fig3, ax3 = plt.subplots()
                    sns.heatmap(cm_rf, annot=True, fmt='.0f', cmap='Greens',
                                xticklabels=['Positif', 'Negatif'], yticklabels=['Positif', 'Negatif'], ax=ax3)
                    ax3.set_xlabel("Prediksi")
                    ax3.set_ylabel("Aktual")
                    st.pyplot(fig3)

                # Metrik
                acc_nb = accuracy_score(df_new['label'], df_new['Prediksi_NB'])
                acc_dt = accuracy_score(df_new['label'], df_new['Prediksi_DT'])
                acc_rf = accuracy_score(df_new['label'], df_new['Prediksi_RF'])
                prec_nb = precision_score(df_new['label'], df_new['Prediksi_NB'], pos_label='Positif')
                prec_dt = precision_score(df_new['label'], df_new['Prediksi_DT'], pos_label='Positif')
                prec_rf = precision_score(df_new['label'], df_new['Prediksi_RF'], pos_label='Positif')
                rec_nb = recall_score(df_new['label'], df_new['Prediksi_NB'], pos_label='Positif')
                rec_dt = recall_score(df_new['label'], df_new['Prediksi_DT'], pos_label='Positif')
                rec_rf = recall_score(df_new['label'], df_new['Prediksi_RF'], pos_label='Positif')
                f1_nb = f1_score(df_new['label'], df_new['Prediksi_NB'], pos_label='Positif')
                f1_dt = f1_score(df_new['label'], df_new['Prediksi_DT'], pos_label='Positif')
                f1_rf = f1_score(df_new['label'], df_new['Prediksi_RF'], pos_label='Positif')

                df_metrik = pd.DataFrame({
                    'Model': ['Naive Bayes', 'Decision Tree', 'Random Forest'],
                    'Akurasi': [round(acc_nb, 2), round(acc_dt, 2), round(acc_rf, 2)],
                    'Presisi': [round(prec_nb, 2), round(prec_dt, 2), round(prec_rf, 2)],
                    'Recall': [round(rec_nb, 2), round(rec_dt, 2), round(rec_rf, 2)],
                    'F1-Score': [round(f1_nb, 2), round(f1_dt, 2), round(f1_rf, 2)]
                })

                st.subheader("Perbandingan Model")
                fig_bar = px.bar(
                    df_metrik.melt(id_vars='Model'),
                    x='value', y='variable', color='Model', orientation='h',
                    barmode='group',
                    labels={'value': 'Skor', 'variable': 'Metrik'},
                    color_discrete_map={'Naive Bayes': '#2980b9', 'Decision Tree': '#e67e22', 'Random Forest': '#27ae60'}
                )
                st.plotly_chart(fig_bar, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Gagal memproses file: {e}")
    else:
        st.info("Silakan unggah file CSV/XLSX terlebih dahulu.")