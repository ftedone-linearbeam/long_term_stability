import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("Analisi Misurazioni Elettriche (mA)")

# Caricamento file Excel
uploaded_file = st.file_uploader("Carica file Excel", type=["xlsx", "xls"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.subheader("Anteprima del file")
    st.dataframe(df.head())

    # Selezione colonna di interesse
    columns = df.columns.tolist()
    selected_column = st.selectbox("Seleziona la colonna delle misurazioni (mA)", columns)

    if selected_column:
        # Estrai i valori
        data = df[selected_column].dropna().reset_index(drop=True)

        # Crea asse temporale (1 valore/sec)
        time = np.arange(len(data))  # tempo in secondi

        # Sidebar per filtri
        st.sidebar.header("Filtri")
        min_val = float(data.min())
        max_val = float(data.max())
        soglia_inf = st.sidebar.slider("Soglia inferiore (mA)", min_val, max_val, min_val)
        soglia_sup = st.sidebar.slider("Soglia superiore (mA)", min_val, max_val, max_val)

        # Filtra i dati
        mask = (data >= soglia_inf) & (data <= soglia_sup)
        filtered_data = data[mask]
        filtered_time = time[mask]

        # Mostra grafico
        st.subheader("Grafico delle misurazioni")
        fig, ax = plt.subplots()
        ax.plot(filtered_time, filtered_data, marker='o', linestyle='-')
        ax.set_xlabel("Tempo (s)")
        ax.set_ylabel("Corrente (mA)")
        ax.grid(True)
        st.pyplot(fig)

        # Mostra tabella filtrata
        st.subheader("Dati filtrati")
        st.dataframe(pd.DataFrame({
            "Tempo (s)": filtered_time,
            "Corrente (mA)": filtered_data
        }))

        # Statistiche
        if not filtered_data.empty:
            minimo = filtered_data.min()
            massimo = filtered_data.max()
            media = filtered_data.mean()
            std_dev = filtered_data.std()
            flatness = ((massimo / minimo - 1) / 2) if minimo != 0 else np.nan

            st.subheader("Statistiche")
            st.markdown(f"""
            - **Minimo**: {minimo:.3f} mA  
            - **Massimo**: {massimo:.3f} mA  
            - **Media**: {media:.3f} mA  
            - **Deviazione standard**: {std_dev:.3f} mA  
            - **Flatness**: {flatness:.3f}
            """)
        else:
            st.warning("Nessun dato nei limiti specificati.")
