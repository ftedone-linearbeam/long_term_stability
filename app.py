import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("analisi misurazioni elettriche (mA)")

# Caricamento file Excel
uploaded_file = st.file_uploader("Carica file Excel", type=["xlsx", "xls"])
if uploaded_file:
    st.sidebar.subheader("Opzioni file")
    has_header = st.sidebar.checkbox("Il file ha intestazione?", value=True)

    try:
        if has_header:
            df = pd.read_excel(uploaded_file, engine="openpyxl", header=0)
        else:
            df = pd.read_excel(uploaded_file, engine="openpyxl", header=None)
            # Rinomina colonne con nomi generici
            df.columns = [f"Colonna {i+1}" for i in range(df.shape[1])]

        st.subheader("Anteprima del file")
        st.dataframe(df.head())

        # Selezione colonna di interesse
        columns = df.columns.tolist()
        selected_column = st.selectbox("Seleziona la colonna delle misurazioni (mA)", columns)

        if selected_column:
            # ... [prosegue il codice precedente]
            data = df[selected_column].dropna().reset_index(drop=True)
            time = np.arange(len(data))

            # Sidebar per filtri
            st.sidebar.header("Filtri")
            min_val = float(data.min())
            max_val = float(data.max())
            soglia_inf = st.sidebar.slider("Soglia inferiore (mA)", min_val, max_val, min_val)
            soglia_sup = st.sidebar.slider("Soglia superiore (mA)", min_val, max_val, max_val)

            mask = (data >= soglia_inf) & (data <= soglia_sup)
            filtered_data = data[mask]
            filtered_time = time[mask]

            # Grafico
            st.subheader("Grafico delle misurazioni")
            fig, ax = plt.subplots()
            ax.plot(filtered_time, filtered_data, marker='o', linestyle='-')
            ax.set_xlabel("Tempo (s)")
            ax.set_ylabel("Corrente (mA)")
            ax.grid(True)
            st.pyplot(fig)

            # Tabella
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

    except Exception as e:
        st.error(f"Errore nella lettura del file: {e}")
