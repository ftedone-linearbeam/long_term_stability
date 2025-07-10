import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from io import BytesIO
import tempfile
import plotly.graph_objs as go

st.title("Long term stability analysis (μA)")

# Caricamento file Excel
uploaded_file = st.file_uploader("Carica file Excel", type=["xlsx", "xls"])
if uploaded_file:
    st.sidebar.subheader("Opzioni file")
    has_header = st.sidebar.checkbox("Il file ha intestazione?", value=False)
    try:
        if has_header:
            df = pd.read_excel(uploaded_file, engine="openpyxl", header=0)
        else:
            df = pd.read_excel(uploaded_file, engine="openpyxl", header=None)
            df.columns = [f"Colonna {i+1}" for i in range(df.shape[1])]

        st.subheader("Anteprima del file")
        st.dataframe(df.head())

        columns = df.columns.tolist()
        selected_column = st.selectbox("Seleziona la colonna delle misurazioni (μA)", columns)

        data = df[selected_column].dropna().reset_index(drop=True)
        if selected_column and not data.empty:
            time = np.arange(len(data))
            # Sidebar filtri
            st.sidebar.header("Filtri")
            min_val = float(data.min())
            max_val = float(data.max())

            if abs(data.mean()-1)<0.2:
                is_normalized=st.sidebar.checkbox("I dati sono già normalizzati?", value=False)
            else:
                is_normalized=st.sidebar.checkbox("I dati sono già normalizzati?", value=True)

            if is_normalized:
                soglia_inf = st.sidebar.number_input("Soglia inferiore (>="+f"{min_val:.2f}"+")", min_val, max_val, min_val)
                soglia_sup = st.sidebar.number_input("Soglia superiore (<="+f"{min_val:.2f}"+")", min_val, max_val, max_val)
            else:
                soglia_inf = st.sidebar.number_input("Soglia inferiore (>="+f"{min_val:.2f}"+"μA)", min_val, max_val, min_val)
                soglia_sup = st.sidebar.number_input("Soglia superiore (<="+f"{min_val:.2f}"+"μA)", min_val, max_val, max_val)
            
            # Filtraggio dati
            mask = (data >= soglia_inf) & (data <= soglia_sup)
            filtered_data = data[mask]
            filtered_time = time[mask]

            # Plot originale
            st.subheader("Grafico delle misurazioni filtrate")

            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=filtered_time, y=filtered_data,
                          mode='lines+markers',
                          name='Filtrato',
                          line=dict(color='blue')))
            fig1.update_layout(
                xaxis_title="Tempo (s)",
                yaxis_title="Corrente (μA)",
                hovermode="x unified"
            )
            st.plotly_chart(fig1, use_container_width=True)

            # Normalizzazione rispetto alla media
            if not filtered_data.empty and not is_normalized:
                media = filtered_data.mean()
                normalized_data = filtered_data / media
                media_norm=normalized_data.mean()

                # Plot normalizzato
                st.subheader("Grafico delle misurazioni normalizzate")
    
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=filtered_time, y=normalized_data,
                              mode='lines+markers',
                              name='Filtrato',
                              line=dict(color='green')))
                fig2.update_layout(
                    xaxis_title="Tempo (s)",
                    yaxis_title="(-)",
                    hovermode="x unified"
                )
                st.plotly_chart(fig2, use_container_width=True)
            
                # Tabella filtrata
                st.subheader("Dati Filtrati")
                st.dataframe(pd.DataFrame({
                    "Tempo (s)": filtered_time,
                    "Corrente (μA)": filtered_data,
                    "Dati normalizzati (-)": normalized_data
                }))

                st.subheader("Download")

                # ---- Esportazione in Excel ----
                export_df = pd.DataFrame({
                    "Tempo (s)": filtered_time,
                    "Corrente (μA)": filtered_data,
                    "Dati normalizzati (-)": normalized_data
                })
                excel_buffer = BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                    export_df.to_excel(writer, index=False, sheet_name="Normalizzati")

                st.download_button(
                    label="Scarica dati normalizzati in Excel",
                    data=excel_buffer.getvalue(),
                    file_name="dati_normalizzati.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

                # ---- Esportazione grafico come immagine PNG ----
                img_buffer = BytesIO()
                fig3, ax3 = plt.subplots()
                ax3.plot(filtered_time, normalized_data, marker='o', linestyle='-', color='green')
                ax3.set_xlabel("Tempo (s)")
                ax3.set_ylabel("(-)")
                ax3.grid(True)
                fig3.savefig(img_buffer, format="png", bbox_inches="tight")
                img_buffer.seek(0)

                st.download_button(
                    label="Scarica grafico normalizzato (PNG)",
                    data=img_buffer,
                    file_name="grafico_normalizzato.png",
                    mime="image/png"
                )
            
            # Statistiche
            if not filtered_data.empty:
                istanti=len(filtered_data)
                minimo = filtered_data.min()
                massimo = filtered_data.max()
                std_dev = filtered_data.std()
                flatness = ((massimo / minimo - 1) / 2) if minimo != 0 else np.nan
                fuori_soglia_mask = (data < soglia_inf) | (data > soglia_sup)
                durata_fuori_soglia = fuori_soglia_mask.sum()

                st.sidebar.subheader("Statistiche")
                st.sidebar.markdown(f"""
                - **Numero istanti analizzati**: {istanti} ({100*istanti/len(data):.2f}%)
                - **Minimo**: {minimo:.3f} μA  
                - **Massimo**: {massimo:.3f} μA  
                - **Media**: {media:.3f} μA  
                - **Media normalizzata**: {media_norm:.3f} μA
                - **Deviazione standard**: {std_dev:.3f} μA  
                - **Flatness**: {flatness:.3f}
                - **Totale secondi fuori soglia (% sul totale)**: {durata_fuori_soglia} s ({durata_fuori_soglia/len(data):.2f}%)
                """)

                # Punto 1: Istogramma distribuzione
                st.subheader("Distribuzione dei valori")
                fig3, ax3 = plt.subplots()
                ax3.hist(filtered_data, bins=30, color='skyblue', edgecolor='black')
                ax3.set_xlabel("Corrente (mA)")
                ax3.set_ylabel("Frequenza")
                st.pyplot(fig3)

                # Punto 2: Trend line (media mobile)
                st.subheader("Trend line (media mobile)")
                window_size = st.sidebar.slider("Finestra media mobile", 1, 500, 5)
                rolling_mean = filtered_data.rolling(window=window_size).mean()
                ewma = filtered_data.ewm(span=window_size, adjust=False).mean()
                fig4, ax4 = plt.subplots()
                ax4.plot(filtered_time, filtered_data, label="Originale", alpha=0.5)
                ax4.plot(filtered_time, rolling_mean, label=f"Media mobile ({window_size})", color='red')
                ax4.plot(filtered_time, ewma, label=f"Media esponenziale (span={window_size})", color='green')
                ax4.set_xlabel("Tempo (s)")
                ax4.set_ylabel("Corrente (mA)")
                ax4.legend()
                ax4.grid(True)
                st.pyplot(fig4)

                # Punto 3: Evidenziazione outlier (> 3σ)
                st.subheader("Outlier (oltre 3 deviazioni standard)")
                upper_outlier = media + 3 * std_dev
                lower_outlier = media - 3 * std_dev
                outlier_mask = (filtered_data > upper_outlier) | (filtered_data < lower_outlier)
                outliers = filtered_data[outlier_mask]
                st.write(f"**Numero di outlier (> ±3σ): {len(outliers)}**")
                if not outliers.empty:
                    st.dataframe(pd.DataFrame({
                        "Tempo (s)": filtered_time[outlier_mask],
                        "Outlier (μA)": outliers
                    }))

                # Punto 4: Durata fuori soglia (in secondi)

            else:
                st.warning("Nessun dato nei limiti specificati.")

    except Exception as e:
        st.error(f"Errore nella lettura del file: {e}")
