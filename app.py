import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from io import BytesIO

# ------------------- Utility Functions -------------------
def read_excel_file(file, has_header):
    header = 0 if has_header else None
    df = pd.read_excel(file, engine="openpyxl", header=header)
    if not has_header:
        df.columns = [f"Colonna {i+1}" for i in range(df.shape[1])]
    return df

def plot_interactive(x, y, x_label, y_label, color='blue', name='Dati'):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name=name, line=dict(color=color)))
    fig.update_layout(xaxis_title=x_label, yaxis_title=y_label, hovermode="x unified")
    return fig

def export_to_excel(df):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name="Normalizzati")
    return buffer.getvalue()

def export_plot_as_png(x, y, x_label, y_label):
    buffer = BytesIO()
    fig, ax = plt.subplots()
    ax.plot(x, y, marker='o', linestyle='-', color='green')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True)
    fig.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    return buffer

def calculate_statistics(data, original_data, soglia_inf, soglia_sup):
    stats = {}
    stats['count'] = len(data)
    stats['min'] = data.min()
    stats['max'] = data.max()
    stats['mean'] = data.mean()
    stats['std'] = data.std()
    stats['flatness'] = ((stats['max'] / stats['min'] - 1) / 2) if stats['min'] != 0 else np.nan
    fuori_soglia = ((original_data < soglia_inf) | (original_data > soglia_sup)).sum()
    stats['fuori_soglia'] = fuori_soglia
    stats['percentuale'] = 100 * stats['count'] / len(original_data)
    return stats

# ------------------- Main App -------------------
st.title("Long term stability analysis (μA)")

if "normalized_estimation" not in st.session_state:
    st.session_state["normalized_estimation"] = False

uploaded_file = st.file_uploader("Carica file Excel", type=["xlsx", "xls"])

if uploaded_file:
    st.sidebar.subheader("Opzioni file")
    has_header = st.sidebar.checkbox("Il file ha intestazione", value=False)

    try:
        df = read_excel_file(uploaded_file, has_header)
        st.subheader("Anteprima del file")
        st.dataframe(df.head())

        columns = df.columns.tolist()
        selected_column = st.selectbox("Seleziona la colonna delle misurazioni (μA)", columns)
        data = df[selected_column].dropna().reset_index(drop=True)

        if not data.empty:
            time = np.arange(len(data))
            min_val, max_val = float(data.min()), float(data.max())

            st.sidebar.header("Filtri")
            if abs(data.mean() - 1) < 0.2:
                st.session_state["normalized_estimation"] = True
            else:
                st.session_state["normalized_estimation"] = False
            
            is_normalized = st.sidebar.checkbox("I dati sono già normalizzati", value=st.session_state["normalized_estimation"],key="checkbox_normalized")

            soglia_inf = st.sidebar.number_input("Soglia inferiore", min_val, max_val, min_val)
            soglia_sup = st.sidebar.number_input("Soglia superiore", min_val, max_val, max_val)

            mask = (data >= soglia_inf) & (data <= soglia_sup)
            filtered_data = data[mask]
            filtered_time = time[mask]

            if not is_normalized:
                st.subheader("Grafico delle misurazioni filtrate")
                fig1 = plot_interactive(filtered_time, filtered_data, "Tempo (s)", "Corrente (μA)", color='blue')
                col1, col2 = st.columns(2)
                with col1:
                    x_min = st.number_input("Limite minimo asse X (s)", min_value=0, max_value=int(time.max()), value=int(filtered_time.min()))
                    x_max = st.number_input("Limite massimo asse X (s)", min_value=0, max_value=int(time.max()), value=int(filtered_time.max()))
                with col2:
                    y_min = st.number_input("Limite minimo asse Y", value=float(filtered_data.min()))
                    y_max = st.number_input("Limite massimo asse Y", value=float(filtered_data.max()))
                fig1.update_xaxes(range=[x_min, x_max])
                fig1.update_yaxes(range=[y_min, y_max])
                st.plotly_chart(fig1, use_container_width=True)
                normalized_data = filtered_data / filtered_data.mean()
            else:
                normalized_data = filtered_data
                st.subheader("Grafico delle misurazioni filtrate e normalizzate")
                fig1 = plot_interactive(filtered_time, normalized_data, "Tempo (s)", "Valori normalizzati (-)", color='blue')
                col1, col2 = st.columns(2)
                with col1:
                    x_min = st.number_input("Limite minimo asse X (s)", min_value=0, max_value=int(time.max()), value=int(filtered_time.min()))
                    x_max = st.number_input("Limite massimo asse X (s)", min_value=0, max_value=int(time.max()), value=int(filtered_time.max()))
                with col2:
                    y_min = st.number_input("Limite minimo asse Y", value=float(normalized_data.min()))
                    y_max = st.number_input("Limite massimo asse Y", value=float(normalized_data.max()))
                fig1.update_xaxes(range=[x_min, x_max])
                fig1.update_yaxes(range=[y_min, y_max])
                st.plotly_chart(fig1, use_container_width=True)

            st.subheader("Grafico delle misurazioni normalizzate")
            fig2 = plot_interactive(filtered_time, normalized_data, "Tempo (s)", "(-)", color='green')
            col1, col2 = st.columns(2)
            with col1:
                x_min = st.number_input("Limite minimo asse X fig2 (s)", min_value=0, max_value=int(time.max()), value=int(filtered_time.min()))
                x_max = st.number_input("Limite massimo asse X fig2 (s)", min_value=0, max_value=int(time.max()), value=int(filtered_time.max()))
            with col2:
                y_min = st.number_input("Limite minimo asse Y fig2", value=float(normalized_data.min()))
                y_max = st.number_input("Limite massimo asse Y fig2", value=float(normalized_data.max()))
            fig2.update_xaxes(range=[x_min, x_max])
            fig2.update_yaxes(range=[y_min, y_max])
            st.plotly_chart(fig2, use_container_width=True)

            media_norm = normalized_data.mean()
            st.markdown(f"**Media normalizzata**: {media_norm:.4f}")

            result_df = pd.DataFrame({
                "Tempo (s)": filtered_time,
                "Dati normalizzati (-)": normalized_data
            })
            if not is_normalized:
                result_df.insert(1, "Corrente (μA)", filtered_data)

            st.subheader("Dati Filtrati")
            st.dataframe(result_df)

            st.subheader("Download")
            st.download_button("Scarica dati normalizzati in Excel", data=export_to_excel(result_df),
                               file_name="dati_normalizzati.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            st.download_button("Scarica grafico normalizzato (PNG)",
                               data=export_plot_as_png(filtered_time, normalized_data, "Tempo (s)", "(-)"),
                               file_name="grafico_normalizzato.png", mime="image/png")

            if not filtered_data.empty:
                stats = calculate_statistics(filtered_data, data, soglia_inf, soglia_sup)
                st.sidebar.subheader("Statistiche")
                st.sidebar.markdown(f"""
                - **Numero istanti analizzati**: {stats['count']} ({stats['percentuale']:.2f}%)
                - **Minimo**: {stats['min']:.3f} μA
                - **Massimo**: {stats['max']:.3f} μA
                - **Media**: {stats['mean']:.3f} μA
                - **Media normalizzata**: {media_norm:.3f}
                - **Deviazione standard**: {stats['std']:.3f} μA
                - **Flatness**: {stats['flatness']:.3f}
                - **Totale secondi fuori soglia**: {stats['fuori_soglia']} ({stats['fuori_soglia']/len(data):.2%})
                """)

                # Distribuzione valori
                st.subheader("Distribuzione dei valori")
                fig3, ax3 = plt.subplots()
                ax3.hist(filtered_data, bins=30, color='skyblue', edgecolor='black')
                ax3.set_xlabel("Corrente (μA)")
                ax3.set_ylabel("Frequenza")
                st.pyplot(fig3)

                # Media mobile
                st.subheader("Trend line (media mobile)")
                window_size = st.sidebar.slider("Finestra media mobile", 1, 500, 5)
                rolling_mean = filtered_data.rolling(window=window_size).mean()
                ewma = filtered_data.ewm(span=window_size, adjust=False).mean()
                fig4, ax4 = plt.subplots()
                ax4.plot(filtered_time, filtered_data, label="Originale", alpha=0.5)
                ax4.plot(filtered_time, rolling_mean, label=f"Media mobile ({window_size})", color='red')
                ax4.plot(filtered_time, ewma, label=f"Media esponenziale (span={window_size})", color='green')
                ax4.set_xlabel("Tempo (s)")
                ax4.set_ylabel("Corrente (μA)")
                ax4.legend()
                ax4.grid(True)
                st.pyplot(fig4)

                # Outlier
                st.subheader("Outlier (oltre 3 deviazioni standard)")
                upper_outlier = stats['mean'] + 3 * stats['std']
                lower_outlier = stats['mean'] - 3 * stats['std']
                outliers = filtered_data[(filtered_data > upper_outlier) | (filtered_data < lower_outlier)]
                st.write(f"**Numero di outlier (> ±3σ): {len(outliers)}**")
                if not outliers.empty:
                    st.dataframe(pd.DataFrame({
                        "Tempo (s)": filtered_time[outliers.index],
                        "Outlier (μA)": outliers
                    }))

    except Exception as e:
        st.error(f"Errore nella lettura del file: {e}")
