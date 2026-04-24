import pandas as pd
import streamlit as st
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
import re

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Dashboard de Análisis de Sentimiento",
    page_icon="🧠",
    layout="wide"
)

# --- 1. FUNCIONES DE PREPARACIÓN Y CARGA ---

@st.cache_resource
def setup_nltk():
    """Descarga los recursos necesarios de NLTK (stopwords)."""
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

def reparar_codificacion(texto):
    """Intenta reparar un texto que fue codificado en UTF-8 pero leído como Latin-1."""
    if not isinstance(texto, str):
        return texto
    try:
        return texto.encode('latin1').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        return texto

@st.cache_data
def cargar_datos_sentimiento():
    """
    Carga los datos desde el archivo CSV pre-procesado de sentimientos.
    """
    try:
        df = pd.read_csv('senti_npl.csv')
        
        # Asegurarse que las columnas de filtro sean strings
        for col in ['Carrera', 'Asignatura', '¿Que opina del profesor?']:
            if col in df.columns:
                df[col] = df[col].astype(str)
        
        # Aseguramos que todas las columnas de probabilidad que usaremos existan y sean numéricas
        prob_cols = [
            '% Sentimiento', '% Emocion', 
            'Prob_Odio', 'Prob_Dirigido', 'Prob_Agresivo'
        ]

        for col in prob_cols:
             if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
             else:
                st.error(f"Error Crítico: La columna requerida '{col}' no se encontró en 'senti_npl.csv'.")
                st.stop()

        return df
    except FileNotFoundError:
        st.error(
            "Error: No se encontró el archivo 'senti_npl.csv'. "
            "Por favor, ejecute primero el script 'analisis_sentimiento.py' para generar el archivo."
        )
        st.stop()
    except Exception as e:
        st.error(f"Ocurrió un error al cargar los datos: {e}")
        st.stop()

# --- 2. FUNCIONES DE RENDERIZADO ---

def renderizar_sidebar(df):
    """
    Renderiza la barra lateral de filtros y devuelve las selecciones.
    """
    st.sidebar.header("Filtros del Dashboard")

    # 1. Filtro de Carrera (Selección Única)
    carrera_options = ['Todas'] + sorted(df['Carrera'].dropna().unique())
    carrera_seleccionada = st.sidebar.selectbox(
        "Seleccione Carrera:",
        options=carrera_options
    )

    # 2. Filtro de Asignatura (Dependiente)
    if carrera_seleccionada == 'Todas':
        df_contexto = df
    else:
        df_contexto = df[df['Carrera'] == carrera_seleccionada]
    
    # Opciones de Asignatura ahora incluye 'Todas'
    asignatura_options = ['Todas'] + sorted(df_contexto['Asignatura'].dropna().unique())
    # Cambiamos multiselect a selectbox
    asignatura_seleccionada = st.sidebar.selectbox(
        "Seleccione Asignatura:",
        options=asignatura_options
    )

    st.sidebar.divider()
    use_dark_theme = st.sidebar.toggle("Modo oscuro en gráficos")
    chart_theme = "plotly_dark" if use_dark_theme else "plotly_white"

    return {
        "carrera": carrera_seleccionada,
        "asignatura": asignatura_seleccionada,
        "chart_theme": chart_theme
    }

def renderizar_histogramas(df_filtrado, chart_theme, carrera_seleccionada, asignatura_seleccionada):
    """
    Muestra los tres histogramas de resultados de NLP.
    """
    st.header("Distribución de Resultados del Análisis de Comentarios")

    # Añadimos el subtítulo dinámico con los filtros seleccionados
    st.subheader(f"Resultados para: {carrera_seleccionada} | {asignatura_seleccionada}")

    if df_filtrado.empty:
        st.warning("No se encontraron comentarios para los filtros seleccionados.")
        return
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Análisis de Sentimiento")
        if 'Sentimiento' in df_filtrado.columns:
            df_sentimiento = df_filtrado['Sentimiento'].value_counts().reset_index(name='Conteo')
            fig_sent = px.bar(
                df_sentimiento, x='Sentimiento', y='Conteo',
                title="Distribución de Sentimiento", color='Sentimiento', text_auto=True
            )
            fig_sent.update_layout(template=chart_theme, showlegend=False)
            st.plotly_chart(fig_sent, use_container_width=True)
        else:
            st.warning("La columna 'Sentimiento' no se encontró.")

    with col2:
        st.subheader("Análisis de Emoción")
        if 'Emocion' in df_filtrado.columns:
            df_emocion = df_filtrado['Emocion'].value_counts().reset_index(name='Conteo')
            fig_emo = px.bar(
                df_emocion, x='Emocion', y='Conteo',
                title="Distribución de Emociones", color='Emocion', text_auto=True
            )
            fig_emo.update_layout(template=chart_theme, showlegend=False)
            st.plotly_chart(fig_emo, use_container_width=True)
        else:
            st.warning("La columna 'Emocion' no se encontró.")

    with col3:
        st.subheader("Análisis de Discurso de Odio")
        if 'Discurso_Odio' in df_filtrado.columns:
            df_odio = df_filtrado['Discurso_Odio'].value_counts().reset_index(name='Conteo')
            fig_odio = px.bar(
                df_odio, x='Discurso_Odio', y='Conteo',
                title="Detección de Discurso de Odio", color='Discurso_Odio', text_auto=True
            )
            fig_odio.update_layout(template=chart_theme, showlegend=False)
            st.plotly_chart(fig_odio, use_container_width=True)
        else:
            st.warning("La columna 'Discurso_Odio' no se encontró.")

def renderizar_wordclouds(df_filtrado, chart_theme):
    """
    Genera y muestra dos nubes de palabras (positiva y negativa)
    a partir de los comentarios filtrados.
    """
    st.header("Nubes de Palabras Clave")

    # 1. Definir Stopwords en español
    stopwords_es = set(nltk.corpus.stopwords.words('spanish'))
    stopwords_es.update([
        'profesor', 'profe', 'clases', 'clase', 'asignatura', 'bien', 'solo', 'si',
        'mas', 'más', 'sin', 'embargo', 'ser', 'muy', 'creo', 'nada', 'gracias',
        'comentarios', 'respecto', 'profesora'
    ])

    # 2. Definir color de fondo basado en el tema
    bg_color = "#0E1117" if chart_theme == "plotly_dark" else "#FFFFFF"
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Palabras Positivas")
        df_pos = df_filtrado[df_filtrado['Sentimiento'] == 'Positivo']
        
        if not df_pos.empty:
            texto_pos = " ".join(
                reparar_codificacion(comentario)
                for comentario in df_pos['¿Que opina del profesor?'].dropna()
            )
            texto_pos = " ".join(
                word for word in re.split(r'\W+', texto_pos.lower()) 
                if word not in stopwords_es and len(word) > 2
            )

            if texto_pos:
                try:
                    wc_pos = WordCloud(
                        background_color=bg_color,
                        stopwords=stopwords_es,
                        width=800, height=400, max_words=100, colormap='Greens'
                    ).generate(texto_pos)
                    
                    fig, ax = plt.subplots()
                    ax.imshow(wc_pos, interpolation='bilinear')
                    ax.axis("off")
                    fig.patch.set_facecolor(bg_color)
                    st.pyplot(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error al generar la nube de palabras positivas: {e}")
            else:
                st.info("No hay suficientes palabras positivas para mostrar una nube.")
        else:
            st.info("No hay comentarios positivos para los filtros seleccionados.")

    with col2:
        st.subheader("Palabras Negativas")
        df_neg = df_filtrado[df_filtrado['Sentimiento'] == 'Negativo']
        
        if not df_neg.empty:
            texto_neg = " ".join(
                reparar_codificacion(comentario)
                for comentario in df_neg['¿Que opina del profesor?'].dropna()
            )
            texto_neg = " ".join(
                word for word in re.split(r'\W+', texto_neg.lower())
                if word not in stopwords_es and len(word) > 2
            )

            if texto_neg:
                try:
                    wc_neg = WordCloud(
                        background_color=bg_color,
                        stopwords=stopwords_es,
                        width=800, height=400, max_words=100, colormap='Reds'
                    ).generate(texto_neg)
                    
                    fig, ax = plt.subplots()
                    ax.imshow(wc_neg, interpolation='bilinear')
                    ax.axis("off")
                    fig.patch.set_facecolor(bg_color)
                    st.pyplot(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error al generar la nube de palabras negativas: {e}")
            else:
                st.info("No hay suficientes palabras negativas para mostrar una nube.")
        else:
            st.info("No hay comentarios negativos para los filtros seleccionados.")

def renderizar_top_comentarios(df_filtrado):
    """
    Muestra los 5 comentarios más positivos y negativos
    basados en la probabilidad del sentimiento.
    """
    st.header("Top 5 Comentarios por Confianza del Modelo")
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Comentarios Positivos")
        df_top_pos = df_filtrado[
            (df_filtrado['Sentimiento'] == 'Positivo') & 
            (df_filtrado['% Sentimiento'].notna())
        ].sort_values(by='% Sentimiento', ascending=False).head(5)

        if not df_top_pos.empty:
            for index, row in df_top_pos.iterrows():
                st.info(f"\"{reparar_codificacion(row['¿Que opina del profesor?'])}\"")
                st.caption(f"Confianza del modelo: {row['% Sentimiento']:.1%}")
                st.divider()
        else:
            st.info("No hay comentarios positivos para mostrar.")

    with col2:
        st.subheader("Comentarios Negativos")
        df_top_neg = df_filtrado[
            (df_filtrado['Sentimiento'] == 'Negativo') & 
            (df_filtrado['% Sentimiento'].notna())
        ].sort_values(by='% Sentimiento', ascending=False).head(5)

        if not df_top_neg.empty:
            for index, row in df_top_neg.iterrows():
                st.error(f"\"{reparar_codificacion(row['¿Que opina del profesor?'])}\"")
                st.caption(f"Confianza del modelo: {row['% Sentimiento']:.1%}")
                st.divider()
        else:
            st.info("No hay comentarios negativos para mostrar.")

def renderizar_analisis_emocion(df_filtrado, chart_theme):
    """
    Muestra un gráfico de barras de emociones (excluyendo 'Otro')
    y un Top 5 de comentarios por cada emoción.
    """
    st.header("Análisis Detallado por Emoción")

    if 'Emocion' not in df_filtrado.columns or '% Emocion' not in df_filtrado.columns:
        st.warning("Las columnas 'Emocion' o '% Emocion' no se encontraron.")
        return

    # 1. Filtrar "Otro" y nulos para el análisis
    df_emociones = df_filtrado[
        (df_filtrado['Emocion'].notna()) & 
        (df_filtrado['Emocion'] != 'Otro')
    ].copy()

    if df_emociones.empty:
        st.info("No hay comentarios con emociones detectadas (excluyendo 'Otro') para los filtros seleccionados.")
        return

    # --- Gráfico de Barras ---
    st.subheader("Distribución de Emociones (excluyendo 'Otro')")
    
    # Creamos el dataframe para la distribución, ordenado por Conteo
    df_distribucion_emocion = df_emociones['Emocion'].value_counts().reset_index(name='Conteo')
    df_distribucion_emocion.columns = ['Emocion', 'Conteo']

    fig_bar_emo = px.bar(
        df_distribucion_emocion,
        x='Conteo', 
        y='Emocion', 
        orientation='h', # Gráfico horizontal para mejor legibilidad
        title="Distribución de Emociones Detectadas",
        text='Conteo',
        color='Emocion'
    )
    fig_bar_emo.update_layout(
        template=chart_theme,
        showlegend=False,
        yaxis_title="Emoción",
        xaxis_title="Cantidad de Comentarios"
    )
    fig_bar_emo.update_yaxes(categoryorder='total ascending') 
    st.plotly_chart(fig_bar_emo, use_container_width=True)

    # --- Top 5 Comentarios por Emoción ---
    st.subheader("Top 5 Comentarios por Emoción")
    
    emociones_list = df_distribucion_emocion['Emocion'].unique()
    
    cols = st.columns(3)

    for i, emocion in enumerate(emociones_list):
        with cols[i % 3]: # Distribuir en 3 columnas
            st.markdown(f"#### {emocion}")
            
            # Cambiamos de .head(3) a .head(5)
            df_top_emo = df_emociones[
                (df_emociones['Emocion'] == emocion) & 
                (df_emociones['% Emocion'].notna())
            ].sort_values(by='% Emocion', ascending=False).head(5)

            if not df_top_emo.empty:
                for index, row in df_top_emo.iterrows():
                    expander_title = f"Confianza: {row['% Emocion']:.1%}"
                    with st.expander(expander_title):
                        st.markdown(f"\"{reparar_codificacion(row['¿Que opina del profesor?'])}\"")
            else:
                st.info(f"No hay comentarios para '{emocion}'.")

# --- INICIO DE LA MODIFICACIÓN ---
def renderizar_analisis_odio(df_filtrado, chart_theme):
    """
    Muestra un gráfico de barras de discurso de odio (excluyendo 'No detectado')
    y un listado completo de comentarios por cada categoría, ordenados por 'Prob_Odio'.
    """
    st.header("Análisis Detallado de Discurso de Odio")

    # 1. Verificar columnas necesarias
    cols_necesarias = ['Discurso_Odio', 'Prob_Odio', 'Prob_Dirigido', 'Prob_Agresivo', '¿Que opina del profesor?']
    if not all(col in df_filtrado.columns for col in cols_necesarias):
        st.warning("No se encontraron todas las columnas de Discurso de Odio o probabilidades en el archivo.")
        return

    # 2. Filtrar "No detectado" y nulos
    df_odio = df_filtrado[
        (df_filtrado['Discurso_Odio'].notna()) & 
        (df_filtrado['Discurso_Odio'] != 'No detectado')
    ].copy()

    if df_odio.empty:
        st.info("No se detectaron comentarios con Discurso de Odio para los filtros seleccionados.")
        return

    # 3. Gráfico de Barras de Distribución (Se mantiene)
    st.subheader("Distribución de Tipos de Discurso de Odio")
    
    df_distribucion_odio = df_odio['Discurso_Odio'].value_counts().reset_index(name='Conteo')
    df_distribucion_odio.columns = ['Tipo de Odio', 'Conteo']

    fig_bar_odio = px.bar(
        df_distribucion_odio,
        x='Conteo', 
        y='Tipo de Odio', 
        orientation='h',
        title="Tipos de Discurso de Odio Detectados",
        text='Conteo',
        color='Tipo de Odio'
    )
    fig_bar_odio.update_layout(
        template=chart_theme,
        showlegend=False,
        yaxis_title="Categoría Detectada",
        xaxis_title="Cantidad de Comentarios"
    )
    fig_bar_odio.update_yaxes(categoryorder='total ascending')
    st.plotly_chart(fig_bar_odio, use_container_width=True)

    # 4. Listado de Comentarios por Categoría, ordenados por Prob_Odio
    st.subheader("Listado de Comentarios Detectados (ordenados por % de Odio)")
    
    col1, col2, col3 = st.columns(3)

    # DataFrame base para los listados, ya ordenado por Prob_Odio
    df_odio_sorted = df_odio.sort_values(by='Prob_Odio', ascending=False)

    # Columna 1: Odio (Hateful)
    with col1:
        st.markdown("#### Comentarios Odiosos (Hateful)")
        df_list_odio = df_odio_sorted[
            df_odio_sorted['Discurso_Odio'].str.contains('Odio', case=False, na=False)
        ]
        
        if not df_list_odio.empty:
            for index, row in df_list_odio.iterrows():
                expander_title = f"Confianza (Odio): {row['Prob_Odio']:.1%}"
                with st.expander(expander_title):
                    st.markdown(f"\"{reparar_codificacion(row['¿Que opina del profesor?'])}\"")
                    st.caption(f"Clasif. Modelo: {row['Discurso_Odio']}")
        else:
            st.info("No hay comentarios 'Odiosos' en esta selección.")

    # Columna 2: Dirigido (Targeted)
    with col2:
        st.markdown("#### Comentarios Dirigidos (Targeted)")
        df_list_dirigido = df_odio_sorted[
            df_odio_sorted['Discurso_Odio'].str.contains('dirigido', case=False, na=False)
        ]
        
        if not df_list_dirigido.empty:
            for index, row in df_list_dirigido.iterrows():
                expander_title = f"Confianza (Odio): {row['Prob_Odio']:.1%}"
                with st.expander(expander_title):
                    st.markdown(f"\"{reparar_codificacion(row['¿Que opina del profesor?'])}\"")
                    st.caption(f"Clasif. Modelo: {row['Discurso_Odio']} | Prob (Dirigido): {row['Prob_Dirigido']:.1%}")
        else:
            st.info("No hay comentarios 'Dirigidos' en esta selección.")

    # Columna 3: Agresivo (Aggressive)
    with col3:
        st.markdown("#### Comentarios Agresivos (Aggressive)")
        df_list_agresivo = df_odio_sorted[
            df_odio_sorted['Discurso_Odio'].str.contains('Agresivo', case=False, na=False)
        ]

        if not df_list_agresivo.empty:
            for index, row in df_list_agresivo.iterrows():
                expander_title = f"Confianza (Odio): {row['Prob_Odio']:.1%}"
                with st.expander(expander_title):
                    st.markdown(f"\"{reparar_codificacion(row['¿Que opina del profesor?'])}\"")
                    st.caption(f"Clasif. Modelo: {row['Discurso_Odio']} | Prob (Agresivo): {row['Prob_Agresivo']:.1%}")
        else:
            st.info("No hay comentarios 'Agresivos' en esta selección.")
# --- FIN DE LA MODIFICACIÓN ---

# --- 3. FLUJO PRINCIPAL DE LA APLICACIÓN ---

st.title("🧠 Dashboard de Análisis de Sentimiento (NLP)")
st.info("Este dashboard analiza los comentarios abiertos de la encuesta de evaluación docente.")

# Ejecutar la configuración de NLTK al inicio
setup_nltk()

# Cargar datos
df_base = cargar_datos_sentimiento()

if df_base is not None:
    # Renderizar Sidebar y obtener filtros
    filtros = renderizar_sidebar(df_base)

    # Aplicar filtros al DataFrame base
    df_filtrado_final = df_base.copy()

    if filtros["carrera"] != 'Todas':
        df_filtrado_final = df_filtrado_final[df_filtrado_final['Carrera'] == filtros["carrera"]]

    if filtros["asignatura"] != 'Todas':
        df_filtrado_final = df_filtrado_final[df_filtrado_final['Asignatura'] == filtros["asignatura"]]

    # Renderizar visualizaciones
    st.markdown("---")
    renderizar_histogramas(
        df_filtrado_final, 
        filtros["chart_theme"], 
        filtros["carrera"], 
        filtros["asignatura"]
    )
    
    st.markdown("---")
    renderizar_wordclouds(df_filtrado_final, filtros["chart_theme"])
    
    st.markdown("---")
    renderizar_top_comentarios(df_filtrado_final)
    
    st.markdown("---")
    renderizar_analisis_emocion(df_filtrado_final, filtros["chart_theme"])
    
    st.markdown("---")
    renderizar_analisis_odio(df_filtrado_final, filtros["chart_theme"])

