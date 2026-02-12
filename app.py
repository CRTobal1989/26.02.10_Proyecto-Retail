import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Simulador de Ventas 2025",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .stApp {
        background-color: #f8f9fa;
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1 {
        color: #4c3fb1 !important;
        font-weight: bold !important;
    }
    h2 {
        color: #5a3891 !important;
        font-weight: bold !important;
        margin-top: 20px !important;
    }
    h3 {
        color: #4c3fb1 !important;
        font-weight: bold !important;
        margin-top: 20px !important;
        margin-bottom: 15px !important;
    }
    h4 {
        color: #5a3891 !important;
        font-weight: bold !important;
    }
    p, span, div {
        color: #2c3e50 !important;
    }
    .stMarkdown {
        color: #2c3e50 !important;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        font-size: 18px;
        font-weight: bold;
        padding: 15px 30px;
        border-radius: 10px;
        border: none;
        width: 100%;
    }
    .stButton>button:hover {
        opacity: 0.9;
    }
    .block-container {
        padding-top: 2rem;
    }
    [data-testid="stMetricLabel"] {
        color: #2c3e50 !important;
    }
    [data-testid="stMetricValue"] {
        color: #1a1a1a !important;
    }
</style>
""", unsafe_allow_html=True)

# Funciones auxiliares
@st.cache_resource
def cargar_modelo():
    """Carga el modelo entrenado"""
    try:
        modelo = joblib.load('../models/modelo_final.joblib')
        return modelo
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        return None

@st.cache_data
def cargar_datos():
    """Carga el dataset de inferencia"""
    try:
        df = pd.read_csv('../data/processed/inferencia_df_transformado.csv')
        df['fecha'] = pd.to_datetime(df['fecha'])
        return df
    except Exception as e:
        st.error(f"❌ Error al cargar los datos: {str(e)}")
        return None

def aplicar_ajustes(df_producto, ajuste_descuento, escenario_competencia):
    """
    Aplica los ajustes de descuento y competencia al dataframe del producto
    """
    df_ajustado = df_producto.copy()
    
    # Ajustar precio_venta según el descuento
    factor_descuento = 1 + (ajuste_descuento / 100)
    df_ajustado['precio_venta'] = df_ajustado['precio_base'] * factor_descuento
    
    # Ajustar precios de competencia según el escenario
    factor_competencia = 1.0
    if escenario_competencia == "Competencia -5%":
        factor_competencia = 0.95
    elif escenario_competencia == "Competencia +5%":
        factor_competencia = 1.05
    
    # Recalcular precio_competencia
    if all(col in df_ajustado.columns for col in ['Amazon', 'Decathlon', 'Deporvillage']):
        df_ajustado['Amazon'] = df_ajustado['Amazon'] * factor_competencia
        df_ajustado['Decathlon'] = df_ajustado['Decathlon'] * factor_competencia
        df_ajustado['Deporvillage'] = df_ajustado['Deporvillage'] * factor_competencia
        df_ajustado['precio_competencia'] = df_ajustado[['Amazon', 'Decathlon', 'Deporvillage']].mean(axis=1)
    
    # Recalcular descuento_pct y ratio_precio
    df_ajustado['descuento_pct'] = 1 - (df_ajustado['precio_venta'] / df_ajustado['precio_base'])
    df_ajustado['ratio_precio'] = df_ajustado['precio_venta'] / df_ajustado['precio_competencia']
    
    return df_ajustado

def predecir_recursivamente(modelo, df_producto):
    """
    Realiza predicciones recursivas actualizando lags día a día
    """
    df_pred = df_producto.copy()
    df_pred = df_pred.sort_values('fecha').reset_index(drop=True)
    
    predicciones = []
    
    # Obtener columnas de features que el modelo espera
    feature_cols = modelo.feature_names_in_
    
    # Verificar que todas las columnas existen
    columnas_faltantes = set(feature_cols) - set(df_pred.columns)
    if columnas_faltantes:
        st.error(f"❌ Faltan columnas en el dataframe: {columnas_faltantes}")
        return df_pred
    
    # Nombres de columnas de lags (sin guión bajo entre lag y número)
    lag_cols = [f'unidades_vendidas_lag{i}' for i in range(1, 8)]
    
    for idx in range(len(df_pred)):
        # Obtener features para predicción
        X = df_pred.loc[[idx], feature_cols]
        
        # Predecir
        pred = modelo.predict(X)[0]
        pred = max(0, pred)  # Evitar predicciones negativas
        predicciones.append(pred)
        
        # Actualizar lags para el siguiente día (si no es el último día)
        if idx < len(df_pred) - 1:
            # Desplazar lags: lag7 ← lag6, lag6 ← lag5, ..., lag2 ← lag1
            for i in range(7, 1, -1):
                df_pred.loc[idx + 1, f'unidades_vendidas_lag{i}'] = df_pred.loc[idx, f'unidades_vendidas_lag{i-1}']
            
            # Actualizar lag1 con la predicción actual
            df_pred.loc[idx + 1, 'unidades_vendidas_lag1'] = pred
            
            # Actualizar media móvil (últimas 7 predicciones)
            if idx >= 6:
                df_pred.loc[idx + 1, 'unidades_vendidas_mm7'] = np.mean(predicciones[-7:])
            else:
                # Si hay menos de 7 predicciones, usar las disponibles más los lags existentes
                predicciones_disponibles = predicciones.copy()
                for i in range(1, 8 - len(predicciones_disponibles)):
                    if f'unidades_vendidas_lag{i}' in df_pred.columns:
                        lag_val = df_pred.loc[idx, f'unidades_vendidas_lag{i}']
                        if not pd.isna(lag_val):
                            predicciones_disponibles.insert(0, lag_val)
                df_pred.loc[idx + 1, 'unidades_vendidas_mm7'] = np.mean(predicciones_disponibles[-7:])
    
    df_pred['unidades_predichas'] = predicciones
    df_pred['ingresos_proyectados'] = df_pred['unidades_predichas'] * df_pred['precio_venta']
    
    return df_pred

def crear_grafico_prediccion(df_resultados):
    """
    Crea el gráfico de predicción diaria con Black Friday destacado
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Configurar estilo
    sns.set_style("whitegrid")
    
    # Graficar predicciones
    ax.plot(df_resultados['fecha'], df_resultados['unidades_predichas'], 
            linewidth=2.5, color='#667eea', marker='o', markersize=5, label='Predicción')
    
    # Marcar Black Friday (28 de noviembre)
    black_friday = pd.to_datetime('2025-11-28')
    if black_friday in df_resultados['fecha'].values:
        idx_bf = df_resultados[df_resultados['fecha'] == black_friday].index[0]
        valor_bf = df_resultados.loc[idx_bf, 'unidades_predichas']
        
        # Línea vertical
        ax.axvline(x=black_friday, color='red', linestyle='--', linewidth=2, alpha=0.7)
        
        # Punto destacado
        ax.scatter([black_friday], [valor_bf], color='red', s=200, zorder=5, 
                  edgecolors='darkred', linewidths=2)
        
        # Anotación
        ax.annotate('🛒 BLACK FRIDAY', 
                   xy=(black_friday, valor_bf), 
                   xytext=(10, 20),
                   textcoords='offset points',
                   fontsize=12,
                   fontweight='bold',
                   color='red',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    # Configurar ejes
    ax.set_xlabel('Fecha', fontsize=12, fontweight='bold')
    ax.set_ylabel('Unidades Vendidas', fontsize=12, fontweight='bold')
    ax.set_title('Predicción de Ventas Diarias - Noviembre 2025', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Rotar etiquetas del eje X
    plt.xticks(rotation=45, ha='right')
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# Cargar modelo y datos
modelo = cargar_modelo()
df_inferencia = cargar_datos()

if modelo is None or df_inferencia is None:
    st.stop()

# SIDEBAR - Controles de Simulación
with st.sidebar:
    st.title("🎮 Controles de Simulación")
    st.markdown("---")
    
    # Selector de producto
    productos = sorted(df_inferencia['nombre'].unique())
    producto_seleccionado = st.selectbox(
        "🏷️ Selecciona un Producto",
        options=productos,
        index=0
    )
    
    st.markdown("---")
    
    # Slider de descuento
    ajuste_descuento = st.slider(
        "💰 Ajuste de Descuento",
        min_value=-50,
        max_value=50,
        value=0,
        step=5,
        format="%d%%",
        help="Ajusta el descuento aplicado sobre el precio base"
    )
    
    st.markdown("---")
    
    # Selector de escenario de competencia
    st.markdown("🏪 **Escenario de Competencia**")
    escenario_competencia = st.radio(
        "",
        options=["Actual (0%)", "Competencia -5%", "Competencia +5%"],
        index=0,
        help="Simula cambios en los precios de la competencia"
    )
    
    st.markdown("---")
    
    # Botón de simulación
    simular = st.button("🚀 Simular Ventas", use_container_width=True)

# ZONA PRINCIPAL
st.title("📊 Dashboard de Simulación de Ventas")
st.markdown(f"### Noviembre 2025 - {producto_seleccionado}")
st.markdown("---")

if simular:
    with st.spinner("⏳ Procesando predicciones recursivas..."):
        # Filtrar datos del producto seleccionado
        df_producto = df_inferencia[df_inferencia['nombre'] == producto_seleccionado].copy()
        
        if df_producto.empty:
            st.error("❌ No hay datos para el producto seleccionado")
            st.stop()
        
        # Aplicar ajustes
        df_ajustado = aplicar_ajustes(df_producto, ajuste_descuento, escenario_competencia)
        
        # Realizar predicciones recursivas
        df_resultados = predecir_recursivamente(modelo, df_ajustado)
        
        # Calcular KPIs
        unidades_totales = df_resultados['unidades_predichas'].sum()
        ingresos_totales = df_resultados['ingresos_proyectados'].sum()
        precio_promedio = df_resultados['precio_venta'].mean()
        descuento_promedio = df_resultados['descuento_pct'].mean() * 100
        
        # 1. KPIs DESTACADOS
        st.markdown("### 📈 Métricas Clave")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="🛒 Unidades Totales",
                value=f"{int(unidades_totales)}"
            )
        
        with col2:
            st.metric(
                label="💶 Ingresos Proyectados",
                value=f"€{ingresos_totales:.2f}"
            )
        
        with col3:
            st.metric(
                label="💵 Precio Promedio",
                value=f"€{precio_promedio:.2f}"
            )
        
        with col4:
            st.metric(
                label="🏷️ Descuento Promedio",
                value=f"{descuento_promedio:.1f}%"
            )
        
        st.markdown("---")
        
        # 2. GRÁFICO DE PREDICCIÓN DIARIA
        st.markdown("### 📉 Evolución de Ventas Diarias")
        fig = crear_grafico_prediccion(df_resultados)
        st.pyplot(fig)
        
        st.markdown("---")
        
        # 3. TABLA DETALLADA
        st.markdown("### 📋 Detalle Diario de Predicciones")
        
        # Preparar tabla
        tabla = df_resultados[['fecha', 'precio_venta', 'precio_competencia', 
                               'descuento_pct', 'unidades_predichas', 'ingresos_proyectados']].copy()
        tabla['dia_semana'] = tabla['fecha'].dt.day_name()
        tabla['fecha_str'] = tabla['fecha'].dt.strftime('%d/%m/%Y')
        tabla['descuento_pct'] = tabla['descuento_pct'] * 100
        
        # Destacar Black Friday
        tabla['es_black_friday'] = tabla['fecha'] == pd.to_datetime('2025-11-28')
        tabla['fecha_display'] = tabla.apply(
            lambda x: f"🛒 {x['fecha_str']}" if x['es_black_friday'] else x['fecha_str'],
            axis=1
        )
        
        # Crear tabla final
        tabla_display = tabla[['fecha_display', 'dia_semana', 'precio_venta', 
                               'precio_competencia', 'descuento_pct', 
                               'unidades_predichas', 'ingresos_proyectados']].copy()
        
        tabla_display.columns = ['Fecha', 'Día', 'Precio Venta (€)', 
                                 'Precio Competencia (€)', 'Descuento (%)', 
                                 'Unidades', 'Ingresos (€)']
        
        # Formatear números
        tabla_display['Precio Venta (€)'] = tabla_display['Precio Venta (€)'].apply(lambda x: f"€{x:.2f}")
        tabla_display['Precio Competencia (€)'] = tabla_display['Precio Competencia (€)'].apply(lambda x: f"€{x:.2f}")
        tabla_display['Descuento (%)'] = tabla_display['Descuento (%)'].apply(lambda x: f"{x:.1f}%")
        tabla_display['Unidades'] = tabla_display['Unidades'].apply(lambda x: f"{int(x)}")
        tabla_display['Ingresos (€)'] = tabla_display['Ingresos (€)'].apply(lambda x: f"€{x:.2f}")
        
        st.dataframe(tabla_display, use_container_width=True, height=600)
        
        st.markdown("---")
        
        # 4. COMPARATIVA DE ESCENARIOS
        st.markdown("### 🔄 Comparativa de Escenarios de Competencia")
        st.markdown("Comparación manteniendo el descuento actual y variando precios de competencia:")
        
        escenarios = ["Actual (0%)", "Competencia -5%", "Competencia +5%"]
        resultados_escenarios = {}
        
        with st.spinner("🔄 Calculando escenarios de competencia..."):
            for escenario in escenarios:
                df_esc = aplicar_ajustes(df_producto, ajuste_descuento, escenario)
                df_pred_esc = predecir_recursivamente(modelo, df_esc)
                resultados_escenarios[escenario] = {
                    'unidades': df_pred_esc['unidades_predichas'].sum(),
                    'ingresos': df_pred_esc['ingresos_proyectados'].sum()
                }
        
        col1, col2, col3 = st.columns(3)
        
        columnas = [col1, col2, col3]
        for idx, (escenario, resultados) in enumerate(resultados_escenarios.items()):
            with columnas[idx]:
                st.markdown(f"#### {escenario}")
                st.metric(
                    label="Unidades Totales",
                    value=f"{int(resultados['unidades'])}"
                )
                st.metric(
                    label="Ingresos Totales",
                    value=f"€{resultados['ingresos']:.2f}"
                )
        
        st.success("✅ Simulación completada con éxito")

else:
    st.info("👆 Configura los parámetros en el panel lateral y pulsa '🚀 Simular Ventas' para comenzar")
    
    # Mostrar información general
    st.markdown("### 📊 Información del Dataset")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Productos", df_inferencia['nombre'].nunique())
        st.metric("Días en Noviembre", df_inferencia['fecha'].nunique())
    
    with col2:
        st.metric("Total Registros", len(df_inferencia))
        st.metric("Categorías", df_inferencia['categoria'].nunique())