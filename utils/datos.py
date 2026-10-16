import pandas as pd

def cargar_y_preparar_datos(ruta: str) -> pd.DataFrame:
    """Lee el CSV y agrupa las ventas por día para el forecasting."""
    df = pd.read_csv(ruta)
    df['fecha'] = pd.to_datetime(df['fecha'])
    
    ventas_diarias = df.groupby(df['fecha'].dt.date)['ventas'].sum().reset_index()
    ventas_diarias['fecha'] = pd.to_datetime(ventas_diarias['fecha'])
    
    ventas_diarias['dia_semana'] = ventas_diarias['fecha'].dt.dayofweek
    ventas_diarias['dia_mes'] = ventas_diarias['fecha'].dt.day
    ventas_diarias['mes'] = ventas_diarias['fecha'].dt.month
    
    return ventas_diarias.sort_values('fecha')