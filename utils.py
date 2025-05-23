from dash import dash_table

def generar_tabla(df, alto='400px'):
    return dash_table.DataTable(
        data=df.to_dict('records'),
        columns=[{'name': i, 'id': i} for i in df.columns],
        style_table={'height': alto, 'overflowY': 'auto', 'overflowX': 'auto'},
        style_cell={'textAlign': 'left', 'minWidth': '80px', 'whiteSpace': 'normal'},
        fixed_rows={'headers': True}
    )