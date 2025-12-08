from JHProgress import JHProgress

class TSF(JHProgress):

    def __init__(self) -> None:
        super().__init__()
        self.table_fundamentals_stationary=super().cargar_desde_sql(self.config['Tiempo']['FundamentalesEstacionarios'])

    
