import requests
import pandas as pd
from io import StringIO
import json
import polars as pl
import yaml
import logging
import sqlalchemy
import urllib
import yfinance as yf
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class JHProgress:

    def __init__(self) -> None:
        with open("config.yaml", "r") as f:
            self.config = yaml.safe_load(f)
            
        self.tickers_cik= self._connect_tickers()

  
        


    def _connect_tickers(self,debug=False) -> pd.DataFrame:
        print('Downloading tickers from the SEC API')
        zeros_mapper={5:'00000',6:'0000',7:'000',8:'00'}

        headers={'User-Agent':'palajnc@gmail.com'}
        tickers=requests.get('https://www.sec.gov/files/company_tickers.json',headers=headers)
        tickers=pd.json_normalize(pd.json_normalize(tickers.json(),max_level=0).values[0])
        tickers['cik_lenght']=tickers['cik_str'].astype(str).apply(len)
        tickers['zeros_lenght']=tickers['cik_lenght'].map(zeros_mapper)
        tickers['cik_str']=f'CIK'+tickers['zeros_lenght']+tickers['cik_str'].astype(str)
        
        if debug:
            tickers.head(10)
        logging.info('Tickers downloaded successfully')
        return tickers

    #Scrappea la información de la SEC
    def scrapping_sec(self,symbol: str)-> dict:
        cik=self.tickers_cik[self.tickers_cik['ticker']==symbol]['cik_str'].values[0]
        url=f'https://data.sec.gov/api/xbrl/companyfacts/{cik}.json'

        headers={'User-Agent':'palajnc@gmail.com'}
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Raise an error for bad responses
            data = response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for {symbol}: {e}")
            data={}

        logging.info(f'Data for {symbol} fetched successfully from SEC')
        return data
    
    def obtener_caracteristica(self,data:dict, symbol: str, caracteristica: str) -> pd.DataFrame:
        resultado = self._dfs(data, caracteristica)
        try:
            df=pd.json_normalize(resultado)
        except:
            df=pd.DataFrame()

        for c in df.columns:
            try:
                df=df.explode(c)
                df=pd.json_normalize(df.to_dict(orient='records'))
            except:
                pass
        logging.info(f'Caracteristica {caracteristica} obtenida para {symbol}')
        return df

    def _dfs(self,node, target):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == target:
                    return value
                result = self._dfs(value, target)
                if result is not None:
                    return result
        elif isinstance(node, list):
            for item in node:
                result = self._dfs(item, target)
                if result is not None:
                    return result
        return None
    
    def cargar_fundamentales(self) -> pd.DataFrame:
        df=pd.DataFrame()
        for symbol in self.config['Simbolos']:
            data=self.scrapping_sec(symbol)
            for metrica in self.config['Metricas']:
                logging.debug(f'Obteniendo {metrica} para {symbol}')
                df_temp=self.obtener_caracteristica(data,symbol,metrica)
                df_temp['Symbol']=symbol
                df_temp['Metrica']=metrica
                df=pd.concat([df,df_temp],ignore_index=True)
        self.df_fundamentals=df
        logging.info('DataFrame de fundamentales creado exitosamente')
        return df
    
    def guardar_fundamentales_sql(self) -> None:

        # Cadena ODBC pura
        odbc_str=self.config['BaseDatos']['String']

        # Codificar para URL
        params = urllib.parse.quote_plus(odbc_str)

        # Crear engine
        engine = sqlalchemy.create_engine(f"mssql+pyodbc:///?odbc_connect={params}")

        #Ejecutar SP
        with engine.begin() as conn:
            conn.execute(sqlalchemy.text("EXEC dbo.SP_Fundamentales"))
            logging.info('Procedimiento almacenado ejecutado exitosamente')


        #db_connection_string='mssql+pyodbc://127.0.0.1/JHProgress?driver={ODBC+Driver+18+for+SQL+Server};trusted_connection=yes'
        table_name=self.config['BaseDatos']['TablaFundamentales']
        #engine = sqlalchemy.create_engine(db_connection_string)
        self.df_fundamentals.to_sql(table_name, engine, if_exists='replace', index=False)
        logging.info(f'DataFrame de fundamentales guardado en la tabla {table_name} exitosamente')

    def descargar_precios_yahoo(self,start,end) -> pd.DataFrame:
        data = yf.download(self.config['Simbolos'], start=start, end=end, progress=False)
        self.df_prices=data
        logging.info('DataFrame de precios descargado exitosamente')
        return data
    
    def retornos_logaritmicos(self,prices) -> pd.DataFrame:
        df_returns=prices[self.config['MetricaCalculoLog']].rolling(window=2).apply(lambda x: np.log(x[1]/x[0]))
        self.df_returns=df_returns
        logging.info('DataFrame de retornos logarítmicos calculado exitosamente')
        return df_returns[self.config['MetricaCalculoLog']]
    
    def cargar_fundamentales_sql(self) -> pd.DataFrame:
        # Cadena ODBC pura
        odbc_str=self.config['BaseDatos']['String']

        # Codificar para URL
        params = urllib.parse.quote_plus(odbc_str)

        # Crear engine
        engine = sqlalchemy.create_engine(f"mssql+pyodbc:///?odbc_connect={params}")

        table_name=self.config['BaseDatos']['FundamentalesProcesada']
        # Leer tabla SQL en DataFrame
        df_fundamentals_sql = pd.read_sql_table(table_name, engine)
        self.df_fundamentals_sql=df_fundamentals_sql
        logging.info(f'DataFrame de fundamentales cargado desde la tabla {table_name} exitosamente')
        return df_fundamentals_sql
    
    def merge_fundamentales_precios(self,fundamentales,prices,symbol) -> pd.DataFrame:
        df=fundamentales.pivot(index=['FECHA_FIN','start','TIPO'],columns=['Symbol','Metrica'],values='VALOR_AJUSTADO')
        df=df[symbol].dropna(how='all')
        df=df.groupby('FECHA_FIN').max()
        ss=df[df['StockholdersEquityNoteStockSplitConversionRatio1']>0].reset_index().groupby('StockholdersEquityNoteStockSplitConversionRatio1',).max()['FECHA_FIN'] #Obtiene los Stock Splits
        df_proc=df.copy()
        for i in ss.index:
            df_proc.loc[:ss[i],'WeightedAverageNumberOfDilutedSharesOutstanding']*=i

        df_proc=df_proc.dropna(subset=['NetIncomeLoss']).drop(columns=['StockholdersEquityNoteStockSplitConversionRatio1'])
        df_proc['EarningsPerShareDiluted']=df_proc['NetIncomeLoss']/df_proc['WeightedAverageNumberOfDilutedSharesOutstanding']
        df_proc['Equity']=df_proc['Assets']-df_proc['Liabilities']
        df_proc['ROE']=df_proc['NetIncomeLoss']/df_proc['Equity']
        df_proc['ROA']=df_proc['NetIncomeLoss']/df_proc['Assets']
        df_proc['BookValuePerShare']=df_proc['Equity']/df_proc['WeightedAverageNumberOfDilutedSharesOutstanding']
        df_proc['SalesPerShare']=df_proc['Revenues']/df_proc['WeightedAverageNumberOfDilutedSharesOutstanding']
        df_proc['AssetTurnover']=df_proc['Revenues']/df_proc['Assets']
        df_proc['DebtToEquity']=df_proc['Liabilities']/df_proc['Equity']
        df_proc['NetProfitMargin']=df_proc['NetIncomeLoss']/df_proc['Revenues']
        df_proc['OperatingMargin']=df_proc['OperatingIncomeLoss']/df_proc['Revenues']
        df_proc.index=df_proc.index.astype('datetime64[ns]')
        df_proc=df_proc.reset_index()
        df_proc['Fin_Periodo']=df_proc['FECHA_FIN'].shift(-1).fillna(pd.Timestamp('2100-12-31'))
        df_proc=df_proc.reset_index().merge(prices['Close'][symbol].to_frame().reset_index(),how='cross')
        df_proc=df_proc[(df_proc['Date']>=df_proc['FECHA_FIN']) & (df_proc['Date']<=df_proc['Fin_Periodo'])]
        df_proc['PE_Ratio']=df_proc[symbol]/df_proc['EarningsPerShareDiluted']
        df_proc['PB_Ratio']=df_proc[symbol]/df_proc['BookValuePerShare']
        df_proc['PS_Ratio']=df_proc[symbol]/df_proc['SalesPerShare']
        df_proc['MarketCap']=df_proc[symbol]*df_proc['WeightedAverageNumberOfDilutedSharesOutstanding']



        logging.info('DataFrame de fundamentales y precios mergeado exitosamente')
        return df_proc


        


