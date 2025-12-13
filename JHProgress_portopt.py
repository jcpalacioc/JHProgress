from JHProgress import JHProgress
import pandas as pd
import numpy as np
from scipy.optimize import basinhopping,minimize
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PortOPT(JHProgress):


    def __init__(self) -> None:
        super().__init__()
        self.log_returns_fwr=self.cargar_desde_sql(self.config['BaseDatos']['VistaForecast'])
        self.log_returns_fwr=pd.pivot_table(self.log_returns_fwr,columns='Simbolo',values='returns',index='Date',aggfunc='max')
        self.log_returns_fwr=self.log_returns_fwr.fillna(self.log_returns_fwr.mean())
        self.log_returns_fwr=self.log_returns_fwr.drop(columns=self.config['SimbolosRecientes'])

        self.w,self.f=self.obtener_pesos_cartera()
        self.w=pd.DataFrame(self.w,self.log_returns_fwr.columns)
        self.prices=self.descargar_precios_yahoo(self.config['Tiempo']['Inicio'],self.config['Tiempo']['Fin'])
        self.r=self.retornos_logaritmicos(self.prices).drop(columns=self.config['SimbolosRecientes'])


    def obtener_pesos_cartera(self):
        f=self.loss_function
        x0=np.zeros(self.log_returns_fwr.shape[1])+0.01
        bounds = [(0, self.config['PortOPT']['MaxLeverage']) for _ in range(self.log_returns_fwr.shape[1])]
        #constraints = ({'type': 'eq', 'fun': lambda w:self.Restriccion_Media(w,self.log_returns_fwr)}) #Dado que puede ser imposible de cumplir
        constraints = {'type': 'ineq', 'fun': lambda w:self.config['PortOPT']['MaxLeverage']-np.sum(w)}

        result=basinhopping(
            f,
            x0,
            minimizer_kwargs={"args":(self.log_returns_fwr),"bounds":bounds,"constraints":constraints},
            niter=30
        )
        return result.x,result.fun


    def Restriccion_Media(self,w,X): #Debe cumplir la media como condicion
        tasa=np.log(1+self.config['PortOPT']['Rd'])/252
        if np.sum(np.abs(w))>1:
            return self.config['PortOPT']['Target'] - np.mean(np.dot(X,w)) + (np.sum(np.abs(w))-1)*tasa
        else:
            return self.config['PortOPT']['Target'] - np.mean(np.dot(X,w))
        

    def loss_function(self,w,R):
        ls=self.desviacion_asimetrica(self.ret_diario(w,R))
        logging.debug(f"Calculando funcion de perdida: {ls}, con R: {self.ret_diario(w,R)} y w: {w}")
        return ls
    
    def ret_diario(self,w,R):
        Xw = np.dot(R, w)
        tasa=np.log(1+self.config['PortOPT']['Rd'])/252

        if np.sum(np.abs(w))>1:
            r=Xw-(np.sum(np.abs(w))-1)*tasa*np.ones(len(Xw))

        r=Xw
        return r

    def desviacion_asimetrica(self,X): #Esta es la funcion de perdida, el objetivo para minimizar
        suma=0
        for r in X:
            if r<self.config['PortOPT']['Target']: # Si es menor a lo esperado, se castiga
                suma+=abs(r-self.config['PortOPT']['Target'])
        return suma
    
    def visualizar_historico(self):
        self.rp=self.r.dot(self.w).cumsum()
        self.rp.plot()

    def simulacion_montecarlo(self,niter):
        dr=pd.date_range(datetime.now(),self.config['Tiempo']['MontecarloForecast'],freq='B')
        predicts_df=pd.DataFrame(index=dr)
        for i in range(niter):
            mdf=pd.DataFrame(index=dr)
            for sym in self.r.columns:
                hist, binds=np.histogram(self.r[sym].dropna(),bins=1000000)
                b=(binds[:-1]+binds[1:])/2
                s=b*hist
                s.sum()
                p=hist/hist.sum()
                idx=p>0

                ret=np.random.choice(s[idx],p=p[idx],size=dr.shape[0])
                mdf[sym]=ret
            mdf['tr']=mdf.dot(self.w)
            predicts_df[i]=mdf['tr']
        return predicts_df.cumsum()