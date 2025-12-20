from JHProgress import JHProgress
import pandas as pd
import numpy as np
from scipy.optimize import basinhopping,minimize
import logging
from datetime import datetime
import yfinance as yfin
from datetime import datetime
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Optimizer():

    def __init__(self) -> None:
        self.val_score=1e6

    def optimizar(self,x0,args,valid_ds,constraints,bounds,target,rd):
        result=minimize(self.loss_funct,x0,args=args,callback=(lambda x: self.early_stopping_callback(x,valid_ds,target,rd)),constraints=constraints,bounds=bounds,options={"maxiter":1000,"eps":1e-10})
        return result.x,result.fun

    def loss_funct(self,w,Ra):
        R=np.dot(Ra,w)
        return self.suma_negativos(R)
    
    def suma_negativos(self,R):
        suma=0
        for r in R:
            if r<0:
                suma+=abs(r)
        return suma
    
    def early_stopping_callback(self,w,valid_ds,target,rd):
        #logging.info(f"Evaluando loss function para w: {w}")
        
        val_score=self.loss_funct(w,valid_ds)
        logging.info(f"Valid score generado : {val_score}, con R: {np.sum(np.dot(valid_ds,w))} y target:{target}")
        if val_score<self.val_score and self.return_constraint(w,valid_ds,target,rd)<=0:
            self.val_score=val_score #Actualiza el mejor score de validacion actual
            #logging.info(f"Nuevo score: {self.val_score}")
            self.best_w=w
        else:
            #logging.info(f"No hay mejoras para el valid_ds")
            
            return True
        
    def return_constraint(self,w,X,target,rd):
        return target-np.sum(np.dot(X,w)) if np.sum(w)<=1 else target+rd*(np.sum(w)-1)-np.sum(np.dot(X,w))

class PortOPT(JHProgress):


    def __init__(self) -> None:
        super().__init__()
        self.log_returns=self._cargar_retornos() #Retornos logaritmicos historicos incluidos los pronosticos
        self.r=self.log_returns.copy()
        self.mcorr=self.log_returns.corr()
        

    def _cargar_retornos(self):
        ret_historicos=yfin.download(self.config['Simbolos'],self.config['PortOPT']['init_train'])['Close']
        ret_historicos=np.log(ret_historicos/ret_historicos.shift(1))
        ret_historicos=ret_historicos.resample(self.config['PortOPT']['freq']).sum()

        forecasted_prices=self.cargar_desde_sql(self.config['BaseDatos']['VistaForecast'])
        forecasted_prices=pd.pivot_table(forecasted_prices,columns='Simbolo',values='PRICE_FW',index='Date',aggfunc='max').dropna(how='all')
        forecasted_prices=np.log(forecasted_prices/forecasted_prices.shift(1))
        forecasted_prices=forecasted_prices.resample(self.config['PortOPT']['freq']).sum()
        ret_historicos=pd.concat([ret_historicos,forecasted_prices])
        return ret_historicos.fillna(0)



    def obtener_pesos_cartera(self,r_target=None,ajustar_atributos=True):
        if r_target==None:
            r_target=self.config['PortOPT']['Target']

        precios_train=self.log_returns[:self.config['PortOPT']['init_valid']]
        precios_valid=self.log_returns[self.config['PortOPT']['init_valid']:]

        logging.info(f"Entrenando los pesos desde: {precios_train.index[0]}, hasta {precios_valid.index[0]}")
        logging.info(f"Validando los pesos desde: {precios_valid.index[0]}, hasta {precios_valid.index[-1]}")

        bounds=[(0, None) for _ in range(precios_train.shape[1])]

        opt=Optimizer()


        years=(pd.to_datetime(self.config['PortOPT']['init_valid'])-pd.to_datetime(self.config['PortOPT']['init_train'])).days/365.25
        years_valid=(precios_valid.index[-1] - precios_valid.index[0]).days/365.25

        constraints = (
            {'type': 'ineq', 'fun': lambda w:self.config['PortOPT']['MaxLeverage']-np.sum(w)}, #Maximo el leverage especificado
            {'type': 'eq', 'fun': lambda w:opt.return_constraint(w,precios_train, (r_target+0.15)*years, self.config['PortOPT']['Rd']*years)} # Garantiza que se respete el target especificado, 0.15 es requerido
        )

        w_train,f_train=opt.optimizar(
            np.zeros(precios_train.shape[1]),
            args=(precios_train),
            valid_ds=precios_valid,
            constraints=constraints,
            bounds=bounds,
            target=r_target*years_valid,
            rd=self.config['PortOPT']['Rd']*years_valid
        )

        try:
            w=opt.best_w
            l=opt.loss_funct(opt.best_w,precios_valid)
        except AttributeError:
            logging.info(f"Cuidado, no se pudo alcanzar el valor del target en validacion: posible sobreajuste o target muy elevado, se asignaran los pesos del train")
            w=w_train
            l=f_train

        logging.info(f"El retorno en validacion es: {np.sum(np.dot(precios_valid,w))}, el riesgo es: {opt.loss_funct(w,precios_valid)}")

        if ajustar_atributos:
            self.w=opt.best_w
            self.R=self.log_returns.dot(self.w)
        return w,l


    def efficient_frontier(self):
        rows=[]
        r=np.arange(100)/100
        for r_sin in r:
            risk=self.obtener_pesos_cartera(r_sin,ajustar_atributos=False)[1]
            rows.append([r_sin,risk])
        return np.array(rows)

        

  
    
    def visualizar_historico(self):
        self.rp=self.r.dot(self.w).cumsum()
        self.rp.plot()
        plt.grid()

    def simulacion_montecarlo(self,niter):
        dr=pd.date_range(datetime.now(),self.config['Tiempo']['MontecarloForecast'],freq=self.config['PortOPT']['freq'])
        predicts_df=pd.DataFrame(index=dr)
        for i in range(niter):
            mdf=pd.DataFrame(index=dr)
            for sym in self.r.columns:
                r=self.r[self.r[sym]!=0][sym]
                hist, binds=np.histogram(self.r.dropna(),bins=1000000)
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
    
    def crear_cartera(self,vm_total: float):
        vm=pd.DataFrame(self.w*vm_total,index=self.r.columns)
        debt=vm_total-np.sum(vm[0])
        vm['cash']=debt
        vm['w']=self.w
        vm=vm[vm['w']>1e-4]
        return vm