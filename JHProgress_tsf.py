from JHProgress import JHProgress
import lightgbm as lgb
import pandas as pd
from datetime import datetime
import logging
import numpy as np
from sklearn.metrics import root_mean_squared_error
import optuna
import json

class TSF(JHProgress):

    def __init__(self) -> None:
        super().__init__()
        self.table_fundamentals_stationary=super().cargar_desde_sql(self.config['BaseDatos']['FundamentalesEstacionarios'])
        self.table_fundamentals_linear=super().cargar_desde_sql(self.config['BaseDatos']['FundamentalesTendencia']).fillna(0) #Dado que los 0 no afectan los linear models

    def feature_engineering(self, dataset, sym,feature_generator):

        tf=1 if sym not in self.config['SimbolosRecientes'] else 0.5
        forecast_horizon=int(self.config['LightGBM']['TestYears']*252*tf)

        dataset=dataset[dataset['Simbolo']==sym]
        dataset['Returns']=np.log(dataset['Price']/dataset['Price'].shift(forecast_horizon)) #Calcular Returns
        
        for ventana in self.config['Ventanas']:
            dataset[f'lag_{ventana}']=dataset[feature_generator].shift(ventana)
            dataset[f'ma_{ventana}']=dataset[feature_generator].rolling(window=ventana).mean()
            dataset[f'std_{ventana}']=dataset[feature_generator].rolling(window=ventana).std()
            dataset[f'skew_{ventana}']=dataset[feature_generator].rolling(window=ventana).skew()
            dataset[f'kurt_{ventana}']=dataset[feature_generator].rolling(window=ventana).kurt()
            dataset[f'total_return_{ventana}']=dataset[feature_generator].rolling(window=ventana).sum()
            dataset[f'min_{ventana}']=dataset[feature_generator].rolling(window=ventana).min()
            dataset[f'max_{ventana}']=dataset[feature_generator].rolling(window=ventana).max()
            dataset[f'first_return_{ventana}']=dataset[feature_generator].rolling(window=ventana).apply(lambda x: x.values[0])
            dataset[f'last_return_{ventana}']=dataset[feature_generator].rolling(window=ventana).apply(lambda x: x.values[-1])

        
        dataset['target']=np.log(dataset['Price'].shift(-forecast_horizon)/dataset['Price']) #Crear Target
        key=f'lag_{self.config["Ventanas"][-2]}' if sym not in self.config['SimbolosRecientes'] else f'lag_{self.config["Ventanas"][-4]}'
        dataset=dataset.dropna(subset=[key]) #Dropear NA
        dataset=dataset.drop(columns=['Price']) #Eliminar Price
        return dataset

    def entrenar_modelos(self):
        self.models={}
        for sym in self.table_fundamentals_stationary['Simbolo'].unique():

            dataset=self.feature_engineering(self.table_fundamentals_stationary, sym,feature_generator='Returns') #Contiene, Train, Valid y Test
            tf=1 if sym not in self.config['SimbolosRecientes'] else 0.5
            if sym not in self.config['SimbolosRecientes']:
                last_train=pd.Timestamp(datetime.now().date().replace(year=datetime.now().year - self.config['LightGBM']['TestYears']*tf - self.config['LightGBM']['ValidationYears']*tf)) 
                last_valid=pd.Timestamp(datetime.now().date().replace(year=datetime.now().year - self.config['LightGBM']['TestYears']*tf))
            else:
                last_train=pd.Timestamp(datetime.now().date()) - pd.DateOffset(months=6 + int(self.config['LightGBM']['ValidationYears']*12*tf))
                last_valid=pd.Timestamp(datetime.now().date()) - pd.DateOffset(months=6)
            
            
            train_data=dataset[dataset['Date']<(last_train)]
            train_data['fold']=pd.qcut(train_data['Date'], self.config['LightGBM']['folds'], labels=False)+1
            
            validation_data=dataset[(dataset['Date']>=last_train) & (dataset['Date']<last_valid)]

            lgb_parameters={
                'objective': self.config['LightGBM']['Metric'],
                'n_estimators': self.config['LightGBM']['NEstimators'],
                'learning_rate': self.config['LightGBM']['LearningRate'],
                'num_leaves': self.config['LightGBM']['NumLeaves'],
                'random_state': self.config['LightGBM']['RandomState'],
                'verbosity': -1,
                'subsample': self.config['LightGBM']['subsample'],
                'colsample_bytree': self.config['LightGBM']['colsample_bytree'],
                'bagging_freq': self.config['LightGBM']['bagging_freq']
            }

            feature_cols=[col for col in dataset.columns if col not in ['Date','Simbolo','target','fold']]
            for fold in range(self.config['LightGBM']['folds']):
                logging.info(f'Entrenando modelo para {sym}, fold {fold+1}/{self.config["LightGBM"]["folds"]}')
                train_fold_data=train_data[train_data['fold']>fold]

                model = lgb.LGBMRegressor(**lgb_parameters)
                
                model.fit(train_fold_data[feature_cols], train_fold_data['target'],
                        eval_set=[(validation_data[feature_cols], validation_data['target'])],callbacks=[lgb.log_evaluation(100),lgb.early_stopping(self.config['LightGBM']['EarlyStoppingRounds'])])
                self.models[sym]=model
                model.booster_.save_model(f'models/Model_{sym}_{fold}.model')
                logging.info(f'Modelo entrenado y guardado para {sym}, fold {fold}, shape de los datos de entrenamiento: {train_fold_data[feature_cols].shape}')

            #Entrenar modelo final con todos los datos de Train + Valid
            model = lgb.LGBMRegressor(**lgb_parameters)            
            model.fit(dataset[dataset['target'].notna()][feature_cols], dataset[dataset['target'].notna()]['target'])
            self.models[sym]=model
            model.booster_.save_model(f'models/Model_{sym}_{fold+1}.model')
            logging.info(f'Modelo entrenado y guardado para {sym}, fold {fold+1}, shape de los datos de entrenamiento: {dataset[feature_cols].shape}')

    def inferencia(self, sym):
        preds= []
        dataset=self.feature_engineering(self.table_fundamentals_stationary, sym)
        feature_cols=[col for col in dataset.columns if col not in ['Date','Simbolo','target']]
        for fold in range(self.config['LightGBM']['folds']+1):
            model=lgb.Booster(model_file=f'models/Model_{sym}_{fold}.model')
            dataset[f'Predicted_Returns_fold_{fold}']=model.predict(dataset[feature_cols])
            preds.append(dataset[f'Predicted_Returns_fold_{fold}'])
        dataset['Predicted_Returns']=np.mean(preds, axis=0)
        return dataset
    
    def hyper_tunning(self,sym):
        dataset=self.feature_engineering(self.table_fundamentals_stationary, sym) #Contiene, Train, Valid y Test
        tf=1 if sym not in self.config['SimbolosRecientes'] else 0.5
        if sym not in self.config['SimbolosRecientes']:
            last_train=pd.Timestamp(datetime.now().date().replace(year=datetime.now().year - self.config['LightGBM']['TestYears']*tf - self.config['LightGBM']['ValidationYears']*tf)) 
            last_valid=pd.Timestamp(datetime.now().date().replace(year=datetime.now().year - self.config['LightGBM']['TestYears']*tf))
        else:
            last_train=pd.Timestamp(datetime.now().date()) - pd.DateOffset(months=6 + int(self.config['LightGBM']['ValidationYears']*12*tf))
            last_valid=pd.Timestamp(datetime.now().date()) - pd.DateOffset(months=6)
        
        
        train_data=dataset[dataset['Date']<(last_train)]
        train_data['fold']=pd.qcut(train_data['Date'], self.config['LightGBM']['folds'], labels=False)+1
        
        validation_data=dataset[(dataset['Date']>=last_train) & (dataset['Date']<last_valid)]

        lgb_parameters={
            'objective': self.config['LightGBM']['Metric'],
            'n_estimators': self.config['LightGBM']['NEstimators'],
            'learning_rate': self.config['LightGBM']['LearningRate'],
            'num_leaves': self.config['LightGBM']['NumLeaves'],
            'random_state': self.config['LightGBM']['RandomState'],
            'verbosity': -1,
            'subsample': self.config['LightGBM']['subsample'],
            'colsample_bytree': self.config['LightGBM']['colsample_bytree'],
            'bagging_freq': self.config['LightGBM']['bagging_freq']
        }

        feature_cols=[col for col in dataset.columns if col not in ['Date','Simbolo','target','fold']]
        self._hyper_tunning(train_data[feature_cols],train_data['target'],validation_data[feature_cols],validation_data['target'])
    
    def _hyper_tunning(self,x_train,y_train,x_valid,y_valid):

        def lgbm_objective(trial, df_fold_train,df_fold_train_target,
                   df_fold_valid, df_fold_valid_target):
    
            lgb_parameters={
                'objective': self.config['LightGBM']['Metric'],
                'n_estimators': self.config['LightGBM']['NEstimators'],
                'learning_rate': trial.suggest_float('learning_rate',1e-8,1,log=True),
                'num_leaves': self.config['LightGBM']['NumLeaves'],
                'random_state': self.config['LightGBM']['RandomState'],
                'verbosity': -1,
                'subsample': self.config['LightGBM']['subsample'],
                'colsample_bytree': self.config['LightGBM']['colsample_bytree'],
                'bagging_freq': self.config['LightGBM']['bagging_freq']
            }
            pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "rmse")
            
            callbacks=[lgb.callback.log_evaluation(period=100)]

            callbacks.append(pruning_callback)
            callbacks.append(lgb.early_stopping(stopping_rounds=100))
            

            
            lgb_model = lgb.LGBMRegressor(**lgb_parameters)
            lgb_model.fit(
                df_fold_train,
                df_fold_train_target,
                eval_set=[(df_fold_valid, df_fold_valid_target)],
                callbacks=callbacks
            )

            # Use the last metric for early stopping
            evals_result = lgb_model.booster_.best_score
            last_metric = list(evals_result.values())[-1]
            trial.set_user_attr('best_model', lgb_model)  # Save the model in the trial
            return last_metric[list(last_metric.keys())[-1]]
        
        # Train a LightGBM model for the current fold
        study = optuna.create_study(direction='minimize',
                    sampler=optuna.samplers.TPESampler(seed=400),
                    pruner=optuna.pruners.MedianPruner(n_startup_trials=100,n_warmup_steps=200))
        study.optimize(lambda trial: lgbm_objective(trial,x_train,y_train,
                                        x_valid,y_valid),
                            n_trials=self.config['LightGBM']['OptunaTrials'],timeout=3600)

        print('Best trial: score {}, params {}'.format(study.best_value, study.best_params))

        best_model = study.trials[study.best_trial.number].user_attrs['best_model']
        lgb_model=best_model

        results={"Params":lgb_model.get_params()}

        # Guardar en archivo
        with open('datos.json', 'w') as f:
            json.dump(results, f, indent=4) # indent=4 para formato legible

        self.feature_importances=best_model.feature_importances_

        logging.info(f"Modelo tunneado efectivamente con los parametros {lgb_model.get_params()}")



        

    
