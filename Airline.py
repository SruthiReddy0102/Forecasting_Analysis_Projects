import pandas as pd
import numpy as np

month=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

Airline=pd.read_csv("C:/Data Science/Assignments/Module-Forecasting/Codes/Airlines Data.csv")


Airline['Month'][0]
a=Airline['Month'][0]
a
a[0:3]

Airline['month']=0

for i in range(96):
    a=Airline['Month'][i]
    Airline['month'][i]=a[0:3]
    
dummy= pd.DataFrame(pd.get_dummies(Airline['month']))

Airline1=pd.concat((Airline,dummy),axis=1)


t=np.arange(1,97)
Airline1['t']=t
t_square=Airline1['t']*Airline1['t']
Airline1['t_square']=t_square

log_Passengers=np.log(Airline1['Passengers'])

Airline1['log_Passengers']=log_Passengers

train=Airline1.head(76)
test=Airline1.tail(20)

Airline1.Passengers.plot()

import statsmodels.formula.api as smf
#linear model
linear_model= smf.ols('Passengers~t',data=train).fit()
linear_model
predlinear= pd.Series(linear_model.predict(pd.DataFrame(test['t'])))
rmse_lin= np.sqrt(np.mean((np.array(test['Passengers'])-np.array(predlinear))**2))
rmse_lin

#quadratic model
quad_model=smf.ols('Passengers~t+t_square',data=train).fit()
predquad=pd.Series(quad_model.predict(pd.DataFrame(test[['t','t_square']])))
predquad

rmse_quad=np.sqrt(np.mean((np.array(test['Passengers'])-np.array(predquad))**2))
rmse_quad

#exponential model
exp_model=smf.ols('log_Passengers~t',data=train).fit()
predexp=pd.Series(exp_model.predict(pd.DataFrame(test['t'])))
rmse_exp= np.sqrt(np.mean((np.array(test['Passengers'])-np.array(np.exp(predexp)))**2))
rmse_exp

#additive seasonality
add_sea=smf.ols('Passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec',data=train).fit()
pred_addsea=pd.Series(add_sea.predict(pd.DataFrame(test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']])))
rmse_add= np.sqrt(np.mean((np.array(test['Passengers'])-np.array(pred_addsea))**2))
rmse_add

#additve with linear
add_sealin=smf.ols('Passengers~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec',data=train).fit()
predaddlin=pd.Series(add_sealin.predict(pd.DataFrame(test[['t','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']])))
rmseaddlin= np.sqrt(np.mean((np.array(test['Passengers'])-np.array(predaddlin))**2))
rmseaddlin

#additive with quadratic
add_seaquad = smf.ols('Passengers~t+t_square+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec',data=train).fit()
predaddquad= pd.Series(add_seaquad.predict(test[['t','t_square','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']]))
rmseaddquad= np.sqrt(np.mean((np.array(test['Passengers'])-np.array(predaddquad))**2))
rmseaddquad

#multiplicative seasonaity
mul_lin= smf.ols('log_Passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec',data=train).fit()
predmul=pd.Series(mul_lin.predict(test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']]))
rmsemul=np.sqrt(np.mean((np.array(test['Passengers'])-np.array(np.exp(predmul)))**2))
rmsemul

#multiplicative additive seasonality
mul_add= smf.ols('log_Passengers~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec',data=train).fit()
predmuladd= pd.Series(mul_add.predict(test[['t','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']]))
rmsemuladd = np.sqrt(np.mean((np.array(test['Passengers'])-np.array(np.exp(predmuladd)))**2))
rmsemuladd

#multiplicative additive quadratic
mul_quad = smf.ols('log_Passengers~t+t_square+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec',data=train).fit()
predmulquad= pd.Series(mul_add.predict(test[['t','t_square','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']]))
rmsemulquad = np.sqrt(np.mean((np.array(test['Passengers'])-np.array(np.exp(predmulquad)))**2))
rmsemulquad

#tabular form of rmse
data={'MODEL': pd.Series(['rmse_add','rmse_exp','rmse_lin','rmse_quad','rmseaddlin','rmseaddquad','rmsemul','rmsemuladd','rmsemulquad']), 'ERROR_VALUES':pd.Series([rmse_add,rmse_exp,rmse_lin,rmse_quad,rmseaddlin,rmseaddquad,rmsemul,rmsemuladd,rmsemulquad])}
table_rmse= pd.DataFrame(data)
table_rmse


#final model is 
finalmodel =smf.ols('log_Passengers~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec',data=Airline1).fit()
predictfinal=pd.Series(finalmodel.predict(Airline1))
predictfinal
