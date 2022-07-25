import pandas as pd
import numpy as np

from scipy import optimize
from scipy import integrate

import dash
dash.__version__
from dash import dcc
from dash import html
from dash.dependencies import Input, Output,State

import plotly.graph_objects as go

import plotly.io as pio
pio.renderers.default = "browser"

import os
print(os.getcwd())

df_analyse = pd.read_csv('../data/processed/COVID_small_flat_table.csv',sep=';')  
df_analyse = df_analyse.sort_values('date',ascending=True)
new_Frame = df_analyse.drop('date',axis=1)
data_top = new_Frame.columns


beta = 0.4
gamma = 0.1
R0 = 0
N0 = 10000000

i=100
n=150

def fit_odeint(x, beta, gamma):
    '''
    helper function for the integration
    '''
    return integrate.odeint(SIR_model_t, (S0, I0, R0), t, args=(beta, gamma))[:,1] # we only would like to get dI

def SIR_model_t(SIR,t,beta,gamma):
    ''' Simple SIR model
        S: susceptible population
        t: time step, mandatory for integral.odeint
        I: infected people
        R: recovered people
        beta: 
        
        overall condition is that the sum of changes (differnces) sum up to 0
        dS+dI+dR=0
        S+I+R= N (constant size of population)
    
    '''
    
    S,I,R=SIR
    dS_dt=-beta*S*I/N0          #S*I is the 
    dI_dt=beta*S*I/N0-gamma*I
    dR_dt=gamma*I
    return dS_dt,dI_dt,dR_dt


ydata1=np.array(new_Frame['Germany'][i:n])
t = np.arange(len(ydata1))
N0 = 80000000
I0 = ydata1[0]
S0 = N0-I0
popt, pcov = optimize.curve_fit(fit_odeint, t, ydata1) ## Train the model / Fit the model
perr = np.sqrt(np.diag(pcov))
fitted1=fit_odeint(t, *popt)

ydata2=np.array(new_Frame['Italy'][i:n])
I0 = ydata2[0]
N0 = 330000000
S0 = N0-I0
popt, pcov = optimize.curve_fit(fit_odeint, t, ydata2) ## Train the model / Fit the model
perr = np.sqrt(np.diag(pcov))
fitted2=fit_odeint(t, *popt)

ydata3=np.array(new_Frame['US'][i:n])
I0 = ydata3[0]
S0 = N0-I0
popt, pcov = optimize.curve_fit(fit_odeint, t, ydata3) ## Train the model / Fit the model
perr = np.sqrt(np.diag(pcov))
fitted3=fit_odeint(t, *popt)


fittedData = pd.DataFrame({'Germany': fitted1,'Italy': fitted2, 'US': fitted3})

ydata = pd.DataFrame({'Germany': ydata1,'Italy': ydata2, 'US': ydata3})

fig = go.Figure()

app = dash.Dash()
app.layout = html.Div([

    dcc.Markdown('''
    #  Applied Data Science on COVID-19 data
    This dashboard shows the dynamic SIR-Model of three pre-selected countries
    '''),

    dcc.Markdown('''
    ## Multi-Select Country for visualization
    '''),


    dcc.Dropdown(
        id='drop_down_country',
        #options=[ {'label': each,'value':each} for each in data_top.unique()],
        options=[ 
            {'label': 'Germany','value': 'Germany'},
            {'label': 'US','value': 'US'},
            {'label': 'Italy','value': 'Italy'}],
        value=['Germany'], # which are pre-selected
        multi=True
    ),

    dcc.Graph(figure=fig, id='main_window_slope1')
])



@app.callback(
    Output(component_id='main_window_slope1', component_property='figure'),
    [Input(component_id='drop_down_country', component_property='value')]
)
def update_figure(country_list):
    
    my_yaxis={'type':"log",
               'title':'Number of Infections'
              }

    traces = []
    for each in country_list:

        #df_plot=df_input_large[df_input_large['country']==each]
        df_plot=ydata
        
        
        
        traces.append(dict(x=t,
                                    y=fittedData[each],
                                    mode='markers+lines',
                                    marker = dict(
                                      size = 3
                                    ),
                                    opacity=0.9,
                                    name=each +"fitted"
                            )
                    )
        traces.append(dict(x=t,
                                    y=ydata[each],
                                    mode='markers',
                                    marker = dict(
                                      size = 5
                                    ),
                                    opacity=0.9,
                                    name=each
                            )
                    )
        

    return {
            'data': traces,
            'layout': dict (
                width=1280,
                height=720,

                xaxis={'title':'Timeline',
                        'tickangle':-45,
                        'nticks':20,
                        'tickfont':dict(size=14,color="#7f7f7f"),
                      },

                yaxis=my_yaxis
        )
    }

if __name__ == '__main__':

    app.run_server(debug=True, use_reloader=False)