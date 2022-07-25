import pandas as pd
import numpy as np

from datetime import datetime


def store_relational_JH_data():
    ''' Transformes the COVID data in a relational data set

    '''

    data_path='../data/raw/covid-19-data/public/data/jhu/total_cases.csv'
    pd_raw=pd.read_csv(data_path)

    pd_relational_model = pd_raw.fillna(method='ffill')
    pd_relational_model=pd_relational_model.set_index('date').stack(level=0).reset_index().rename(columns={'level_1':'country',
                                                                        0:'confirmed'},)
    pd_relational_model['date']=pd_relational_model.date.astype('datetime64[ns]')
    pd_relational_model.to_csv('../data/processed/Relational_data.csv',sep=';', index=False)

    print(' Number of rows stored: '+str(pd_relational_model.shape[0]))

if __name__ == '__main__':

    store_relational_JH_data()