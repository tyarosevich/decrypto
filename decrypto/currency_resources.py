from urllib.request import urlopen
import certifi
import json
import pandas as pd
from datetime import datetime
import pytz
from aws_resources import get_secret
from db_resources import db_write, get_db_engine

lst_index_codes = ['^TYX', '^FVX', '^TNX', '^HSCE', '^SPGSCI', '^STI', '^BVSP', '^MXX', '^GSPTSE', '^VIX', '^DJI', '^FCHI', '^N100', 'IMOEX.ME', '^AXJO', '^NZ50', '^N225', '^TWII', '^BFX', '^TA125.TA', '^KLSE', '^BSESN', '^STOXX50E', '^IXIC', '^KS11', '^JN0U.JO', '000001.SS', '^CASE30', '^GSPC', '^FTSE', '^BUK100P', '^AORD', '^RUT', '^GDAXI', '^JKSE', '^HSI', '^NSEI', '^NSEBANK', '^SP500TR', '^DJITR', '^RUTTR', '^XNDX', '^RUITR', '^RUATR', '^RMCCTR', '^AEX', '^IRX', '^DJT', '^NDX', '^RUA', '^RUI', '^RVX', '^VXN', '^OVX', '^GVZ', '^VVIX', '^VXSLV', '^IBEX', '^SSMI', 'DX-Y.NYB', '^XAX', '^MID', '^NYA', 'TX60.TS', 'XU100.IS']
lst_index_codes_formatted = ['%5ETYX', '%5EFVX', '%5ETNX', '%5EHSCE', '%5ESPGSCI', '%5ESTI', '%5EBVSP', '%5EMXX', '%5EGSPTSE', '%5EVIX', '%5EDJI', '%5EFCHI', '%5EN100', 'IMOEX.ME', '%5EAXJO', '%5ENZ50', '%5EN225', '%5ETWII', '%5EBFX', '%5ETA125.TA', '%5EKLSE', '%5EBSESN', '%5ESTOXX50E', '%5EIXIC', '%5EKS11', '%5EJN0U.JO', '000001.SS', '%5ECASE30', '%5EGSPC', '%5EFTSE', '%5EBUK100P', '%5EAORD', '%5ERUT', '%5EGDAXI', '%5EJKSE', '%5EHSI', '%5ENSEI', '%5ENSEBANK', '%5ESP500TR', '%5EDJITR', '%5ERUTTR', '%5EXNDX', '%5ERUITR', '%5ERUATR', '%5ERMCCTR', '%5EAEX', '%5EIRX', '%5EDJT', '%5ENDX', '%5ERUA', '%5ERUI', '%5ERVX', '%5EVXN', '%5EOVX', '%5EGVZ', '%5EVVIX', '%5EVXSLV', '%5EIBEX', '%5ESSMI', 'DX-Y.NYB', '%5EXAX', '%5EMID', '%5ENYA', 'TX60.TS', 'XU100.IS']


def _get_jsonparsed_data(url):
    """
    Receive the content of ``url``, parse it as JSON and return the object.

    Parameters
    ----------
    url : str

    Returns
    -------
    dict
    """
    response = urlopen(url, cafile=certifi.where())
    data = response.read().decode("utf-8")
    return json.loads(data)

def _get_stock_indexes():

    stock_secrets = get_secret('stock-api-secrets')
    dct_auth = json.loads(stock_secrets['SecretString'])
    api_key = dct_auth['stock_api_key']

    url = ("https://financialmodelingprep.com/api/v3/quote/{}?apikey={}").format(",".join(lst_index_codes_formatted), api_key)
    result = _get_jsonparsed_data(url)
    return pd.DataFrame(result)

def update_indexes_to_db():
    
    df_raw = _get_stock_indexes()
    df_raw['symbol_sort'] = pd.Categorical(
        df_raw['symbol'],
        categories=lst_index_codes,
        ordered=True
    )
    df_raw.sort_values('symbol_sort', inplace=True)
    cols = ['created_at'] + lst_index_codes
    df_update = pd.DataFrame(columns=cols)
    naive_start_time = datetime.now(pytz.utc)
    update_vals = [naive_start_time] + df_raw['price'].to_list()
    df_update.loc[0] = update_vals

    engine = get_db_engine()
    db_write(df_update, 'raw_stock_indexes', engine)





