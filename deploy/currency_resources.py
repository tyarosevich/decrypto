# A module to query a finance API for stock market indexes and crypto values, then format and write these
# values as rows to two separate db tables, with each column being an index or crypto. These are intended
# to serve as parts of the training features downstream.


from urllib.request import urlopen
import certifi
import json
import pandas as pd
from datetime import datetime
import pytz
from aws_resources import get_secret
from db_resources import db_write, get_db_engine
# Note this assumes AWS Lambda, with its preconfigured logger.
import logging
logging.getLogger().setLevel(logging.INFO)

# These are the stock index codes as they appear in the wild, and as formatted for financialmodelingprep API.
# lst_index_codes = ['^TYX', '^FVX', '^TNX', '^HSCE', '^SPGSCI', '^STI', '^BVSP', '^MXX', '^GSPTSE', '^VIX', '^DJI', '^FCHI', '^N100', 'IMOEX.ME', '^AXJO', '^NZ50', '^N225', '^TWII', '^BFX', '^TA125.TA', '^KLSE', '^BSESN', '^STOXX50E', '^IXIC', '^KS11', '^JN0U.JO', '000001.SS', '^CASE30', '^GSPC', '^FTSE', '^BUK100P', '^AORD', '^RUT', '^GDAXI', '^JKSE', '^HSI', '^NSEI', '^NSEBANK', '^SP500TR', '^DJITR', '^RUTTR', '^XNDX', '^RUITR', '^RUATR', '^RMCCTR', '^AEX', '^IRX', '^DJT', '^NDX', '^RUA', '^RUI', '^RVX', '^VXN', '^OVX', '^GVZ', '^VVIX', '^VXSLV', '^IBEX', '^SSMI', 'DX-Y.NYB', '^XAX', '^MID', '^NYA', 'TX60.TS', 'XU100.IS']
# lst_index_codes_formatted = ['%5ETYX', '%5EFVX', '%5ETNX', '%5EHSCE', '%5ESPGSCI', '%5ESTI', '%5EBVSP', '%5EMXX', '%5EGSPTSE', '%5EVIX', '%5EDJI', '%5EFCHI', '%5EN100', 'IMOEX.ME', '%5EAXJO', '%5ENZ50', '%5EN225', '%5ETWII', '%5EBFX', '%5ETA125.TA', '%5EKLSE', '%5EBSESN', '%5ESTOXX50E', '%5EIXIC', '%5EKS11', '%5EJN0U.JO', '000001.SS', '%5ECASE30', '%5EGSPC', '%5EFTSE', '%5EBUK100P', '%5EAORD', '%5ERUT', '%5EGDAXI', '%5EJKSE', '%5EHSI', '%5ENSEI', '%5ENSEBANK', '%5ESP500TR', '%5EDJITR', '%5ERUTTR', '%5EXNDX', '%5ERUITR', '%5ERUATR', '%5ERMCCTR', '%5EAEX', '%5EIRX', '%5EDJT', '%5ENDX', '%5ERUA', '%5ERUI', '%5ERVX', '%5EVXN', '%5EOVX', '%5EGVZ', '%5EVVIX', '%5EVXSLV', '%5EIBEX', '%5ESSMI', 'DX-Y.NYB', '%5EXAX', '%5EMID', '%5ENYA', 'TX60.TS', 'XU100.IS']


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

def _get_finance_api_key():
    """
    A wrapper to collect API info and query API.
    :return:
    :rtype: dict
    """

    stock_secrets = get_secret('stock_api_secrets')
    dct_auth = json.loads(stock_secrets['SecretString'])
    return dct_auth['stock_api_key']

def _get_stock_indexes(api_key):
    """
    A simple wrapper to use the API query function.
    :param api_key:
    :type api_key:
    :return:
    :rtype: DataFrame
    """
    
    lst_index_codes_formatted = ['%5ETYX', '%5EFVX', '%5ETNX', '%5EHSCE', '%5ESPGSCI', '%5ESTI', '%5EBVSP', '%5EMXX', '%5EGSPTSE', '%5EVIX', '%5EDJI', '%5EFCHI', '%5EN100', 'IMOEX.ME', '%5EAXJO', '%5ENZ50', '%5EN225', '%5ETWII', '%5EBFX', '%5ETA125.TA', '%5EKLSE', '%5EBSESN', '%5ESTOXX50E', '%5EIXIC', '%5EKS11', '%5EJN0U.JO', '000001.SS', '%5ECASE30', '%5EGSPC', '%5EFTSE', '%5EBUK100P', '%5EAORD', '%5ERUT', '%5EGDAXI', '%5EJKSE', '%5EHSI', '%5ENSEI', '%5ENSEBANK', '%5ESP500TR', '%5EDJITR', '%5ERUTTR', '%5EXNDX', '%5ERUITR', '%5ERUATR', '%5ERMCCTR', '%5EAEX', '%5EIRX', '%5EDJT', '%5ENDX', '%5ERUA', '%5ERUI', '%5ERVX', '%5EVXN', '%5EOVX', '%5EGVZ', '%5EVVIX', '%5EVXSLV', '%5EIBEX', '%5ESSMI', 'DX-Y.NYB', '%5EXAX', '%5EMID', '%5ENYA', 'TX60.TS', 'XU100.IS']

    url = ("https://financialmodelingprep.com/api/v3/quote/{}?apikey={}").format(",".join(lst_index_codes_formatted), api_key)
    result = _get_jsonparsed_data(url)
    return pd.DataFrame(result)

def _update_stock_indexes(api_key, engine):
    """
    A top function to get new stock indexes from the API, format a row, and write it to the database.
    :param api_key:
    :type api_key: str
    :param engine:
    :type engine: SQLAlchey Engine
    :return:
    :rtype: None
    """
    
    lst_index_codes = ['^TYX', '^FVX', '^TNX', '^HSCE', '^SPGSCI', '^STI', '^BVSP', '^MXX', '^GSPTSE', '^VIX', '^DJI', '^FCHI', '^N100', 'IMOEX.ME', '^AXJO', '^NZ50', '^N225', '^TWII', '^BFX', '^TA125.TA', '^KLSE', '^BSESN', '^STOXX50E', '^IXIC', '^KS11', '^JN0U.JO', '000001.SS', '^CASE30', '^GSPC', '^FTSE', '^BUK100P', '^AORD', '^RUT', '^GDAXI', '^JKSE', '^HSI', '^NSEI', '^NSEBANK', '^SP500TR', '^DJITR', '^RUTTR', '^XNDX', '^RUITR', '^RUATR', '^RMCCTR', '^AEX', '^IRX', '^DJT', '^NDX', '^RUA', '^RUI', '^RVX', '^VXN', '^OVX', '^GVZ', '^VVIX', '^VXSLV', '^IBEX', '^SSMI', 'DX-Y.NYB', '^XAX', '^MID', '^NYA', 'TX60.TS', 'XU100.IS']

    df_raw = _get_stock_indexes(api_key)
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

    db_write(df_update, 'raw_stock_indexes', engine)

def update_indexes_to_db():
    """
    A public function to access the module and update both stock indexes and crypto values to the project db.
    :return:
    :rtype: None
    """
    engine = get_db_engine()
    api_key = _get_finance_api_key()

    _update_stock_indexes(api_key, engine)
    _update_crypto_prices(api_key, engine)

def _update_crypto_prices(api_key, engine):
    """
    A top wrapper to query the API for crypto prices, format them to a row, and write them to the db.
    :param api_key:
    :type api_key: str
    :param engine:
    :type engine: SQLAlchemy Engine
    :return:
    :rtype: None
    """
    lst_crypto_codes = ['BTCUSD', 'ETHUSD']
    
    df_raw = _get_crypto_prices(api_key, lst_crypto_codes)
    df_raw['symbol_sort'] = pd.Categorical(
        df_raw['symbol'],
        categories=lst_crypto_codes,
        ordered=True
    )
    df_raw.sort_values('symbol_sort', inplace=True)
    cols = ['created_at'] + lst_crypto_codes
    df_update = pd.DataFrame(columns=cols)
    naive_start_time = datetime.now(pytz.utc)
    update_vals = [naive_start_time] + df_raw['price'].to_list()
    df_update.loc[0] = update_vals
    
    

    engine = get_db_engine()
    db_write(df_update, 'raw_crypto_prices', engine)

def _get_crypto_prices(api_key, lst_crypto_codes):
    """
    A simple wrapper to query the API for stock index values.
    :param api_key:
    :type api_key: str
    :param lst_crypto_codes:
    :type lst_crypto_codes: list
    :return:
    :rtype: DataFrame
    """
    
    url = 'https://financialmodelingprep.com/api/v3/quote/{}?apikey={}'.format(",".join(lst_crypto_codes), api_key)
    result = _get_jsonparsed_data(url)
    return pd.DataFrame(result)

