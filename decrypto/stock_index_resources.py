import json
from pathlib import Path
import pandas as pd

# Need to add key from env.
api_key = None

# Load all stock index info for this API.
path_stock_indexes = '/home/tyarosevich/Projects/decrypto/data/stock_index_codes.json'
with open(path_stock_indexes)as json_file:
    lst_stock_indexes_all = json.load(json_file)['majorIndexesList']

# Get the codes for the desired indexes.
lst_stock_indexes_used = ["Dow Jones", "Nasdaq", "S&P 500", "NYSE Arca Technology 100 Index", "NYSE Composite Index"]
lst_stock_index_codes = [dct['ticker'] for dct in lst_stock_indexes_all if dct['indexName'] in lst_stock_indexes_used]
lst_stock_index_codes_formatted = [w.replace('.', '%5E') for w in lst_stock_index_codes]
#%%
#!/usr/bin/env python


from urllib.request import urlopen
import certifi
import json

def get_jsonparsed_data(url):
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

url = ("https://financialmodelingprep.com/api/v3/quote/{}?apikey={}").format(",".join(lst_stock_index_codes_formatted), api_key)

dct_response_parsed = get_jsonparsed_data(url)

#%%
try:
    # For Python 3.0 and later
    from urllib.request import urlopen
except ImportError:
    # Fall back to Python 2's urllib2
    from urllib2 import urlopen

import certifi
import json

def get_jsonparsed_data(url):
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

url = ("https://financialmodelingprep.com/api/v3/quote/%5EGSPC,%5EDJI,%5EIXIC?apikey=96fabfa2f33c6915608e6ebe52f7750b")
print(get_jsonparsed_data(url))


#%%
url = "https://financialmodelingprep.com/api/v3/symbol/available-indexes?apikey={}".format(api_key)
indexes = get_jsonparsed_data(url)

# So the results from this should provide the codes, replace carats and periods with %5E

#%%
lst_index_codes = [x['symbol'] for x in indexes]
lst_stock_index_codes_formatted = [w.replace('^', '%5E') for w in lst_index_codes]
url = ("https://financialmodelingprep.com/api/v3/quote/{}?apikey={}").format(",".join(lst_stock_index_codes_formatted), api_key)
result = get_jsonparsed_data(url)
#%%
df_result = pd.DataFrame(result)


#%%

df_symbol_lookup = df_result[['symbol', 'name']]
out_path = Path('~/Projects/decrypto/temp/index_lookup_table.csv')
df_symbol_lookup.to_csv(out_path, index=False, encoding = 'utf8')

#%%
sql_add = "` INT, `".join(lst_index_codes)

#%%

index_cols =['%5ETYX', '%5EFVX', '%5ETNX', '%5EHSCE', '%5ESPGSCI', '%5ESTI', '%5EBVSP', '%5EMXX', '%5EGSPTSE', '%5EVIX', '%5EDJI', '%5EFCHI', '%5EN100', 'IMOEX.ME', '%5EAXJO', '%5ENZ50', '%5EN225', '%5ETWII', '%5EBFX', '%5ETA125.TA', '%5EKLSE', '%5EBSESN', '%5ESTOXX50E', '%5EIXIC', '%5EKS11', '%5EJN0U.JO', '000001.SS', '%5ECASE30', '%5EGSPC', '%5EFTSE', '%5EBUK100P', '%5EAORD', '%5ERUT', '%5EGDAXI', '%5EJKSE', '%5EHSI', '%5ENSEI', '%5ENSEBANK', '%5ESP500TR', '%5EDJITR', '%5ERUTTR', '%5EXNDX', '%5ERUITR', '%5ERUATR', '%5ERMCCTR', '%5EAEX', '%5EIRX', '%5EDJT', '%5ENDX', '%5ERUA', '%5ERUI', '%5ERVX', '%5EVXN', '%5EOVX', '%5EGVZ', '%5EVVIX', '%5EVXSLV', '%5EIBEX', '%5ESSMI', 'DX-Y.NYB', '%5EXAX', '%5EMID', '%5ENYA', 'TX60.TS', 'XU100.IS']

#%%
df_result['symbol_sort'] = pd.Categorical(
    df_result['symbol'],
    categories = lst_index_codes,
    ordered = True
)
df_result.sort_values('symbol_sort', inplace=True)

#%%
from datetime import datetime
import pytz

cols = ['created_at'] + lst_index_codes
df_update = pd.DataFrame(columns=cols)
naive_start_time = datetime.now(pytz.utc)
update_vals = [naive_start_time] + df_result['price'].to_list()
df_update.loc[0] = update_vals

#%% Crypto

url = 'https://financialmodelingprep.com/api/v3/quote/{}?apikey={}'.format(",".join(['BTCUSD', 'ETHUSD']), api_key)
result = get_jsonparsed_data(url)
df_result = pd.DataFrame(result)
