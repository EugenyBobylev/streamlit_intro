from typing import Any

from binance import AsyncClient
from binance.enums import HistoricalKlinesType
import pandas as pd
from pandas import DataFrame

from utils import insert_date_column_df


class BinanceAsyncClient:
    """Asynchronous Context Manager for binance AsyncClient"""
    def __init__(self):
        self.client: AsyncClient  = None

    async def __aenter__(self):
        self.client = await AsyncClient.create()
        return self.client

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client is not None:
            await self.client.close_connection()


async def get_client() -> AsyncClient:
    client = await AsyncClient.create()
    return client


async def get_history_klines(client: AsyncClient, symbol: str, interval,
                             start_str: str = Any, end_str: str = Any) -> tuple[str, str, str, list[dict]]:
    klines = await client.get_historical_klines(symbol, interval, start_str, end_str,
                                                klines_type=HistoricalKlinesType.FUTURES)
    return symbol, interval, start_str, klines


async def get_history_klines_2(client: AsyncClient, symbol: str, interval: str,
                               limit: int = 1000) -> (str, str, int, list[dict]):
    klines = await client.get_historical_klines(symbol, interval, limit=limit, klines_type=HistoricalKlinesType.FUTURES)
    return symbol, interval, limit, klines


def create_ohlc_df(klines: list[list]) -> DataFrame:
    columns = ['Open_time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time',
               'Quote_asset_volume', 'Number_of_trades', 'Taker_buy_base_asset_volume',
               'Taker_buy_quote_asset_volume', 'Ignore']

    df = pd.DataFrame(klines, columns=columns, )
    df = df.drop(['Quote_asset_volume', 'Number_of_trades', 'Taker_buy_base_asset_volume',
                  'Taker_buy_quote_asset_volume', 'Ignore'], axis=1)
    df = df.astype({'Open': 'float64', 'High': 'float64', 'Low': 'float64', 'Close': 'float64', 'Volume': 'float64'})
    df = insert_date_column_df(df)
    return df


async def load_df(symbol: str, interval: str, start_str: str, end_str: str) -> DataFrame:
    """
    Load klines from binance, create ohlc_df and prepare df
    :param symbol:
    :param interval:
    :param start_str:
    :param end_str:
    :return:
    """
    client = None
    try:
        client = await get_client()
        data = (symbol, interval, start_str, end_str)
        _, _, _, klines = await get_history_klines(client, *data)
        df = create_ohlc_df(klines)
        df = insert_date_column_df(df)
    finally:
        if client:
            await client.close_connection()
    return df


async def load_df_2(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    """
    Load klines from binance, create ohlc_df and prepare df
    :param symbol:
    :param interval:
    :param limit:
    :return:
    """
    client = None
    try:
        client = await get_client()
        data = (symbol, interval, limit)
        _, _, _, klines = await get_history_klines_2(client, *data)
        df = create_ohlc_df(klines)
        df = insert_date_column_df(df)
    finally:
        if client:
            await client.close_connection()
    return df


async def load_df_3(symbol: str, interval: str, limit: int) -> tuple[str, str, pd.DataFrame]:
    """Load klines from binance, create ohlc_df and prepare df"""
    df = await load_df_2(symbol, interval, limit)
    return symbol, interval, df