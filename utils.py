import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path

import pandas as pd
from pandas import DataFrame

from config import SingletonMeta


@dataclass
class StockInterval(metaclass=SingletonMeta):
    """
    StockInterval - работа с временными интервалами на бирже и данными
    """
    one_minute: float = 60.0          # 1m
    three_minutes: float = 180.0      # 3m
    five_minutes: float = 300.0       # 5m
    ten_minutes: float = 600.0        # 10m
    fifteen_minutes: float = 900.0    # 15m
    thirty_minutes: float = 1800.0    # 30m
    one_hour: float = 3600.0          # 1h
    two_hours: float = 7200.0         # 2h
    four_hours: float = 14400.0       # 4h
    eight_hours: float = 28800.0      # 8h
    twelve_hours: float = 43200.0     # 12h
    one_day: float = 86400.0          # 1d
    base_interval = '1m'
    all_intervals = ('3m', '5m', '10m', '15m', '30m', '1h', '2h', '4h', '8h', '12h', '1d')
    _intervals: dict[float, str] = field(default_factory=dict)
    _times: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        self._intervals.update(
            {
                60.0: '1m',
                180.0: '3m',
                300.0: '5m',
                600.0: '10m',
                900.0: '15m',
                1800.0: '30m',
                3600.0: '1h',
                7200.0: '2h',
                14400.0: '4h',
                28800.0: '8h',
                43200.0: '12h',
                86400.0: '1d',
            })
        self._times.update(
            {
                '1m': 60.0,
                '3m': 180.0,
                '5m': 300.0,
                '10m': 600.0,
                '15m': 900.0,
                '30m': 1800.0,
                '1h': 3600.0,
                '2h': 7200.0,
                '4h': 14400.0,
                '8h': 28800.0,
                '12h': 43200.0,
                '1d': 86400.0,
            })

    def get_interval(self, seconds: float) -> str:
        return self._intervals.get(seconds, '')

    def get_time(self, interval: str) -> float | None:
        return self._times.get(interval, None)

    def is_3m(self, seconds: float | pd.Series) -> bool | pd.Series:
        result = seconds % self.three_minutes
        return True if result == 0.0 else False

    def is_5m(self, seconds: float) -> bool:
        result = seconds % self.five_minutes
        return True if result == 0 else False

    def is_10m(self, seconds: float) -> bool:
        result = seconds % self.ten_minutes
        return True if result == 0 else False

    def is_15m(self, seconds: float) -> bool:
        result = seconds % self.fifteen_minutes
        return True if result == 0 else False

    def is_30m(self, seconds: float) -> bool:
        result = seconds % self.thirty_minutes
        return True if result == 0 else False

    def is_1h(self, seconds: float) -> bool:
        result = seconds % self.one_hour
        return True if result == 0 else False

    def is_2h(self, seconds: float) -> bool:
        result = seconds % self.two_hours
        return True if result == 0 else False

    def is_4h(self, seconds: float) -> bool:
        result = seconds % self.four_hours
        return True if result == 0 else False

    def is_8h(self, seconds: float) -> bool:
        result = seconds % self.eight_hours
        return True if result == 0 else False

    def is_12h(self, seconds: float) -> bool:
        result = seconds % self.twelve_hours
        return True if result == 0 else False

    def is_1d(self, seconds: float) -> bool:
        result = seconds % self.one_day
        return True if result == 0 else False

    def calc_all_intervals(self, seconds: float) -> (bool, bool, bool, bool, bool, bool, bool, bool, bool, bool, bool):
        idx = StockInterval.all_intervals
        if self.is_1d(seconds):
            return pd.Series([True, True, True, True, True, True, True, True, True, True, True], index=idx)
        if self.is_12h(seconds):
            return pd.Series([True, True, True, True, True, True, True, True, True, True, False], index=idx)
        if self.is_8h(seconds):
            return pd.Series([True, True, True, True, True, True, True, True, True, False, False], index=idx)
        if self.is_4h(seconds):
            return pd.Series([True, True, True, True, True, True, True, True, False, False, False], index=idx)
        if self.is_2h(seconds):
            return pd.Series([True, True, True, True, True, True, True, False, False, False, False], index=idx)
        if self.is_1h(seconds):
            return pd.Series([True, True, True, True, True, True, False, False, False, False, False], index=idx)
        if self.is_30m(seconds):
            return pd.Series([True, True, True, True, True, False, False, False, False, False, False], index=idx)
        if self.is_15m(seconds):
            return pd.Series([True, True, False, True, False, False, False, False, False, False, False], index=idx)
        if self.is_10m(seconds):
            return pd.Series([False, True, True, False, False, False, False, False, False, False, False], index=idx)
        if self.is_5m(seconds):
            return pd.Series([False, True, False, False, False, False, False, False, False, False, False], index=idx)
        if self.is_3m(seconds):
            return pd.Series([True, False, False, False, False, False, False, False, False, False, False], index=idx)
        return pd.Series([False, False, False, False, False, False, False, False, False, False, False], index=idx)


class DataFrameConverter(metaclass=SingletonMeta):
    """Конвертация данных 1m в старшие тайм-фреймы: 3m, 5m, 10m, 15m, 30m, 1h, 2h, 4h, 8h, 12h, 1d"""

    @staticmethod
    def _record_from_slice(df_slice: DataFrame, interval: str) -> tuple:
        """Рассчитать OHLCV из набора данных 1m"""
        close_time = int(df_slice.iloc[0]['Open_time'] + (StockInterval().get_time(interval) * 1000) - 1)
        record = (
            df_slice.iloc[0]['Date'],
            df_slice.iloc[0]['Open_time'],
            df_slice.iloc[0]['Open'],
            df_slice['High'].max(),
            df_slice['Low'].min(),
            df_slice.iloc[len(df_slice) - 1]['Close'],
            df_slice['Volume'].sum(),
            close_time,
        )
        return record

    @classmethod
    def extract_df_by_interval(cls, df_1m: DataFrame, interval: str, only_last: bool = False) -> DataFrame | None:
        """
        Извлечь данные по указанному интервалу из набора 1 минутных данных
        :param df_1m: исходный ohlcv дата фрейм с интервалом 1 мин
        :param interval: строка формата '1m', '3m', '5m', '10m', '15m', '30m', '1h', '2h', '4h', '8h', '12h', '1d'
        :param only_last: признак для расчета только последнего интервала, по умолчанию False
        :return:
        """
        columns = ['Date', 'Open_time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time']
        if interval == '1m':
            _df = df_1m[-1:][columns]
            return _df

        intervals = StockInterval.all_intervals
        if interval not in intervals:
            return None

        indexes: list[int] = list(df_1m[df_1m[interval]].index)
        records = []
        if not only_last:
            # обработка всех интервалов кроме последнего
            for idx1, idx2 in zip(indexes, indexes[1:]):
                df_slice = df_1m.iloc[idx1:idx2]
                record = cls._record_from_slice(df_slice, interval)
                records.append(record)

        # последний кусок
        idx1 = indexes[-1]
        df_slice = df_1m.iloc[idx1:]
        record = cls._record_from_slice(df_slice, interval)
        records.append(record)

        _df = pd.DataFrame(records, columns=columns, )
        return _df

    @classmethod
    def to_df_with_intervals(cls, df_1m) -> pd.DataFrame:
        """Добавить столбцы для расчета набора данных производных тайм-фреймов"""
        intervals = list(StockInterval.all_intervals)
        tf = StockInterval()
        df_1m[intervals] = df_1m.apply(lambda row: tf.calc_all_intervals(row.Open_time / 1000), axis=1)
        return df_1m

    @classmethod
    def convert_to(cls, df_1m: DataFrame, interval: str) -> DataFrame | None:
        """
        Собрать DataFrame заданного интервала из одно минутного DataFrame
        :param df_1m: одно минутный DataFrame
        :param interval: 3m 5m 10m 15m 30m 1h 2h 4h 8h 12h 1d
        :return:
        """
        intervals = list(StockInterval.all_intervals)
        if interval not in intervals and interval != '1m':
            return None

        df_1m = cls.to_df_with_intervals(df_1m)
        _df = cls.extract_df_by_interval(df_1m, interval)
        return _df


def measure_time(func):
    """
    Измерить время работы функции
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        elapsed_time = end - start
        print(f'{func.__name__} took {elapsed_time} sec.')
        return result

    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start = time.time()
        result = await func(*args, **kwargs)
        end = time.time()
        elapsed_time = end - start
        print(f'{func.__name__} took {elapsed_time} sec.')
        return result

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return wrapper


def csv2df(filename) -> DataFrame | None:
    path = Path(filename)
    if path.exists():
        df = pd.read_csv(filename)
        return df
    return None


def insert_date_column_df(df: DataFrame) -> DataFrame:
    """
    insert Date column, set index Open_time column
    """
    df['Date'] = pd.to_datetime(df['Open_time'], unit='ms')
    # df.set_index('Open_time', inplace=True)

    # shift column 'date' to first position
    first_column = df.pop('Date')
    df.insert(0, 'Date', first_column)

    return df


def ts2datetime(ts: int) -> str:
    dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')
    return dt


def datetime2str(date_time: datetime, trim_seconds: bool = False) -> str:
    """Вернуть строковое представление времени, trim_seconds - признак обнуления секунд"""
    as_str = date_time.strftime('%Y-%m-%d %H:%M:00') if trim_seconds else date_time.strftime('%Y-%m-%d %H:%M:%S')
    return as_str


def create_filename_csv(data, count) -> str:
    symbol, interval, start_str, end_str = data
    start = start_str.replace('-', '').replace(' ', '_').replace(':', '')
    file_name = f'{symbol.lower()}_{interval}_{start}_{count}.csv'
    return file_name


def create_filename_csv_2(data, count) -> str:
    symbol, interval, limit = data
    file_name = f'{symbol.lower()}_{interval}_{count}.csv'
    return file_name


def df2csv(df: DataFrame, filename_csv: str, index=True, mode: str = 'w'):
    df.to_csv(filename_csv, index=index, mode=mode)


def pd_set_display_options(max_columns=None, max_rows=None, max_width=None, min_rows=12):
    """
    Установка максимального количества строк и колонок для печати данных в dataframe
    """
    pd.set_option('display.max_rows', max_rows)
    pd.set_option('display.max_columns', max_columns)
    pd.set_option('display.width', max_width)
    pd.set_option('display.min_rows', min_rows)