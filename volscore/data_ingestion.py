"""
Data ingestion module for the VolSSCORE package.

This module provides functions for fetching stock universe data, historical prices,
and option chains using yfinance API. It includes caching mechanisms to avoid
excessive API calls and proper error handling.
"""

import os
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union, Tuple
import time
import joblib
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

# Constants
DEFAULT_CACHE_DIR = Path("data")
PRICE_CACHE_DIR = DEFAULT_CACHE_DIR / "prices"
OPTIONS_CACHE_DIR = DEFAULT_CACHE_DIR / "options"
CACHE_EXPIRY = 24  # hours


def ensure_cache_dirs() -> None:
    """Ensure that cache directories exist."""
    PRICE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OPTIONS_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def load_sp500(file_path: str = "sp500.csv") -> pd.DataFrame:
    """
    Load S&P 500 universe from CSV file.
    
    Args:
        file_path: Path to the CSV file containing S&P 500 tickers.
        
    Returns:
        DataFrame containing ticker information.
        
    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} tickers from {file_path}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading S&P 500 data: {str(e)}")
        raise


def fetch_stock_universe(sector: Optional[str] = None, 
                        industry: Optional[str] = None,
                        min_market_cap: Optional[float] = None) -> pd.DataFrame:
    """
    Fetch stock universe from S&P 500 with optional filters.
    
    Args:
        sector: Filter by sector name.
        industry: Filter by industry name.
        min_market_cap: Minimum market capitalization in dollars.
        
    Returns:
        DataFrame containing filtered ticker information.
    """
    try:
        df = load_sp500()
        
        # Apply filters if provided
        if sector:
            df = df[df['sector'] == sector]
        if industry:
            df = df[df['industry'] == industry]
        if min_market_cap:
            df = df[df['market_cap'] >= min_market_cap]
            
        logger.info(f"Filtered universe contains {len(df)} stocks")
        return df
    except Exception as e:
        logger.error(f"Error fetching stock universe: {str(e)}")
        raise


def get_cache_path(ticker: str, data_type: str, period: str) -> Path:
    """
    Generate cache file path for the given ticker and data type.
    
    Args:
        ticker: Stock ticker symbol.
        data_type: Type of data ('price' or 'options').
        period: Period for which data is cached ('1y', '6m', etc.).
        
    Returns:
        Path object to the cache file.
    """
    if data_type == 'price':
        return PRICE_CACHE_DIR / f"{ticker}_{period}.joblib"
    elif data_type == 'options':
        return OPTIONS_CACHE_DIR / f"{ticker}_{period}.joblib"
    else:
        raise ValueError(f"Unknown data type: {data_type}")


def is_cache_valid(cache_path: Path) -> bool:
    """
    Check if the cache file is valid (exists and not expired).
    
    Args:
        cache_path: Path to the cache file.
        
    Returns:
        Boolean indicating if cache is valid.
    """
    if not cache_path.exists():
        return False
    
    # Check if cache is expired
    mod_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
    expiry_time = datetime.now() - timedelta(hours=CACHE_EXPIRY)
    
    return mod_time > expiry_time


def fetch_historical_prices(tickers: Union[List[str], str], 
                           period: str = "1y", 
                           interval: str = "1d",
                           use_cache: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Fetch historical price data for a list of tickers using yfinance.
    
    Args:
        tickers: List of ticker symbols or a single ticker symbol.
        period: Time period to fetch data for (e.g., "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max").
        interval: Data interval (e.g., "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo").
        use_cache: Whether to use cached data if available.
        
    Returns:
        Dictionary mapping ticker symbols to DataFrames with historical price data.
        
    Raises:
        ValueError: If invalid parameters are provided.
    """
    ensure_cache_dirs()
    
    # Convert single ticker to list
    if isinstance(tickers, str):
        tickers = [tickers]
    
    results = {}
    failed_tickers = []
    
    for ticker in tickers:
        try:
            cache_path = get_cache_path(ticker, 'price', period)
            
            # Try to load from cache if enabled and valid
            if use_cache and is_cache_valid(cache_path):
                try:
                    data = joblib.load(cache_path)
                    logger.debug(f"Loaded {ticker} price data from cache")
                    results[ticker] = data
                    continue
                except Exception as e:
                    logger.warning(f"Failed to load {ticker} from cache: {str(e)}")
            
            # Fetch data from yfinance
            logger.info(f"Fetching historical data for {ticker}")
            ticker_obj = yf.Ticker(ticker)
            data = ticker_obj.history(period=period, interval=interval)
            
            # Handle empty data
            if data.empty:
                logger.warning(f"No historical data found for {ticker}")
                failed_tickers.append(ticker)
                continue
                
            # Save to cache
            joblib.dump(data, cache_path)
            
            results[ticker] = data
            
            # Avoid rate limiting
            time.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {str(e)}")
            failed_tickers.append(ticker)
    
    if failed_tickers:
        logger.warning(f"Failed to fetch data for {len(failed_tickers)} tickers: {', '.join(failed_tickers)}")
    
    return results


def fetch_option_chains(tickers: Union[List[str], str], 
                       expiration_date: Optional[str] = None,
                       use_cache: bool = True) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Fetch option chain data for a list of tickers using yfinance.
    
    Args:
        tickers: List of ticker symbols or a single ticker symbol.
        expiration_date: Specific expiration date to fetch (format: 'YYYY-MM-DD'). 
                         If None, fetches the nearest monthly expiration.
        use_cache: Whether to use cached data if available.
        
    Returns:
        Nested dictionary mapping ticker symbols to dictionaries of option expiration 
        dates to DataFrames with call and put option data.
        
    Raises:
        ValueError: If invalid parameters are provided.
    """
    ensure_cache_dirs()
    
    # Convert single ticker to list
    if isinstance(tickers, str):
        tickers = [tickers]
    
    results = {}
    failed_tickers = []
    
    for ticker in tickers:
        try:
            cache_key = f"{ticker}_{'nearest' if expiration_date is None else expiration_date}"
            cache_path = get_cache_path(ticker, 'options', cache_key)
            
            # Try to load from cache if enabled and valid
            if use_cache and is_cache_valid(cache_path):
                try:
                    data = joblib.load(cache_path)
                    logger.debug(f"Loaded {ticker} option data from cache")
                    results[ticker] = data
                    continue
                except Exception as e:
                    logger.warning(f"Failed to load {ticker} options from cache: {str(e)}")
            
            # Fetch data from yfinance
            logger.info(f"Fetching option chain for {ticker}")
            ticker_obj = yf.Ticker(ticker)
            
            # Get available expiration dates
            expirations = ticker_obj.options
            
            if not expirations:
                logger.warning(f"No options data available for {ticker}")
                failed_tickers.append(ticker)
                continue
            
            # Determine which expiration to use
            if expiration_date is None:
                # Find the nearest monthly expiration
                today = datetime.now()
                target_expirations = [exp for exp in expirations 
                                     if datetime.strptime(exp, '%Y-%m-%d') > today]
                
                if not target_expirations:
                    logger.warning(f"No future options available for {ticker}")
                    failed_tickers.append(ticker)
                    continue
                    
                expiration_to_use = min(target_expirations, 
                                       key=lambda x: abs(datetime.strptime(x, '%Y-%m-%d') - today))
            else:
                # Use specified expiration if available
                if expiration_date not in expirations:
                    logger.warning(f"Specified expiration {expiration_date} not available for {ticker}")
                    failed_tickers.append(ticker)
                    continue
                expiration_to_use = expiration_date
            
            # Fetch option chain for the determined expiration
            option_chain = ticker_obj.option_chain(expiration_to_use)
            
            # Organize data
            options_data = {
                expiration_to_use: {
                    'calls': option_chain.calls,
                    'puts': option_chain.puts
                }
            }
            
            # Save to cache
            joblib.dump(options_data, cache_path)
            
            results[ticker] = options_data
            
            # Avoid rate limiting
            time.sleep(0.2)
            
        except Exception as e:
            logger.error(f"Error fetching options for {ticker}: {str(e)}")
            failed_tickers.append(ticker)
    
    if failed_tickers:
        logger.warning(f"Failed to fetch options for {len(failed_tickers)} tickers: {', '.join(failed_tickers)}")
    
    return results


def clear_cache(data_type: Optional[str] = None, tickers: Optional[List[str]] = None) -> None:
    """
    Clear cache files.
    
    Args:
        data_type: Type of data to clear ('price', 'options', or None for all).
        tickers: List of tickers to clear cache for (None for all).
    """
    if data_type == 'price' or data_type is None:
        _clear_cache_dir(PRICE_CACHE_DIR, tickers)
    
    if data_type == 'options' or data_type is None:
        _clear_cache_dir(OPTIONS_CACHE_DIR, tickers)


def _clear_cache_dir(directory: Path, tickers: Optional[List[str]] = None) -> None:
    """
    Clear cache files from a specific directory.
    
    Args:
        directory: Path to the cache directory.
        tickers: List of tickers to clear cache for (None for all).
    """
    try:
        if not directory.exists():
            return
            
        for file_path in directory.glob('*.joblib'):
            if tickers is None:
                file_path.unlink()
            else:
                # Extract ticker from filename (format: TICKER_period.joblib)
                file_ticker = file_path.name.split('_')[0]
                if file_ticker in tickers:
                    file_path.unlink()
        
        logger.info(f"Cleared cache in {directory}")
    except Exception as e:
        logger.error(f"Error clearing cache in {directory}: {str(e)}")