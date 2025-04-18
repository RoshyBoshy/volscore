"""
Metrics module for the VolSSCORE package.

This module provides functions for calculating volatility metrics, including
realized volatility (RV), at-the-money implied volatility (IV), and the 
VolSSCORE metric which combines multiple volatility factors to identify
potentially overpriced options.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple
import logging
import math
from scipy.stats import norm, percentileofscore
from datetime import datetime, timedelta

# Set up logging
logger = logging.getLogger(__name__)

# Constants
TRADING_DAYS_PER_YEAR = 252
CALENDAR_DAYS_PER_YEAR = 365


def calculate_realized_volatility(prices: pd.DataFrame, 
                                 window: int = 30,
                                 annualize: bool = True) -> float:
    """
    Calculate realized volatility from historical prices.
    
    Args:
        prices: DataFrame with historical price data (must have 'Close' column).
        window: Number of trading days to calculate volatility over.
        annualize: Whether to annualize the volatility.
        
    Returns:
        Realized volatility as a decimal (not percentage).
        
    Raises:
        ValueError: If input data doesn't have enough points or is invalid.
    """
    try:
        if len(prices) < window + 1:
            raise ValueError(f"Not enough data points. Need at least {window + 1}, got {len(prices)}")
            
        # Calculate daily log returns
        log_returns = np.log(prices['Close'] / prices['Close'].shift(1)).dropna()
        
        # Get the most recent window of returns
        recent_returns = log_returns.iloc[-window:]
        
        # Calculate standard deviation
        std_dev = recent_returns.std()
        
        # Annualize if requested
        if annualize:
            std_dev *= math.sqrt(TRADING_DAYS_PER_YEAR)
            
        return std_dev
    except Exception as e:
        logger.error(f"Error calculating realized volatility: {str(e)}")
        raise


def calculate_historical_volatility_series(prices: pd.DataFrame, 
                                          window: int = 30,
                                          annualize: bool = True) -> pd.Series:
    """
    Calculate historical volatility series as a rolling window.
    
    Args:
        prices: DataFrame with historical price data (must have 'Close' column).
        window: Rolling window size in trading days.
        annualize: Whether to annualize the volatility.
        
    Returns:
        Series of historical volatility values.
    """
    try:
        # Calculate daily log returns
        log_returns = np.log(prices['Close'] / prices['Close'].shift(1)).dropna()
        
        # Calculate rolling standard deviation
        rolling_std = log_returns.rolling(window=window).std()
        
        # Annualize if requested
        if annualize:
            rolling_std *= math.sqrt(TRADING_DAYS_PER_YEAR)
            
        return rolling_std
    except Exception as e:
        logger.error(f"Error calculating historical volatility series: {str(e)}")
        raise


def calculate_implied_volatility(option_chain: Dict[str, pd.DataFrame], 
                                underlying_price: float,
                                risk_free_rate: float = 0.05) -> float:
    """
    Calculate at-the-money implied volatility from option chain.
    
    Args:
        option_chain: Dictionary with 'calls' and 'puts' DataFrames.
        underlying_price: Current price of the underlying stock.
        risk_free_rate: Annual risk-free interest rate as a decimal.
        
    Returns:
        At-the-money implied volatility as a decimal (not percentage).
        
    Raises:
        ValueError: If no suitable options found or calculation fails.
    """
    try:
        calls = option_chain['calls']
        puts = option_chain['puts']
        
        # Find the nearest ATM call and put options
        call_diffs = abs(calls['strike'] - underlying_price)
        put_diffs = abs(puts['strike'] - underlying_price)
        
        nearest_call_idx = call_diffs.idxmin()
        nearest_put_idx = put_diffs.idxmin()
        
        atm_call = calls.loc[nearest_call_idx]
        atm_put = puts.loc[nearest_put_idx]
        
        # Use average of call and put IV if both are available
        iv_values = []
        
        # Check if call IV is available (yfinance provides 'impliedVolatility')
        if 'impliedVolatility' in atm_call:
            iv_values.append(atm_call['impliedVolatility'])
        
        # Check if put IV is available
        if 'impliedVolatility' in atm_put:
            iv_values.append(atm_put['impliedVolatility'])
            
        # If yfinance provides the IV, use it
        if iv_values:
            return np.mean(iv_values)
            
        # Otherwise, estimate IV using the average of put and call prices
        # and the Black-Scholes approximation
        expiration_date_str = next(iter(option_chain.keys()))
        expiration_date = datetime.strptime(expiration_date_str, '%Y-%m-%d')
        days_to_expiry = (expiration_date - datetime.now()).days
        
        if days_to_expiry <= 0:
            raise ValueError("Option already expired")
            
        t = days_to_expiry / CALENDAR_DAYS_PER_YEAR
        
        # Use call for initial estimate
        call_price = atm_call['lastPrice']
        call_strike = atm_call['strike']
        iv_call = _estimate_implied_volatility(
            option_price=call_price,
            underlying_price=underlying_price,
            strike_price=call_strike,
            time_to_expiry=t,
            risk_free_rate=risk_free_rate,
            option_type='call'
        )
        
        # Use put for initial estimate
        put_price = atm_put['lastPrice']
        put_strike = atm_put['strike']
        iv_put = _estimate_implied_volatility(
            option_price=put_price,
            underlying_price=underlying_price,
            strike_price=put_strike,
            time_to_expiry=t,
            risk_free_rate=risk_free_rate,
            option_type='put'
        )
        
        # Use average of call and put IV estimates
        return (iv_call + iv_put) / 2
        
    except Exception as e:
        logger.error(f"Error calculating implied volatility: {str(e)}")
        raise


def _estimate_implied_volatility(option_price: float, 
                               underlying_price: float,
                               strike_price: float,
                               time_to_expiry: float,
                               risk_free_rate: float,
                               option_type: str,
                               precision: float = 0.00001,
                               max_iterations: int = 100) -> float:
    """
    Estimate implied volatility using the Newton-Raphson method.
    
    Args:
        option_price: Observed market price of the option.
        underlying_price: Current price of the underlying asset.
        strike_price: Strike price of the option.
        time_to_expiry: Time to expiry in years.
        risk_free_rate: Annual risk-free rate as a decimal.
        option_type: 'call' or 'put'.
        precision: Desired precision for the estimate.
        max_iterations: Maximum number of iterations.
        
    Returns:
        Estimated implied volatility.
    """
    # Initial volatility estimate
    sigma = 0.3
    
    for i in range(max_iterations):
        price, vega = _black_scholes(
            underlying_price=underlying_price,
            strike_price=strike_price,
            time_to_expiry=time_to_expiry,
            risk_free_rate=risk_free_rate,
            sigma=sigma,
            option_type=option_type,
            return_vega=True
        )
        
        # Calculate the difference
        price_diff = price - option_price
        
        # Check if we have reached desired precision
        if abs(price_diff) < precision:
            return sigma
            
        # Update volatility estimate using Newton-Raphson
        if vega == 0:
            # Avoid division by zero
            sigma = sigma * 1.5
        else:
            sigma = sigma - price_diff / vega
            
        # Ensure sigma is positive and reasonable
        sigma = max(0.001, min(sigma, 5.0))
    
    # If we reach here, we failed to converge
    logger.warning(f"Implied volatility estimation did not converge: {option_type}, S={underlying_price}, K={strike_price}")
    return sigma


def _black_scholes(underlying_price: float, 
                 strike_price: float,
                 time_to_expiry: float,
                 risk_free_rate: float,
                 sigma: float,
                 option_type: str,
                 return_vega: bool = False) -> Union[float, Tuple[float, float]]:
    """
    Calculate option price using the Black-Scholes model.
    
    Args:
        underlying_price: Current price of the underlying asset.
        strike_price: Strike price of the option.
        time_to_expiry: Time to expiry in years.
        risk_free_rate: Annual risk-free rate as a decimal.
        sigma: Volatility of the underlying asset.
        option_type: 'call' or 'put'.
        return_vega: Whether to return the vega as well.
        
    Returns:
        Option price according to Black-Scholes, and vega if return_vega is True.
    """
    if time_to_expiry <= 0:
        if option_type == 'call':
            return max(0, underlying_price - strike_price)
        else:
            return max(0, strike_price - underlying_price)
            
    # Calculate d1 and d2
    d1 = (np.log(underlying_price / strike_price) + 
          (risk_free_rate + 0.5 * sigma**2) * time_to_expiry) / (sigma * np.sqrt(time_to_expiry))
    d2 = d1 - sigma * np.sqrt(time_to_expiry)
    
    # Calculate option price
    if option_type == 'call':
        price = (underlying_price * norm.cdf(d1) - 
                strike_price * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2))
    else:  # put
        price = (strike_price * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) - 
                underlying_price * norm.cdf(-d1))
    
    if return_vega:
        # Calculate vega
        vega = underlying_price * np.sqrt(time_to_expiry) * norm.pdf(d1)
        return price, vega
    else:
        return price


def calculate_vol_premium(iv: float, rv: float) -> float:
    """
    Calculate volatility premium (IV - RV).
    
    Args:
        iv: Implied volatility.
        rv: Realized volatility.
        
    Returns:
        Volatility premium.
    """
    return iv - rv


def calculate_sector_premium(iv: float, sector_avg_iv: float) -> float:
    """
    Calculate sector premium (IV - sector average IV).
    
    Args:
        iv: Stock's implied volatility.
        sector_avg_iv: Average implied volatility for the sector.
        
    Returns:
        Sector premium.
    """
    return iv - sector_avg_iv


def calculate_hist_percentile(current_iv: float, 
                             historical_iv_series: pd.Series) -> float:
    """
    Calculate historical percentile rank of current IV vs past values.
    
    Args:
        current_iv: Current implied volatility.
        historical_iv_series: Series of historical implied volatility values.
        
    Returns:
        Percentile rank (0-100).
        
    Raises:
        ValueError: If historical_iv_series is empty.
    """
    if historical_iv_series.empty:
        raise ValueError("Historical IV series is empty")
        
    return percentileofscore(historical_iv_series.dropna(), current_iv)


def normalize_metric(value: float, 
                    min_val: Optional[float] = None, 
                    max_val: Optional[float] = None) -> float:
    """
    Normalize a metric to a 0-1 range.
    
    Args:
        value: Value to normalize.
        min_val: Minimum value for normalization (default: 0).
        max_val: Maximum value for normalization (default: 2*value).
        
    Returns:
        Normalized value between 0 and 1.
    """
    # Default min and max if not provided
    min_val = 0 if min_val is None else min_val
    max_val = 2 * value if max_val is None else max_val
    
    # Ensure we don't divide by zero
    if max_val == min_val:
        return 0.5
        
    # Normalize to 0-1 range
    normalized = (value - min_val) / (max_val - min_val)
    
    # Clip to 0-1 range
    return max(0, min(normalized, 1))


def calculate_volscore(vol_premium: float, 
                      sector_premium: float, 
                      hist_percentile: float,
                      vol_premium_range: Optional[Tuple[float, float]] = None,
                      sector_premium_range: Optional[Tuple[float, float]] = None,
                      weights: Optional[Dict[str, float]] = None) -> float:
    """
    Calculate the VolSSCORE metric from its components.
    
    Formula: VolSSCORE = 0.5*norm_vol_premium + 0.3*norm_sector_premium + 0.2*norm_hist_percentile
    
    Args:
        vol_premium: Volatility premium (IV - RV).
        sector_premium: Sector premium (IV - sector average IV).
        hist_percentile: Historical percentile (0-100).
        vol_premium_range: (min, max) range for normalizing vol_premium.
        sector_premium_range: (min, max) range for normalizing sector_premium.
        weights: Dictionary of component weights (default: {'vol_premium': 0.5, 'sector_premium': 0.3, 'hist_percentile': 0.2}).
        
    Returns:
        VolSSCORE metric (0-1 scale).
    """
    # Default weights
    if weights is None:
        weights = {
            'vol_premium': 0.5,
            'sector_premium': 0.3,
            'hist_percentile': 0.2
        }
    
    # Normalize components
    if vol_premium_range:
        norm_vol_premium = normalize_metric(vol_premium, vol_premium_range[0], vol_premium_range[1])
    else:
        norm_vol_premium = normalize_metric(vol_premium, 0, 0.3)
        
    if sector_premium_range:
        norm_sector_premium = normalize_metric(sector_premium, sector_premium_range[0], sector_premium_range[1])
    else:
        norm_sector_premium = normalize_metric(sector_premium, -0.1, 0.1)
        
    # Historical percentile is already in 0-100 range, convert to 0-1
    norm_hist_percentile = hist_percentile / 100
    
    # Calculate weighted score
    volscore = (weights['vol_premium'] * norm_vol_premium +
               weights['sector_premium'] * norm_sector_premium +
               weights['hist_percentile'] * norm_hist_percentile)
    
    return volscore


def calculate_stock_metrics(ticker: str, 
                          price_data: pd.DataFrame, 
                          option_data: Dict[str, Dict[str, pd.DataFrame]],
                          sector_data: Dict[str, Dict],
                          historical_iv: Optional[pd.Series] = None) -> Dict:
    """
    Calculate all VolSSCORE metrics for a single stock.
    
    Args:
        ticker: Stock ticker symbol.
        price_data: Historical price data for the stock.
        option_data: Option chain data for the stock.
        sector_data: Dictionary containing sector metrics.
        historical_iv: Series of historical implied volatility (optional).
        
    Returns:
        Dictionary with calculated metrics.
    """
    try:
        # Extract current price
        current_price = price_data['Close'].iloc[-1]
        
        # Calculate realized volatility (30-day)
        rv = calculate_realized_volatility(price_data, window=30)
        
        # Extract option chain for the nearest expiration
        expiration = next(iter(option_data.keys()))
        option_chain = option_data[expiration]
        
        # Calculate implied volatility
        iv = calculate_implied_volatility(option_chain, current_price)
        
        # Calculate volatility premium
        vol_premium = calculate_vol_premium(iv, rv)
        
        # Get sector information
        sector = sector_data[ticker]['sector']
        sector_avg_iv = sector_data[ticker]['sector_avg_iv']
        
        # Calculate sector premium
        sector_premium = calculate_sector_premium(iv, sector_avg_iv)
        
        # Calculate historical percentile if historical IV is provided
        if historical_iv is not None and not historical_iv.empty:
            hist_percentile = calculate_hist_percentile(iv, historical_iv)
        else:
            # Use a default value (median) if historical data not available
            hist_percentile = 50
            
        # Calculate VolSSCORE
        volscore = calculate_volscore(vol_premium, sector_premium, hist_percentile)
        
        # Compile results
        return {
            'ticker': ticker,
            'sector': sector,
            'current_price': current_price,
            'iv': iv,
            'rv': rv,
            'vol_premium': vol_premium,
            'sector_avg_iv': sector_avg_iv,
            'sector_premium': sector_premium,
            'hist_percentile': hist_percentile,
            'volscore': volscore
        }
        
    except Exception as e:
        logger.error(f"Error calculating metrics for {ticker}: {str(e)}")
        raise