"""
Screening module for the VolSSCORE package.

This module provides functions for screening stocks based on volatility metrics,
with customizable filters for vol_premium, hist_percentile, bid-ask spreads, and
open interest to identify potential trading opportunities.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Callable, Tuple, Any

# Set up logging
logger = logging.getLogger(__name__)


def screen_stocks(metrics_data: pd.DataFrame,
                 filters: Optional[List[Dict[str, Any]]] = None,
                 min_volscore: Optional[float] = None,
                 top_n: Optional[int] = None,
                 sort_by: str = 'volscore',
                 ascending: bool = False) -> pd.DataFrame:
    """
    Screen stocks based on VolSSCORE metrics and custom filters.
    
    Args:
        metrics_data: DataFrame containing volatility metrics for multiple stocks.
        filters: List of filter dictionaries, each with keys 'column', 'operator', 'value'.
                 Operators can be '>', '<', '>=', '<=', '==', '!='.
        min_volscore: Minimum VolSSCORE value to include (shorthand for a volscore filter).
        top_n: Return only the top N results after filtering and sorting.
        sort_by: Column to sort results by.
        ascending: Whether to sort in ascending order.
        
    Returns:
        DataFrame with filtered and sorted results.
        
    Example:
        >>> filters = [
        ...     {'column': 'vol_premium', 'operator': '>', 'value': 0.1},
        ...     {'column': 'hist_percentile', 'operator': '>=', 'value': 80}
        ... ]
        >>> results = screen_stocks(metrics_data, filters, min_volscore=0.7)
    """
    try:
        if metrics_data.empty:
            logger.warning("Empty metrics data provided to screening function")
            return pd.DataFrame()
            
        # Create a copy to avoid modifying the original DataFrame
        filtered_data = metrics_data.copy()
        
        # Apply minimum volscore filter if provided
        if min_volscore is not None:
            filtered_data = filtered_data[filtered_data['volscore'] >= min_volscore]
            
        # Apply custom filters if provided
        if filters:
            for filter_dict in filters:
                column = filter_dict['column']
                operator = filter_dict['operator']
                value = filter_dict['value']
                
                if column not in filtered_data.columns:
                    logger.warning(f"Column '{column}' not found in metrics data")
                    continue
                    
                # Apply the filter based on the operator
                if operator == '>':
                    filtered_data = filtered_data[filtered_data[column] > value]
                elif operator == '<':
                    filtered_data = filtered_data[filtered_data[column] < value]
                elif operator == '>=':
                    filtered_data = filtered_data[filtered_data[column] >= value]
                elif operator == '<=':
                    filtered_data = filtered_data[filtered_data[column] <= value]
                elif operator == '==':
                    filtered_data = filtered_data[filtered_data[column] == value]
                elif operator == '!=':
                    filtered_data = filtered_data[filtered_data[column] != value]
                else:
                    logger.warning(f"Unsupported operator: {operator}")
        
        # Sort results
        if sort_by in filtered_data.columns:
            filtered_data = filtered_data.sort_values(by=sort_by, ascending=ascending)
        else:
            logger.warning(f"Sort column '{sort_by}' not found, using default sorting")
            
        # Limit to top N if requested
        if top_n is not None and top_n > 0:
            filtered_data = filtered_data.head(top_n)
            
        return filtered_data
        
    except Exception as e:
        logger.error(f"Error in stock screening: {str(e)}")
        raise


def filter_by_vol_premium(metrics_data: pd.DataFrame,
                         min_premium: Optional[float] = None,
                         max_premium: Optional[float] = None,
                         premium_percentile: Optional[float] = None) -> pd.DataFrame:
    """
    Filter stocks by volatility premium (IV - RV).
    
    Args:
        metrics_data: DataFrame containing volatility metrics.
        min_premium: Minimum volatility premium.
        max_premium: Maximum volatility premium.
        premium_percentile: Alternative to min_premium, filters stocks above a percentile of the premium distribution.
        
    Returns:
        Filtered DataFrame.
    
    Example:
        >>> # Get stocks with vol_premium > 0.1
        >>> high_premium_stocks = filter_by_vol_premium(metrics_data, min_premium=0.1)
        >>> # Get stocks in the top 20% of vol_premium
        >>> top_premium_stocks = filter_by_vol_premium(metrics_data, premium_percentile=80)
    """
    try:
        if metrics_data.empty:
            return pd.DataFrame()
            
        # Create a copy to avoid modifying the original DataFrame
        filtered_data = metrics_data.copy()
        
        if 'vol_premium' not in filtered_data.columns:
            logger.error("vol_premium column not found in metrics data")
            return pd.DataFrame()
            
        # Apply percentile filter if specified
        if premium_percentile is not None:
            if not 0 <= premium_percentile <= 100:
                logger.warning(f"Invalid percentile value: {premium_percentile}, must be between 0 and 100")
                return filtered_data
                
            threshold = np.percentile(filtered_data['vol_premium'].dropna(), premium_percentile)
            filtered_data = filtered_data[filtered_data['vol_premium'] >= threshold]
            return filtered_data
            
        # Apply min and max filters
        if min_premium is not None:
            filtered_data = filtered_data[filtered_data['vol_premium'] >= min_premium]
            
        if max_premium is not None:
            filtered_data = filtered_data[filtered_data['vol_premium'] <= max_premium]
            
        return filtered_data
        
    except Exception as e:
        logger.error(f"Error filtering by volatility premium: {str(e)}")
        raise


def filter_by_hist_percentile(metrics_data: pd.DataFrame,
                             min_percentile: Optional[float] = None,
                             max_percentile: Optional[float] = None) -> pd.DataFrame:
    """
    Filter stocks by historical implied volatility percentile.
    
    Args:
        metrics_data: DataFrame containing volatility metrics.
        min_percentile: Minimum historical IV percentile (0-100).
        max_percentile: Maximum historical IV percentile (0-100).
        
    Returns:
        Filtered DataFrame.
    
    Example:
        >>> # Get stocks with IV in the top 20% of their historical range
        >>> high_iv_stocks = filter_by_hist_percentile(metrics_data, min_percentile=80)
    """
    try:
        if metrics_data.empty:
            return pd.DataFrame()
            
        # Create a copy to avoid modifying the original DataFrame
        filtered_data = metrics_data.copy()
        
        if 'hist_percentile' not in filtered_data.columns:
            logger.error("hist_percentile column not found in metrics data")
            return pd.DataFrame()
            
        # Apply min and max filters
        if min_percentile is not None:
            if not 0 <= min_percentile <= 100:
                logger.warning(f"Invalid min_percentile value: {min_percentile}, must be between 0 and 100")
            else:
                filtered_data = filtered_data[filtered_data['hist_percentile'] >= min_percentile]
            
        if max_percentile is not None:
            if not 0 <= max_percentile <= 100:
                logger.warning(f"Invalid max_percentile value: {max_percentile}, must be between 0 and 100")
            else:
                filtered_data = filtered_data[filtered_data['hist_percentile'] <= max_percentile]
            
        return filtered_data
        
    except Exception as e:
        logger.error(f"Error filtering by historical percentile: {str(e)}")
        raise


def filter_by_spread(options_data: Dict[str, Dict[str, pd.DataFrame]],
                    max_spread_pct: float = 0.1) -> List[str]:
    """
    Filter stocks by option bid-ask spread percentage.
    
    Args:
        options_data: Dictionary mapping tickers to option chain data.
        max_spread_pct: Maximum acceptable bid-ask spread as a percentage of the mid price.
        
    Returns:
        List of tickers that pass the filter.
    
    Example:
        >>> # Get tickers with tight option spreads (< 5% of option price)
        >>> liquid_options = filter_by_spread(options_data, max_spread_pct=0.05)
    """
    try:
        if not options_data:
            return []
            
        passing_tickers = []
        
        for ticker, expirations in options_data.items():
            # Get the first expiration date
            expiration = next(iter(expirations.keys()))
            option_chain = expirations[expiration]
            
            calls = option_chain['calls']
            puts = option_chain['puts']
            
            # Check for required columns
            required_columns = ['ask', 'bid']
            if not all(col in calls.columns for col in required_columns) or \
               not all(col in puts.columns for col in required_columns):
                logger.warning(f"Bid/ask columns missing for {ticker}, skipping spread filter")
                continue
                
            # Calculate spread percentage for calls
            calls['mid_price'] = (calls['ask'] + calls['bid']) / 2
            calls['spread_pct'] = (calls['ask'] - calls['bid']) / calls['mid_price']
            
            # Calculate spread percentage for puts
            puts['mid_price'] = (puts['ask'] + puts['bid']) / 2
            puts['spread_pct'] = (puts['ask'] - puts['bid']) / puts['mid_price']
            
            # Filter out rows with zero or NaN mid prices to avoid division by zero
            calls_spreads = calls[calls['mid_price'] > 0]['spread_pct'].dropna()
            puts_spreads = puts[puts['mid_price'] > 0]['spread_pct'].dropna()
            
            # Check if we have enough data
            if len(calls_spreads) == 0 and len(puts_spreads) == 0:
                logger.warning(f"No valid spread data for {ticker}")
                continue
                
            # Combine spreads and calculate median
            all_spreads = pd.concat([calls_spreads, puts_spreads])
            median_spread = all_spreads.median()
            
            # Check if the median spread is below the threshold
            if median_spread <= max_spread_pct:
                passing_tickers.append(ticker)
                
        return passing_tickers
        
    except Exception as e:
        logger.error(f"Error filtering by spread: {str(e)}")
        raise


def filter_by_open_interest(options_data: Dict[str, Dict[str, pd.DataFrame]],
                          min_total_oi: int = 1000,
                          min_atm_oi: int = 100) -> List[str]:
    """
    Filter stocks by option open interest.
    
    Args:
        options_data: Dictionary mapping tickers to option chain data.
        min_total_oi: Minimum total open interest across all strikes.
        min_atm_oi: Minimum open interest for at-the-money options.
        
    Returns:
        List of tickers that pass the filter.
    
    Example:
        >>> # Get tickers with liquid options (total OI > 5000, ATM OI > 200)
        >>> liquid_options = filter_by_open_interest(options_data, min_total_oi=5000, min_atm_oi=200)
    """
    try:
        if not options_data:
            return []
            
        passing_tickers = []
        
        for ticker, expirations in options_data.items():
            # Get the first expiration date
            expiration = next(iter(expirations.keys()))
            option_chain = expirations[expiration]
            
            calls = option_chain['calls']
            puts = option_chain['puts']
            
            # Check for required columns
            if 'openInterest' not in calls.columns or 'openInterest' not in puts.columns:
                logger.warning(f"Open interest column missing for {ticker}, skipping OI filter")
                continue
                
            # Get total open interest
            total_call_oi = calls['openInterest'].sum()
            total_put_oi = puts['openInterest'].sum()
            total_oi = total_call_oi + total_put_oi
            
            # Check total open interest
            if total_oi < min_total_oi:
                continue
                
            # Find ATM options (need stock price)
            underlying_price = None
            if 'underlying_price' in calls.columns:
                underlying_price = calls['underlying_price'].iloc[0]
            elif 'lastPrice' in calls.columns and 'strike' in calls.columns:
                # Estimate underlying price from ATM call-put parity
                call_values = calls[['strike', 'lastPrice']].copy()
                put_values = puts[['strike', 'lastPrice']].copy()
                
                # Join on strike
                combined = call_values.merge(
                    put_values, 
                    on='strike', 
                    suffixes=('_call', '_put')
                )
                
                # Find the strike where call and put prices are closest
                combined['price_diff'] = abs(combined['lastPrice_call'] - combined['lastPrice_put'])
                min_diff_idx = combined['price_diff'].idxmin()
                
                if not pd.isna(min_diff_idx):
                    atm_strike = combined.loc[min_diff_idx, 'strike']
                    underlying_price = atm_strike
            
            # If we can't determine the underlying price, skip ATM check
            if underlying_price is None:
                logger.warning(f"Cannot determine underlying price for {ticker}, skipping ATM OI check")
                # Pass if total OI check is satisfied
                passing_tickers.append(ticker)
                continue
                
            # Find the nearest ATM strikes
            calls['dist_from_atm'] = abs(calls['strike'] - underlying_price)
            puts['dist_from_atm'] = abs(puts['strike'] - underlying_price)
            
            # Get the ATM call and put indices
            atm_call_idx = calls['dist_from_atm'].idxmin()
            atm_put_idx = puts['dist_from_atm'].idxmin()
            
            # Get the ATM open interest
            atm_call_oi = calls.loc[atm_call_idx, 'openInterest']
            atm_put_oi = puts.loc[atm_put_idx, 'openInterest']
            atm_oi = atm_call_oi + atm_put_oi
            
            # Check ATM open interest
            if atm_oi >= min_atm_oi:
                passing_tickers.append(ticker)
                
        return passing_tickers
        
    except Exception as e:
        logger.error(f"Error filtering by open interest: {str(e)}")
        raise


def combine_filters(metrics_data: pd.DataFrame,
                   filter_results: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Combine the results of multiple filters.
    
    Args:
        metrics_data: DataFrame containing volatility metrics.
        filter_results: Dictionary mapping filter names to lists of passing tickers.
        
    Returns:
        Filtered DataFrame with only tickers that pass all filters.
    
    Example:
        >>> filter_results = {
        ...     'spread': filter_by_spread(options_data),
        ...     'open_interest': filter_by_open_interest(options_data)
        ... }
        >>> final_results = combine_filters(metrics_data, filter_results)
    """
    try:
        if metrics_data.empty or not filter_results:
            return metrics_data
            
        # Get tickers that pass all filters
        all_tickers = set()
        first_filter = True
        
        for filter_name, tickers in filter_results.items():
            ticker_set = set(tickers)
            
            if first_filter:
                all_tickers = ticker_set
                first_filter = False
            else:
                all_tickers &= ticker_set
                
        # Filter the metrics data to include only passing tickers
        if 'ticker' in metrics_data.columns:
            return metrics_data[metrics_data['ticker'].isin(all_tickers)]
        else:
            logger.error("ticker column not found in metrics data")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error combining filters: {str(e)}")
        raise


def rank_opportunities(filtered_data: pd.DataFrame, 
                      weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """
    Rank opportunities based on a weighted combination of metrics.
    
    Args:
        filtered_data: DataFrame with filtered stock metrics.
        weights: Dictionary mapping metrics to weights for ranking.
                 Default: {'volscore': 0.7, 'vol_premium': 0.2, 'hist_percentile': 0.1}
        
    Returns:
        DataFrame with added 'rank_score' column, sorted by rank.
    
    Example:
        >>> weights = {'volscore': 0.5, 'vol_premium': 0.3, 'open_interest': 0.2}
        >>> ranked_results = rank_opportunities(filtered_data, weights)
    """
    try:
        if filtered_data.empty:
            return filtered_data
            
        # Default weights
        if weights is None:
            weights = {
                'volscore': 0.7,
                'vol_premium': 0.2,
                'hist_percentile': 0.1
            }
            
        # Create a copy to avoid modifying the original DataFrame
        ranked_data = filtered_data.copy()
        
        # Normalize each metric to 0-1 scale for consistent weighting
        for metric in weights.keys():
            if metric in ranked_data.columns:
                min_val = ranked_data[metric].min()
                max_val = ranked_data[metric].max()
                
                # Avoid division by zero
                if max_val > min_val:
                    ranked_data[f'{metric}_norm'] = (ranked_data[metric] - min_val) / (max_val - min_val)
                else:
                    ranked_data[f'{metric}_norm'] = 0.5  # Default to middle value if all values are the same
            else:
                logger.warning(f"Metric '{metric}' not found in data, skipping in ranking")
                weights[metric] = 0  # Remove weight for missing metric
                
        # Renormalize weights if needed
        weight_sum = sum(weights.values())
        if weight_sum > 0:
            normalized_weights = {k: v/weight_sum for k, v in weights.items()}
        else:
            logger.warning("Sum of weights is zero, using equal weights")
            # Fall back to equal weights
            valid_metrics = [m for m in weights.keys() if f'{m}_norm' in ranked_data.columns]
            normalized_weights = {m: 1/len(valid_metrics) for m in valid_metrics} if valid_metrics else {}
        
        # Calculate rank score
        ranked_data['rank_score'] = 0
        for metric, weight in normalized_weights.items():
            if f'{metric}_norm' in ranked_data.columns:
                ranked_data['rank_score'] += weight * ranked_data[f'{metric}_norm']
                
        # Sort by rank score
        ranked_data = ranked_data.sort_values('rank_score', ascending=False)
        
        # Clean up temporary normalization columns
        for metric in weights.keys():
            if f'{metric}_norm' in ranked_data.columns:
                ranked_data = ranked_data.drop(f'{metric}_norm', axis=1)
                
        return ranked_data
        
    except Exception as e:
        logger.error(f"Error ranking opportunities: {str(e)}")
        raise


def get_sector_performance(metrics_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate sector-level volatility metrics.
    
    Args:
        metrics_data: DataFrame containing stock-level volatility metrics.
        
    Returns:
        DataFrame with sector-level metrics.
    """
    try:
        if metrics_data.empty or 'sector' not in metrics_data.columns:
            return pd.DataFrame()
            
        # Group by sector and calculate aggregate metrics
        sector_metrics = metrics_data.groupby('sector').agg({
            'iv': 'mean',
            'rv': 'mean',
            'vol_premium': 'mean',
            'volscore': 'mean',
            'ticker': 'count'
        }).reset_index()
        
        # Rename ticker count column
        sector_metrics = sector_metrics.rename(columns={'ticker': 'stock_count'})
        
        # Sort by average volscore
        sector_metrics = sector_metrics.sort_values('volscore', ascending=False)
        
        return sector_metrics
        
    except Exception as e:
        logger.error(f"Error calculating sector performance: {str(e)}")
        raise