"""
Unit tests for the volscore.metrics module.

This module contains unit tests for the volatility metrics calculation
functions in the volscore.metrics module.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytest
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from volscore.metrics import (
    calculate_realized_volatility,
    calculate_historical_volatility_series,
    calculate_implied_volatility,
    calculate_vol_premium,
    calculate_sector_premium,
    calculate_hist_percentile,
    normalize_metric,
    calculate_volscore,
    _black_scholes,
    _estimate_implied_volatility
)


class TestRealizedVolatility(unittest.TestCase):
    """Test cases for realized volatility calculations."""

    def setUp(self):
        """Set up test data."""
        # Create a sample price dataframe
        dates = pd.date_range(start='2023-01-01', periods=60, freq='D')
        # Create a price series with known volatility
        np.random.seed(42)  # for reproducibility
        daily_returns = np.random.normal(loc=0.0005, scale=0.01, size=60)
        prices = 100 * np.exp(np.cumsum(daily_returns))
        
        self.price_data = pd.DataFrame({
            'Date': dates,
            'Close': prices
        }).set_index('Date')

    def test_calculate_realized_volatility(self):
        """Test the realized volatility calculation."""
        # Calculate 30-day realized volatility
        rv = calculate_realized_volatility(self.price_data, window=30, annualize=True)
        
        # Volatility should be a positive float
        self.assertGreater(rv, 0)
        
        # Manual calculation for verification
        log_returns = np.log(self.price_data['Close'] / self.price_data['Close'].shift(1)).dropna()
        recent_returns = log_returns.iloc[-30:]
        expected_rv = recent_returns.std() * np.sqrt(252)  # annualized
        
        # Values should be close
        self.assertAlmostEqual(rv, expected_rv, places=6)
        
    def test_calculate_realized_volatility_not_annualized(self):
        """Test realized volatility calculation without annualization."""
        # Calculate non-annualized volatility
        rv = calculate_realized_volatility(self.price_data, window=30, annualize=False)
        
        # Manual calculation for verification
        log_returns = np.log(self.price_data['Close'] / self.price_data['Close'].shift(1)).dropna()
        recent_returns = log_returns.iloc[-30:]
        expected_rv = recent_returns.std()
        
        # Values should be close
        self.assertAlmostEqual(rv, expected_rv, places=6)
        
    def test_insufficient_data(self):
        """Test behavior with insufficient data."""
        # Create a small dataframe with less than window data points
        small_df = self.price_data.iloc[-10:]
        
        # Should raise ValueError
        with self.assertRaises(ValueError):
            calculate_realized_volatility(small_df, window=30)
            
    def test_historical_volatility_series(self):
        """Test the historical volatility series calculation."""
        # Calculate rolling volatility
        vol_series = calculate_historical_volatility_series(self.price_data, window=10, annualize=True)
        
        # Series should have the right length
        expected_length = len(self.price_data) - 1  # lose one observation for returns calculation
        self.assertEqual(len(vol_series), expected_length)
        
        # The first 9 values should be NaN (need 10 points for the rolling window)
        self.assertTrue(vol_series.iloc[:9].isna().all())
        
        # The rest should have values
        self.assertTrue(vol_series.iloc[9:].notna().all())


class TestImpliedVolatility(unittest.TestCase):
    """Test cases for implied volatility calculations."""
    
    def setUp(self):
        """Set up test data."""
        # Create a mock option chain
        self.option_chain = {
            'calls': pd.DataFrame({
                'strike': [95, 100, 105, 110],
                'lastPrice': [7.5, 5.0, 3.0, 1.5],
                'bid': [7.3, 4.8, 2.8, 1.3],
                'ask': [7.7, 5.2, 3.2, 1.7],
                'impliedVolatility': [0.25, 0.23, 0.22, 0.21]
            }),
            'puts': pd.DataFrame({
                'strike': [95, 100, 105, 110],
                'lastPrice': [2.0, 3.5, 6.0, 9.5],
                'bid': [1.8, 3.3, 5.8, 9.3],
                'ask': [2.2, 3.7, 6.2, 9.7],
                'impliedVolatility': [0.26, 0.24, 0.23, 0.22]
            })
        }
        
        # Underlying price
        self.underlying_price = 100.0
        
        # Mock option chain with dates
        expiration_date = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
        self.dated_option_chain = {
            expiration_date: {
                'calls': self.option_chain['calls'],
                'puts': self.option_chain['puts']
            }
        }
        
    def test_calculate_implied_volatility(self):
        """Test the implied volatility calculation using provided IVs."""
        # Calculate IV
        iv = calculate_implied_volatility(self.option_chain, self.underlying_price)
        
        # Check that IV is between call and put IV
        call_iv = self.option_chain['calls']['impliedVolatility'].iloc[1]  # ATM call
        put_iv = self.option_chain['puts']['impliedVolatility'].iloc[1]  # ATM put
        expected_iv = (call_iv + put_iv) / 2
        
        self.assertAlmostEqual(iv, expected_iv, places=6)
        
    def test_black_scholes(self):
        """Test the Black-Scholes option pricing model."""
        # Parameters
        S = 100.0  # underlying price
        K = 100.0  # strike price
        t = 1.0    # time to expiry (1 year)
        r = 0.05   # risk-free rate
        sigma = 0.2  # volatility
        
        # Calculate call and put prices
        call_price = _black_scholes(S, K, t, r, sigma, 'call')
        put_price = _black_scholes(S, K, t, r, sigma, 'put')
        
        # Expected values (calculated using a reference implementation)
        expected_call = 10.45
        expected_put = 5.57
        
        # Test with a 5% tolerance due to potential implementation differences
        self.assertAlmostEqual(call_price, expected_call, delta=expected_call*0.05)
        self.assertAlmostEqual(put_price, expected_put, delta=expected_put*0.05)
        
    def test_estimate_implied_volatility(self):
        """Test the implied volatility estimation function."""
        # Parameters
        S = 100.0  # underlying price
        K = 100.0  # strike price
        t = 1.0    # time to expiry (1 year)
        r = 0.05   # risk-free rate
        sigma = 0.2  # true volatility
        
        # Calculate call price using Black-Scholes
        call_price = _black_scholes(S, K, t, r, sigma, 'call')
        
        # Estimate IV from the price
        estimated_iv = _estimate_implied_volatility(
            call_price, S, K, t, r, 'call', precision=0.00001, max_iterations=100
        )
        
        # Estimated IV should be close to original sigma
        self.assertAlmostEqual(estimated_iv, sigma, places=3)


class TestVolatilityMetrics(unittest.TestCase):
    """Test cases for volatility premium metrics and scoring."""
    
    def test_calculate_vol_premium(self):
        """Test volatility premium calculation."""
        iv = 0.25  # 25% implied volatility
        rv = 0.20  # 20% realized volatility
        
        premium = calculate_vol_premium(iv, rv)
        expected_premium = 0.05  # 5% premium
        
        self.assertEqual(premium, expected_premium)
        
    def test_calculate_sector_premium(self):
        """Test sector premium calculation."""
        iv = 0.25  # 25% implied volatility
        sector_avg_iv = 0.22  # 22% sector average IV
        
        premium = calculate_sector_premium(iv, sector_avg_iv)
        expected_premium = 0.03  # 3% premium
        
        self.assertEqual(premium, expected_premium)
        
    def test_calculate_hist_percentile(self):
        """Test historical percentile calculation."""
        current_iv = 0.25  # 25% current implied volatility
        historical_iv = pd.Series([0.15, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30])
        
        percentile = calculate_hist_percentile(current_iv, historical_iv)
        expected_percentile = 62.5  # 62.5th percentile (5th out of 8 values)
        
        self.assertEqual(percentile, expected_percentile)
        
    def test_normalize_metric(self):
        """Test metric normalization."""
        # Test with default min/max
        value = 0.1
        normalized = normalize_metric(value)
        self.assertEqual(normalized, 0.5)  # 0.1 / (2 * 0.1)
        
        # Test with custom min/max
        value = 15
        normalized = normalize_metric(value, min_val=10, max_val=20)
        self.assertEqual(normalized, 0.5)  # (15 - 10) / (20 - 10)
        
        # Test clipping at 0 and 1
        self.assertEqual(normalize_metric(5, min_val=0, max_val=10), 0.5)
        self.assertEqual(normalize_metric(15, min_val=0, max_val=10), 1.0)  # clipped to 1
        self.assertEqual(normalize_metric(-5, min_val=0, max_val=10), 0.0)  # clipped to 0
        
    def test_calculate_volscore(self):
        """Test VolSSCORE calculation."""
        vol_premium = 0.05  # 5% volatility premium
        sector_premium = 0.03  # 3% sector premium
        hist_percentile = 75.0  # 75th percentile
        
        # Default weights
        volscore = calculate_volscore(vol_premium, sector_premium, hist_percentile)
        
        # Expected calculation:
        # norm_vol_premium = vol_premium / 0.3 = 0.05 / 0.3 = 0.167 (clipped to range [0,1])
        # norm_sector_premium = (sector_premium - (-0.1)) / (0.1 - (-0.1)) = 0.03 / 0.2 = 0.65
        # norm_hist_percentile = hist_percentile / 100 = 0.75
        # volscore = 0.5*0.167 + 0.3*0.65 + 0.2*0.75 = 0.0835 + 0.195 + 0.15 = 0.4285
        expected_volscore = 0.5 * (0.05 / 0.3) + 0.3 * ((0.03 - (-0.1)) / 0.2) + 0.2 * (75.0 / 100)
        expected_volscore = min(1.0, max(0.0, expected_volscore))  # Ensure in [0,1] range
        
        self.assertAlmostEqual(volscore, expected_volscore, places=6)
        
        # Test with custom weights
        custom_weights = {'vol_premium': 0.6, 'sector_premium': 0.2, 'hist_percentile': 0.2}
        volscore = calculate_volscore(vol_premium, sector_premium, hist_percentile, weights=custom_weights)
        
        expected_volscore = 0.6 * (0.05 / 0.3) + 0.2 * ((0.03 - (-0.1)) / 0.2) + 0.2 * (75.0 / 100)
        expected_volscore = min(1.0, max(0.0, expected_volscore))  # Ensure in [0,1] range
        
        self.assertAlmostEqual(volscore, expected_volscore, places=6)


# Run tests if executed as script
if __name__ == '__main__':
    unittest.main()