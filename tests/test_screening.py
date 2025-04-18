"""
Unit tests for the volscore.screening module.

This module contains unit tests for the stock screening functions in the
volscore.screening module.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from volscore.screening import (
    screen_stocks,
    filter_by_vol_premium,
    filter_by_hist_percentile,
    filter_by_spread,
    filter_by_open_interest,
    combine_filters,
    rank_opportunities,
    get_sector_performance
)


class TestScreening(unittest.TestCase):
    """Test cases for stock screening functions."""
    
    def setUp(self):
        """Set up test data."""
        # Create a sample metrics dataframe
        self.metrics_data = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
            'sector': ['Technology', 'Technology', 'Communication Services', 
                       'Consumer Discretionary', 'Communication Services'],
            'iv': [0.25, 0.22, 0.30, 0.35, 0.28],
            'rv': [0.20, 0.18, 0.20, 0.25, 0.15],
            'vol_premium': [0.05, 0.04, 0.10, 0.10, 0.13],
            'sector_avg_iv': [0.24, 0.24, 0.29, 0.30, 0.29],
            'sector_premium': [0.01, -0.02, 0.01, 0.05, -0.01],
            'hist_percentile': [70, 60, 85, 90, 95],
            'volscore': [0.65, 0.55, 0.80, 0.85, 0.90]
        })
        
        # Create sample option data
        self.options_data = {
            'AAPL': {
                '2023-06-16': {
                    'calls': pd.DataFrame({
                        'strike': [150, 155, 160, 165, 170],
                        'bid': [10.0, 7.0, 5.0, 3.0, 1.5],
                        'ask': [10.5, 7.5, 5.5, 3.5, 2.0],
                        'openInterest': [5000, 4000, 3000, 2000, 1000]
                    }),
                    'puts': pd.DataFrame({
                        'strike': [150, 155, 160, 165, 170],
                        'bid': [2.0, 3.0, 5.0, 7.0, 10.0],
                        'ask': [2.5, 3.5, 5.5, 7.5, 10.5],
                        'openInterest': [3000, 3500, 4000, 2500, 1500]
                    })
                }
            },
            'MSFT': {
                '2023-06-16': {
                    'calls': pd.DataFrame({
                        'strike': [250, 255, 260, 265, 270],
                        'bid': [15.0, 12.0, 9.0, 6.0, 3.0],
                        'ask': [15.5, 12.5, 9.5, 6.5, 3.5],
                        'openInterest': [2000, 1500, 1000, 500, 200]
                    }),
                    'puts': pd.DataFrame({
                        'strike': [250, 255, 260, 265, 270],
                        'bid': [3.0, 5.0, 8.0, 11.0, 15.0],
                        'ask': [3.5, 5.5, 8.5, 11.5, 15.5],
                        'openInterest': [1800, 1600, 1400, 1200, 1000]
                    })
                }
            },
            'GOOGL': {
                '2023-06-16': {
                    'calls': pd.DataFrame({
                        'strike': [100, 105, 110, 115, 120],
                        'bid': [8.0, 5.0, 3.0, 1.5, 0.5],
                        'ask': [8.2, 5.2, 3.2, 1.7, 0.7],
                        'openInterest': [8000, 7000, 6000, 5000, 4000]
                    }),
                    'puts': pd.DataFrame({
                        'strike': [100, 105, 110, 115, 120],
                        'bid': [1.0, 2.0, 4.0, 7.0, 11.0],
                        'ask': [1.2, 2.2, 4.2, 7.2, 11.2],
                        'openInterest': [7000, 6000, 5000, 4000, 3000]
                    })
                }
            },
            'AMZN': {
                '2023-06-16': {
                    'calls': pd.DataFrame({
                        'strike': [120, 125, 130, 135, 140],
                        'bid': [9.0, 6.0, 4.0, 2.0, 1.0],
                        'ask': [10.0, 7.0, 5.0, 3.0, 2.0],
                        'openInterest': [500, 400, 300, 200, 100]
                    }),
                    'puts': pd.DataFrame({
                        'strike': [120, 125, 130, 135, 140],
                        'bid': [2.0, 4.0, 7.0, 10.0, 14.0],
                        'ask': [3.0, 5.0, 8.0, 11.0, 15.0],
                        'openInterest': [400, 350, 300, 250, 200]
                    })
                }
            },
            'META': {
                '2023-06-16': {
                    'calls': pd.DataFrame({
                        'strike': [200, 205, 210, 215, 220],
                        'bid': [12.0, 9.0, 6.0, 4.0, 2.0],
                        'ask': [13.0, 10.0, 7.0, 5.0, 3.0],
                        'openInterest': [3000, 2500, 2000, 1500, 1000]
                    }),
                    'puts': pd.DataFrame({
                        'strike': [200, 205, 210, 215, 220],
                        'bid': [2.0, 4.0, 6.0, 9.0, 12.0],
                        'ask': [3.0, 5.0, 7.0, 10.0, 13.0],
                        'openInterest': [2500, 2000, 1500, 1000, 500]
                    })
                }
            }
        }

    def test_screen_stocks(self):
        """Test the main screening function."""
        # Test with no filters
        results = screen_stocks(self.metrics_data)
        self.assertEqual(len(results), 5)  # All stocks included
        
        # Test with minimum volscore
        results = screen_stocks(self.metrics_data, min_volscore=0.8)
        self.assertEqual(len(results), 3)  # GOOGL, AMZN, META
        
        # Test with custom filters
        filters = [
            {'column': 'vol_premium', 'operator': '>', 'value': 0.05},
            {'column': 'hist_percentile', 'operator': '>=', 'value': 85}
        ]
        results = screen_stocks(self.metrics_data, filters=filters)
        self.assertEqual(len(results), 3)  # GOOGL, AMZN, META
        
        # Test sorting
        results = screen_stocks(self.metrics_data, sort_by='iv', ascending=True)
        self.assertEqual(results.iloc[0]['ticker'], 'MSFT')  # Lowest IV
        
        # Test top_n
        results = screen_stocks(self.metrics_data, sort_by='volscore', top_n=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results.iloc[0]['ticker'], 'META')  # Highest volscore
        self.assertEqual(results.iloc[1]['ticker'], 'AMZN')  # Second highest
        
    def test_filter_by_vol_premium(self):
        """Test filtering by volatility premium."""
        # Test minimum premium
        results = filter_by_vol_premium(self.metrics_data, min_premium=0.1)
        self.assertEqual(len(results), 3)  # GOOGL, AMZN, META
        
        # Test maximum premium
        results = filter_by_vol_premium(self.metrics_data, max_premium=0.05)
        self.assertEqual(len(results), 2)  # AAPL, MSFT
        
        # Test both min and max
        results = filter_by_vol_premium(self.metrics_data, min_premium=0.05, max_premium=0.1)
        self.assertEqual(len(results), 3)  # AAPL, GOOGL, AMZN
        
        # Test percentile
        results = filter_by_vol_premium(self.metrics_data, premium_percentile=80)
        # This should get the top 20% based on vol_premium
        expected_tickers = ['META']  # Top 20% of 5 stocks = 1 stock (META with highest premium)
        self.assertEqual(set(results['ticker'].tolist()), set(expected_tickers))
        
    def test_filter_by_hist_percentile(self):
        """Test filtering by historical IV percentile."""
        # Test minimum percentile
        results = filter_by_hist_percentile(self.metrics_data, min_percentile=80)
        self.assertEqual(len(results), 3)  # GOOGL, AMZN, META
        
        # Test maximum percentile
        results = filter_by_hist_percentile(self.metrics_data, max_percentile=70)
        self.assertEqual(len(results), 2)  # AAPL, MSFT
        
        # Test both min and max
        results = filter_by_hist_percentile(self.metrics_data, min_percentile=70, max_percentile=90)
        self.assertEqual(len(results), 3)  # AAPL, GOOGL, AMZN
        
        # Test invalid percentile values
        results = filter_by_hist_percentile(self.metrics_data, min_percentile=110)
        self.assertEqual(len(results), 5)  # All stocks (invalid min ignored)
        
        results = filter_by_hist_percentile(self.metrics_data, max_percentile=-10)
        self.assertEqual(len(results), 5)  # All stocks (invalid max ignored)
        
    def test_filter_by_spread(self):
        """Test filtering by option spread width."""
        # Test with moderate spread requirement
        passing_tickers = filter_by_spread(self.options_data, max_spread_pct=0.1)
        self.assertEqual(set(passing_tickers), {'GOOGL'})  # GOOGL has narrowest spreads
        
        # Test with looser spread requirement
        passing_tickers = filter_by_spread(self.options_data, max_spread_pct=0.2)
        self.assertEqual(set(passing_tickers), {'AAPL', 'GOOGL', 'META'})
        
        # Test with very tight spread requirement
        passing_tickers = filter_by_spread(self.options_data, max_spread_pct=0.05)
        self.assertEqual(passing_tickers, [])  # No stocks pass
        
    def test_filter_by_open_interest(self):
        """Test filtering by option open interest."""
        # Test with high OI requirement
        passing_tickers = filter_by_open_interest(self.options_data, min_total_oi=10000)
        self.assertEqual(set(passing_tickers), {'GOOGL'})  # GOOGL has highest OI
        
        # Test with moderate OI requirement
        passing_tickers = filter_by_open_interest(self.options_data, min_total_oi=5000)
        self.assertEqual(set(passing_tickers), {'AAPL', 'GOOGL', 'META'})
        
        # Test with ATM OI requirement
        passing_tickers = filter_by_open_interest(self.options_data, min_total_oi=0, min_atm_oi=5000)
        # This would normally use 'underlying_price' to determine ATM options
        # Our mock data doesn't have this, so the function should skip ATM OI check
        # and just rely on total OI
        self.assertEqual(set(passing_tickers), {'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'})
        
    def test_combine_filters(self):
        """Test combining multiple filter results."""
        # Test with two filter groups
        filter_results = {
            'vol_premium': ['GOOGL', 'AMZN', 'META'],  # vol_premium >= 0.1
            'hist_percentile': ['GOOGL', 'AMZN', 'META']  # hist_percentile >= 80
        }
        
        combined = combine_filters(self.metrics_data, filter_results)
        self.assertEqual(set(combined['ticker'].tolist()), {'GOOGL', 'AMZN', 'META'})
        
        # Test with additional filter that restricts further
        filter_results['spread'] = ['AAPL', 'GOOGL', 'META']  # max_spread_pct <= 0.2
        
        combined = combine_filters(self.metrics_data, filter_results)
        self.assertEqual(set(combined['ticker'].tolist()), {'GOOGL', 'META'})
        
        # Test with no passing stocks
        filter_results['oi'] = ['GOOGL']  # min_total_oi >= 10000
        filter_results['another_filter'] = ['META']  # some other filter passing only META
        
        combined = combine_filters(self.metrics_data, filter_results)
        self.assertEqual(len(combined), 0)  # No stocks pass all filters
        
    def test_rank_opportunities(self):
        """Test ranking opportunities."""
        # Test with default weights
        ranked = rank_opportunities(self.metrics_data)
        
        # Check that order is preserved
        self.assertEqual(ranked.iloc[0]['ticker'], 'META')  # Highest volscore
        self.assertEqual(ranked.iloc[1]['ticker'], 'AMZN')
        self.assertEqual(ranked.iloc[2]['ticker'], 'GOOGL')
        self.assertEqual(ranked.iloc[3]['ticker'], 'AAPL')
        self.assertEqual(ranked.iloc[4]['ticker'], 'MSFT')  # Lowest volscore
        
        # Test with custom weights
        weights = {'vol_premium': 0.8, 'hist_percentile': 0.2}
        ranked = rank_opportunities(self.metrics_data, weights)
        
        # With these weights, META should still be first (highest vol_premium and hist_percentile)
        self.assertEqual(ranked.iloc[0]['ticker'], 'META')
        
        # Check for rank_score column
        self.assertTrue('rank_score' in ranked.columns)
        
    def test_get_sector_performance(self):
        """Test sector performance summary."""
        sector_metrics = get_sector_performance(self.metrics_data)
        
        # Should have one row per sector
        self.assertEqual(len(sector_metrics), 3)  # Technology, Communication Services, Consumer Discretionary
        
        # Check calculated values
        tech_row = sector_metrics[sector_metrics['sector'] == 'Technology']
        self.assertEqual(tech_row['stock_count'].iloc[0], 2)  # AAPL, MSFT
        
        # Check average IV for Technology sector
        expected_avg_iv = (0.25 + 0.22) / 2
        self.assertAlmostEqual(tech_row['iv'].iloc[0], expected_avg_iv)
        
        # Check sorting
        # Should be sorted by volscore, highest first
        self.assertEqual(sector_metrics.iloc[0]['sector'], 'Communication Services')  # Highest avg volscore


# Run tests if executed as script
if __name__ == '__main__':
    unittest.main()