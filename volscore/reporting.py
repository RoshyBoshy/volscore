"""
Reporting module for the VolSSCORE package.

This module provides functions for generating reports, visualizations, and
exporting results from the VolSSCORE analysis to various formats including
CSV files, console tables, and interactive charts.
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from typing import List, Dict, Optional, Union, Tuple, Any
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Set up logging
logger = logging.getLogger(__name__)

# Constants
DEFAULT_REPORTS_DIR = Path("reports")


def ensure_reports_dir(reports_dir: Optional[Path] = None) -> Path:
    """
    Ensure that reports directory exists.
    
    Args:
        reports_dir: Directory path for reports (default: "reports").
        
    Returns:
        Path to the reports directory.
    """
    if reports_dir is None:
        reports_dir = DEFAULT_REPORTS_DIR
        
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir


def export_to_csv(data: pd.DataFrame, 
                 filename: str,
                 reports_dir: Optional[Path] = None,
                 include_timestamp: bool = True) -> str:
    """
    Export results to a CSV file.
    
    Args:
        data: DataFrame containing results to export.
        filename: Name of the output file.
        reports_dir: Directory to save the file in.
        include_timestamp: Whether to add a timestamp to the filename.
        
    Returns:
        Path to the saved file.
        
    Raises:
        ValueError: If data is empty or filename is invalid.
    """
    try:
        if data.empty:
            raise ValueError("Cannot export empty data")
            
        # Ensure filename has .csv extension
        if not filename.endswith('.csv'):
            filename += '.csv'
            
        # Add timestamp if requested
        if include_timestamp:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            name_parts = filename.split('.')
            filename = f"{name_parts[0]}_{timestamp}.{name_parts[1]}"
            
        # Ensure reports directory exists
        reports_dir = ensure_reports_dir(reports_dir)
        
        # Create full file path
        file_path = reports_dir / filename
        
        # Export to CSV
        data.to_csv(file_path, index=False)
        logger.info(f"Exported data to {file_path}")
        
        return str(file_path)
        
    except Exception as e:
        logger.error(f"Error exporting to CSV: {str(e)}")
        raise


def print_results_table(data: pd.DataFrame,
                       title: Optional[str] = None,
                       columns: Optional[List[str]] = None,
                       max_rows: int = 25,
                       format_dict: Optional[Dict[str, str]] = None) -> None:
    """
    Print results as a formatted table to the console.
    
    Args:
        data: DataFrame containing results to print.
        title: Optional title for the table.
        columns: List of columns to include (None for all).
        max_rows: Maximum number of rows to display.
        format_dict: Dictionary mapping column names to format strings.
        
    Example:
        >>> format_dict = {
        ...     'volscore': '{:.2f}',
        ...     'iv': '{:.1%}',
        ...     'rv': '{:.1%}'
        ... }
        >>> print_results_table(results, title="Top VolSSCORE Results", format_dict=format_dict)
    """
    try:
        if data.empty:
            print("No data to display")
            return
            
        # Select subset of columns if specified
        if columns is not None:
            valid_columns = [col for col in columns if col in data.columns]
            if not valid_columns:
                logger.warning("None of the specified columns found in data")
                display_data = data
            else:
                display_data = data[valid_columns]
        else:
            display_data = data
            
        # Limit number of rows
        if len(display_data) > max_rows:
            display_data = display_data.head(max_rows)
            rows_message = f"\nShowing top {max_rows} of {len(data)} rows"
        else:
            rows_message = f"\nShowing all {len(data)} rows"
            
        # Apply formatting if provided
        if format_dict:
            formatted_data = display_data.copy()
            for col, fmt in format_dict.items():
                if col in formatted_data.columns:
                    formatted_data[col] = formatted_data[col].apply(
                        lambda x: fmt.format(x) if pd.notnull(x) else ''
                    )
            table_data = formatted_data
        else:
            table_data = display_data
        
        # Print title if provided
        if title:
            print(f"\n{title}")
            
        # Print the table
        print(tabulate(table_data, headers='keys', tablefmt='psql', showindex=False))
        print(rows_message)
        
    except Exception as e:
        logger.error(f"Error printing results table: {str(e)}")
        print(f"Error printing results: {str(e)}")


def generate_volatility_chart(stock_data: pd.DataFrame,
                             price_history: Dict[str, pd.DataFrame],
                             ticker: str,
                             lookback_periods: List[int] = [30, 60, 90],
                             output_file: Optional[str] = None,
                             reports_dir: Optional[Path] = None,
                             show_plot: bool = True,
                             figsize: Tuple[int, int] = (12, 8)) -> Optional[str]:
    """
    Generate a chart showing historical price, realized volatility, and implied volatility.
    
    Args:
        stock_data: DataFrame containing volatility metrics.
        price_history: Dictionary mapping tickers to price history DataFrames.
        ticker: Stock ticker to generate chart for.
        lookback_periods: List of periods to calculate realized volatility for.
        output_file: Filename to save the chart to (optional).
        reports_dir: Directory to save the chart in.
        show_plot: Whether to display the plot interactively.
        figsize: Figure size as (width, height) tuple.
        
    Returns:
        Path to the saved file if output_file is provided, None otherwise.
    """
    try:
        if ticker not in price_history:
            logger.warning(f"No price history found for {ticker}")
            return None
            
        # Get price history for the ticker
        prices = price_history[ticker]
        
        if prices.empty:
            logger.warning(f"Empty price history for {ticker}")
            return None
            
        # Create figure and axes
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        
        # Plot price history on the first axis
        prices['Close'].plot(ax=ax1, label='Close Price', color='blue')
        ax1.set_title(f'{ticker} Price and Volatility', fontsize=14)
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        
        # Calculate and plot realized volatility for different lookback periods
        from volscore.metrics import calculate_historical_volatility_series
        
        for period in lookback_periods:
            rv_series = calculate_historical_volatility_series(prices, window=period)
            rv_series.plot(ax=ax2, label=f'{period}-day RV', alpha=0.7)
            
        # Add current implied volatility as a horizontal line
        if 'iv' in stock_data.columns:
            ticker_data = stock_data[stock_data['ticker'] == ticker]
            if not ticker_data.empty:
                current_iv = ticker_data['iv'].iloc[0]
                ax2.axhline(y=current_iv, color='red', linestyle='--', 
                          label=f'Current IV: {current_iv:.1%}')
                
        ax2.set_ylabel('Volatility', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left')
        
        # Format y-axis as percentage
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
        
        # Adjust layout
        fig.tight_layout()
        
        # Save the figure if output file is specified
        if output_file:
            reports_dir = ensure_reports_dir(reports_dir)
            
            # Ensure filename has extension
            if not output_file.endswith(('.png', '.jpg', '.pdf')):
                output_file += '.png'
                
            # Create full file path
            file_path = reports_dir / output_file
            
            # Save figure
            fig.savefig(file_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved volatility chart to {file_path}")
            
            saved_path = str(file_path)
        else:
            saved_path = None
            
        # Display the plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
            
        return saved_path
        
    except Exception as e:
        logger.error(f"Error generating volatility chart: {str(e)}")
        plt.close()  # Ensure figure is closed in case of error
        raise


def generate_sector_comparison(metrics_data: pd.DataFrame,
                              output_file: Optional[str] = None,
                              reports_dir: Optional[Path] = None,
                              show_plot: bool = True,
                              plot_type: str = 'bar',
                              figsize: Tuple[int, int] = (12, 8)) -> Optional[str]:
    """
    Generate a chart comparing volatility metrics across sectors.
    
    Args:
        metrics_data: DataFrame containing volatility metrics with 'sector' column.
        output_file: Filename to save the chart to (optional).
        reports_dir: Directory to save the chart in.
        show_plot: Whether to display the plot interactively.
        plot_type: Type of plot ('bar' or 'scatter').
        figsize: Figure size as (width, height) tuple.
        
    Returns:
        Path to the saved file if output_file is provided, None otherwise.
    """
    try:
        if metrics_data.empty or 'sector' not in metrics_data.columns:
            logger.warning("No sector data available for comparison")
            return None
            
        # Calculate sector-level aggregates
        sector_metrics = metrics_data.groupby('sector').agg({
            'iv': 'mean',
            'rv': 'mean',
            'vol_premium': 'mean',
            'volscore': 'mean',
            'ticker': 'count'
        }).reset_index()
        
        # Rename ticker count column
        sector_metrics = sector_metrics.rename(columns={'ticker': 'stock_count'})
        
        # Create figure
        plt.figure(figsize=figsize)
        
        if plot_type == 'bar':
            # Create bar chart
            sector_metrics = sector_metrics.sort_values('volscore', ascending=False)
            
            # Create stacked bar chart
            ax = sector_metrics.plot(
                x='sector',
                y=['iv', 'rv'],
                kind='bar',
                stacked=False,
                alpha=0.7,
                color=['blue', 'green']
            )
            
            # Add volscore as a line on secondary axis
            ax2 = ax.twinx()
            sector_metrics.plot(
                x='sector',
                y='volscore',
                kind='line',
                marker='o',
                color='red',
                ax=ax2
            )
            
            # Set labels and title
            ax.set_xlabel('Sector', fontsize=12)
            ax.set_ylabel('Volatility', fontsize=12)
            ax2.set_ylabel('VolSSCORE', fontsize=12)
            plt.title('Sector Volatility Comparison', fontsize=14)
            
            # Format y-axis as percentage for volatility
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
            
            # Add legend
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines + lines2, ['Implied Volatility', 'Realized Volatility', 'VolSSCORE'], 
                    loc='upper right')
            
            # Remove redundant legend from second axis
            ax2.get_legend().remove()
            
            # Rotate x-axis labels for readability
            plt.xticks(rotation=45, ha='right')
            
        elif plot_type == 'scatter':
            # Create scatter plot of IV vs RV colored by sector
            fig = px.scatter(
                metrics_data,
                x='rv',
                y='iv',
                color='sector',
                hover_name='ticker',
                hover_data=['volscore', 'vol_premium'],
                title='Implied vs Realized Volatility by Sector',
                labels={
                    'rv': 'Realized Volatility',
                    'iv': 'Implied Volatility',
                    'sector': 'Sector'
                },
                opacity=0.7,
                size='volscore',
                size_max=15
            )
            
            # Add diagonal line (IV = RV)
            min_val = min(metrics_data['rv'].min(), metrics_data['iv'].min())
            max_val = max(metrics_data['rv'].max(), metrics_data['iv'].max())
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    line=dict(color='black', dash='dash'),
                    name='IV = RV'
                )
            )
            
            # Format axes as percentages
            fig.update_layout(
                xaxis_tickformat='.0%',
                yaxis_tickformat='.0%'
            )
            
            # Display the plot
            if show_plot:
                fig.show()
                
            # Save the plot if output file is specified
            if output_file:
                reports_dir = ensure_reports_dir(reports_dir)
                
                # Ensure filename has extension
                if not output_file.endswith(('.html', '.png', '.jpg', '.pdf')):
                    output_file += '.html'
                    
                # Create full file path
                file_path = reports_dir / output_file
                
                # Save figure
                fig.write_html(str(file_path))
                logger.info(f"Saved sector comparison to {file_path}")
                
                return str(file_path)
                
            return None
            
        # Adjust layout
        plt.tight_layout()
        
        # Save the figure if output file is specified
        if output_file:
            reports_dir = ensure_reports_dir(reports_dir)
            
            # Ensure filename has extension
            if not output_file.endswith(('.png', '.jpg', '.pdf')):
                output_file += '.png'
                
            # Create full file path
            file_path = reports_dir / output_file
            
            # Save figure
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved sector comparison to {file_path}")
            
            saved_path = str(file_path)
        else:
            saved_path = None
            
        # Display the plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close()
            
        return saved_path
        
    except Exception as e:
        logger.error(f"Error generating sector comparison: {str(e)}")
        plt.close()  # Ensure figure is closed in case of error
        raise


def generate_vol_premium_histogram(metrics_data: pd.DataFrame,
                                  output_file: Optional[str] = None,
                                  reports_dir: Optional[Path] = None,
                                  show_plot: bool = True,
                                  figsize: Tuple[int, int] = (10, 6)) -> Optional[str]:
    """
    Generate a histogram of volatility premium distribution.
    
    Args:
        metrics_data: DataFrame containing volatility metrics.
        output_file: Filename to save the chart to (optional).
        reports_dir: Directory to save the chart in.
        show_plot: Whether to display the plot interactively.
        figsize: Figure size as (width, height) tuple.
        
    Returns:
        Path to the saved file if output_file is provided, None otherwise.
    """
    try:
        if metrics_data.empty or 'vol_premium' not in metrics_data.columns:
            logger.warning("No volatility premium data available")
            return None
            
        # Create figure
        plt.figure(figsize=figsize)
        
        # Create histogram with kernel density estimate
        sns.histplot(metrics_data['vol_premium'], kde=True, bins=30)
        
        # Add vertical line at mean
        mean_premium = metrics_data['vol_premium'].mean()
        plt.axvline(mean_premium, color='red', linestyle='--', 
                  label=f'Mean: {mean_premium:.1%}')
        
        # Add vertical lines at percentiles
        for percentile in [25, 75]:
            p_val = np.percentile(metrics_data['vol_premium'], percentile)
            plt.axvline(p_val, color='green', linestyle=':', 
                      label=f'{percentile}th Percentile: {p_val:.1%}')
            
        # Set labels and title
        plt.xlabel('Volatility Premium (IV - RV)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Distribution of Volatility Premium', fontsize=14)
        
        # Format x-axis as percentage
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
        
        # Add legend
        plt.legend()
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the figure if output file is specified
        if output_file:
            reports_dir = ensure_reports_dir(reports_dir)
            
            # Ensure filename has extension
            if not output_file.endswith(('.png', '.jpg', '.pdf')):
                output_file += '.png'
                
            # Create full file path
            file_path = reports_dir / output_file
            
            # Save figure
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved vol premium histogram to {file_path}")
            
            saved_path = str(file_path)
        else:
            saved_path = None
            
        # Display the plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close()
            
        return saved_path
        
    except Exception as e:
        logger.error(f"Error generating vol premium histogram: {str(e)}")
        plt.close()  # Ensure figure is closed in case of error
        raise


def generate_top_opportunities_chart(metrics_data: pd.DataFrame,
                                    top_n: int = 10,
                                    output_file: Optional[str] = None,
                                    reports_dir: Optional[Path] = None,
                                    show_plot: bool = True,
                                    figsize: Tuple[int, int] = (12, 8)) -> Optional[str]:
    """
    Generate a chart showing the top opportunities by VolSSCORE.
    
    Args:
        metrics_data: DataFrame containing volatility metrics.
        top_n: Number of top opportunities to display.
        output_file: Filename to save the chart to (optional).
        reports_dir: Directory to save the chart in.
        show_plot: Whether to display the plot interactively.
        figsize: Figure size as (width, height) tuple.
        
    Returns:
        Path to the saved file if output_file is provided, None otherwise.
    """
    try:
        if metrics_data.empty or 'volscore' not in metrics_data.columns:
            logger.warning("No VolSSCORE data available")
            return None
            
        # Sort and get top N opportunities
        top_data = metrics_data.sort_values('volscore', ascending=False).head(top_n)
        
        if top_data.empty:
            logger.warning("No data available after filtering")
            return None
            
        # Create figure
        plt.figure(figsize=figsize)
        
        # Create horizontal bar chart
        bars = plt.barh(top_data['ticker'], top_data['volscore'], color='skyblue')
        
        # Add value labels to the bars
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{width:.2f}', va='center')
            
        # Set labels and title
        plt.xlabel('VolSSCORE', fontsize=12)
        plt.ylabel('Ticker', fontsize=12)
        plt.title(f'Top {top_n} Opportunities by VolSSCORE', fontsize=14)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the figure if output file is specified
        if output_file:
            reports_dir = ensure_reports_dir(reports_dir)
            
            # Ensure filename has extension
            if not output_file.endswith(('.png', '.jpg', '.pdf')):
                output_file += '.png'
                
            # Create full file path
            file_path = reports_dir / output_file
            
            # Save figure
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved top opportunities chart to {file_path}")
            
            saved_path = str(file_path)
        else:
            saved_path = None
            
        # Display the plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close()
            
        return saved_path
        
    except Exception as e:
        logger.error(f"Error generating top opportunities chart: {str(e)}")
        plt.close()  # Ensure figure is closed in case of error
        raise


def generate_interactive_dashboard(metrics_data: pd.DataFrame,
                                  output_file: str = "dashboard.html",
                                  reports_dir: Optional[Path] = None) -> str:
    """
    Generate an interactive HTML dashboard with Plotly.
    
    Args:
        metrics_data: DataFrame containing volatility metrics.
        output_file: Filename to save the dashboard to.
        reports_dir: Directory to save the dashboard in.
        
    Returns:
        Path to the saved dashboard HTML file.
    """
    try:
        if metrics_data.empty:
            logger.warning("No data available for dashboard")
            return ""
            
        # Ensure reports directory exists
        reports_dir = ensure_reports_dir(reports_dir)
        
        # Ensure filename has .html extension
        if not output_file.endswith('.html'):
            output_file += '.html'
            
        # Create full file path
        file_path = reports_dir / output_file
        
        # Create a subplot figure
        fig = go.Figure()
        
        # Add scatter plot of IV vs RV
        fig.add_trace(
            go.Scatter(
                x=metrics_data['rv'],
                y=metrics_data['iv'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=metrics_data['volscore'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title='VolSSCORE')
                ),
                text=metrics_data['ticker'],
                hovertemplate='<b>%{text}</b><br>' +
                             'IV: %{y:.1%}<br>' +
                             'RV: %{x:.1%}<br>' +
                             'VolSSCORE: %{marker.color:.2f}',
                name='Stocks'
            )
        )
        
        # Add diagonal line (IV = RV)
        min_val = min(metrics_data['rv'].min(), metrics_data['iv'].min())
        max_val = max(metrics_data['rv'].max(), metrics_data['iv'].max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='black', dash='dash'),
                name='IV = RV'
            )
        )
        
        # Update layout
        fig.update_layout(
            title='Implied vs Realized Volatility',
            xaxis_title='Realized Volatility (30-day)',
            yaxis_title='Implied Volatility (ATM)',
            xaxis_tickformat='.0%',
            yaxis_tickformat='.0%',
            hovermode='closest',
            width=900,
            height=600
        )
        
        # Write to HTML file
        fig.write_html(
            str(file_path),
            full_html=True,
            include_plotlyjs='cdn'
        )
        
        logger.info(f"Generated interactive dashboard at {file_path}")
        
        return str(file_path)
        
    except Exception as e:
        logger.error(f"Error generating interactive dashboard: {str(e)}")
        raise


def generate_complete_report(metrics_data: pd.DataFrame,
                            price_history: Dict[str, pd.DataFrame],
                            report_name: str = "volscore_report",
                            reports_dir: Optional[Path] = None,
                            top_n: int = 10) -> List[str]:
    """
    Generate a complete set of reports and visualizations.
    
    Args:
        metrics_data: DataFrame containing volatility metrics.
        price_history: Dictionary mapping tickers to price history DataFrames.
        report_name: Base name for report files.
        reports_dir: Directory to save reports in.
        top_n: Number of top opportunities to include in detailed reports.
        
    Returns:
        List of paths to generated report files.
    """
    try:
        if metrics_data.empty:
            logger.warning("No data available for report generation")
            return []
            
        # Ensure reports directory exists
        reports_dir = ensure_reports_dir(reports_dir)
        
        # Add timestamp to report name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_base = f"{report_name}_{timestamp}"
        
        generated_files = []
        
        # 1. Export full results to CSV
        csv_path = export_to_csv(
            data=metrics_data,
            filename=f"{report_base}_full.csv",
            reports_dir=reports_dir,
            include_timestamp=False
        )
        generated_files.append(csv_path)
        
        # 2. Generate sector comparison chart
        sector_chart_path = generate_sector_comparison(
            metrics_data=metrics_data,
            output_file=f"{report_base}_sectors.png",
            reports_dir=reports_dir,
            show_plot=False
        )
        if sector_chart_path:
            generated_files.append(sector_chart_path)
            
        # 3. Generate volatility premium histogram
        hist_path = generate_vol_premium_histogram(
            metrics_data=metrics_data,
            output_file=f"{report_base}_premium_dist.png",
            reports_dir=reports_dir,
            show_plot=False
        )
        if hist_path:
            generated_files.append(hist_path)
            
        # 4. Generate top opportunities chart
        top_chart_path = generate_top_opportunities_chart(
            metrics_data=metrics_data,
            top_n=top_n,
            output_file=f"{report_base}_top_{top_n}.png",
            reports_dir=reports_dir,
            show_plot=False
        )
        if top_chart_path:
            generated_files.append(top_chart_path)
            
        # 5. Generate interactive dashboard
        dashboard_path = generate_interactive_dashboard(
            metrics_data=metrics_data,
            output_file=f"{report_base}_dashboard.html",
            reports_dir=reports_dir
        )
        if dashboard_path:
            generated_files.append(dashboard_path)
            
        # 6. Generate individual charts for top stocks
        top_stocks = metrics_data.sort_values('volscore', ascending=False).head(top_n)
        
        for _, row in top_stocks.iterrows():
            ticker = row['ticker']
            try:
                chart_path = generate_volatility_chart(
                    stock_data=metrics_data,
                    price_history=price_history,
                    ticker=ticker,
                    output_file=f"{report_base}_{ticker}.png",
                    reports_dir=reports_dir,
                    show_plot=False
                )
                if chart_path:
                    generated_files.append(chart_path)
            except Exception as e:
                logger.warning(f"Error generating chart for {ticker}: {str(e)}")
                
        # 7. Export top stocks to CSV
        top_csv_path = export_to_csv(
            data=top_stocks,
            filename=f"{report_base}_top_{top_n}.csv",
            reports_dir=reports_dir,
            include_timestamp=False
        )
        generated_files.append(top_csv_path)
        
        logger.info(f"Generated {len(generated_files)} report files")
        
        return generated_files
        
    except Exception as e:
        logger.error(f"Error generating complete report: {str(e)}")
        raise