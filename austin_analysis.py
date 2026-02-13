#!/usr/bin/env python3
"""
Austin, TX Taxi Dataset — Complete Analysis Pipeline
=====================================================
Author: Sumesh Chakkaravarthi
Course: ALY6110 — Big Data Management & Analytics
Dataset: RideAustin_Weather.csv (~911K rides, June 2016 – Feb 2017)

This script is the main analysis pipeline for my Austin taxi project.
I structured it into 4 phases that run sequentially:
  1. Data Cleaning & Preprocessing — remove bad records, engineer features
  2. Exploratory Data Analysis (EDA) — generate 11 visualizations
  3. Machine Learning Modeling — train Gradient Boosting & Extra Trees
  4. Parquet Integration — merge my Austin data with classmates' cities

I chose these 4 phases because they follow the standard data science
workflow: clean → explore → model → integrate.
"""

# ── Standard library imports ──────────────────────────────────────────────────
import os            # For file path operations (joining paths, making directories)
import json          # For saving results as JSON files (cleaning_summary, eda_results, etc.)
import warnings      # To suppress non-critical warnings during execution

# ── Third-party data science imports ──────────────────────────────────────────
import numpy as np   # Numerical computing — used for math operations, random noise, polyfit
import pandas as pd  # Data manipulation — the core library for loading, filtering, grouping data

# ── Visualization imports ─────────────────────────────────────────────────────
import matplotlib                    # Base plotting library
matplotlib.use('Agg')                # Use non-interactive backend so plots save to file without displaying
import matplotlib.pyplot as plt      # Pyplot interface — I use this for all my charts
import matplotlib.ticker as ticker   # For customizing axis tick formatting
import seaborn as sns                # Statistical visualization — built on top of matplotlib, great for heatmaps

from datetime import datetime        # To track how long the full pipeline takes to run

# Suppress warnings like FutureWarning from pandas/sklearn — keeps console output clean
warnings.filterwarnings('ignore')

# ── File Path Configuration ───────────────────────────────────────────────────
# I use os.path to build paths dynamically so it works on any machine.
# __file__ refers to this script's location, so all paths are relative to it.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Directory where this script lives
DATA_PATH = os.path.join(BASE_DIR, 'RideAustin_Weather.csv')  # Raw dataset (~166 MB)
PARQUET_PATH = os.path.join(BASE_DIR, 'taxi_ml_training 1.parquet')  # Classmates' combined data
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')  # Where all plots and results go
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Create outputs/ folder if it doesn't exist yet

# ── Global Plot Styling (Dark Theme) ─────────────────────────────────────────
# I set a dark GitHub-inspired theme for all my matplotlib charts.
# rcParams is matplotlib's global config — setting it here applies to every plot.
plt.rcParams.update({
    'figure.facecolor': '#0d1117',   # Dark background for the figure canvas
    'axes.facecolor': '#161b22',     # Slightly lighter background for plot area
    'axes.edgecolor': '#30363d',     # Subtle border around each axis
    'axes.labelcolor': '#c9d1d9',    # Light gray for axis labels
    'text.color': '#c9d1d9',         # Light gray for all text elements
    'xtick.color': '#8b949e',        # Muted color for x-axis tick marks
    'ytick.color': '#8b949e',        # Muted color for y-axis tick marks
    'grid.color': '#21262d',         # Very subtle grid lines
    'figure.dpi': 150,               # High resolution for crisp output
    'font.size': 11,                 # Base font size
    'font.family': 'sans-serif',     # Clean sans-serif font family
})

# ── Color Palette ────────────────────────────────────────────────────────────
# I defined a consistent color palette used across all visualizations.
# This ensures visual consistency and a professional look.
COLORS = {
    'primary': '#58a6ff',     # Blue — used for main bars and primary elements
    'secondary': '#f78166',   # Orange — used for secondary data series or highlights
    'accent': '#7ee787',      # Green — used for positive indicators and accents
    'warning': '#d29922',     # Yellow — used for warnings or surge-related visuals
    'purple': '#bc8cff',      # Purple — used for weekend highlighting
    'pink': '#f778ba',        # Pink — used sparingly for variety
    'gradient': ['#58a6ff', '#7ee787', '#f78166', '#d29922', '#bc8cff', '#f778ba'],  # Full palette for multi-category charts
    'heatmap': 'YlOrRd',      # Yellow-Orange-Red colormap for heatmaps
}


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 : DATA CLEANING & PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def load_and_clean_data():
    """
    Load the raw RideAustin_Weather.csv and perform comprehensive data cleaning.

    My cleaning strategy follows a funnel approach — I start with the full dataset
    and progressively remove invalid records. Each step targets a specific data
    quality issue (bad timestamps, impossible durations, outlier distances, etc.).

    After cleaning, I engineer new features like fare_amount (calculated from
    RideAustin's published fare structure) and time-based features for the ML models.

    Returns:
        pd.DataFrame: Cleaned dataframe with engineered features, ready for EDA and ML.
    """
    print("\n" + "="*70)
    print("  PHASE 1 : DATA CLEANING & PREPROCESSING")
    print("="*70)

    # Load the CSV file into a pandas DataFrame
    # low_memory=False tells pandas to read the entire file to infer column types accurately
    # (otherwise pandas might guess types from the first few chunks and get it wrong)
    df = pd.read_csv(DATA_PATH, low_memory=False)
    raw_count = len(df)  # Save the original count so I can report how many rows were removed
    print(f"  📂 Loaded {raw_count:,} raw records ({29} columns)")

    # ── 1a. Parse datetime columns ─────────────────────────────────────────
    # The timestamps come in as strings; I need to convert them to proper datetime
    # objects so I can calculate trip duration and extract time features.
    # errors='coerce' converts unparseable values to NaT (Not a Time) instead of crashing.
    df['started_on'] = pd.to_datetime(df['started_on'], errors='coerce')   # Trip start time
    df['completed_on'] = pd.to_datetime(df['completed_on'], errors='coerce')  # Trip end time
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  # Weather observation date

    # Drop rows where datetime parsing failed (NaT values)
    # These are records with corrupted or missing timestamp data — unusable
    before = len(df)
    df.dropna(subset=['started_on', 'completed_on'], inplace=True)
    print(f"  ⏱  Removed {before - len(df):,} rows with invalid timestamps")

    # ── 1b. Compute trip duration ──────────────────────────────────────────
    # Calculate duration by subtracting start from end time.
    # .dt.total_seconds() converts the timedelta to a numeric value in seconds.
    df['trip_duration_seconds'] = (df['completed_on'] - df['started_on']).dt.total_seconds()

    # Remove negative or zero durations — these are clearly data errors
    # (a trip can't take 0 or negative seconds)
    before = len(df)
    df = df[df['trip_duration_seconds'] > 0]
    print(f"  ⛔ Removed {before - len(df):,} rows with non-positive duration")

    # Remove trips longer than 2 hours (7200 seconds)
    # A normal taxi ride shouldn't exceed 2 hours — these are likely GPS/app errors
    before = len(df)
    df = df[df['trip_duration_seconds'] <= 7200]
    print(f"  ⛔ Removed {before - len(df):,} rows with duration > 2 hours")

    # ── 1c. Clean distance ─────────────────────────────────────────────────
    # The raw 'distance_travelled' column is in meters (SI units from the GPS)
    # First, remove trips with zero or missing distance — these are cancelled or invalid
    before = len(df)
    df = df[df['distance_travelled'] > 0]
    print(f"  📏 Removed {before - len(df):,} rows with zero/missing distance")

    # Remove extreme distances > 100 miles (160,934 meters)
    # Austin metro is ~30 miles across, so 100+ mile trips are almost certainly errors
    before = len(df)
    df = df[df['distance_travelled'] <= 160934]
    print(f"  📏 Removed {before - len(df):,} rows with distance > 100 miles")

    # Convert meters to miles for easier interpretation (1 mile = 1609.34 meters)
    df['trip_distance_miles'] = df['distance_travelled'] / 1609.34

    # ── 1d. Filter Austin coordinates only ─────────────────────────────────
    # Some records have GPS coordinates outside Austin (errors or airport trips).
    # I defined a bounding box around Austin, TX to keep only local rides:
    #   Latitude: 30.0° to 30.6° N (downtown Austin area)
    #   Longitude: -98.1° to -97.4° W
    # The end-location box is slightly wider to allow for trips ending just outside city limits.
    before = len(df)
    austin_mask = (
        (df['start_location_lat'].between(30.0, 30.6)) &
        (df['start_location_long'].between(-98.1, -97.4)) &
        (df['end_location_lat'].between(29.5, 31.0)) &
        (df['end_location_long'].between(-98.5, -97.0))
    )
    df = df[austin_mask]
    print(f"  🗺  Removed {before - len(df):,} rows outside Austin, TX bbox")

    # ── 1e. Extract time features ──────────────────────────────────────────
    # These engineered features help the ML models capture temporal patterns:
    df['hour'] = df['started_on'].dt.hour                     # Hour of day (0–23) for demand patterns
    df['dow'] = df['started_on'].dt.day_name().str[:3]        # Abbreviated day name (Mon, Tue, etc.) for EDA labels
    df['month'] = df['started_on'].dt.month                   # Month number (1–12) for seasonal trends
    df['day_of_week'] = df['started_on'].dt.dayofweek         # Numeric day (0=Monday … 6=Sunday) for ML
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)  # Binary flag: 1=weekend, 0=weekday

    # ── 1f. Engineer fare_amount ───────────────────────────────────────────
    # The raw dataset doesn't include fare amounts directly, so I calculated them
    # using RideAustin's published fare structure (from their website/documentation):
    #   Base fare:   $1.50  (flat fee charged at trip start)
    #   Per mile:    $1.10  (distance-based component)
    #   Per minute:  $0.20  (time-based component, covers traffic/waiting)
    #   Minimum fare: $5.00 (even very short trips cost at least this)
    #   Surge multiplier: applied on top of the calculated fare

    base_fare = 1.50
    per_mile = 1.10
    per_minute = 0.20
    min_fare = 5.00

    # Calculate the base fare: fixed base + distance component + time component
    raw_fare = base_fare + (df['trip_distance_miles'] * per_mile) + \
               ((df['trip_duration_seconds'] / 60) * per_minute)
    # Enforce minimum fare — no ride costs less than $5.00
    raw_fare = raw_fare.clip(lower=min_fare)

    # Apply surge multiplier to the base fare
    # In this dataset: surge_factor=0 means normal pricing (no surge),
    # 1 means 1x surge (same as normal), and values >1 are actual surge multipliers
    surge = df['surge_factor'].copy()
    surge = surge.replace(0, 1.0)  # Treat 0 as 1x (no surge effect)
    df['fare_amount'] = raw_fare * surge

    # Add small random noise (±5%) to simulate real-world fare variation
    # (rounding, tips partial, minor route differences, etc.)
    np.random.seed(42)  # Set seed for reproducibility — same results every run
    noise = np.random.uniform(0.95, 1.05, len(df))  # Random values between 0.95 and 1.05
    df['fare_amount'] = (df['fare_amount'] * noise).round(2)  # Apply noise and round to cents

    # ── 1g. Clean categorical columns ──────────────────────────────────────
    df['requested_car_category'] = df['requested_car_category'].fillna('REGULAR')

    # ── 1h. Clean weather columns ──────────────────────────────────────────
    weather_cols = ['PRCP', 'TMAX', 'TMIN', 'AWND', 'GustSpeed2', 'Fog', 'HeavyFog', 'Thunder']
    for col in weather_cols:
        df[col] = df[col].fillna(0)

    # ── Summary ────────────────────────────────────────────────────────────
    cleaned_count = len(df)
    removed = raw_count - cleaned_count
    print(f"\n  ✅ Cleaning complete: {cleaned_count:,} rows retained ({removed:,} removed, {removed/raw_count*100:.1f}%)")
    print(f"  📅 Date range: {df['started_on'].min().date()} → {df['started_on'].max().date()}")
    print(f"  💰 Fare range: ${df['fare_amount'].min():.2f} – ${df['fare_amount'].max():.2f} (median ${df['fare_amount'].median():.2f})")
    print(f"  📏 Distance range: {df['trip_distance_miles'].min():.2f} – {df['trip_distance_miles'].max():.2f} miles")
    print(f"  ⏱  Duration range: {df['trip_duration_seconds'].min():.0f} – {df['trip_duration_seconds'].max():.0f} seconds")

    # Save cleaning summary
    cleaning_summary = {
        'raw_records': raw_count,
        'cleaned_records': cleaned_count,
        'records_removed': removed,
        'pct_removed': round(removed/raw_count*100, 1),
        'date_range': f"{df['started_on'].min().date()} to {df['started_on'].max().date()}",
        'fare_median': round(df['fare_amount'].median(), 2),
        'fare_mean': round(df['fare_amount'].mean(), 2),
        'distance_median_miles': round(df['trip_distance_miles'].median(), 2),
        'duration_median_seconds': round(df['trip_duration_seconds'].median(), 0),
        'total_columns': len(df.columns),
        'surge_pct': round((df['surge_factor'] > 1).mean() * 100, 1),
    }
    with open(os.path.join(OUTPUT_DIR, 'cleaning_summary.json'), 'w') as f:
        json.dump(cleaning_summary, f, indent=2)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 : EXPLORATORY DATA ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def run_eda(df):
    """Generate all EDA visualizations."""
    print("\n" + "="*70)
    print("  PHASE 2 : EXPLORATORY DATA ANALYSIS")
    print("="*70)

    eda_results = {}

    # ── 2a. Fare Distribution ──────────────────────────────────────────────
    print("  📊 Plotting fare distribution...")
    fig, ax = plt.subplots(figsize=(10, 6))
    fare_clipped = df['fare_amount'].clip(upper=df['fare_amount'].quantile(0.99))
    ax.hist(fare_clipped, bins=60, color=COLORS['primary'], alpha=0.85, edgecolor='none')
    ax.axvline(df['fare_amount'].median(), color=COLORS['secondary'], linestyle='--', linewidth=2,
               label=f"Median: ${df['fare_amount'].median():.2f}")
    ax.axvline(df['fare_amount'].mean(), color=COLORS['accent'], linestyle='--', linewidth=2,
               label=f"Mean: ${df['fare_amount'].mean():.2f}")
    ax.set_xlabel('Fare Amount ($)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Number of Trips', fontsize=13, fontweight='bold')
    ax.set_title('Fare Distribution — Austin, TX', fontsize=16, fontweight='bold', pad=15)
    ax.legend(fontsize=11, facecolor='#161b22', edgecolor='#30363d')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '01_fare_distribution.png'), bbox_inches='tight')
    plt.close()
    eda_results['fare_skewness'] = round(df['fare_amount'].skew(), 2)

    # ── 2b. Trip Distance Distribution ─────────────────────────────────────
    print("  📊 Plotting distance distribution...")
    fig, ax = plt.subplots(figsize=(10, 6))
    dist_clipped = df['trip_distance_miles'].clip(upper=df['trip_distance_miles'].quantile(0.99))
    ax.hist(dist_clipped, bins=60, color=COLORS['accent'], alpha=0.85, edgecolor='none')
    ax.axvline(df['trip_distance_miles'].median(), color=COLORS['secondary'], linestyle='--', linewidth=2,
               label=f"Median: {df['trip_distance_miles'].median():.2f} mi")
    ax.set_xlabel('Trip Distance (miles)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Number of Trips', fontsize=13, fontweight='bold')
    ax.set_title('Trip Distance Distribution — Austin, TX', fontsize=16, fontweight='bold', pad=15)
    ax.legend(fontsize=11, facecolor='#161b22', edgecolor='#30363d')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '02_distance_distribution.png'), bbox_inches='tight')
    plt.close()

    # ── 2c. Hourly Demand Trend ────────────────────────────────────────────
    print("  📊 Plotting hourly demand trend...")
    hourly = df.groupby('hour').agg(
        trip_count=('hour', 'count'),
        avg_fare=('fare_amount', 'mean'),
        avg_surge=('surge_factor', 'mean')
    ).reset_index()

    fig, ax1 = plt.subplots(figsize=(12, 6))
    bars = ax1.bar(hourly['hour'], hourly['trip_count'], color=COLORS['primary'], alpha=0.8,
                   edgecolor='none', width=0.7)
    ax1.set_xlabel('Hour of Day', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Number of Trips', fontsize=13, fontweight='bold', color=COLORS['primary'])
    ax1.set_xticks(range(24))

    # Create a second y-axis (twinx) for the average fare line
    ax2 = ax1.twinx()
    ax2.plot(hourly['hour'], hourly['avg_fare'], color=COLORS['secondary'], linewidth=2.5,
             marker='o', markersize=6, label='Avg Fare ($)')  # Line plot overlaid on bars
    ax2.set_ylabel('Average Fare ($)', fontsize=13, fontweight='bold', color=COLORS['secondary'])

    ax1.set_title('Hourly Ride Demand & Average Fare — Austin, TX',
                  fontsize=16, fontweight='bold', pad=15)
    ax2.legend(loc='upper left', fontsize=11, facecolor='#161b22', edgecolor='#30363d')
    ax1.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '03_hourly_demand.png'), bbox_inches='tight')
    plt.close()

    peak_hour = hourly.loc[hourly['trip_count'].idxmax(), 'hour']
    eda_results['peak_demand_hour'] = int(peak_hour)

    # ── 2d. Day of Week Patterns ───────────────────────────────────────────
    print("  📊 Plotting day-of-week patterns...")
    dow_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    dow_data = df.groupby('dow').agg(
        trip_count=('dow', 'count'),
        avg_fare=('fare_amount', 'mean')
    ).reindex(dow_order).reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_colors = [COLORS['purple'] if d in ['Fri', 'Sat', 'Sun'] else COLORS['primary'] for d in dow_order]
    ax.bar(dow_data['dow'], dow_data['trip_count'], color=bar_colors, alpha=0.85, edgecolor='none')
    ax.set_xlabel('Day of Week', fontsize=13, fontweight='bold')
    ax.set_ylabel('Number of Trips', fontsize=13, fontweight='bold')
    ax.set_title('Ride Demand by Day of Week — Austin, TX\n(Purple = Weekend/Friday)',
                 fontsize=15, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '04_day_of_week.png'), bbox_inches='tight')
    plt.close()

    busiest_day = dow_data.loc[dow_data['trip_count'].idxmax(), 'dow']
    eda_results['busiest_day'] = busiest_day

    # ── 2e. Surge Factor Analysis ──────────────────────────────────────────
    print("  📊 Plotting surge factor analysis...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Surge distribution
    surge_counts = df['surge_factor'].value_counts().sort_index()
    surge_counts_filtered = surge_counts[surge_counts.index > 0]
    ax1.bar(surge_counts_filtered.index.astype(str), surge_counts_filtered.values,
            color=COLORS['warning'], alpha=0.85, edgecolor='none')
    ax1.set_xlabel('Surge Multiplier', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Trips', fontsize=12, fontweight='bold')
    ax1.set_title('Surge Factor Distribution', fontsize=14, fontweight='bold', pad=10)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')

    # Surge impact on fare
    surge_fare = df.groupby(pd.cut(df['surge_factor'], bins=[0, 1, 1.5, 2, 3, 6],
                                     labels=['1x', '1-1.5x', '1.5-2x', '2-3x', '3x+'])) \
                   ['fare_amount'].mean()
    ax2.barh(surge_fare.index.astype(str), surge_fare.values,
             color=COLORS['gradient'][:5], alpha=0.85, edgecolor='none')
    ax2.set_xlabel('Average Fare ($)', fontsize=12, fontweight='bold')
    ax2.set_title('Average Fare by Surge Level', fontsize=14, fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.3, axis='x')

    plt.suptitle('Surge Pricing Analysis — Austin, TX', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '05_surge_analysis.png'), bbox_inches='tight')
    plt.close()

    surge_pct = (df['surge_factor'] > 1).mean() * 100
    eda_results['surge_trip_pct'] = round(surge_pct, 1)

    # ── 2f. Weather Impact Analysis ────────────────────────────────────────
    print("  📊 Plotting weather impact...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Temperature vs Trips
    temp_data = df.groupby('TMAX').agg(trip_count=('TMAX', 'count'),
                                        avg_fare=('fare_amount', 'mean')).reset_index()
    axes[0, 0].scatter(temp_data['TMAX'], temp_data['trip_count'],
                       c=COLORS['secondary'], alpha=0.7, s=40)
    axes[0, 0].set_xlabel('Max Temperature (°F)', fontweight='bold')
    axes[0, 0].set_ylabel('Trip Count', fontweight='bold')
    axes[0, 0].set_title('Temperature vs Trip Volume', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    # Top-right: Rain intensity categories vs average fare
    # pd.cut bins the continuous precipitation values into human-readable categories
    df['rain_category'] = pd.cut(df['PRCP'], bins=[-0.01, 0, 0.1, 0.5, 3],
                                  labels=['No Rain', 'Light', 'Moderate', 'Heavy'])
    rain_fare = df.groupby('rain_category', observed=True)['fare_amount'].mean()  # observed=True avoids warnings
    axes[0, 1].bar(rain_fare.index.astype(str), rain_fare.values,
                   color=[COLORS['primary'], COLORS['accent'], COLORS['warning'], COLORS['secondary']],
                   alpha=0.85, edgecolor='none')
    axes[0, 1].set_xlabel('Rain Intensity', fontweight='bold')
    axes[0, 1].set_ylabel('Average Fare ($)', fontweight='bold')
    axes[0, 1].set_title('Rain Impact on Fares', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Fog impact
    fog_data = df.groupby('Fog').agg(avg_fare=('fare_amount', 'mean'),
                                      trip_count=('Fog', 'count')).reset_index()
    labels = ['No Fog', 'Fog']
    axes[1, 0].bar(labels[:len(fog_data)], fog_data['avg_fare'],
                   color=[COLORS['primary'], COLORS['warning']][:len(fog_data)],
                   alpha=0.85, edgecolor='none')
    axes[1, 0].set_ylabel('Average Fare ($)', fontweight='bold')
    axes[1, 0].set_title('Fog Impact on Fares', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Wind speed vs trips
    axes[1, 1].scatter(df.groupby('AWND')['fare_amount'].mean().index,
                       df.groupby('AWND')['fare_amount'].mean().values,
                       c=COLORS['purple'], alpha=0.7, s=40)
    axes[1, 1].set_xlabel('Avg Wind Speed (mph)', fontweight='bold')
    axes[1, 1].set_ylabel('Average Fare ($)', fontweight='bold')
    axes[1, 1].set_title('Wind Speed vs Avg Fare', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('Weather Impact on Ride-Hailing — Austin, TX',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '06_weather_impact.png'), bbox_inches='tight')
    plt.close()

    # ── 2g. Distance vs Fare (Scatter + Regression) ───────────────────────
    print("  📊 Plotting distance vs fare...")
    sample = df.sample(min(10000, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(10, 7))

    scatter = ax.scatter(sample['trip_distance_miles'], sample['fare_amount'],
                         c=sample['surge_factor'], cmap='YlOrRd', alpha=0.4, s=15,
                         edgecolors='none')
    plt.colorbar(scatter, ax=ax, label='Surge Factor', pad=0.02)

    # Regression line
    z = np.polyfit(sample['trip_distance_miles'], sample['fare_amount'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(0, sample['trip_distance_miles'].quantile(0.99), 100)
    ax.plot(x_line, p(x_line), color=COLORS['accent'], linewidth=2.5, linestyle='--',
            label=f'Linear fit: y = {z[0]:.2f}x + {z[1]:.2f}')

    corr = df['trip_distance_miles'].corr(df['fare_amount'])
    ax.set_xlabel('Trip Distance (miles)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Fare Amount ($)', fontsize=13, fontweight='bold')
    ax.set_title(f'Distance vs Fare — Austin, TX  (r = {corr:.3f})',
                 fontsize=16, fontweight='bold', pad=15)
    ax.legend(fontsize=11, facecolor='#161b22', edgecolor='#30363d')
    ax.set_xlim(0, sample['trip_distance_miles'].quantile(0.99))
    ax.set_ylim(0, sample['fare_amount'].quantile(0.99))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '07_distance_vs_fare.png'), bbox_inches='tight')
    plt.close()

    eda_results['distance_fare_correlation'] = round(corr, 3)

    # ── 2h. Correlation Heatmap ────────────────────────────────────────────
    print("  📊 Plotting correlation heatmap...")
    corr_cols = ['trip_distance_miles', 'trip_duration_seconds', 'fare_amount',
                 'surge_factor', 'hour', 'TMAX', 'TMIN', 'PRCP', 'AWND', 'Fog']
    corr_labels = ['Distance', 'Duration', 'Fare', 'Surge', 'Hour',
                   'Max Temp', 'Min Temp', 'Precip.', 'Wind', 'Fog']
    corr_matrix = df[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdYlBu_r',
                center=0, square=True, linewidths=0.5,
                xticklabels=corr_labels, yticklabels=corr_labels,
                cbar_kws={'shrink': 0.8}, ax=ax,
                annot_kws={'size': 10, 'fontweight': 'bold'})
    ax.set_title('Feature Correlation Matrix — Austin, TX',
                 fontsize=16, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '08_correlation_heatmap.png'), bbox_inches='tight')
    plt.close()

    # ── 2i. Car Category Breakdown ─────────────────────────────────────────
    print("  📊 Plotting car category breakdown...")
    cat_data = df.groupby('requested_car_category').agg(
        count=('requested_car_category', 'count'),
        avg_fare=('fare_amount', 'mean'),
        avg_distance=('trip_distance_miles', 'mean')
    ).sort_values('count', ascending=False).reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Pie chart
    colors_pie = COLORS['gradient'][:len(cat_data)]
    wedges, texts, autotexts = ax1.pie(cat_data['count'], labels=cat_data['requested_car_category'],
                                        autopct='%1.1f%%', colors=colors_pie,
                                        textprops={'color': '#c9d1d9', 'fontsize': 11},
                                        pctdistance=0.75, startangle=90)
    for autotext in autotexts:
        autotext.set_fontweight('bold')
    ax1.set_title('Trip Share by Car Category', fontsize=14, fontweight='bold', pad=10)

    # Average fare by category
    ax2.barh(cat_data['requested_car_category'], cat_data['avg_fare'],
             color=colors_pie, alpha=0.85, edgecolor='none')
    ax2.set_xlabel('Average Fare ($)', fontsize=12, fontweight='bold')
    ax2.set_title('Average Fare by Car Category', fontsize=14, fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.3, axis='x')

    plt.suptitle('Car Category Analysis — Austin, TX', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '09_car_category.png'), bbox_inches='tight')
    plt.close()

    # ── 2j. Monthly Trend ─────────────────────────────────────────────────
    print("  📊 Plotting monthly trend...")
    monthly = df.groupby(df['started_on'].dt.to_period('M')).agg(
        trip_count=('started_on', 'count'),
        avg_fare=('fare_amount', 'mean')
    ).reset_index()
    monthly['started_on'] = monthly['started_on'].astype(str)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.bar(range(len(monthly)), monthly['trip_count'], color=COLORS['primary'],
            alpha=0.8, edgecolor='none')
    ax1.set_xlabel('Month', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Number of Trips', fontsize=13, fontweight='bold', color=COLORS['primary'])
    ax1.set_xticks(range(len(monthly)))
    ax1.set_xticklabels(monthly['started_on'], rotation=45)

    ax2 = ax1.twinx()
    ax2.plot(range(len(monthly)), monthly['avg_fare'], color=COLORS['secondary'],
             linewidth=2.5, marker='o', markersize=8, label='Avg Fare')
    ax2.set_ylabel('Average Fare ($)', fontsize=13, fontweight='bold', color=COLORS['secondary'])

    ax1.set_title('Monthly Trip Volume & Average Fare — Austin, TX',
                  fontsize=16, fontweight='bold', pad=15)
    ax2.legend(loc='upper left', fontsize=11, facecolor='#161b22', edgecolor='#30363d')
    ax1.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '10_monthly_trend.png'), bbox_inches='tight')
    plt.close()

    # ── 2k. Heatmap: Hour x Day of Week ───────────────────────────────────
    # A 2D heatmap reveals demand hotspots across both time dimensions simultaneously.
    # pivot_table creates a matrix with days as rows and hours as columns.
    print("  📊 Plotting hour×day heatmap...")
    pivot = df.pivot_table(values='fare_amount', index='dow', columns='hour',
                           aggfunc='count')  # Count trips per hour-day combination
    pivot = pivot.reindex(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])  # Force day order

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(pivot, cmap='YlOrRd', ax=ax, linewidths=0.3,
                cbar_kws={'label': 'Trip Count'})
    ax.set_xlabel('Hour of Day', fontsize=13, fontweight='bold')
    ax.set_ylabel('Day of Week', fontsize=13, fontweight='bold')
    ax.set_title('Ride Demand Heatmap — Austin, TX (Hour × Day)',
                 fontsize=16, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '11_demand_heatmap.png'), bbox_inches='tight')
    plt.close()

    # Save EDA results
    with open(os.path.join(OUTPUT_DIR, 'eda_results.json'), 'w') as f:
        json.dump(eda_results, f, indent=2)

    print(f"\n  ✅ EDA complete — 11 visualizations saved to {OUTPUT_DIR}")
    print(f"  📈 Key findings:")
    print(f"     • Fare skewness: {eda_results['fare_skewness']} (right-skewed)")
    print(f"     • Peak demand hour: {eda_results['peak_demand_hour']}:00")
    print(f"     • Busiest day: {eda_results['busiest_day']}")
    print(f"     • Distance–Fare correlation: r = {eda_results['distance_fare_correlation']}")
    print(f"     • Trips with surge: {eda_results['surge_trip_pct']}%")

    return eda_results


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 : MACHINE LEARNING MODELS
# ══════════════════════════════════════════════════════════════════════════════

def run_ml_models(df):
    """
    Train two ML models for fare prediction: Gradient Boosting and Extra Trees.

    I chose these two models specifically because my classmates already used
    Linear Regression and Random Forest — I wanted to try different algorithms
    to bring unique contributions to the group project.

    Gradient Boosting builds trees sequentially, where each new tree corrects
    errors from the previous ones. Extra Trees (Extremely Randomized Trees)
    is similar to Random Forest but adds more randomness in how splits are chosen.

    Args:
        df: Cleaned DataFrame with engineered features
    Returns:
        dict: Model performance metrics (R², MAE, RMSE) for both models
    """
    print("\n" + "="*70)
    print("  PHASE 3 : MACHINE LEARNING MODELS")
    print("="*70)

    # Import scikit-learn modules here (only needed in this phase)
    from sklearn.model_selection import train_test_split    # Split data into train/test sets
    from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor  # My two models
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # Evaluation metrics
    from sklearn.preprocessing import LabelEncoder   # Convert categorical text to numbers

    # ── Feature preparation ────────────────────────────────────────────────
    # These 12 features are what the model uses to predict fare_amount.
    # I selected features that intuitively affect pricing: distance, duration,
    # time of day, surge, and weather conditions.
    features = ['trip_distance_miles', 'trip_duration_seconds', 'hour', 'day_of_week',
                'is_weekend', 'surge_factor', 'TMAX', 'TMIN', 'PRCP', 'AWND',
                'Fog', 'Thunder']

    # LabelEncoder converts categorical text (REGULAR, SUV, PREMIUM, etc.) into
    # numeric values (0, 1, 2, etc.) because ML models only work with numbers.
    le = LabelEncoder()
    df['car_category_encoded'] = le.fit_transform(df['requested_car_category'])
    features.append('car_category_encoded')  # Now we have 13 features total

    target = 'fare_amount'  # This is what we're predicting

    X = df[features].copy()  # Feature matrix (independent variables)
    y = df[target].copy()    # Target vector (dependent variable — what we predict)

    # Fill any remaining NaN values with 0 (e.g., missing weather data)
    X = X.fillna(0)

    # Split data: 80% for training, 20% for testing
    # random_state=42 ensures the same split every time (reproducibility)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"  📊 Training set: {len(X_train):,} | Test set: {len(X_test):,}")

    model_results = {}

    # ── Model 1: Gradient Boosting Regressor ───────────────────────────────
    # Gradient Boosting works by building trees one at a time. Each new tree
    # tries to fix the mistakes (residuals) of the previous trees.
    # Key hyperparameters I tuned:
    #   n_estimators=200: build 200 trees (more = better but slower)
    #   max_depth=6: each tree can be up to 6 levels deep (controls complexity)
    #   learning_rate=0.1: how much each tree contributes (lower = more conservative)
    #   subsample=0.8: each tree uses 80% of data (adds randomness, prevents overfitting)
    #   min_samples_leaf=20: each leaf node needs at least 20 samples
    print("  🤖 Training Gradient Boosting Regressor...")
    gb_model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        min_samples_leaf=20,
        random_state=42
    )
    gb_model.fit(X_train, y_train)    # Train the model on training data
    gb_pred = gb_model.predict(X_test) # Generate predictions on test data

    # Calculate performance metrics for Gradient Boosting:
    #   R² (coefficient of determination): 1.0 = perfect, 0 = average, negative = terrible
    #   MAE (Mean Absolute Error): average dollar difference between predicted and actual
    #   RMSE (Root Mean Squared Error): like MAE but penalizes large errors more
    gb_metrics = {
        'model': 'Gradient Boosting Regressor',
        'r2_score': round(r2_score(y_test, gb_pred), 4),
        'mae': round(mean_absolute_error(y_test, gb_pred), 4),
        'rmse': round(np.sqrt(mean_squared_error(y_test, gb_pred)), 4),
    }
    model_results['gradient_boosting'] = gb_metrics
    print(f"     R² = {gb_metrics['r2_score']} | MAE = ${gb_metrics['mae']:.2f} | RMSE = ${gb_metrics['rmse']:.2f}")

    # ── Model 2: Extra Trees Regressor ────────────────────────────────────
    # Extra Trees (Extremely Randomized Trees) is an ensemble method similar to
    # Random Forest but with more randomization — it picks random split thresholds
    # instead of searching for the best splits. This makes it faster but sometimes
    # less accurate than carefully tuned models.
    #   n_jobs=-1: use all CPU cores for parallel training (faster)
    #   max_features='sqrt': each tree considers sqrt(n_features) features per split
    print("  🤖 Training Extra Trees Regressor...")
    et_model = ExtraTreesRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=10,
        max_features='sqrt',
        n_jobs=-1,
        random_state=42
    )
    et_model.fit(X_train, y_train)
    et_pred = et_model.predict(X_test)

    et_metrics = {
        'model': 'Extra Trees Regressor',
        'r2_score': round(r2_score(y_test, et_pred), 4),
        'mae': round(mean_absolute_error(y_test, et_pred), 4),
        'rmse': round(np.sqrt(mean_squared_error(y_test, et_pred)), 4),
    }
    model_results['extra_trees'] = et_metrics
    print(f"     R² = {et_metrics['r2_score']} | MAE = ${et_metrics['mae']:.2f} | RMSE = ${et_metrics['rmse']:.2f}")

    # ── Feature Importance Plot ────────────────────────────────────────────
    print("  📊 Plotting feature importance...")
    feature_names_display = ['Distance (mi)', 'Duration (s)', 'Hour', 'Day of Week',
                             'Weekend', 'Surge Factor', 'Max Temp', 'Min Temp',
                             'Precipitation', 'Wind Speed', 'Fog', 'Thunder', 'Car Type']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # GB feature importance
    gb_imp = pd.Series(gb_model.feature_importances_, index=feature_names_display).sort_values()
    gb_imp.plot(kind='barh', ax=ax1, color=COLORS['primary'], alpha=0.85, edgecolor='none')
    ax1.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
    ax1.set_title('Gradient Boosting', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')

    # ET feature importance
    et_imp = pd.Series(et_model.feature_importances_, index=feature_names_display).sort_values()
    et_imp.plot(kind='barh', ax=ax2, color=COLORS['accent'], alpha=0.85, edgecolor='none')
    ax2.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
    ax2.set_title('Extra Trees', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')

    plt.suptitle('Feature Importance Comparison — Austin, TX',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '12_feature_importance.png'), bbox_inches='tight')
    plt.close()

    # ── Model Comparison Bar Chart ─────────────────────────────────────────
    print("  📊 Plotting model comparison...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics_names = ['R² Score', 'MAE ($)', 'RMSE ($)']
    gb_vals = [gb_metrics['r2_score'], gb_metrics['mae'], gb_metrics['rmse']]
    et_vals = [et_metrics['r2_score'], et_metrics['mae'], et_metrics['rmse']]

    for i, (ax, name, gv, ev) in enumerate(zip(axes, metrics_names, gb_vals, et_vals)):
        x = np.arange(2)
        bars = ax.bar(x, [gv, ev], color=[COLORS['primary'], COLORS['accent']],
                      alpha=0.85, edgecolor='none', width=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(['Gradient\nBoosting', 'Extra\nTrees'], fontsize=11)
        ax.set_title(name, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, [gv, ev]):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{val:.4f}' if i == 0 else f'${val:.2f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=11,
                    color='#c9d1d9')

    plt.suptitle('Model Performance Comparison — Austin, TX',
                 fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '13_model_comparison.png'), bbox_inches='tight')
    plt.close()

    # ── Actual vs Predicted Plot ───────────────────────────────────────────
    print("  📊 Plotting actual vs predicted...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    sample_idx = np.random.choice(len(y_test), min(5000, len(y_test)), replace=False)
    y_test_s = y_test.iloc[sample_idx]

    ax1.scatter(y_test_s, gb_pred[sample_idx], alpha=0.3, s=8,
                color=COLORS['primary'], edgecolors='none')
    ax1.plot([0, y_test_s.max()], [0, y_test_s.max()], 'r--', linewidth=1.5, alpha=0.8)
    ax1.set_xlabel('Actual Fare ($)', fontweight='bold')
    ax1.set_ylabel('Predicted Fare ($)', fontweight='bold')
    ax1.set_title(f'Gradient Boosting (R² = {gb_metrics["r2_score"]})',
                  fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    ax2.scatter(y_test_s, et_pred[sample_idx], alpha=0.3, s=8,
                color=COLORS['accent'], edgecolors='none')
    ax2.plot([0, y_test_s.max()], [0, y_test_s.max()], 'r--', linewidth=1.5, alpha=0.8)
    ax2.set_xlabel('Actual Fare ($)', fontweight='bold')
    ax2.set_ylabel('Predicted Fare ($)', fontweight='bold')
    ax2.set_title(f'Extra Trees (R² = {et_metrics["r2_score"]})',
                  fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Actual vs Predicted Fare — Austin, TX',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '14_actual_vs_predicted.png'), bbox_inches='tight')
    plt.close()

    # ── Residual Analysis ──────────────────────────────────────────────────
    print("  📊 Plotting residual analysis...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    gb_residuals = y_test_s - gb_pred[sample_idx]
    et_residuals = y_test_s - et_pred[sample_idx]

    ax1.scatter(gb_pred[sample_idx], gb_residuals, alpha=0.3, s=8,
                color=COLORS['primary'], edgecolors='none')
    ax1.axhline(y=0, color='r', linestyle='--', linewidth=1.5)
    ax1.set_xlabel('Predicted Fare ($)', fontweight='bold')
    ax1.set_ylabel('Residual ($)', fontweight='bold')
    ax1.set_title('Gradient Boosting Residuals', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    ax2.scatter(et_pred[sample_idx], et_residuals, alpha=0.3, s=8,
                color=COLORS['accent'], edgecolors='none')
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=1.5)
    ax2.set_xlabel('Predicted Fare ($)', fontweight='bold')
    ax2.set_ylabel('Residual ($)', fontweight='bold')
    ax2.set_title('Extra Trees Residuals', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Residual Analysis — Austin, TX', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '15_residual_analysis.png'), bbox_inches='tight')
    plt.close()

    # Save model results
    with open(os.path.join(OUTPUT_DIR, 'model_results.json'), 'w') as f:
        json.dump(model_results, f, indent=2)

    # ── Model Accuracy Summary ─────────────────────────────────────────────
    best = 'Extra Trees' if et_metrics['r2_score'] > gb_metrics['r2_score'] else 'Gradient Boosting'
    gb_accuracy_pct = gb_metrics['r2_score'] * 100
    et_accuracy_pct = et_metrics['r2_score'] * 100

    print(f"\n  ✅ ML modeling complete — 4 visualizations saved")
    print(f"\n  {'─'*58}")
    print(f"  {'MODEL ACCURACY REPORT':^58}")
    print(f"  {'─'*58}")
    print(f"  {'Model':<28} {'R²':>8} {'MAE':>10} {'RMSE':>10}")
    print(f"  {'─'*58}")
    print(f"  {'Gradient Boosting':<28} {gb_metrics['r2_score']:>8.4f} {gb_metrics['mae']:>9.2f}$ {gb_metrics['rmse']:>9.2f}$")
    print(f"  {'Extra Trees':<28} {et_metrics['r2_score']:>8.4f} {et_metrics['mae']:>9.2f}$ {et_metrics['rmse']:>9.2f}$")
    print(f"  {'─'*58}")
    print(f"  Gradient Boosting Accuracy : {gb_accuracy_pct:.2f}%")
    print(f"  Extra Trees Accuracy       : {et_accuracy_pct:.2f}%")
    print(f"  {'─'*58}")
    print(f"  🏆 Best model: {best} ({max(gb_accuracy_pct, et_accuracy_pct):.2f}% accuracy)")

    return model_results


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 : PARQUET INTEGRATION
# ══════════════════════════════════════════════════════════════════════════════

def integrate_parquet(df):
    """
    Map my Austin data to the classmates' parquet schema and merge into one combined file.

    Each teammate cleaned their own city's data and saved it as parquet. Parquet is
    a columnar storage format (from Apache Arrow) that's much faster to read/write
    than CSV and compresses well — ideal for big data workflows.

    I need to map my column names to match the agreed-upon schema so all 5 cities
    can be analyzed together in one DataFrame.

    Args:
        df: Cleaned Austin DataFrame
    Returns:
        pd.DataFrame: Austin data mapped to the shared schema
    """
    print("\n" + "="*70)
    print("  PHASE 4 : PARQUET INTEGRATION")
    print("="*70)

    # Read existing combined parquet
    combined = pd.read_parquet(PARQUET_PATH)
    print(f"  📂 Existing parquet: {len(combined):,} rows, cities: {combined['city'].unique().tolist()}")

    # Map Austin data to matching schema
    # Schema: city, pickup_datetime, dropoff_datetime, trip_distance_miles,
    #         trip_duration_seconds, fare_amount, hour, dow, month
    austin_mapped = pd.DataFrame({
        'city': 'Austin',
        'pickup_datetime': df['started_on'].dt.tz_localize('US/Central', ambiguous='NaT', nonexistent='NaT').dt.tz_convert('UTC'),
        'dropoff_datetime': df['completed_on'].dt.tz_localize('US/Central', ambiguous='NaT', nonexistent='NaT').dt.tz_convert('UTC'),
        'trip_distance_miles': df['trip_distance_miles'],
        'trip_duration_seconds': df['trip_duration_seconds'],
        'fare_amount': df['fare_amount'],
        'hour': df['hour'].astype('int32'),
        'dow': pd.Categorical(df['dow']),
        'month': df['month'].astype('float64'),
    })

    print(f"  📊 Austin mapped: {len(austin_mapped):,} rows")

    # Save Austin-only parquet
    austin_path = os.path.join(OUTPUT_DIR, 'austin_cleaned.parquet')
    austin_mapped.to_parquet(austin_path, index=False)
    print(f"  💾 Saved Austin-only parquet: {austin_path}")

    # Combine
    combined_new = pd.concat([combined, austin_mapped], ignore_index=True)
    combined_path = os.path.join(OUTPUT_DIR, 'taxi_ml_training_combined.parquet')
    combined_new.to_parquet(combined_path, index=False)

    print(f"  💾 Saved combined parquet: {combined_path}")
    print(f"  📊 Combined dataset: {len(combined_new):,} total rows")
    print(f"  🌆 Cities: {combined_new['city'].unique().tolist()}")
    print(f"  📊 Rows per city:")
    for city, count in combined_new.groupby('city').size().items():
        print(f"     • {city}: {count:,}")

    return austin_mapped


# ══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\n" + "★"*70)
    print("  AUSTIN, TX TAXI DATASET — COMPLETE ANALYSIS PIPELINE")
    print("  ALY6110 DBMS Class Project")
    print("  Dataset: RideAustin_Weather.csv")
    print("★"*70)

    start_time = datetime.now()

    # Phase 1: Clean
    df = load_and_clean_data()

    # Phase 2: EDA
    eda_results = run_eda(df)

    # Phase 3: ML
    model_results = run_ml_models(df)

    # Phase 4: Parquet Integration
    austin_mapped = integrate_parquet(df)

    elapsed = (datetime.now() - start_time).total_seconds()
    print("\n" + "★"*70)
    print(f"  ✅ ALL PHASES COMPLETE in {elapsed:.1f} seconds")
    print(f"  📁 Output directory: {OUTPUT_DIR}")
    print(f"  📊 Generated files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
        print(f"     • {f} ({size/1024:.0f} KB)")
    print("★"*70 + "\n")
