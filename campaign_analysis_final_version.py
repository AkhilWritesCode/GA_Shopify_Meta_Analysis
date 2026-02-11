import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io
import duckdb
import tempfile
import os

# Page configuration
st.set_page_config(
    page_title="Campaign Analysis - Multi-Week Campaign",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.input-form {
    background-color: #f8f9fa;
    padding: 20px;
    border-radius: 10px;
    margin: 20px 0;
    border: 2px solid #007bff;
}
.report-section {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 10px;
    margin: 20px 0;
    border: 1px solid #dee2e6;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.campaign-week {
    background-color: #f0f8ff;
    padding: 10px;
    border-radius: 5px;
    margin: 5px 0;
    border: 1px solid #b0d4f1;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_convert_data(uploaded_file, file_type="ga"):
    """Load data and convert to parquet for faster processing"""
    try:
        # Load data
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Preprocess based on file type
        if file_type == "ga":
            df = preprocess_ga_data(df)
        elif file_type == "shopify":
            df = preprocess_shopify_data(df)
        elif file_type == "meta":
            df = preprocess_meta_data(df)
        
        return df
    except Exception as e:
        return None, str(e)

def preprocess_ga_data(df):
    """Preprocess GA data"""
    df = df.copy()
    
    # Parse date column
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Remove rows with invalid dates
    df = df.dropna(subset=['Date'])
    
    # Ensure numeric columns are numeric
    numeric_columns = ['Sessions', 'Total users', 'New users', 'Items viewed', 
                      'Add to carts', 'Total purchasers', 'Engaged sessions']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Clean up text columns
    text_columns = ['Session source']
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    
    return df

def preprocess_shopify_data(df):
    """Preprocess Shopify data"""
    df = df.copy()
    
    # Parse date column
    df['Day'] = pd.to_datetime(df['Day'], errors='coerce')
    
    # Remove rows with invalid dates
    df = df.dropna(subset=['Day'])
    
    # Ensure numeric columns are numeric
    numeric_columns = ['Net sales', 'Net items sold', 'Orders', 'Average order value', 
                      'Discounts', 'Gross margin', 'Customers', 'New customers']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    return df

def preprocess_meta_data(df):
    """Preprocess Meta ad data"""
    df = df.copy()
    
    # Parse date column
    df['Day'] = pd.to_datetime(df['Day'], errors='coerce')
    
    # Remove rows with invalid dates
    df = df.dropna(subset=['Day'])
    
    # Ensure numeric columns are numeric
    # Handle both possible column name variations
    numeric_columns = []
    if 'Impressions' in df.columns:
        numeric_columns.append('Impressions')
    elif 'Impression' in df.columns:
        numeric_columns.append('Impression')
    
    if 'Amount spent (USD)' in df.columns:
        numeric_columns.append('Amount spent (USD)')
    elif 'Ad Spend' in df.columns:
        numeric_columns.append('Ad Spend')
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Clean up Region column
    if 'Region' in df.columns:
        df['Region'] = df['Region'].astype(str).str.strip()
    
    return df

def calculate_weeks_in_period(start_date, end_date):
    """Calculate number of weeks in a period, rounded to nearest whole number"""
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    days = (end - start).days + 1
    weeks = days / 7
    # Round to nearest whole number of weeks (minimum 1)
    return max(1, round(weeks))

def calculate_percentage_change(base_value, campaign_value):
    """Calculate percentage change between base and campaign values"""
    if base_value == 0:
        return "N/A" if campaign_value == 0 else "∞"
    
    change = ((campaign_value - base_value) / base_value) * 100
    return f"{change:+.1f}%"

def create_analysis_with_duckdb(ga_data, shopify_data, meta_data, regions, 
                               base_week_start, base_week_end,
                               campaign_weeks, control_regions, google_sources, 
                               base_week_method, campaign_display_method, campaign_calculation_method, 
                               region_column, shopify_region_column, meta_region_column):
    """Create analysis using DuckDB for faster processing with multiple campaign weeks"""
    
    # Initialize DuckDB connection
    conn = duckdb.connect()
    
    try:
        # Register dataframes with DuckDB
        conn.register('ga_data', ga_data)
        conn.register('shopify_data', shopify_data)
        if meta_data is not None and not meta_data.empty:
            conn.register('meta_data', meta_data)
        
        # Calculate weeks for averaging (always rounded to nearest whole number)
        base_week_weeks = calculate_weeks_in_period(base_week_start, base_week_end)
        
        # Base weeks are ALWAYS averaged (divided by number of weeks)
        base_divisor = base_week_weeks  # Always divide base week by its weeks
        
        # Create Google sources filter
        google_sources_str = "', '".join(google_sources)
        google_filter = f'"Session source" IN (\'{google_sources_str}\')' if google_sources else "1=0"
        
        results = []
        
        # Process target regions
        target_regions = [r for r in regions if r not in control_regions]
        
        for region in target_regions:
            # GA Base Week query
            ga_base_query = f"""
            SELECT 
                SUM(Sessions) as total_sessions,
                SUM(CASE WHEN {google_filter} THEN Sessions ELSE 0 END) as google_sessions
            FROM ga_data 
            WHERE "{region_column}" = '{region}' 
            AND Date >= '{base_week_start}' 
            AND Date <= '{base_week_end}'
            """
            
            # Shopify Base Week query
            shopify_base_query = f"""
            SELECT SUM("Net sales") as net_sales
            FROM shopify_data 
            WHERE "{shopify_region_column}" = '{region}' 
            AND Day >= '{base_week_start}' 
            AND Day <= '{base_week_end}'
            """
            
            # Meta Base Week query (if Meta data is available)
            meta_base_query = None
            if meta_data is not None and not meta_data.empty and meta_region_column:
                meta_base_query = f"""
                SELECT 
                    SUM(Impressions) as impressions,
                    SUM("Amount spent (USD)") as ad_spend
                FROM meta_data 
                WHERE "{meta_region_column}" = '{region}' 
                AND Day >= '{base_week_start}' 
                AND Day <= '{base_week_end}'
                """
            
            # Execute base week queries
            ga_base_result = conn.execute(ga_base_query).fetchone()
            shopify_base_result = conn.execute(shopify_base_query).fetchone()
            meta_base_result = conn.execute(meta_base_query).fetchone() if meta_base_query else (0, 0)
            
            # Calculate base week metrics (always averaged)
            sessions_total_base = (ga_base_result[0] or 0) / base_divisor
            sessions_google_base = (ga_base_result[1] or 0) / base_divisor
            net_sales_base = (shopify_base_result[0] or 0) / base_divisor
            impressions_base = (meta_base_result[0] or 0) / base_divisor
            ad_spend_base = (meta_base_result[1] or 0) / base_divisor
            
            # Initialize result row
            result_row = {
                'Region': region,
                'Sessions_Total_Base': sessions_total_base,
                'Sessions_Google_Base': sessions_google_base,
                'Net_Sales_Base': net_sales_base,
                'Impressions_Base': impressions_base,
                'Ad_Spend_Base': ad_spend_base
            }
            
            # Process campaign weeks
            campaign_sessions_total = []
            campaign_sessions_google = []
            campaign_net_sales = []
            campaign_impressions = []
            campaign_ad_spend = []
            
            for i, week in enumerate(campaign_weeks):
                week_start = week['start']
                week_end = week['end']
                week_label = week['label']
                
                # Calculate divisor for this campaign week
                campaign_week_weeks = calculate_weeks_in_period(week_start, week_end)
                campaign_divisor = campaign_week_weeks if campaign_calculation_method == "Average (÷weeks)" else 1
                
                # GA Campaign query for this week
                ga_campaign_query = f"""
                SELECT 
                    SUM(Sessions) as total_sessions,
                    SUM(CASE WHEN {google_filter} THEN Sessions ELSE 0 END) as google_sessions
                FROM ga_data 
                WHERE "{region_column}" = '{region}' 
                AND Date >= '{week_start}' 
                AND Date <= '{week_end}'
                """
                
                # Shopify Campaign query for this week
                shopify_campaign_query = f"""
                SELECT SUM("Net sales") as net_sales
                FROM shopify_data 
                WHERE "{shopify_region_column}" = '{region}' 
                AND Day >= '{week_start}' 
                AND Day <= '{week_end}'
                """
                
                # Meta Campaign query for this week (if Meta data is available)
                meta_campaign_query = None
                if meta_data is not None and not meta_data.empty and meta_region_column:
                    meta_campaign_query = f"""
                    SELECT 
                        SUM(Impressions) as impressions,
                        SUM("Amount spent (USD)") as ad_spend
                    FROM meta_data 
                    WHERE "{meta_region_column}" = '{region}' 
                    AND Day >= '{week_start}' 
                    AND Day <= '{week_end}'
                    """
                
                # Execute campaign queries
                ga_campaign_result = conn.execute(ga_campaign_query).fetchone()
                shopify_campaign_result = conn.execute(shopify_campaign_query).fetchone()
                meta_campaign_result = conn.execute(meta_campaign_query).fetchone() if meta_campaign_query else (0, 0)
                
                # Calculate campaign metrics
                sessions_total_campaign = (ga_campaign_result[0] or 0) / campaign_divisor
                sessions_google_campaign = (ga_campaign_result[1] or 0) / campaign_divisor
                net_sales_campaign = (shopify_campaign_result[0] or 0) / campaign_divisor
                impressions_campaign = (meta_campaign_result[0] or 0) / campaign_divisor
                ad_spend_campaign = (meta_campaign_result[1] or 0) / campaign_divisor
                
                # Store individual week data
                campaign_sessions_total.append(sessions_total_campaign)
                campaign_sessions_google.append(sessions_google_campaign)
                campaign_net_sales.append(net_sales_campaign)
                campaign_impressions.append(impressions_campaign)
                campaign_ad_spend.append(ad_spend_campaign)
                
                if campaign_display_method == "Separate Columns":
                    # Add individual week columns
                    result_row[f'Sessions_Total_Campaign_Week_{i+1}'] = sessions_total_campaign
                    result_row[f'Sessions_Google_Campaign_Week_{i+1}'] = sessions_google_campaign
                    result_row[f'Net_Sales_Campaign_Week_{i+1}'] = net_sales_campaign
                    result_row[f'Impressions_Campaign_Week_{i+1}'] = impressions_campaign
                    result_row[f'Ad_Spend_Campaign_Week_{i+1}'] = ad_spend_campaign
                    
                    # Calculate percentage changes for individual weeks
                    result_row[f'Sessions_Total_Change_Week_{i+1}'] = calculate_percentage_change(sessions_total_base, sessions_total_campaign)
                    result_row[f'Sessions_Google_Change_Week_{i+1}'] = calculate_percentage_change(sessions_google_base, sessions_google_campaign)
                    result_row[f'Net_Sales_Change_Week_{i+1}'] = calculate_percentage_change(net_sales_base, net_sales_campaign)
                    result_row[f'Impressions_Change_Week_{i+1}'] = calculate_percentage_change(impressions_base, impressions_campaign)
                    result_row[f'Ad_Spend_Change_Week_{i+1}'] = calculate_percentage_change(ad_spend_base, ad_spend_campaign)
            
            # If combined display, calculate combined metrics
            if campaign_display_method == "Combined Column":
                if campaign_calculation_method == "Average (÷weeks)":
                    combined_sessions_total = sum(campaign_sessions_total) / len(campaign_sessions_total)
                    combined_sessions_google = sum(campaign_sessions_google) / len(campaign_sessions_google)
                    combined_net_sales = sum(campaign_net_sales) / len(campaign_net_sales)
                    combined_impressions = sum(campaign_impressions) / len(campaign_impressions)
                    combined_ad_spend = sum(campaign_ad_spend) / len(campaign_ad_spend)
                else:  # Sum
                    combined_sessions_total = sum(campaign_sessions_total)
                    combined_sessions_google = sum(campaign_sessions_google)
                    combined_net_sales = sum(campaign_net_sales)
                    combined_impressions = sum(campaign_impressions)
                    combined_ad_spend = sum(campaign_ad_spend)
                
                result_row['Sessions_Total_Campaign_Combined'] = combined_sessions_total
                result_row['Sessions_Google_Campaign_Combined'] = combined_sessions_google
                result_row['Net_Sales_Campaign_Combined'] = combined_net_sales
                result_row['Impressions_Campaign_Combined'] = combined_impressions
                result_row['Ad_Spend_Campaign_Combined'] = combined_ad_spend
                
                # Calculate percentage changes for combined
                result_row['Sessions_Total_Change_Combined'] = calculate_percentage_change(sessions_total_base, combined_sessions_total)
                result_row['Sessions_Google_Change_Combined'] = calculate_percentage_change(sessions_google_base, combined_sessions_google)
                result_row['Net_Sales_Change_Combined'] = calculate_percentage_change(net_sales_base, combined_net_sales)
                result_row['Impressions_Change_Combined'] = calculate_percentage_change(impressions_base, combined_impressions)
                result_row['Ad_Spend_Change_Combined'] = calculate_percentage_change(ad_spend_base, combined_ad_spend)
            
            results.append(result_row)
        
        return results, base_divisor, conn
        
    except Exception as e:
        conn.close()
        raise e

def process_control_regions_duckdb(conn, control_regions, google_sources, 
                                  base_week_start, base_week_end,
                                  campaign_weeks, region_column, shopify_region_column, meta_region_column,
                                  base_divisor, campaign_display_method, campaign_calculation_method):
    """Process control regions using DuckDB aggregation"""
    
    if not control_regions:
        return None
    
    # Create control regions filter
    control_regions_str = "', '".join(control_regions)
    control_filter = f'"{region_column}" IN (\'{control_regions_str}\')'
    shopify_control_filter = f'"{shopify_region_column}" IN (\'{control_regions_str}\')'
    
    # Check if Meta data is available
    has_meta_data = False
    meta_control_filter = None
    if meta_region_column:
        try:
            # Try to query meta_data to see if it exists
            conn.execute("SELECT COUNT(*) FROM meta_data LIMIT 1").fetchone()
            has_meta_data = True
            meta_control_filter = f'"{meta_region_column}" IN (\'{control_regions_str}\')'
        except:
            has_meta_data = False
    
    # Create Google sources filter
    google_sources_str = "', '".join(google_sources)
    google_filter = f'"Session source" IN (\'{google_sources_str}\')' if google_sources else "1=0"
    
    # Calculate control region count
    control_region_count = len(control_regions)
    
    # Aggregate GA data for control regions - base week
    ga_control_base_query = f"""
    SELECT 
        SUM(Sessions) as sessions_base,
        SUM(CASE WHEN {google_filter} THEN Sessions ELSE 0 END) as google_sessions_base
    FROM ga_data 
    WHERE {control_filter}
    AND Date >= '{base_week_start}' 
    AND Date <= '{base_week_end}'
    """
    
    # Aggregate Shopify data for control regions - base week
    shopify_control_base_query = f"""
    SELECT 
        SUM("Net sales") as sales_base
    FROM shopify_data 
    WHERE {shopify_control_filter}
    AND Day >= '{base_week_start}' 
    AND Day <= '{base_week_end}'
    """
    
    # Aggregate Meta data for control regions - base week (if available)
    meta_control_base_query = None
    if has_meta_data:
        meta_control_base_query = f"""
        SELECT 
            SUM(Impressions) as impressions_base,
            SUM("Amount spent (USD)") as ad_spend_base
        FROM meta_data 
        WHERE {meta_control_filter}
        AND Day >= '{base_week_start}' 
        AND Day <= '{base_week_end}'
        """
    
    # Execute base week queries
    ga_control_base_result = conn.execute(ga_control_base_query).fetchone()
    shopify_control_base_result = conn.execute(shopify_control_base_query).fetchone()
    meta_control_base_result = conn.execute(meta_control_base_query).fetchone() if meta_control_base_query else (0, 0)
    
    # Calculate base week metrics (always averaged)
    sessions_total_base = (ga_control_base_result[0] or 0) / (control_region_count * base_divisor)
    sessions_google_base = (ga_control_base_result[1] or 0) / (control_region_count * base_divisor)
    net_sales_base = (shopify_control_base_result[0] or 0) / (control_region_count * base_divisor)
    impressions_base = (meta_control_base_result[0] or 0) / (control_region_count * base_divisor)
    ad_spend_base = (meta_control_base_result[1] or 0) / (control_region_count * base_divisor)
    
    # Initialize result row
    result_row = {
        'Region': 'Control set',
        'Sessions_Total_Base': sessions_total_base,
        'Sessions_Google_Base': sessions_google_base,
        'Net_Sales_Base': net_sales_base,
        'Impressions_Base': impressions_base,
        'Ad_Spend_Base': ad_spend_base
    }
    
    # Process campaign weeks for control regions
    campaign_sessions_total = []
    campaign_sessions_google = []
    campaign_net_sales = []
    campaign_impressions = []
    campaign_ad_spend = []
    
    for i, week in enumerate(campaign_weeks):
        week_start = week['start']
        week_end = week['end']
        
        # Calculate divisor for this campaign week
        campaign_week_weeks = calculate_weeks_in_period(week_start, week_end)
        campaign_divisor = campaign_week_weeks if campaign_calculation_method == "Average (÷weeks)" else 1
        
        # GA Campaign query for this week
        ga_campaign_query = f"""
        SELECT 
            SUM(Sessions) as sessions_campaign,
            SUM(CASE WHEN {google_filter} THEN Sessions ELSE 0 END) as google_sessions_campaign
        FROM ga_data 
        WHERE {control_filter}
        AND Date >= '{week_start}' 
        AND Date <= '{week_end}'
        """
        
        # Shopify Campaign query for this week
        shopify_campaign_query = f"""
        SELECT SUM("Net sales") as sales_campaign
        FROM shopify_data 
        WHERE {shopify_control_filter}
        AND Day >= '{week_start}' 
        AND Day <= '{week_end}'
        """
        
        # Meta Campaign query for this week (if available)
        meta_campaign_query = None
        if has_meta_data:
            meta_campaign_query = f"""
            SELECT 
                SUM(Impressions) as impressions_campaign,
                SUM("Amount spent (USD)") as ad_spend_campaign
            FROM meta_data 
            WHERE {meta_control_filter}
            AND Day >= '{week_start}' 
            AND Day <= '{week_end}'
            """
        
        # Execute campaign queries
        ga_campaign_result = conn.execute(ga_campaign_query).fetchone()
        shopify_campaign_result = conn.execute(shopify_campaign_query).fetchone()
        meta_campaign_result = conn.execute(meta_campaign_query).fetchone() if meta_campaign_query else (0, 0)
        
        # Calculate campaign metrics
        sessions_total_campaign = (ga_campaign_result[0] or 0) / (control_region_count * campaign_divisor)
        sessions_google_campaign = (ga_campaign_result[1] or 0) / (control_region_count * campaign_divisor)
        net_sales_campaign = (shopify_campaign_result[0] or 0) / (control_region_count * campaign_divisor)
        impressions_campaign = (meta_campaign_result[0] or 0) / (control_region_count * campaign_divisor)
        ad_spend_campaign = (meta_campaign_result[1] or 0) / (control_region_count * campaign_divisor)
        
        # Store individual week data
        campaign_sessions_total.append(sessions_total_campaign)
        campaign_sessions_google.append(sessions_google_campaign)
        campaign_net_sales.append(net_sales_campaign)
        campaign_impressions.append(impressions_campaign)
        campaign_ad_spend.append(ad_spend_campaign)
        
        if campaign_display_method == "Separate Columns":
            # Add individual week columns
            result_row[f'Sessions_Total_Campaign_Week_{i+1}'] = sessions_total_campaign
            result_row[f'Sessions_Google_Campaign_Week_{i+1}'] = sessions_google_campaign
            result_row[f'Net_Sales_Campaign_Week_{i+1}'] = net_sales_campaign
            result_row[f'Impressions_Campaign_Week_{i+1}'] = impressions_campaign
            result_row[f'Ad_Spend_Campaign_Week_{i+1}'] = ad_spend_campaign
            
            # Calculate percentage changes for individual weeks
            result_row[f'Sessions_Total_Change_Week_{i+1}'] = calculate_percentage_change(sessions_total_base, sessions_total_campaign)
            result_row[f'Sessions_Google_Change_Week_{i+1}'] = calculate_percentage_change(sessions_google_base, sessions_google_campaign)
            result_row[f'Net_Sales_Change_Week_{i+1}'] = calculate_percentage_change(net_sales_base, net_sales_campaign)
            result_row[f'Impressions_Change_Week_{i+1}'] = calculate_percentage_change(impressions_base, impressions_campaign)
            result_row[f'Ad_Spend_Change_Week_{i+1}'] = calculate_percentage_change(ad_spend_base, ad_spend_campaign)
    
    # If combined display, calculate combined metrics
    if campaign_display_method == "Combined Column":
        if campaign_calculation_method == "Average (÷weeks)":
            combined_sessions_total = sum(campaign_sessions_total) / len(campaign_sessions_total)
            combined_sessions_google = sum(campaign_sessions_google) / len(campaign_sessions_google)
            combined_net_sales = sum(campaign_net_sales) / len(campaign_net_sales)
            combined_impressions = sum(campaign_impressions) / len(campaign_impressions)
            combined_ad_spend = sum(campaign_ad_spend) / len(campaign_ad_spend)
        else:  # Sum
            combined_sessions_total = sum(campaign_sessions_total)
            combined_sessions_google = sum(campaign_sessions_google)
            combined_net_sales = sum(campaign_net_sales)
            combined_impressions = sum(campaign_impressions)
            combined_ad_spend = sum(campaign_ad_spend)
        
        result_row['Sessions_Total_Campaign_Combined'] = combined_sessions_total
        result_row['Sessions_Google_Campaign_Combined'] = combined_sessions_google
        result_row['Net_Sales_Campaign_Combined'] = combined_net_sales
        result_row['Impressions_Campaign_Combined'] = combined_impressions
        result_row['Ad_Spend_Campaign_Combined'] = combined_ad_spend
        
        # Calculate percentage changes for combined
        result_row['Sessions_Total_Change_Combined'] = calculate_percentage_change(sessions_total_base, combined_sessions_total)
        result_row['Sessions_Google_Change_Combined'] = calculate_percentage_change(sessions_google_base, combined_sessions_google)
        result_row['Net_Sales_Change_Combined'] = calculate_percentage_change(net_sales_base, combined_net_sales)
        result_row['Impressions_Change_Combined'] = calculate_percentage_change(impressions_base, combined_impressions)
        result_row['Ad_Spend_Change_Combined'] = calculate_percentage_change(ad_spend_base, combined_ad_spend)
    
    return result_row

def create_display_dataframes(analysis_df, base_label, campaign_weeks, campaign_display_method):
    """Create formatted dataframes for display"""
    
    # Check if Meta columns exist in the dataframe
    has_meta_data = 'Impressions_Base' in analysis_df.columns
    
    if campaign_display_method == "Separate Columns":
        # Create Base Week vs Individual Campaign Weeks comparison
        df = pd.DataFrame()
        df['Region'] = analysis_df['Region']
        df[f'Sessions Total - {base_label}'] = analysis_df['Sessions_Total_Base'].apply(lambda x: f"{x:,.0f}")
        
        # Add columns for each campaign week
        for i in range(len(campaign_weeks)):
            week_label = campaign_weeks[i]['label']
            df[f'Sessions Total - {week_label}'] = analysis_df[f'Sessions_Total_Campaign_Week_{i+1}'].apply(lambda x: f"{x:,.0f}")
            df[f'Sessions Total - %Change ({week_label})'] = analysis_df[f'Sessions_Total_Change_Week_{i+1}']
        
        # Add Google sessions columns
        df[f'Sessions Google - {base_label}'] = analysis_df['Sessions_Google_Base'].apply(lambda x: f"{x:,.0f}")
        for i in range(len(campaign_weeks)):
            week_label = campaign_weeks[i]['label']
            df[f'Sessions Google - {week_label}'] = analysis_df[f'Sessions_Google_Campaign_Week_{i+1}'].apply(lambda x: f"{x:,.0f}")
            df[f'Sessions Google - %Change ({week_label})'] = analysis_df[f'Sessions_Google_Change_Week_{i+1}']
        
        # Add Net Sales columns
        df[f'Net Sales - {base_label}'] = analysis_df['Net_Sales_Base'].apply(lambda x: f"${x:,.0f}")
        for i in range(len(campaign_weeks)):
            week_label = campaign_weeks[i]['label']
            df[f'Net Sales - {week_label}'] = analysis_df[f'Net_Sales_Campaign_Week_{i+1}'].apply(lambda x: f"${x:,.0f}")
            df[f'Net Sales - %Change ({week_label})'] = analysis_df[f'Net_Sales_Change_Week_{i+1}']
        
        # Add Meta Impressions columns if available
        if has_meta_data:
            df[f'Impressions - {base_label}'] = analysis_df['Impressions_Base'].apply(lambda x: f"{x:,.0f}")
            for i in range(len(campaign_weeks)):
                week_label = campaign_weeks[i]['label']
                df[f'Impressions - {week_label}'] = analysis_df[f'Impressions_Campaign_Week_{i+1}'].apply(lambda x: f"{x:,.0f}")
                df[f'Impressions - %Change ({week_label})'] = analysis_df[f'Impressions_Change_Week_{i+1}']
            
            # Add Meta Ad Spend columns
            df[f'Ad Spend - {base_label}'] = analysis_df['Ad_Spend_Base'].apply(lambda x: f"${x:,.2f}")
            for i in range(len(campaign_weeks)):
                week_label = campaign_weeks[i]['label']
                df[f'Ad Spend - {week_label}'] = analysis_df[f'Ad_Spend_Campaign_Week_{i+1}'].apply(lambda x: f"${x:,.2f}")
                df[f'Ad Spend - %Change ({week_label})'] = analysis_df[f'Ad_Spend_Change_Week_{i+1}']
        
        return df
        
    else:  # Combined Column
        # Create Base Week vs Combined Campaign comparison
        df = pd.DataFrame()
        df['Region'] = analysis_df['Region']
        df[f'Sessions Total - {base_label}'] = analysis_df['Sessions_Total_Base'].apply(lambda x: f"{x:,.0f}")
        df[f'Sessions Total - Campaign Combined'] = analysis_df['Sessions_Total_Campaign_Combined'].apply(lambda x: f"{x:,.0f}")
        df['Sessions Total - %Change'] = analysis_df['Sessions_Total_Change_Combined']
        df[f'Sessions Google - {base_label}'] = analysis_df['Sessions_Google_Base'].apply(lambda x: f"{x:,.0f}")
        df[f'Sessions Google - Campaign Combined'] = analysis_df['Sessions_Google_Campaign_Combined'].apply(lambda x: f"{x:,.0f}")
        df['Sessions Google - %Change'] = analysis_df['Sessions_Google_Change_Combined']
        df[f'Net Sales - {base_label}'] = analysis_df['Net_Sales_Base'].apply(lambda x: f"${x:,.0f}")
        df[f'Net Sales - Campaign Combined'] = analysis_df['Net_Sales_Campaign_Combined'].apply(lambda x: f"${x:,.0f}")
        df['Net Sales - %Change'] = analysis_df['Net_Sales_Change_Combined']
        
        # Add Meta columns if available
        if has_meta_data:
            df[f'Impressions - {base_label}'] = analysis_df['Impressions_Base'].apply(lambda x: f"{x:,.0f}")
            df[f'Impressions - Campaign Combined'] = analysis_df['Impressions_Campaign_Combined'].apply(lambda x: f"{x:,.0f}")
            df['Impressions - %Change'] = analysis_df['Impressions_Change_Combined']
            df[f'Ad Spend - {base_label}'] = analysis_df['Ad_Spend_Base'].apply(lambda x: f"${x:,.2f}")
            df[f'Ad Spend - Campaign Combined'] = analysis_df['Ad_Spend_Campaign_Combined'].apply(lambda x: f"${x:,.2f}")
            df['Ad Spend - %Change'] = analysis_df['Ad_Spend_Change_Combined']
        
        return df

def create_csv_export_data(df, base_label, campaign_weeks, campaign_display_method):
    """Create CSV data that matches the exact display format"""
    
    # This is a simplified version - you can expand based on the display format
    csv_lines = []
    
    if campaign_display_method == "Separate Columns":
        csv_lines.append(f"Base Week ({base_label}) vs Individual Campaign Weeks Comparison")
    else:
        csv_lines.append(f"Base Week ({base_label}) vs Combined Campaign Comparison")
    
    csv_lines.append("")
    
    # Convert dataframe to CSV format
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_content = csv_buffer.getvalue()
    
    return csv_content

def render_campaign_weeks_input(section_id):
    """Render the campaign weeks input section"""
    
    # Initialize campaign weeks in session state if not exists
    if f'campaign_weeks_{section_id}' not in st.session_state:
        st.session_state[f'campaign_weeks_{section_id}'] = []
    
    st.subheader("📅 Campaign Weeks Configuration")
    
    # Display existing campaign weeks
    campaign_weeks = st.session_state[f'campaign_weeks_{section_id}']
    
    if campaign_weeks:
        st.write("**Current Campaign Weeks:**")
        for i, week in enumerate(campaign_weeks):
            st.markdown(f"""
            <div class="campaign-week">
                <strong>{week['label']}</strong>: {week['start']} to {week['end']}
            </div>
            """, unsafe_allow_html=True)
    
    # Add new campaign week
    st.write("**Add New Campaign Week:**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        new_week_label = st.text_input(
            "Week Label", 
            value=f"Campaign Week {len(campaign_weeks) + 1}",
            key=f"new_week_label_{section_id}"
        )
    
    with col2:
        new_week_start = st.date_input(
            "Week Start",
            key=f"new_week_start_{section_id}"
        )
    
    with col3:
        new_week_end = st.date_input(
            "Week End",
            key=f"new_week_end_{section_id}"
        )
    
    # Buttons for managing campaign weeks
    button_col1, button_col2, button_col3 = st.columns(3)
    
    with button_col1:
        if st.button("➕ Add Week", key=f"add_week_{section_id}"):
            if new_week_start <= new_week_end:
                new_week = {
                    'label': new_week_label,
                    'start': new_week_start,
                    'end': new_week_end
                }
                st.session_state[f'campaign_weeks_{section_id}'].append(new_week)
                st.rerun()
            else:
                st.error("Start date must be before or equal to end date")
    
    with button_col2:
        if campaign_weeks and st.button("🗑️ Remove Last Week", key=f"remove_week_{section_id}"):
            st.session_state[f'campaign_weeks_{section_id}'].pop()
            st.rerun()
    
    with button_col3:
        if campaign_weeks and st.button("🗑️ Clear All Weeks", key=f"clear_weeks_{section_id}"):
            st.session_state[f'campaign_weeks_{section_id}'] = []
            st.rerun()
    
    return campaign_weeks

def render_analysis_section(ga_data, shopify_data, meta_data, section_id):
    """Render a complete analysis section with input form and report display"""
    
    # Import required modules
    from datetime import date, timedelta, datetime
    
    # Validate input data
    if ga_data is None or ga_data.empty:
        st.error("GA data is not available or empty")
        return
    
    if shopify_data is None or shopify_data.empty:
        st.error("Shopify data is not available or empty")
        return
    
    # Meta data is optional
    has_meta_data = meta_data is not None and not meta_data.empty
    
    # Initialize this section's data in session state if it doesn't exist
    if f'section_{section_id}' not in st.session_state:
        st.session_state[f'section_{section_id}'] = {
            'report_generated': False,
            'analysis_df': None,
            'config': None,
            'timestamp': None
        }
    
    section_data = st.session_state[f'section_{section_id}']
    
    # Input form (always visible and in the same place)
    st.markdown(f"""
    <div class="input-form">
        <h3>🔧 Analysis Configuration #{section_id}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Get date range from GA data for defaults
    if ga_data is not None and not ga_data.empty and 'Date' in ga_data.columns:
        try:
            min_date = ga_data['Date'].min().date()
            max_date = ga_data['Date'].max().date()
        except Exception as e:
            st.error(f"Error accessing Date column: {str(e)}")
            # Use fallback dates
            max_date = date.today()
            min_date = max_date - timedelta(days=30)
    else:
        st.error("Date column not found in GA data")
        # Use fallback dates
        max_date = date.today()
        min_date = max_date - timedelta(days=30)
    
    # Column selection
    st.subheader("📊 Column Configuration")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ga_columns = list(ga_data.columns) if ga_data is not None and not ga_data.empty else []
        if not ga_columns:
            st.error("No GA data columns available")
            region_column = None
        else:
            region_column = st.selectbox(
                "Select Region Column from GA Data",
                options=ga_columns,
                index=next((i for i, col in enumerate(ga_columns) if 'region' in col.lower()), 0),
                help="Select the column that contains region information",
                key=f"region_col_{section_id}"
            )
    
    with col2:
        shopify_columns = list(shopify_data.columns) if shopify_data is not None and not shopify_data.empty else []
        if not shopify_columns:
            st.error("No Shopify data columns available")
            shopify_region_column = None
        else:
            shopify_region_column = st.selectbox(
                "Select Region Column from Shopify Data",
                options=shopify_columns,
                index=next((i for i, col in enumerate(shopify_columns) if 'region' in col.lower()), 0),
                help="Select the column that contains region information in Shopify data",
                key=f"shopify_region_col_{section_id}"
            )
    
    with col3:
        if has_meta_data:
            meta_columns = list(meta_data.columns)
            meta_region_column = st.selectbox(
                "Select Region Column from Meta Data",
                options=meta_columns,
                index=next((i for i, col in enumerate(meta_columns) if 'region' in col.lower()), 0),
                help="Select the column that contains region information in Meta data",
                key=f"meta_region_col_{section_id}"
            )
        else:
            st.info("Meta data not uploaded")
            meta_region_column = None
    
    # Session source configuration
    st.subheader("🔍 Session Source Configuration")
    
    all_sources = []
    if ga_data is not None and not ga_data.empty and 'Session source' in ga_data.columns:
        try:
            # Handle mixed data types and NaN values
            source_series = ga_data['Session source'].dropna()  # Remove NaN values
            unique_sources = source_series.unique()
            
            # Convert all values to strings and filter out empty ones
            string_sources = [str(source).strip() for source in unique_sources if str(source).strip() and str(source).lower() != 'nan']
            
            # Sort the cleaned sources
            all_sources = sorted(string_sources)
        except Exception as e:
            st.error(f"Error accessing Session source column: {str(e)}")
            all_sources = []
    elif ga_data is not None and not ga_data.empty:
        st.warning("'Session source' column not found in GA data")
    
    google_sources = st.multiselect(
        "Select Google Session Sources",
        options=all_sources,
        default=[source for source in all_sources if 'google' in source.lower()],
        help="Select which session sources should be counted as Google sessions",
        key=f"google_sources_{section_id}"
    )
    
    # Base week calculation method
    st.subheader("📊 Calculation Method")
    base_week_method = st.radio(
        "Base Week Calculation",
        options=["Average (÷weeks)", "Sum (Total)"],
        index=0,
        help="Base weeks are ALWAYS averaged by number of weeks. This is kept for consistency.",
        key=f"base_week_method_{section_id}"
    )
    
    # Campaign calculation and display methods
    col1, col2 = st.columns(2)
    
    with col1:
        campaign_calculation_method = st.radio(
            "Campaign Week Calculation",
            options=["Average (÷weeks)", "Sum (Total)"],
            index=0,
            help="Choose how to calculate individual campaign weeks",
            key=f"campaign_calc_method_{section_id}"
        )
    
    with col2:
        campaign_display_method = st.radio(
            "Campaign Display Method",
            options=["Separate Columns", "Combined Column"],
            index=0,
            help="Display each campaign week separately or combine them into one column",
            key=f"campaign_display_method_{section_id}"
        )
    
    st.info("ℹ️ Base weeks are automatically averaged by their respective number of weeks (rounded to nearest whole number)")
    
    # Period configuration
    st.subheader("📅 Base Week Configuration")
    
    st.write(f"**Available Date Range:** {min_date} to {max_date}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Base Week:**")
        base_week_start = st.date_input(
            "Base Week Start", 
            value=min_date, 
            min_value=min_date, 
            max_value=max_date,
            key=f"base_start_{section_id}"
        )
        base_week_end = st.date_input(
            "Base Week End", 
            value=min_date + timedelta(days=20), 
            min_value=min_date, 
            max_value=max_date,
            key=f"base_end_{section_id}"
        )
    
    with col2:
        st.write("**Campaign Weeks:**")
        st.info("Configure campaign weeks below in the Campaign Weeks Configuration section")
    
    # Validation
    if base_week_start > base_week_end:
        st.error("Base week start date must be before end date")
        return
    
    # Campaign weeks configuration
    campaign_weeks = render_campaign_weeks_input(section_id)
    
    if not campaign_weeks:
        st.warning("Please add at least one campaign week.")
        return
    
    # Show week calculations
    st.subheader("📊 Week Calculations")
    
    base_weeks = calculate_weeks_in_period(base_week_start, base_week_end)
    
    calc_col1, calc_col2 = st.columns(2)
    
    with calc_col1:
        base_days = (base_week_end - base_week_start).days + 1
        st.write(f"**Base Week:** {base_days} days → {base_weeks} weeks (averaged)")
    
    with calc_col2:
        st.write("**Campaign Weeks:**")
        for week in campaign_weeks:
            campaign_days = (week['end'] - week['start']).days + 1
            campaign_weeks_calc = calculate_weeks_in_period(week['start'], week['end'])
            campaign_method = "averaged" if campaign_calculation_method == "Average (÷weeks)" else "total"
            st.write(f"• {week['label']}: {campaign_days} days → {campaign_weeks_calc} weeks ({campaign_method})")
    
    # Labels
    st.subheader("🏷️ Period Labels")
    label_col1, label_col2 = st.columns(2)
    
    with label_col1:
        base_label = st.text_input("Base Week Label", value="Base week", key=f"base_label_{section_id}")
    with label_col2:
        st.info("Campaign week labels are configured in the Campaign Weeks section above")
    
    # Region selection
    st.subheader("🌍 Region Configuration")
    
    # Safe region extraction with proper validation
    available_regions = []
    if ga_data is not None and not ga_data.empty and region_column and region_column in ga_data.columns:
        try:
            # Handle mixed data types and NaN values
            region_series = ga_data[region_column].dropna()  # Remove NaN values
            unique_regions = region_series.unique()
            
            # Convert all values to strings and filter out empty ones
            string_regions = [str(region).strip() for region in unique_regions if str(region).strip() and str(region).lower() != 'nan']
            
            # Sort the cleaned regions
            available_regions = sorted(string_regions)
        except Exception as e:
            st.error(f"Error accessing region column '{region_column}': {str(e)}")
            available_regions = []
    
    st.write(f"**Available Regions from '{region_column or 'N/A'}' ({len(available_regions)}):**")
    
    # Show data quality info if there are issues
    if ga_data is not None and not ga_data.empty and region_column and region_column in ga_data.columns:
        total_rows = len(ga_data)
        non_null_rows = ga_data[region_column].notna().sum()
        if non_null_rows < total_rows:
            st.info(f"Note: {total_rows - non_null_rows} rows have missing region values (out of {total_rows} total)")
    
    st.info("ℹ️ Region names will be normalized (lowercase, spaces removed) during analysis to ensure matching across datasets")
    
    if len(available_regions) <= 10:
        st.write(", ".join(available_regions))
    else:
        st.write(f"{', '.join(available_regions[:10])}... and {len(available_regions)-10} more")
    
    region_col1, region_col2 = st.columns(2)
    
    with region_col1:
        selected_regions = st.multiselect(
            "Select Target Regions",
            options=available_regions,
            default=available_regions[:3] if len(available_regions) >= 3 else available_regions,
            help="Select regions to include in the analysis",
            key=f"selected_regions_{section_id}"
        )
    
    with region_col2:
        control_regions = st.multiselect(
            "Select Control Regions",
            options=available_regions,
            help="Select which regions should be labeled as 'Control set'",
            key=f"control_regions_{section_id}"
        )
    
    if not selected_regions:
        st.warning("Please select at least one region for analysis.")
        return
    
    # Generate/Update analysis button
    button_col1, button_col2 = st.columns(2)
    
    with button_col1:
        if section_data['report_generated']:
            generate_button = st.button("🔄 Update Report", type="primary", key=f"update_{section_id}")
        else:
            generate_button = st.button("🚀 Generate Analysis", type="primary", key=f"generate_{section_id}")
    
    with button_col2:
        if section_data['report_generated']:
            if st.button("📊 Generate Another Report", type="secondary", key=f"generate_another_{section_id}"):
                # Add a new section
                if 'next_section_id' not in st.session_state:
                    st.session_state.next_section_id = 2
                else:
                    st.session_state.next_section_id += 1
                
                if 'active_sections' not in st.session_state:
                    st.session_state.active_sections = [1]
                
                st.session_state.active_sections.append(st.session_state.next_section_id)
                st.rerun()
    
    # Generate analysis if button clicked
    if generate_button:
        with st.spinner("Generating analysis..."):
            # Debug: Show what variables we have
            st.write("🔍 **Debug Info:**")
            st.write(f"- Region column: {region_column}")
            st.write(f"- Shopify region column: {shopify_region_column}")
            st.write(f"- Selected regions: {selected_regions}")
            st.write(f"- Campaign weeks: {len(campaign_weeks) if campaign_weeks else 0}")
            st.write(f"- Base week start: {base_week_start}")
            st.write(f"- Base week end: {base_week_end}")
            
            # Validate required parameters before analysis
            if not region_column:
                st.error("Please select a valid region column from GA data")
                return
            
            if not shopify_region_column:
                st.error("Please select a valid region column from Shopify data")
                return
            
            if not selected_regions:
                st.error("Please select at least one target region")
                return
            
            if not campaign_weeks:
                st.error("Please add at least one campaign week")
                return
            
            # Normalize region columns across all datasets
            st.info("🔄 Normalizing region data across datasets...")
            
            # Create copies of the data to avoid modifying the original cached data
            ga_data_normalized = ga_data.copy()
            shopify_data_normalized = shopify_data.copy()
            meta_data_normalized = meta_data.copy() if meta_data is not None and not meta_data.empty else None
            
            # Normalize GA region column: lowercase and remove all spaces
            if region_column in ga_data_normalized.columns:
                ga_data_normalized[region_column] = ga_data_normalized[region_column].astype(str).str.lower().str.replace(r'\s+', '', regex=True)
            
            # Normalize Shopify region column: lowercase and remove all spaces
            if shopify_region_column in shopify_data_normalized.columns:
                shopify_data_normalized[shopify_region_column] = shopify_data_normalized[shopify_region_column].astype(str).str.lower().str.replace(r'\s+', '', regex=True)
            
            # Normalize Meta region column: lowercase and remove all spaces (if Meta data exists)
            if meta_data_normalized is not None and meta_region_column and meta_region_column in meta_data_normalized.columns:
                meta_data_normalized[meta_region_column] = meta_data_normalized[meta_region_column].astype(str).str.lower().str.replace(r'\s+', '', regex=True)
            
            # Normalize selected regions and control regions to match
            selected_regions_normalized = [str(r).lower().replace(' ', '') for r in selected_regions]
            control_regions_normalized = [str(r).lower().replace(' ', '') for r in control_regions]
            
            st.success("✅ Region data normalized successfully!")
            
            try:
                results, base_divisor, conn = create_analysis_with_duckdb(
                    ga_data_normalized, shopify_data_normalized, meta_data_normalized, selected_regions_normalized,
                    base_week_start, base_week_end,
                    campaign_weeks, control_regions_normalized, google_sources, 
                    base_week_method, campaign_display_method, campaign_calculation_method,
                    region_column, shopify_region_column, meta_region_column
                )
                
                # Process control regions if any
                if control_regions_normalized:
                    control_result = process_control_regions_duckdb(
                        conn, control_regions_normalized, google_sources,
                        base_week_start, base_week_end,
                        campaign_weeks, region_column, shopify_region_column, meta_region_column,
                        base_divisor, campaign_display_method, campaign_calculation_method
                    )
                    if control_result:
                        results.append(control_result)
                
                # Close DuckDB connection
                conn.close()
                
                # Convert to DataFrame
                analysis_df = pd.DataFrame(results)
                
                # Store the results in session state
                st.session_state[f'section_{section_id}'] = {
                    'report_generated': True,
                    'analysis_df': analysis_df,
                    'config': {
                        'region_column': region_column,
                        'shopify_region_column': shopify_region_column,
                        'google_sources': google_sources,
                        'base_week_method': base_week_method,
                        'campaign_calculation_method': campaign_calculation_method,
                        'campaign_display_method': campaign_display_method,
                        'base_week_start': base_week_start,
                        'base_week_end': base_week_end,
                        'campaign_weeks': campaign_weeks,
                        'base_label': base_label,
                        'selected_regions': selected_regions,
                        'control_regions': control_regions
                    },
                    'timestamp': datetime.now()
                }
                
                st.success(f"✅ Analysis #{section_id} generated successfully!")
                st.rerun()
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                st.error("Please check your data and configuration settings.")
                # Add more detailed error information
                import traceback
                st.code(traceback.format_exc())
                return
                
            except Exception as e:
                st.error(f"Error generating analysis: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    # Display report if it exists (right below the input form)
    if section_data['report_generated']:
        st.markdown("---")
        
        # Report header
        st.markdown(f"""
        <div class="report-section">
            <h3>📊 Analysis Report #{section_id}</h3>
            <p><strong>Generated:</strong> {section_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        analysis_df = section_data['analysis_df']
        config = section_data['config']
        
        # Show configuration summary
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📅 Period Configuration:**")
            st.write(f"• Base Week: {config['base_week_start']} to {config['base_week_end']}")
            st.write(f"• Campaign Weeks: {len(config['campaign_weeks'])} weeks")
            
        with col2:
            st.write("**🌍 Region Configuration:**")
            st.write(f"• Target Regions: {', '.join(config['selected_regions'])}")
            if config['control_regions']:
                st.write(f"• Control Regions: {', '.join(config['control_regions'])}")
            st.write(f"• Google Sources: {len(config['google_sources'])} selected")
            st.write(f"• Display: {config['campaign_display_method']}")
        
        # Create display dataframes
        display_df = create_display_dataframes(analysis_df, config['base_label'], 
                                             config['campaign_weeks'], config['campaign_display_method'])
        
        # Display tables
        st.subheader(f"📊 {config['base_label']} vs Campaign Comparison")
        st.dataframe(display_df, use_container_width=True)
        
        # Create CSV data for download
        csv_data = create_csv_export_data(analysis_df, config['base_label'], 
                                        config['campaign_weeks'], config['campaign_display_method'])
        
        # Download button
        st.download_button(
            label=f"📥 Download Report #{section_id} as CSV",
            data=csv_data,
            file_name=f"campaign_analysis_report_{section_id}_{section_data['timestamp'].strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key=f"download_report_{section_id}"
        )

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Campaign Analysis - Multi-Week Campaign</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'active_sections' not in st.session_state:
        st.session_state.active_sections = [1]  # Start with section 1
    if 'next_section_id' not in st.session_state:
        st.session_state.next_section_id = 2
    
    # Sidebar for file uploads ONLY
    st.sidebar.header("📁 Dataset Upload")
    
    # GA data upload
    ga_file = st.sidebar.file_uploader(
        "Upload Merged GA Data (CSV/Excel/Parquet)",
        type=['csv', 'xlsx', 'xls', 'parquet'],
        help="Upload your merged Google Analytics data file",
        key="ga_upload"
    )
    
    # Shopify data upload
    shopify_file = st.sidebar.file_uploader(
        "Upload Shopify Data (CSV/Excel/Parquet)",
        type=['csv', 'xlsx', 'xls', 'parquet'],
        help="Upload your Shopify data file",
        key="shopify_upload"
    )
    
    # Meta data upload (optional)
    meta_file = st.sidebar.file_uploader(
        "Upload Meta Ad Data (CSV/Excel) - Optional",
        type=['csv', 'xlsx', 'xls'],
        help="Upload your Meta ad data file (Day, Region, Impression, Ad Spend)",
        key="meta_upload"
    )
    
    # Initialize data variables
    ga_data = None
    shopify_data = None
    meta_data = None
    
    # Data preview in sidebar
    if ga_file and shopify_file:
        st.sidebar.markdown("---")
        st.sidebar.header("📊 Dataset Info")
        
        # Load data with caching
        with st.spinner("Loading and optimizing data..."):
            ga_data = load_and_convert_data(ga_file, "ga")
            shopify_data = load_and_convert_data(shopify_file, "shopify")
            
            if ga_data is None:
                st.error("Error loading GA data")
                return
            if shopify_data is None:
                st.error("Error loading Shopify data")
                return
            
            # Load Meta data if provided (optional)
            if meta_file:
                meta_data = load_and_convert_data(meta_file, "meta")
                if meta_data is None:
                    st.warning("Error loading Meta data - continuing without it")
                    meta_data = None
        
        st.sidebar.success("✅ Data loaded!")
        st.sidebar.write(f"**GA Data:** {len(ga_data):,} rows")
        st.sidebar.write(f"**Shopify Data:** {len(shopify_data):,} rows")
        if meta_data is not None and not meta_data.empty:
            st.sidebar.write(f"**Meta Data:** {len(meta_data):,} rows")
        
        if not ga_data.empty:
            min_date = ga_data['Date'].min().date()
            max_date = ga_data['Date'].max().date()
            st.sidebar.write(f"**Date Range:** {min_date} to {max_date}")
    
    # Main content area
    if ga_file is None or shopify_file is None:
        st.info("👆 Please upload both merged GA data and Shopify data files in the sidebar to begin analysis")
        
        st.markdown("""
        ### Expected Data Formats:
        
        **Merged GA Data should contain:**
        - Date, Region, Session source, Sessions, Total users, New users
        - Items viewed, Add to carts, Total purchasers, Engaged sessions
        - Average session duration, Items added to cart
        
        **Shopify Data should contain:**
        - Day, Shipping postal code, Shipping region, Net sales, Net items sold
        - Orders, Average order value, Discounts, Gross margin
        - Customers, New customers, DMA
        
        **Meta Ad Data (Optional) should contain:**
        - Day, Region, Impressions, Amount spent (USD)
        
        ### New Features:
        - **Multi-Week Campaigns:** Add multiple campaign weeks individually
        - **Flexible Display:** Show campaign weeks separately or combined
        - **Calculation Options:** Average or sum campaign weeks
        - **Meta Ad Integration:** Include Meta ad metrics (Impressions, Ad Spend) in your analysis
        """)
        return
    
    # Only render analysis sections if data is properly loaded
    if ga_data is not None and shopify_data is not None and not ga_data.empty and not shopify_data.empty:
        # Render all active analysis sections
        for section_id in st.session_state.active_sections:
            render_analysis_section(ga_data, shopify_data, meta_data, section_id)
            
            # Add separator between sections (except for the last one)
            if section_id != st.session_state.active_sections[-1]:
                st.markdown("---")
                st.markdown("---")
    else:
        st.error("Data is not properly loaded. Please check your uploaded files.")
    
    # Data preview
    if ga_file and shopify_file:
        with st.expander("👀 Data Preview"):
            tabs = ["GA Data", "Shopify Data"]
            if meta_data is not None and not meta_data.empty:
                tabs.append("Meta Data")
            
            tab_objects = st.tabs(tabs)
            
            with tab_objects[0]:
                if not ga_data.empty:
                    st.write(f"**Date Range:** {ga_data['Date'].min()} to {ga_data['Date'].max()}")
                    st.write(f"**Memory Usage:** {ga_data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
                    st.dataframe(ga_data.head(10), use_container_width=True)
            
            with tab_objects[1]:
                if not shopify_data.empty:
                    st.write(f"**Memory Usage:** {shopify_data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
                    st.dataframe(shopify_data.head(10), use_container_width=True)
            
            if len(tab_objects) > 2:
                with tab_objects[2]:
                    if not meta_data.empty:
                        st.write(f"**Date Range:** {meta_data['Day'].min()} to {meta_data['Day'].max()}")
                        st.write(f"**Memory Usage:** {meta_data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
                        st.dataframe(meta_data.head(10), use_container_width=True)

if __name__ == "__main__":
    main()