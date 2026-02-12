import streamlit as st
import pandas as pd
import numpy as np
import duckdb
from datetime import datetime, timedelta
from io import StringIO
import tempfile
import os

st.set_page_config(page_title="Campaign DMA Analysis", layout="wide")

def generate_analysis_output(df, analysis, idx, conn):
    """Generate and display analysis output"""
    
    # Define Meta sources
    meta_sources = ["fbig", "fb social", "facebook organic", "instagram organic", 
                  "ig social", "instagram", "ig", "meta"]
    
    def calculate_exclusive_shares_optimized(start_date, end_date, dma_list, clusters_dict, counted_sources):
        """Calculate exclusive shares with priority ordering"""
        # If multiple DMAs, combine them
        if isinstance(dma_list, list):
            dma_str = "', '".join(dma_list)
            dma_condition = f"DMA IN ('{dma_str}')"
        else:
            dma_condition = f"DMA = '{dma_list}'"
        
        results = {}
        current_excluded = list(counted_sources)
        
        for cluster_name, cluster_sources in clusters_dict.items():
            sources_str = "', '".join(cluster_sources)
            
            if current_excluded:
                exclude_str = "', '".join(current_excluded)
                exclude_clause = f"AND source NOT IN ('{exclude_str}')"
            else:
                exclude_clause = ""
            
            query = f"""
            WITH journey_data AS (
                SELECT DISTINCT JourneyName
                FROM campaign_data
                WHERE entry_timestamp::DATE >= '{start_date}'
                AND entry_timestamp::DATE <= '{end_date}'
                AND {dma_condition}
            ),
            cluster_journeys AS (
                SELECT DISTINCT JourneyName
                FROM campaign_data
                WHERE entry_timestamp::DATE >= '{start_date}'
                AND entry_timestamp::DATE <= '{end_date}'
                AND {dma_condition}
                AND source IN ('{sources_str}')
                {exclude_clause}
            )
            SELECT 
                (SELECT COUNT(*) FROM journey_data) as total,
                (SELECT COUNT(*) FROM cluster_journeys) as cluster_count
            """
            
            result = conn.execute(query).fetchone()
            if result:
                total, count = result
                share = count / total if total > 0 else 0
                results[cluster_name] = {'share': share, 'count': count}
            
            current_excluded.extend(cluster_sources)
        
        return results, current_excluded
    
    # Extract analysis config
    clusters = analysis['clusters']
    campaign_weeks = analysis['campaign_weeks']
    base_start = analysis['base_start']
    base_end = analysis['base_end']
    base_weeks = analysis['base_weeks']
    target_dmas = analysis['target_dmas']
    control_dmas = analysis.get('control_dmas', [])
    use_control = analysis.get('use_control', False)
    combine_target = analysis.get('combine_target', False)
    
    # Pre-calculate all sources
    all_sources = df['source'].unique().tolist()
    
    # Determine how to process target DMAs
    if combine_target:
        target_groups = [target_dmas]  # Single group with all DMAs
        target_labels = [" + ".join(target_dmas)]
    else:
        target_groups = [[dma] for dma in target_dmas]  # Separate group for each DMA
        target_labels = target_dmas
    
    # For each target DMA or group
    for target_group, target_label in zip(target_groups, target_labels):
        st.write(f"### {target_label}")
        
        rows = []
        
        # Base period - Target
        base_row = {'Share of journeys': 'Base'}
        
        # Calculate Meta and clusters
        base_results, counted_sources = calculate_exclusive_shares_optimized(
            base_start, base_end, target_group, 
            {'Meta': meta_sources, **clusters}, 
            []
        )
        
        base_row['Meta'] = f"{base_results['Meta']['share']:.2%} ({base_results['Meta']['count']})"
        
        # Other clusters
        for cluster_name in clusters.keys():
            base_row[cluster_name] = f"{base_results[cluster_name]['share']:.2%} ({base_results[cluster_name]['count']})"
        
        # Others
        others_sources = [s for s in all_sources if s not in counted_sources]
        if others_sources:
            others_results, _ = calculate_exclusive_shares_optimized(
                base_start, base_end, target_group,
                {'Others': others_sources},
                counted_sources
            )
            base_row['Others'] = f"{others_results['Others']['share']:.2%} ({others_results['Others']['count']})"
        else:
            base_row['Others'] = "0.00% (0)"
        
        rows.append(base_row)
        
        # Campaign weeks - Target
        for camp_name, camp_data in campaign_weeks.items():
            camp_row = {'Share of journeys': camp_name}
            
            camp_results, counted_sources = calculate_exclusive_shares_optimized(
                camp_data['start'], camp_data['end'], target_group,
                {'Meta': meta_sources, **clusters},
                []
            )
            
            camp_row['Meta'] = f"{camp_results['Meta']['share']:.2%} ({camp_results['Meta']['count']})"
            
            for cluster_name in clusters.keys():
                camp_row[cluster_name] = f"{camp_results[cluster_name]['share']:.2%} ({camp_results[cluster_name]['count']})"
            
            # Others
            if others_sources:
                others_results, _ = calculate_exclusive_shares_optimized(
                    camp_data['start'], camp_data['end'], target_group,
                    {'Others': others_sources},
                    counted_sources
                )
                camp_row['Others'] = f"{others_results['Others']['share']:.2%} ({others_results['Others']['count']})"
            else:
                camp_row['Others'] = "0.00% (0)"
            
            rows.append(camp_row)
        
        # Control rows (only if use_control is True)
        if use_control and control_dmas:
            # Control average - Base
            control_row = {'Share of journeys': 'Control (Avg)'}
            
            # Calculate for all control DMAs combined
            ctrl_results, _ = calculate_exclusive_shares_optimized(
                base_start, base_end, control_dmas,
                {'Meta': meta_sources, **clusters},
                []
            )
            
            control_row['Meta'] = f"{ctrl_results['Meta']['share']:.2%} ({ctrl_results['Meta']['count']})"
            
            for cluster_name in clusters.keys():
                control_row[cluster_name] = f"{ctrl_results[cluster_name]['share']:.2%} ({ctrl_results[cluster_name]['count']})"
            
            # Others
            if others_sources:
                others_ctrl_results, _ = calculate_exclusive_shares_optimized(
                    base_start, base_end, control_dmas,
                    {'Others': others_sources},
                    counted_sources
                )
                control_row['Others'] = f"{others_ctrl_results['Others']['share']:.2%} ({others_ctrl_results['Others']['count']})"
            else:
                control_row['Others'] = "0.00% (0)"
            
            rows.append(control_row)
            
            # Control average - Campaign weeks
            for camp_name, camp_data in campaign_weeks.items():
                camp_control_row = {'Share of journeys': f'Control (Avg) - {camp_name}'}
                
                ctrl_results, _ = calculate_exclusive_shares_optimized(
                    camp_data['start'], camp_data['end'], control_dmas,
                    {'Meta': meta_sources, **clusters},
                    []
                )
                
                camp_control_row['Meta'] = f"{ctrl_results['Meta']['share']:.2%} ({ctrl_results['Meta']['count']})"
                
                for cluster_name in clusters.keys():
                    camp_control_row[cluster_name] = f"{ctrl_results[cluster_name]['share']:.2%} ({ctrl_results[cluster_name]['count']})"
                
                # Others
                if others_sources:
                    others_ctrl_results, _ = calculate_exclusive_shares_optimized(
                        camp_data['start'], camp_data['end'], control_dmas,
                        {'Others': others_sources},
                        counted_sources
                    )
                    camp_control_row['Others'] = f"{others_ctrl_results['Others']['share']:.2%} ({others_ctrl_results['Others']['count']})"
                else:
                    camp_control_row['Others'] = "0.00% (0)"
                
                rows.append(camp_control_row)
        
        # Display table
        result_df = pd.DataFrame(rows)
        cols = ['Share of journeys', 'Meta'] + list(clusters.keys()) + ['Others']
        result_df = result_df[cols]
        st.dataframe(result_df, use_container_width=True, hide_index=True)
        
        # Download CSV button
        csv = result_df.to_csv(index=False)
        st.download_button(
            label=f"📥 Download CSV - {target_label}",
            data=csv,
            file_name=f"analysis_{idx+1}_{target_label.replace(' ', '_').replace('+', 'and')}.csv",
            mime="text/csv",
            key=f"download_{idx}_{target_label}"
        )
        
        st.divider()

# Initialize session state
if 'clusters' not in st.session_state:
    st.session_state.clusters = {}
if 'campaign_weeks' not in st.session_state:
    st.session_state.campaign_weeks = {}
if 'df' not in st.session_state:
    st.session_state.df = None
if 'duckdb_conn' not in st.session_state:
    st.session_state.duckdb_conn = None
if 'parquet_path' not in st.session_state:
    st.session_state.parquet_path = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'analyses' not in st.session_state:
    st.session_state.analyses = []
if 'analysis_counter' not in st.session_state:
    st.session_state.analysis_counter = 0
if 'show_new_analysis_form' not in st.session_state:
    st.session_state.show_new_analysis_form = False

st.title("Campaign DMA Analysis Tool")

# File upload
uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=['csv', 'xlsx'])

if uploaded_file is not None:
    with st.spinner("Loading and optimizing dataset..."):
        # Read file based on extension
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Display available columns and let user select region column
        st.write("**Column Mapping**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            journey_col = st.selectbox("Journey Name Column", options=df.columns.tolist(), 
                                       index=df.columns.tolist().index('JourneyName') if 'JourneyName' in df.columns else 0)
        with col2:
            timestamp_col = st.selectbox("Timestamp Column", options=df.columns.tolist(),
                                        index=df.columns.tolist().index('entry_timestamp') if 'entry_timestamp' in df.columns else 0)
        with col3:
            source_col = st.selectbox("Source Column", options=df.columns.tolist(),
                                     index=df.columns.tolist().index('source') if 'source' in df.columns else 0)
        
        region_col = st.selectbox("Region/DMA Column", options=df.columns.tolist(),
                                 index=df.columns.tolist().index('DMA') if 'DMA' in df.columns else 0)
        
        if st.button("Load Dataset"):
            # Rename columns to standard names
            df = df.rename(columns={
                journey_col: 'JourneyName',
                timestamp_col: 'entry_timestamp',
                source_col: 'source',
                region_col: 'DMA'
            })
            
            # Filter for entry_sequence = 1 only (first touch)
            if 'entry_sequence' in df.columns:
                df = df[df['entry_sequence'] == 1]
                st.info(f"Filtered to entry_sequence = 1: {len(df)} rows")
            
            # Clean and convert all object columns to strings
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].fillna('').astype(str)
                    # Replace empty strings with 'unknown' for source column
                    if col == 'source':
                        df[col] = df[col].replace('', 'unknown')
            
            # Convert timestamp
            df['entry_timestamp'] = pd.to_datetime(df['entry_timestamp'])
            
            # Save to parquet for faster access
            temp_dir = tempfile.gettempdir()
            parquet_path = os.path.join(temp_dir, 'campaign_data.parquet')
            df.to_parquet(parquet_path, index=False)
            
            # Initialize DuckDB connection
            conn = duckdb.connect(':memory:')
            conn.execute(f"CREATE TABLE campaign_data AS SELECT * FROM read_parquet('{parquet_path}')")
            
            st.session_state.df = df
            st.session_state.duckdb_conn = conn
            st.session_state.parquet_path = parquet_path
            st.session_state.data_loaded = True
            st.rerun()

if st.session_state.get('data_loaded', False):
    df = st.session_state.df
    conn = st.session_state.duckdb_conn
    
    st.success(f"Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
    
    # Display available columns
    with st.expander("View Dataset Info"):
        st.write("Columns:", list(df.columns))
        st.write("Date range:", df['entry_timestamp'].min(), "to", df['entry_timestamp'].max())
        st.dataframe(df.head())
    
    st.divider()
    
    # Source Cluster Management
    st.subheader("📊 Source Cluster Management")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        unique_sources = sorted(df['source'].unique().tolist())
        st.write(f"Available sources: {len(unique_sources)}")
        
        # Add new cluster
        with st.expander("➕ Add New Cluster", expanded=len(st.session_state.clusters) == 0):
            cluster_name = st.text_input("Cluster Name", key="new_cluster_name")
            selected_sources = st.multiselect(
                "Select sources for this cluster",
                options=unique_sources,
                key="new_cluster_sources"
            )
            
            if st.button("Add Cluster", key="add_cluster_btn"):
                if cluster_name and selected_sources:
                    st.session_state.clusters[cluster_name] = selected_sources
                    st.success(f"Cluster '{cluster_name}' added!")
                    st.rerun()
                else:
                    st.warning("Please provide cluster name and select sources")
    
    with col2:
        st.write("**Current Clusters:**")
        if st.session_state.clusters:
            for cluster_name, sources in st.session_state.clusters.items():
                with st.container():
                    st.write(f"**{cluster_name}** ({len(sources)} sources)")
                    if st.button(f"🗑️ Delete", key=f"del_{cluster_name}"):
                        del st.session_state.clusters[cluster_name]
                        st.rerun()
        else:
            st.info("No clusters defined yet")
    
    st.divider()
    
    # Period Selection
    st.subheader("📅 Period Selection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Base Week Period**")
        base_start = st.date_input("Base Start Date", value=df['entry_timestamp'].min(), key="base_start")
        base_end = st.date_input("Base End Date", value=df['entry_timestamp'].min() + timedelta(days=6), key="base_end")
        
        base_days = (base_end - base_start).days + 1
        base_weeks = round(base_days / 7)
        st.info(f"Base period: {base_days} days ≈ {base_weeks} week(s)")
    
    with col2:
        st.write("**Campaign Weeks**")
        with st.expander("➕ Add Campaign Week", expanded=len(st.session_state.campaign_weeks) == 0):
            campaign_name = st.text_input("Campaign Week Name", key="new_campaign_name")
            campaign_start = st.date_input("Campaign Start Date", key="new_campaign_start")
            campaign_end = st.date_input("Campaign End Date", key="new_campaign_end")
            
            if st.button("Add Campaign Week", key="add_campaign_btn"):
                if campaign_name:
                    campaign_days = (campaign_end - campaign_start).days + 1
                    campaign_weeks = round(campaign_days / 7)
                    st.session_state.campaign_weeks[campaign_name] = {
                        'start': campaign_start,
                        'end': campaign_end,
                        'days': campaign_days,
                        'weeks': campaign_weeks
                    }
                    st.success(f"Campaign '{campaign_name}' added!")
                    st.rerun()
        
        if st.session_state.campaign_weeks:
            st.write("**Defined Campaign Weeks:**")
            for camp_name, camp_data in st.session_state.campaign_weeks.items():
                st.write(f"**{camp_name}**: {camp_data['days']} days ≈ {camp_data['weeks']} week(s)")
                if st.button(f"🗑️ Delete", key=f"del_camp_{camp_name}"):
                    del st.session_state.campaign_weeks[camp_name]
                    st.rerun()
    
    st.divider()
    
    # DMA Selection
    st.subheader("🗺️ DMA Selection")
    
    unique_dmas = sorted(df['DMA'].unique().tolist())
    
    target_dmas = st.multiselect("Select Target DMAs", options=unique_dmas, key="target_dmas")
    if len(target_dmas) > 1:
        combine_target = st.checkbox("Combine target DMAs into one analysis", value=False, key="combine_target")
    else:
        combine_target = False
    
    st.divider()
    
    # Generate Analysis Button
    if st.button("🚀 Generate Analysis", type="primary", key="generate_analysis"):
        if not st.session_state.clusters:
            st.error("Please define at least one source cluster")
        elif not st.session_state.campaign_weeks:
            st.error("Please define at least one campaign week")
        elif not target_dmas:
            st.error("Please select at least one target DMA")
        else:
            # Store analysis configuration
            analysis_config = {
                'id': st.session_state.analysis_counter,
                'clusters': dict(st.session_state.clusters),
                'campaign_weeks': dict(st.session_state.campaign_weeks),
                'base_start': base_start,
                'base_end': base_end,
                'base_weeks': base_weeks,
                'target_dmas': target_dmas,
                'control_dmas': [],
                'use_control': False,
                'combine_target': combine_target
            }
            st.session_state.analyses.append(analysis_config)
            st.session_state.analysis_counter += 1
            st.session_state.show_new_analysis_form = False
            st.success("Analysis generated!")
            st.rerun()
    
    # Display all generated analyses
    if st.session_state.analyses:
        st.divider()
        st.header("📊 Generated Analyses")
        
        for idx, analysis in enumerate(st.session_state.analyses):
            with st.container():
                col1, col2 = st.columns([6, 1])
                with col1:
                    st.subheader(f"Analysis {idx + 1}")
                with col2:
                    if st.button("🗑️ Delete", key=f"delete_analysis_{idx}"):
                        st.session_state.analyses.pop(idx)
                        st.rerun()
                
                # Generate analysis output
                generate_analysis_output(df, analysis, idx, st.session_state.duckdb_conn)
        
        # Add new analysis button at the bottom
        st.divider()
        if st.button("➕ Add New Analysis", key="add_new_analysis_btn", type="secondary"):
            st.session_state.show_new_analysis_form = True
            st.rerun()
    
    # Show new analysis form if requested
    if st.session_state.show_new_analysis_form:
        st.divider()
        st.header("🆕 Configure New Analysis")
        
        # Source Cluster Management for new analysis
        st.subheader("📊 Source Cluster Management")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            unique_sources = sorted(df['source'].unique().tolist())
            st.write(f"Available sources: {len(unique_sources)}")
            
            # Add new cluster
            with st.expander("➕ Add New Cluster", expanded=True):
                cluster_name_new = st.text_input("Cluster Name", key="new_cluster_name_2")
                selected_sources_new = st.multiselect(
                    "Select sources for this cluster",
                    options=unique_sources,
                    key="new_cluster_sources_2"
                )
                
                if st.button("Add Cluster", key="add_cluster_btn_2"):
                    if cluster_name_new and selected_sources_new:
                        st.session_state.clusters[cluster_name_new] = selected_sources_new
                        st.success(f"Cluster '{cluster_name_new}' added!")
                        st.rerun()
                    else:
                        st.warning("Please provide cluster name and select sources")
        
        with col2:
            st.write("**Current Clusters:**")
            if st.session_state.clusters:
                for cluster_name, sources in st.session_state.clusters.items():
                    with st.container():
                        st.write(f"**{cluster_name}** ({len(sources)} sources)")
                        if st.button(f"🗑️ Delete", key=f"del_{cluster_name}_2"):
                            del st.session_state.clusters[cluster_name]
                            st.rerun()
            else:
                st.info("No clusters defined yet")
        
        st.divider()
        
        # Period Selection for new analysis
        st.subheader("📅 Period Selection")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Base Week Period**")
            base_start_new = st.date_input("Base Start Date", value=df['entry_timestamp'].min(), key="base_start_2")
            base_end_new = st.date_input("Base End Date", value=df['entry_timestamp'].min() + timedelta(days=6), key="base_end_2")
            
            base_days_new = (base_end_new - base_start_new).days + 1
            base_weeks_new = round(base_days_new / 7)
            st.info(f"Base period: {base_days_new} days ≈ {base_weeks_new} week(s)")
        
        with col2:
            st.write("**Campaign Weeks**")
            with st.expander("➕ Add Campaign Week", expanded=True):
                campaign_name_new = st.text_input("Campaign Week Name", key="new_campaign_name_2")
                campaign_start_new = st.date_input("Campaign Start Date", key="new_campaign_start_2")
                campaign_end_new = st.date_input("Campaign End Date", key="new_campaign_end_2")
                
                if st.button("Add Campaign Week", key="add_campaign_btn_2"):
                    if campaign_name_new:
                        campaign_days_new = (campaign_end_new - campaign_start_new).days + 1
                        campaign_weeks_new = round(campaign_days_new / 7)
                        st.session_state.campaign_weeks[campaign_name_new] = {
                            'start': campaign_start_new,
                            'end': campaign_end_new,
                            'days': campaign_days_new,
                            'weeks': campaign_weeks_new
                        }
                        st.success(f"Campaign '{campaign_name_new}' added!")
                        st.rerun()
            
            if st.session_state.campaign_weeks:
                st.write("**Defined Campaign Weeks:**")
                for camp_name, camp_data in st.session_state.campaign_weeks.items():
                    st.write(f"**{camp_name}**: {camp_data['days']} days ≈ {camp_data['weeks']} week(s)")
                    if st.button(f"🗑️ Delete", key=f"del_camp_{camp_name}_2"):
                        del st.session_state.campaign_weeks[camp_name]
                        st.rerun()
        
        st.divider()
        
        # DMA Selection for new analysis
        st.subheader("🗺️ DMA Selection")
        
        unique_dmas = sorted(df['DMA'].unique().tolist())
        
        target_dmas_new = st.multiselect("Select Target DMAs", options=unique_dmas, key="target_dmas_2")
        if len(target_dmas_new) > 1:
            combine_target_new = st.checkbox("Combine target DMAs into one analysis", value=False, key="combine_target_2")
        else:
            combine_target_new = False
        
        st.divider()
        
        # Generate New Analysis Button
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Generate New Analysis", type="primary", key="generate_analysis_2"):
                if not st.session_state.clusters:
                    st.error("Please define at least one source cluster")
                elif not st.session_state.campaign_weeks:
                    st.error("Please define at least one campaign week")
                elif not target_dmas_new:
                    st.error("Please select at least one target DMA")
                else:
                    # Store analysis configuration
                    analysis_config = {
                        'id': st.session_state.analysis_counter,
                        'clusters': dict(st.session_state.clusters),
                        'campaign_weeks': dict(st.session_state.campaign_weeks),
                        'base_start': base_start_new,
                        'base_end': base_end_new,
                        'base_weeks': base_weeks_new,
                        'target_dmas': target_dmas_new,
                        'control_dmas': [],
                        'use_control': False,
                        'combine_target': combine_target_new
                    }
                    st.session_state.analyses.append(analysis_config)
                    st.session_state.analysis_counter += 1
                    st.session_state.show_new_analysis_form = False
                    st.success("Analysis generated!")
                    st.rerun()
        
        with col2:
            if st.button("❌ Cancel", key="cancel_new_analysis"):
                st.session_state.show_new_analysis_form = False
                st.rerun()
        
else:
    st.info("👆 Please upload a CSV or Excel file to begin")
