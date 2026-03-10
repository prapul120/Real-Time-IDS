"""
Streamlit Dashboard for Real-Time Intrusion Detection System

This module provides a web-based dashboard for monitoring IDS activity
in real-time with visualizations and alerts.
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import sys
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from realtime_detection import RealtimeDetector

# Page configuration
st.set_page_config(
    page_title="Real-Time IDS Monitor",
    page_icon="🛡️",
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
        margin-bottom: 1rem;
    }
    .status-running {
        color: #28a745;
        font-weight: bold;
    }
    .status-stopped {
        color: #dc3545;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .alert-attack {
        background-color: #f8d7da;
        color: #721c24;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .alert-normal {
        background-color: #d4edda;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .stProgress > div > div {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'stats' not in st.session_state:
    st.session_state.stats = {
        'total_packets': 0,
        'normal_packets': 0,
        'attack_packets': 0
    }


def initialize_detector(use_simulation=True):
    """Initialize the real-time detector."""
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'ids_model.pkl')
    
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}. Please train the model first.")
        return None
    
    try:
        detector = RealtimeDetector(model_path, use_simulation=use_simulation)
        return detector
    except Exception as e:
        st.error(f"Error initializing detector: {e}")
        return None


def update_callback(detection_record):
    """Callback function for detection updates."""
    st.session_state.detection_history.append(detection_record)
    
    # Keep only last 1000 records
    if len(st.session_state.detection_history) > 1000:
        st.session_state.detection_history = st.session_state.detection_history[-1000:]


def start_detection():
    """Start the detection process."""
    if st.session_state.detector is None:
        st.session_state.detector = initialize_detector(use_simulation=True)
    
    if st.session_state.detector:
        st.session_state.detector.register_callback(update_callback)
        st.session_state.detector.start_detection()
        st.session_state.is_running = True


def stop_detection():
    """Stop the detection process."""
    if st.session_state.detector:
        st.session_state.detector.stop_detection()
        st.session_state.detector.unregister_callback(update_callback)
        st.session_state.is_running = False


def get_current_stats():
    """Get current detection statistics."""
    if st.session_state.detector:
        return st.session_state.detector.get_statistics()
    return st.session_state.stats


def create_traffic_chart(normal_pct, attack_pct):
    """Create traffic distribution chart."""
    fig = go.Figure(data=[
        go.Bar(
            name='Normal Traffic',
            x=['Traffic Type'],
            y=[normal_pct],
            marker_color='#28a745',
            text=f'{normal_pct:.1f}%',
            textposition='auto'
        ),
        go.Bar(
            name='Attack Traffic',
            x=['Traffic Type'],
            y=[attack_pct],
            marker_color='#dc3545',
            text=f'{attack_pct:.1f}%',
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title='Traffic Distribution',
        barmode='group',
        yaxis_title='Percentage (%)',
        showlegend=True,
        height=300
    )
    
    return fig


def create_pie_chart(normal_count, attack_count):
    """Create pie chart for traffic distribution."""
    fig = go.Figure(data=[go.Pie(
        labels=['Normal', 'Attack'],
        values=[normal_count, attack_count],
        hole=0.4,
        marker_colors=['#28a745', '#dc3545']
    )])
    
    fig.update_layout(
        title='Traffic Distribution',
        showlegend=True,
        height=300
    )
    
    return fig


def create_timeline_chart(history):
    """Create timeline chart of detections."""
    if not history:
        return go.Figure()
    
    df = pd.DataFrame(history)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%H:%M:%S')
    
    # Group by minute and count
    df['minute'] = df['timestamp'].dt.floor('min')
    timeline = df.groupby(['minute', 'result']).size().unstack(fill_value=0)
    
    fig = go.Figure()
    
    if 'Normal' in timeline.columns:
        fig.add_trace(go.Scatter(
            x=timeline.index,
            y=timeline['Normal'],
            name='Normal',
            line=dict(color='#28a745'),
            mode='lines+markers'
        ))
    
    if 'Attack' in timeline.columns:
        fig.add_trace(go.Scatter(
            x=timeline.index,
            y=timeline['Attack'],
            name='Attack',
            line=dict(color='#dc3545'),
            mode='lines+markers'
        ))
    
    fig.update_layout(
        title='Detection Timeline',
        xaxis_title='Time',
        yaxis_title='Packet Count',
        height=300
    )
    
    return fig


# Main app
def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🛡️ Real-Time IDS Monitor</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Control Panel")
        
        # Status indicator
        if st.session_state.is_running:
            st.markdown('<p class="status-running">● Detection Running</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-stopped">● Detection Stopped</p>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Control buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("▶️ Start", use_container_width=True, disabled=st.session_state.is_running):
                start_detection()
                st.rerun()
        
        with col2:
            if st.button("⏹️ Stop", use_container_width=True, disabled=not st.session_state.is_running):
                stop_detection()
                st.rerun()
        
        st.markdown("---")
        
        # Settings
        st.subheader("Settings")
        
        if st.button("🔄 Reset Statistics", use_container_width=True):
            if st.session_state.detector:
                st.session_state.detector.reset_statistics()
            st.session_state.detection_history = []
            st.session_state.stats = {
                'total_packets': 0,
                'normal_packets': 0,
                'attack_packets': 0
            }
            st.rerun()
        
        if st.button("🗑️ Clear Alerts", use_container_width=True):
            if st.session_state.detector:
                st.session_state.detector.clear_log()
            st.session_state.detection_history = []
            st.rerun()
        
        st.markdown("---")
        
        # About
        st.subheader("About")
        st.info("""
        This dashboard monitors network traffic in real-time and 
        uses machine learning to detect potential intrusions.
        
        **Models Used:**
        - Decision Tree
        - Random Forest
        
        **Features:**
        - Real-time packet capture
        - Attack detection
        - Alert logging
        """)
    
    # Main content
    stats = get_current_stats()
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📊 Packets Analyzed",
            value=stats['total_packets'],
            delta=None
        )
    
    with col2:
        normal_pct = stats.get('normal_percentage', 0)
        st.metric(
            label="✅ Normal Traffic",
            value=f"{normal_pct:.1f}%",
            delta=None
        )
    
    with col3:
        attack_pct = stats.get('attack_percentage', 0)
        st.metric(
            label="⚠️ Attack Traffic",
            value=f"{attack_pct:.1f}%",
            delta=None
        )
    
    with col4:
        duration = stats.get('duration', 0)
        st.metric(
            label="⏱️ Duration (s)",
            value=f"{duration:.1f}",
            delta=None
        )
    
    st.markdown("---")
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Traffic Distribution")
        if stats['total_packets'] > 0:
            pie_chart = create_pie_chart(stats['normal_packets'], stats['attack_packets'])
            st.plotly_chart(pie_chart, use_container_width=True)
        else:
            st.info("No data available yet. Start detection to see traffic distribution.")
    
    with col2:
        st.subheader("Detection Timeline")
        if st.session_state.detection_history:
            timeline_chart = create_timeline_chart(st.session_state.detection_history)
            st.plotly_chart(timeline_chart, use_container_width=True)
        else:
            st.info("No detection data available yet.")
    
    st.markdown("---")
    
    # Alerts section
    st.subheader("🚨 Recent Alerts")
    
    # Get recent attacks
    recent_alerts = [r for r in st.session_state.detection_history if r['prediction'] == 1][-10:]
    
    if recent_alerts:
        alerts_df = pd.DataFrame(recent_alerts)
        alerts_df = alerts_df[['timestamp', 'src_ip', 'dst_ip', 'protocol', 'result']]
        alerts_df.columns = ['Time', 'Source IP', 'Destination IP', 'Protocol', 'Status']
        
        # Style the dataframe
        def highlight_attack(val):
            if val == 'Attack':
                return 'background-color: #f8d7da; color: #721c24;'
            return ''
        
        styled_df = alerts_df.style.applymap(highlight_attack, subset=['Status'])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
    else:
        st.info("No alerts detected. Start detection to monitor for intrusions.")
    
    st.markdown("---")
    
    # Recent packets table
    st.subheader("📋 Recent Packets")
    
    recent_packets = st.session_state.detection_history[-20:]
    
    if recent_packets:
        packets_df = pd.DataFrame(recent_packets)
        packets_df = packets_df[['timestamp', 'src_ip', 'dst_ip', 'protocol', 'packet_length', 'result']]
        packets_df.columns = ['Time', 'Source IP', 'Destination IP', 'Protocol', 'Length', 'Status']
        
        # Style the dataframe
        def color_status(val):
            if val == 'Attack':
                return 'color: #dc3545; font-weight: bold;'
            return 'color: #28a745;'
        
        styled_df = packets_df.style.applymap(color_status, subset=['Status'])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
    else:
        st.info("No packets captured yet. Start detection to see packet information.")
    
    # Auto-refresh when running
    if st.session_state.is_running:
        time.sleep(1)
        st.rerun()


if __name__ == "__main__":
    main()
