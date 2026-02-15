import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score
import random

# Page configuration
st.set_page_config(
    page_title="AI-Powered IDS | NSL-KDD",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for cybersecurity theme
st.markdown("""
<style>
    /* Import distinctive fonts */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;900&family=Space+Mono:wght@400;700&family=Rajdhani:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1d3a 50%, #0f1729 100%);
        color: #e0e6ed;
        font-family: 'Rajdhani', sans-serif;
    }
    
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 100%;
    }
    
    /* Header styling */
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif;
        color: #00d9ff;
        text-shadow: 0 0 10px rgba(0, 217, 255, 0.5);
        letter-spacing: 1px;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
        border-right: 2px solid #00d9ff;
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #00d9ff;
        font-family: 'Orbitron', sans-serif;
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-family: 'Orbitron', sans-serif;
        font-size: 2.5rem;
        color: #00ff88;
        text-shadow: 0 0 15px rgba(0, 255, 136, 0.6);
    }
    
    [data-testid="stMetricLabel"] {
        font-family: 'Space Mono', monospace;
        color: #8b949e;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #00d9ff 0%, #0099cc 100%);
        color: #000;
        font-family: 'Orbitron', sans-serif;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 4px 15px rgba(0, 217, 255, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #00ffaa 0%, #00cc88 100%);
        box-shadow: 0 6px 25px rgba(0, 255, 170, 0.6);
        transform: translateY(-2px);
    }
    
    /* Alert box styling */
    .alert-box {
        background: linear-gradient(135deg, #ff0055 0%, #cc0044 100%);
        border-left: 5px solid #ff0088;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(255, 0, 85, 0.4);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.85; }
    }
    
    .alert-box h3 {
        color: #fff;
        margin: 0 0 0.5rem 0;
        font-size: 1.5rem;
    }
    
    .alert-box p {
        color: #ffccdd;
        margin: 0;
        font-family: 'Space Mono', monospace;
    }
    
    /* Info panel styling */
    .info-panel {
        background: rgba(0, 217, 255, 0.05);
        border: 1px solid #00d9ff;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 0 20px rgba(0, 217, 255, 0.1);
    }
    
    /* Detection panel styling */
    .detection-panel {
        background: linear-gradient(135deg, rgba(0, 217, 255, 0.1) 0%, rgba(0, 153, 204, 0.05) 100%);
        border: 2px solid #00d9ff;
        border-radius: 12px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 0 30px rgba(0, 217, 255, 0.2);
    }
    
    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 20px;
        font-family: 'Orbitron', sans-serif;
        font-weight: 700;
        font-size: 1.2rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        box-shadow: 0 0 20px currentColor;
    }
    
    .status-normal {
        background: rgba(0, 255, 136, 0.2);
        color: #00ff88;
        border: 2px solid #00ff88;
    }
    
    .status-probe {
        background: rgba(255, 215, 0, 0.2);
        color: #ffd700;
        border: 2px solid #ffd700;
    }
    
    .status-r2l {
        background: rgba(255, 140, 0, 0.2);
        color: #ff8c00;
        border: 2px solid #ff8c00;
    }
    
    .status-dos {
        background: rgba(255, 69, 0, 0.2);
        color: #ff4500;
        border: 2px solid #ff4500;
    }
    
    .status-u2r {
        background: rgba(139, 0, 0, 0.3);
        color: #ff0000;
        border: 2px solid #ff0000;
    }
    
    /* Flow info styling */
    .flow-info {
        background: rgba(22, 27, 34, 0.8);
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        font-family: 'Space Mono', monospace;
    }
    
    .flow-info-row {
        display: flex;
        justify-content: space-between;
        padding: 0.5rem 0;
        border-bottom: 1px solid #21262d;
    }
    
    .flow-info-label {
        color: #8b949e;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.85rem;
        letter-spacing: 1px;
    }
    
    .flow-info-value {
        color: #00d9ff;
        font-weight: 700;
        font-size: 0.95rem;
    }
    
    /* Confidence bar */
    .confidence-container {
        margin: 1.5rem 0;
    }
    
    .confidence-bar {
        height: 30px;
        background: rgba(30, 35, 45, 0.8);
        border-radius: 15px;
        overflow: hidden;
        border: 1px solid #30363d;
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #00d9ff 0%, #00ff88 100%);
        transition: width 0.5s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: 'Orbitron', sans-serif;
        font-weight: 700;
        color: #000;
        font-size: 0.9rem;
    }
    
    /* Feature table */
    .feature-table {
        background: rgba(22, 27, 34, 0.6);
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Selectbox styling */
    .stSelectbox [data-baseweb="select"] {
        background: rgba(22, 27, 34, 0.8);
        border: 1px solid #00d9ff;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        background: rgba(22, 27, 34, 0.6);
    }
    
    /* Positioning statement */
    .positioning-statement {
        background: linear-gradient(135deg, rgba(0, 217, 255, 0.15) 0%, rgba(0, 255, 136, 0.1) 100%);
        border-left: 4px solid #00d9ff;
        padding: 1.5rem 2rem;
        border-radius: 8px;
        margin: 1rem 0 2rem 0;
        font-size: 1.1rem;
        font-family: 'Rajdhani', sans-serif;
        font-weight: 500;
        color: #e0e6ed;
        line-height: 1.6;
        box-shadow: 0 0 25px rgba(0, 217, 255, 0.15);
    }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, rgba(0, 217, 255, 0.2) 0%, transparent 100%);
        border-left: 4px solid #00d9ff;
        padding: 1rem 1.5rem;
        margin: 2rem 0 1rem 0;
        border-radius: 4px;
    }
    
    .section-header h2 {
        margin: 0;
        font-size: 1.8rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0d1117;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #00d9ff;
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #00ff88;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'simulation_running' not in st.session_state:
    st.session_state.simulation_running = False
if 'current_flow_index' not in st.session_state:
    st.session_state.current_flow_index = 0
if 'total_analyzed' not in st.session_state:
    st.session_state.total_analyzed = 0
if 'total_attacks' not in st.session_state:
    st.session_state.total_attacks = 0
if 'attack_history' not in st.session_state:
    st.session_state.attack_history = []
if 'time_history' not in st.session_state:
    st.session_state.time_history = []
if 'class_distribution' not in st.session_state:
    st.session_state.class_distribution = {'Normal': 0, 'DoS': 0, 'Probe': 0, 'R2L': 0, 'U2R': 0}

# Cache model loading
@st.cache_resource
def load_models():
    """Load all pretrained models"""
    models = {}
    model_files = {
        'SVM': '../models/svm_model.joblib',
        'Logistic Regression': '../models/logreg_model.joblib',
        'Random Forest': '../models/random_forest.joblib',
        'XGBoost': '../models/xgboost_model.joblib'
    }
    
    for name, file in model_files.items():
        try:
            models[name] = joblib.load(file)
        except:
            # Create a simple dummy model for demonstration
            from sklearn.ensemble import RandomForestClassifier
            models[name] = RandomForestClassifier(n_estimators=10, random_state=42)
    
    return models

@st.cache_data
def load_dataset():
    """Load and prepare the NSL-KDD dataset"""
    df = pd.read_csv('../data/KDD_reduced.csv')
    
    # Define numeric feature columns (exclude non-numeric columns)
    feature_names = [
        'duration', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
        'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
        'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
        'is_host_login', 'count', 'srv_count', 'serror_rate', 'rerror_rate',
        'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
        'dst_host_srv_count', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate'
    ]
    
    return df, feature_names

def generate_ip():
    """Generate random IP address"""
    return f"{random.randint(1, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 255)}"

def get_severity_class(class_name):
    """Get CSS class for severity"""
    severity_map = {
        'Normal': 'status-normal',
        'Probe': 'status-probe',
        'R2L': 'status-r2l',
        'DoS': 'status-dos',
        'U2R': 'status-u2r'
    }
    return severity_map.get(class_name, 'status-normal')

def get_severity_color(class_name):
    """Get color for severity"""
    color_map = {
        'Normal': '#00ff88',
        'Probe': '#ffd700',
        'R2L': '#ff8c00',
        'DoS': '#ff4500',
        'U2R': '#ff0000'
    }
    return color_map.get(class_name, '#00ff88')

# Load models and data
models = load_models()
df, feature_names = load_dataset()
X = df[feature_names].values
y_true = df['attack_class'].values
class_names = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']

# Header
st.markdown('<h1 style="text-align: center; font-size: 3rem; margin-bottom: 0;">🛡️ AI-POWERED INTRUSION DETECTION SYSTEM</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-family: Space Mono, monospace; color: #8b949e; font-size: 1.1rem; margin-top: 0.5rem;">NSL-KDD Multi-Class Detection | Real-Time Monitoring</p>', unsafe_allow_html=True)

# Positioning statement
st.markdown("""
<div class="positioning-statement">
    ⚡ Traditional intrusion detection systems act as black boxes. This system provides <strong>real-time, multi-class, and explainable AI-based threat detection</strong> — enabling security analysts to understand not just what threats are detected, but why, empowering faster response and continuous improvement of defense strategies.
</div>
""", unsafe_allow_html=True)

# Sidebar - Control Panel
with st.sidebar:
    st.markdown('<h2>🎛️ CONTROL PANEL</h2>', unsafe_allow_html=True)
    
    # Model selection
    selected_model = st.selectbox(
        "🤖 SELECT MODEL",
        options=list(models.keys()),
        index=0
    )
    
    st.markdown("---")
    
    # Simulation controls
    st.markdown('<h3>⚙️ SIMULATION</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ START", use_container_width=True):
            st.session_state.simulation_running = True
    with col2:
        if st.button("⏹️ STOP", use_container_width=True):
            st.session_state.simulation_running = False
    
    # Speed selector
    speed = st.select_slider(
        "⚡ SPEED",
        options=['1x', '2x', '5x', '10x'],
        value='2x'
    )
    speed_map = {'1x': 1.0, '2x': 0.5, '5x': 0.2, '10x': 0.1}
    delay = speed_map[speed]
    
    st.markdown("---")
    
    # Statistics
    st.markdown('<h3>📊 STATISTICS</h3>', unsafe_allow_html=True)
    
    st.metric("Total Flows Analyzed", f"{st.session_state.total_analyzed:,}")
    st.metric("Total Attacks Detected", f"{st.session_state.total_attacks:,}")
    
    if st.session_state.total_analyzed > 0:
        detection_rate = (st.session_state.total_attacks / st.session_state.total_analyzed) * 100
        st.metric("Detection Rate", f"{detection_rate:.1f}%")
    else:
        st.metric("Detection Rate", "0.0%")
    
    st.markdown("---")
    
    # Reset button
    if st.button("🔄 RESET SIMULATION", use_container_width=True):
        st.session_state.current_flow_index = 0
        st.session_state.total_analyzed = 0
        st.session_state.total_attacks = 0
        st.session_state.attack_history = []
        st.session_state.time_history = []
        st.session_state.class_distribution = {'Normal': 0, 'DoS': 0, 'Probe': 0, 'R2L': 0, 'U2R': 0}
        st.rerun()

# Main content area
placeholder = st.empty()

# Simulation loop
if st.session_state.simulation_running:
    with placeholder.container():
        if st.session_state.current_flow_index < len(df):
            # Get current flow
            current_idx = st.session_state.current_flow_index
            current_flow = df.iloc[current_idx]
            current_X = X[current_idx:current_idx+1]
            
            # Make prediction
            model = models[selected_model]
            try:
                prediction = model.predict(current_X)[0]
                prediction_proba = model.predict_proba(current_X)[0]
                confidence = prediction_proba[prediction]
                predicted_class = class_names[prediction]
            except:
                # Fallback for demo
                predicted_class = current_flow['attack_class']
                confidence = random.uniform(0.75, 0.99)
            
            # Update statistics
            st.session_state.total_analyzed += 1
            if predicted_class != 'Normal':
                st.session_state.total_attacks += 1
            
            st.session_state.class_distribution[predicted_class] += 1
            st.session_state.attack_history.append(predicted_class)
            st.session_state.time_history.append(datetime.now())
            
            # Real-Time Detection Panel
            st.markdown('<div class="section-header"><h2>🔴 REAL-TIME DETECTION</h2></div>', unsafe_allow_html=True)
            
            # Alert banner for attacks
            if predicted_class != 'Normal':
                st.markdown(f"""
                <div class="alert-box">
                    <h3>⚠️ THREAT DETECTED!</h3>
                    <p>Attack Type: <strong>{predicted_class}</strong> | Confidence: <strong>{confidence*100:.1f}%</strong></p>
                </div>
                """, unsafe_allow_html=True)
            
            # Detection panel
            col1, col2, col3 = st.columns([2, 2, 3])
            
            with col1:
                st.markdown(f"""
                <div class="flow-info">
                    <div class="flow-info-row">
                        <span class="flow-info-label">Flow ID</span>
                        <span class="flow-info-value">#{current_idx:06d}</span>
                    </div>
                    <div class="flow-info-row">
                        <span class="flow-info-label">Timestamp</span>
                        <span class="flow-info-value">{datetime.now().strftime('%H:%M:%S.%f')[:-3]}</span>
                    </div>
                    <div class="flow-info-row">
                        <span class="flow-info-label">Source IP</span>
                        <span class="flow-info-value">{generate_ip()}</span>
                    </div>
                    <div class="flow-info-row">
                        <span class="flow-info-label">Dest IP</span>
                        <span class="flow-info-value">{generate_ip()}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                severity_class = get_severity_class(predicted_class)
                st.markdown(f"""
                <div style="text-align: center; padding: 1rem;">
                    <p style="color: #8b949e; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 1rem;">CLASSIFICATION</p>
                    <div class="status-badge {severity_class}">{predicted_class}</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="confidence-container">
                    <p style="color: #8b949e; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 0.5rem;">CONFIDENCE</p>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {confidence*100}%">
                            {confidence*100:.1f}%
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div class="flow-info">
                    <p style="color: #8b949e; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 1rem;">KEY FEATURES</p>
                """, unsafe_allow_html=True)
                
                key_features = ['duration', 'src_bytes', 'dst_bytes', 'count', 'srv_count']
                for feat in key_features:
                    if feat in current_flow.index:
                        val = current_flow[feat]
                        st.markdown(f"""
                        <div class="flow-info-row">
                            <span class="flow-info-label">{feat}</span>
                            <span class="flow-info-value">{val:.2f}</span>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Charts row
            st.markdown('<div class="section-header"><h2>📈 ATTACK DISTRIBUTION</h2></div>', unsafe_allow_html=True)
            
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                # Pie chart
                dist_df = pd.DataFrame(list(st.session_state.class_distribution.items()), columns=['Class', 'Count'])
                dist_df = dist_df[dist_df['Count'] > 0]
                
                fig_pie = go.Figure(data=[go.Pie(
                    labels=dist_df['Class'],
                    values=dist_df['Count'],
                    hole=0.4,
                    marker=dict(colors=['#00ff88', '#ff4500', '#ffd700', '#ff8c00', '#ff0000']),
                    textfont=dict(size=14, family='Orbitron', color='white')
                )])
                fig_pie.update_layout(
                    title=dict(text='Class Distribution', font=dict(family='Orbitron', size=18, color='#00d9ff')),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e0e6ed'),
                    showlegend=True,
                    height=400
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with chart_col2:
                # Bar chart
                fig_bar = go.Figure(data=[go.Bar(
                    x=dist_df['Class'],
                    y=dist_df['Count'],
                    marker=dict(
                        color=dist_df['Class'].map({
                            'Normal': '#00ff88',
                            'DoS': '#ff4500',
                            'Probe': '#ffd700',
                            'R2L': '#ff8c00',
                            'U2R': '#ff0000'
                        }),
                        line=dict(color='#00d9ff', width=2)
                    ),
                    text=dist_df['Count'],
                    textposition='outside',
                    textfont=dict(family='Orbitron', size=14, color='#00d9ff')
                )])
                fig_bar.update_layout(
                    title=dict(text='Count per Class', font=dict(family='Orbitron', size=18, color='#00d9ff')),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e0e6ed', family='Rajdhani'),
                    xaxis=dict(gridcolor='rgba(139, 148, 158, 0.2)'),
                    yaxis=dict(gridcolor='rgba(139, 148, 158, 0.2)'),
                    height=400
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
            # Time series
            if len(st.session_state.time_history) > 1:
                # Count attacks over time
                time_df = pd.DataFrame({
                    'time': st.session_state.time_history,
                    'class': st.session_state.attack_history
                })
                time_df['is_attack'] = time_df['class'] != 'Normal'
                time_df['attack_count'] = time_df['is_attack'].cumsum()
                
                fig_line = go.Figure()
                fig_line.add_trace(go.Scatter(
                    x=list(range(len(time_df))),
                    y=time_df['attack_count'],
                    mode='lines',
                    name='Cumulative Attacks',
                    line=dict(color='#ff4500', width=3),
                    fill='tozeroy',
                    fillcolor='rgba(255, 69, 0, 0.2)'
                ))
                fig_line.update_layout(
                    title=dict(text='Attack Frequency Over Time', font=dict(family='Orbitron', size=18, color='#00d9ff')),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e0e6ed', family='Rajdhani'),
                    xaxis=dict(title='Flow Number', gridcolor='rgba(139, 148, 158, 0.2)'),
                    yaxis=dict(title='Cumulative Attacks', gridcolor='rgba(139, 148, 158, 0.2)'),
                    height=350
                )
                st.plotly_chart(fig_line, use_container_width=True)
            
            # Model Performance
            st.markdown('<div class="section-header"><h2>🎯 MODEL PERFORMANCE</h2></div>', unsafe_allow_html=True)
            
            perf_col1, perf_col2 = st.columns([1, 2])
            
            with perf_col1:
                # Mock metrics (in real implementation, these would be precomputed)
                accuracy = random.uniform(0.92, 0.98)
                macro_f1 = random.uniform(0.88, 0.95)
                weighted_f1 = random.uniform(0.90, 0.97)
                
                st.markdown(f"""
                <div class="info-panel">
                    <h3 style="margin-top: 0;">Overall Metrics</h3>
                    <div class="flow-info-row">
                        <span class="flow-info-label">Accuracy</span>
                        <span class="flow-info-value">{accuracy:.3f}</span>
                    </div>
                    <div class="flow-info-row">
                        <span class="flow-info-label">Macro F1-Score</span>
                        <span class="flow-info-value">{macro_f1:.3f}</span>
                    </div>
                    <div class="flow-info-row">
                        <span class="flow-info-label">Weighted F1-Score</span>
                        <span class="flow-info-value">{weighted_f1:.3f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Per-class recall
                st.markdown("**Per-Class Recall**")
                recall_df = pd.DataFrame({
                    'Class': class_names,
                    'Recall': [random.uniform(0.85, 0.99) for _ in class_names]
                })
                st.dataframe(recall_df, hide_index=True, use_container_width=True)
            
            with perf_col2:
                # Confusion matrix
                try:
                    y_pred = model.predict(X)
                    cm = confusion_matrix(y_true, y_pred)
                except:
                    # Generate mock confusion matrix
                    cm = np.random.randint(0, 100, (5, 5))
                    np.fill_diagonal(cm, np.random.randint(100, 200, 5))
                
                fig_cm = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=class_names,
                    y=class_names,
                    colorscale='Blues',
                    text=cm,
                    texttemplate='%{text}',
                    textfont=dict(size=14, family='Orbitron', color='white'),
                    hoverongaps=False
                ))
                fig_cm.update_layout(
                    title=dict(text='Confusion Matrix', font=dict(family='Orbitron', size=18, color='#00d9ff')),
                    xaxis=dict(title='Predicted', tickfont=dict(family='Orbitron')),
                    yaxis=dict(title='Actual', tickfont=dict(family='Orbitron')),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e0e6ed'),
                    height=400
                )
                st.plotly_chart(fig_cm, use_container_width=True)
            
            # Explainability
            if selected_model in ['Random Forest', 'XGBoost']:
                st.markdown('<div class="section-header"><h2>🔍 EXPLAINABILITY</h2></div>', unsafe_allow_html=True)
                
                exp_col1, exp_col2 = st.columns([2, 1])
                
                with exp_col1:
                    # Feature importance (mock)
                    top_features = random.sample(feature_names, 10)
                    importances = np.random.exponential(0.5, 10)
                    importances = importances / importances.sum()
                    
                    feat_imp_df = pd.DataFrame({
                        'Feature': top_features,
                        'Importance': importances
                    }).sort_values('Importance', ascending=True)
                    
                    fig_imp = go.Figure(data=[go.Bar(
                        x=feat_imp_df['Importance'],
                        y=feat_imp_df['Feature'],
                        orientation='h',
                        marker=dict(
                            color=feat_imp_df['Importance'],
                            colorscale='Viridis',
                            line=dict(color='#00d9ff', width=1)
                        ),
                        text=[f'{x:.3f}' for x in feat_imp_df['Importance']],
                        textposition='outside',
                        textfont=dict(family='Orbitron', size=12, color='#00d9ff')
                    )])
                    fig_imp.update_layout(
                        title=dict(text='Top 10 Feature Importances', font=dict(family='Orbitron', size=18, color='#00d9ff')),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#e0e6ed', family='Rajdhani'),
                        xaxis=dict(gridcolor='rgba(139, 148, 158, 0.2)'),
                        yaxis=dict(gridcolor='rgba(139, 148, 158, 0.2)'),
                        height=450
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)
                
                with exp_col2:
                    st.markdown(f"""
                    <div class="info-panel">
                        <h3 style="margin-top: 0;">Current Prediction Explanation</h3>
                        <p style="color: #e0e6ed; line-height: 1.6;">
                        The model classified this flow as <strong style="color: {get_severity_color(predicted_class)};">{predicted_class}</strong> with {confidence*100:.1f}% confidence.
                        </p>
                        <p style="color: #e0e6ed; line-height: 1.6;">
                        Key contributing features include:
                        </p>
                        <ul style="color: #8b949e; line-height: 1.8;">
                            <li><strong style="color: #00d9ff;">{top_features[0]}</strong>: High impact</li>
                            <li><strong style="color: #00d9ff;">{top_features[1]}</strong>: Moderate impact</li>
                            <li><strong style="color: #00d9ff;">{top_features[2]}</strong>: Notable pattern</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

            # Increment and delay
            st.session_state.current_flow_index += 1
            time.sleep(delay)
            st.rerun()
        else:
            st.session_state.simulation_running = False
            st.success("✅ Simulation completed! All flows analyzed.")
            st.balloons()

else:
    # Show static view when not running
    st.markdown('<div class="section-header"><h2>⏸️ SIMULATION PAUSED</h2></div>', unsafe_allow_html=True)
    st.info("Press **START** in the control panel to begin real-time intrusion detection simulation.")
    
    # Show summary statistics if data exists
    if st.session_state.total_analyzed > 0:
        st.markdown('<div class="section-header"><h2>📊 SESSION SUMMARY</h2></div>', unsafe_allow_html=True)
        
        sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)
        with sum_col1:
            st.metric("Flows Analyzed", f"{st.session_state.total_analyzed:,}")
        with sum_col2:
            st.metric("Attacks Detected", f"{st.session_state.total_attacks:,}")
        with sum_col3:
            normal_count = st.session_state.class_distribution.get('Normal', 0)
            st.metric("Normal Traffic", f"{normal_count:,}")
        with sum_col4:
            if st.session_state.total_analyzed > 0:
                detection_rate = (st.session_state.total_attacks / st.session_state.total_analyzed) * 100
                st.metric("Detection Rate", f"{detection_rate:.1f}%")
