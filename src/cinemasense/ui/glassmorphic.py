"""
Apple VisionOS-inspired Glassmorphic UI Components
Premium, futuristic, calm design system

Requirements: 11.1, 11.2, 11.3, 11.4, 11.5
"""

import streamlit as st
from typing import Optional, List, Dict, Any
import base64
from pathlib import Path


def inject_glassmorphic_css():
    """
    Inject premium glassmorphic CSS styles.
    
    Requirements:
    - 11.1: Inject CSS styles for glass card effects, gradients, and animations
    - 11.2: Style all Streamlit components with glassmorphic theme
    - 11.3: Provide smooth transitions and micro-interactions
    - 11.4: Use consistent color palette with accent gradients (cyan to purple)
    - 11.5: Hide default Streamlit branding elements
    """
    css = """
    <style>
    /* ===== GLASSMORPHIC DESIGN SYSTEM v2.0 ===== */
    /* Apple VisionOS-inspired premium UI */
    
    /* ===== ROOT CSS VARIABLES ===== */
    :root {
        /* Glass effects */
        --glass-bg: rgba(255, 255, 255, 0.05);
        --glass-bg-hover: rgba(255, 255, 255, 0.08);
        --glass-bg-active: rgba(255, 255, 255, 0.12);
        --glass-border: rgba(255, 255, 255, 0.1);
        --glass-border-hover: rgba(255, 255, 255, 0.2);
        --glass-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        --glass-shadow-hover: 0 12px 40px rgba(0, 0, 0, 0.4);
        --glass-blur: blur(20px);
        --glass-blur-heavy: blur(40px);

        /* Accent colors - cyan to purple gradient */
        --accent-primary: #00D4FF;
        --accent-secondary: #7B61FF;
        --accent-tertiary: #FF6B6B;
        --accent-gradient: linear-gradient(135deg, #00D4FF 0%, #7B61FF 100%);
        --accent-gradient-hover: linear-gradient(135deg, #00E5FF 0%, #8B71FF 100%);
        --accent-gradient-active: linear-gradient(135deg, #00C4EF 0%, #6B51EF 100%);
        --accent-gradient-full: linear-gradient(135deg, #00D4FF 0%, #7B61FF 50%, #FF6B6B 100%);
        
        /* Text colors */
        --text-primary: rgba(255, 255, 255, 0.95);
        --text-secondary: rgba(255, 255, 255, 0.7);
        --text-muted: rgba(255, 255, 255, 0.5);
        --text-disabled: rgba(255, 255, 255, 0.3);
        
        /* Status colors */
        --success: #00E676;
        --success-bg: rgba(0, 230, 118, 0.15);
        --warning: #FFD600;
        --warning-bg: rgba(255, 214, 0, 0.15);
        --error: #FF5252;
        --error-bg: rgba(255, 82, 82, 0.15);
        --info: #00D4FF;
        --info-bg: rgba(0, 212, 255, 0.15);
        
        /* Spacing */
        --spacing-xs: 4px;
        --spacing-sm: 8px;
        --spacing-md: 16px;
        --spacing-lg: 24px;
        --spacing-xl: 32px;
        --spacing-2xl: 48px;
        
        /* Border radius */
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
        --radius-xl: 24px;
        --radius-full: 9999px;
        
        /* Transitions */
        --transition-fast: all 0.15s cubic-bezier(0.4, 0, 0.2, 1);
        --transition-smooth: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        --transition-slow: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
        --transition-bounce: all 0.4s cubic-bezier(0.68, -0.55, 0.265, 1.55);
        
        /* Z-index layers */
        --z-base: 1;
        --z-dropdown: 100;
        --z-sticky: 200;
        --z-modal: 300;
        --z-tooltip: 400;
        --z-toast: 500;
    }

    /* ===== DARK THEME BACKGROUND ===== */
    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 50%, #0f0f1a 100%);
        background-attachment: fixed;
    }
    
    /* Animated background gradient */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: 
            radial-gradient(ellipse at 20% 20%, rgba(0, 212, 255, 0.08) 0%, transparent 50%),
            radial-gradient(ellipse at 80% 80%, rgba(123, 97, 255, 0.08) 0%, transparent 50%);
        pointer-events: none;
        z-index: 0;
    }
    
    /* ===== MAIN CONTAINER ===== */
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
    }
    
    /* ===== GLASS CARD EFFECT ===== */
    .glass-card {
        background: var(--glass-bg);
        backdrop-filter: var(--glass-blur);
        -webkit-backdrop-filter: var(--glass-blur);
        border: 1px solid var(--glass-border);
        border-radius: var(--radius-xl);
        padding: var(--spacing-lg);
        box-shadow: var(--glass-shadow);
        transition: var(--transition-smooth);
        position: relative;
        overflow: hidden;
    }
    
    .glass-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
    }
    
    .glass-card:hover {
        background: var(--glass-bg-hover);
        border-color: var(--glass-border-hover);
        transform: translateY(-2px);
        box-shadow: var(--glass-shadow-hover);
    }

    /* ===== BUTTON STYLING ===== */
    .stButton > button {
        background: var(--accent-gradient) !important;
        color: white !important;
        border: none !important;
        border-radius: var(--radius-md) !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        letter-spacing: 0.5px !important;
        transition: var(--transition-smooth) !important;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3) !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: var(--transition-smooth);
    }
    
    .stButton > button:hover {
        background: var(--accent-gradient-hover) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(0, 212, 255, 0.4) !important;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:active {
        transform: translateY(0) scale(0.98) !important;
        box-shadow: 0 2px 10px rgba(0, 212, 255, 0.3) !important;
    }
    
    .stButton > button:focus {
        outline: none !important;
        box-shadow: 0 0 0 3px rgba(0, 212, 255, 0.3), 0 4px 15px rgba(0, 212, 255, 0.3) !important;
    }
    
    /* Secondary button style */
    .stButton > button[kind="secondary"] {
        background: var(--glass-bg) !important;
        border: 1px solid var(--glass-border) !important;
        box-shadow: none !important;
    }
    
    .stButton > button[kind="secondary"]:hover {
        background: var(--glass-bg-hover) !important;
        border-color: var(--accent-primary) !important;
    }

    /* ===== METRICS STYLING ===== */
    [data-testid="stMetric"] {
        background: var(--glass-bg);
        backdrop-filter: var(--glass-blur);
        -webkit-backdrop-filter: var(--glass-blur);
        border: 1px solid var(--glass-border);
        border-radius: var(--radius-lg);
        padding: 20px;
        transition: var(--transition-smooth);
        position: relative;
        overflow: hidden;
    }
    
    [data-testid="stMetric"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: var(--accent-gradient);
        opacity: 0;
        transition: var(--transition-smooth);
    }
    
    [data-testid="stMetric"]:hover {
        border-color: var(--accent-primary);
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.2);
        transform: translateY(-2px);
    }
    
    [data-testid="stMetric"]:hover::before {
        opacity: 1;
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--text-secondary) !important;
        font-size: 12px !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-size: 28px !important;
        font-weight: 700;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 14px !important;
    }
    
    [data-testid="stMetricDelta"] svg {
        width: 14px;
        height: 14px;
    }

    /* ===== INPUT FIELDS ===== */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: var(--glass-bg) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: var(--radius-md) !important;
        color: var(--text-primary) !important;
        padding: 12px 16px !important;
        transition: var(--transition-smooth) !important;
        font-size: 14px !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: var(--accent-primary) !important;
        box-shadow: 0 0 0 3px rgba(0, 212, 255, 0.15), 0 0 20px rgba(0, 212, 255, 0.1) !important;
        background: var(--glass-bg-hover) !important;
    }
    
    .stTextInput > div > div > input::placeholder,
    .stTextArea > div > div > textarea::placeholder {
        color: var(--text-muted) !important;
    }
    
    /* ===== SELECT BOX ===== */
    .stSelectbox > div > div {
        background: var(--glass-bg) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: var(--radius-md) !important;
        transition: var(--transition-smooth) !important;
    }
    
    .stSelectbox > div > div:hover {
        border-color: var(--glass-border-hover) !important;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: var(--accent-primary) !important;
        box-shadow: 0 0 0 3px rgba(0, 212, 255, 0.15) !important;
    }
    
    .stSelectbox [data-baseweb="select"] > div {
        background: transparent !important;
        color: var(--text-primary) !important;
    }

    /* ===== SLIDER STYLING ===== */
    .stSlider > div > div > div {
        background: var(--glass-bg) !important;
        border-radius: var(--radius-full) !important;
    }
    
    .stSlider > div > div > div > div {
        background: var(--accent-gradient) !important;
        border-radius: var(--radius-full) !important;
    }
    
    .stSlider > div > div > div > div > div {
        background: white !important;
        border: 2px solid var(--accent-primary) !important;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3), 0 0 10px rgba(0, 212, 255, 0.3) !important;
        transition: var(--transition-smooth) !important;
        width: 18px !important;
        height: 18px !important;
    }
    
    .stSlider > div > div > div > div > div:hover {
        transform: scale(1.2) !important;
        box-shadow: 0 2px 15px rgba(0, 0, 0, 0.4), 0 0 15px rgba(0, 212, 255, 0.5) !important;
    }
    
    .stSlider > div > div > div > div > div:active {
        transform: scale(1.1) !important;
    }
    
    /* Slider labels */
    .stSlider label {
        color: var(--text-secondary) !important;
    }
    
    .stSlider [data-testid="stTickBarMin"],
    .stSlider [data-testid="stTickBarMax"] {
        color: var(--text-muted) !important;
    }

    /* ===== PROGRESS BAR ===== */
    .stProgress > div > div > div {
        background: var(--glass-bg) !important;
        border-radius: var(--radius-full) !important;
        overflow: hidden;
    }
    
    .stProgress > div > div > div > div {
        background: var(--accent-gradient) !important;
        border-radius: var(--radius-full) !important;
        position: relative;
        overflow: hidden;
    }
    
    .stProgress > div > div > div > div::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        animation: progressShimmer 2s infinite;
    }
    
    @keyframes progressShimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    /* ===== SPINNER / LOADING ===== */
    .stSpinner > div {
        border-color: var(--accent-primary) transparent transparent transparent !important;
    }
    
    /* Custom loading animation */
    .loading-pulse {
        animation: loadingPulse 1.5s ease-in-out infinite;
    }
    
    @keyframes loadingPulse {
        0%, 100% { opacity: 0.4; transform: scale(0.95); }
        50% { opacity: 1; transform: scale(1); }
    }

    /* ===== TABS STYLING ===== */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--glass-bg);
        border-radius: var(--radius-md);
        padding: 4px;
        gap: 4px;
        border: 1px solid var(--glass-border);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: var(--radius-sm);
        color: var(--text-secondary);
        padding: 10px 20px;
        transition: var(--transition-smooth);
        font-weight: 500;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: var(--glass-bg-hover);
        color: var(--text-primary);
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--accent-gradient) !important;
        color: white !important;
        box-shadow: 0 2px 10px rgba(0, 212, 255, 0.3);
    }
    
    .stTabs [data-baseweb="tab-highlight"] {
        display: none;
    }
    
    .stTabs [data-baseweb="tab-border"] {
        display: none;
    }
    
    /* ===== EXPANDER STYLING ===== */
    .streamlit-expanderHeader {
        background: var(--glass-bg) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: var(--radius-md) !important;
        color: var(--text-primary) !important;
        font-weight: 500;
        transition: var(--transition-smooth);
        padding: 12px 16px !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: var(--glass-bg-hover) !important;
        border-color: var(--glass-border-hover) !important;
    }
    
    .streamlit-expanderContent {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid var(--glass-border) !important;
        border-top: none !important;
        border-radius: 0 0 var(--radius-md) var(--radius-md) !important;
        padding: 16px !important;
    }

    /* ===== FILE UPLOADER ===== */
    [data-testid="stFileUploader"] {
        background: var(--glass-bg);
        border: 2px dashed var(--glass-border);
        border-radius: var(--radius-lg);
        padding: 40px;
        transition: var(--transition-smooth);
        position: relative;
        overflow: hidden;
    }
    
    [data-testid="stFileUploader"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: var(--accent-gradient);
        opacity: 0;
        transition: var(--transition-smooth);
        z-index: 0;
        pointer-events: none;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: var(--accent-primary);
        background: rgba(0, 212, 255, 0.05);
    }
    
    [data-testid="stFileUploader"]:hover::before {
        opacity: 0.05;
    }
    
    [data-testid="stFileUploader"] > div {
        position: relative;
        z-index: 1;
    }
    
    /* File uploader button */
    [data-testid="stFileUploader"] button {
        background: var(--glass-bg) !important;
        border: 1px solid var(--glass-border) !important;
        color: var(--text-primary) !important;
        position: relative !important;
        z-index: 10 !important;
        cursor: pointer !important;
    }
    
    [data-testid="stFileUploader"] button:hover {
        background: var(--glass-bg-hover) !important;
        border-color: var(--accent-primary) !important;
    }
    
    /* Ensure file input is clickable */
    [data-testid="stFileUploader"] input[type="file"] {
        cursor: pointer !important;
    }
    
    [data-testid="stFileUploader"] section {
        position: relative;
        z-index: 5;
    }
    
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {
        cursor: pointer !important;
    }

    /* ===== SIDEBAR STYLING ===== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(10, 10, 15, 0.98) 0%, rgba(26, 26, 46, 0.98) 100%);
        backdrop-filter: var(--glass-blur-heavy);
        -webkit-backdrop-filter: var(--glass-blur-heavy);
        border-right: 1px solid var(--glass-border);
    }
    
    [data-testid="stSidebar"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(ellipse at 50% 0%, rgba(0, 212, 255, 0.05) 0%, transparent 70%);
        pointer-events: none;
    }
    
    [data-testid="stSidebar"] .block-container {
        padding: 2rem 1rem;
    }
    
    /* Sidebar radio buttons */
    [data-testid="stSidebar"] .stRadio > div {
        gap: 4px;
    }
    
    [data-testid="stSidebar"] .stRadio label {
        background: var(--glass-bg);
        border: 1px solid transparent;
        border-radius: var(--radius-sm);
        padding: 10px 16px;
        margin: 2px 0;
        transition: var(--transition-smooth);
        cursor: pointer;
    }
    
    [data-testid="stSidebar"] .stRadio label:hover {
        background: var(--glass-bg-hover);
        border-color: var(--glass-border);
    }
    
    [data-testid="stSidebar"] .stRadio label[data-checked="true"] {
        background: var(--accent-gradient);
        border-color: transparent;
        box-shadow: 0 2px 10px rgba(0, 212, 255, 0.3);
    }

    /* ===== ALERT BOXES ===== */
    .stAlert {
        background: var(--glass-bg) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: var(--radius-md) !important;
        backdrop-filter: var(--glass-blur);
        -webkit-backdrop-filter: var(--glass-blur);
        padding: 16px !important;
    }
    
    /* Success message */
    .stSuccess, [data-testid="stNotification"][data-type="success"] {
        border-left: 4px solid var(--success) !important;
        background: var(--success-bg) !important;
    }
    
    /* Warning message */
    .stWarning, [data-testid="stNotification"][data-type="warning"] {
        border-left: 4px solid var(--warning) !important;
        background: var(--warning-bg) !important;
    }
    
    /* Error message */
    .stError, [data-testid="stNotification"][data-type="error"] {
        border-left: 4px solid var(--error) !important;
        background: var(--error-bg) !important;
    }
    
    /* Info message */
    .stInfo, [data-testid="stNotification"][data-type="info"] {
        border-left: 4px solid var(--info) !important;
        background: var(--info-bg) !important;
    }
    
    /* ===== DATAFRAME STYLING ===== */
    .stDataFrame {
        background: var(--glass-bg);
        border: 1px solid var(--glass-border);
        border-radius: var(--radius-md);
        overflow: hidden;
    }
    
    .stDataFrame [data-testid="stDataFrameResizable"] {
        background: transparent;
    }
    
    /* Table headers */
    .stDataFrame th {
        background: var(--glass-bg-hover) !important;
        color: var(--text-primary) !important;
        font-weight: 600;
        border-bottom: 1px solid var(--glass-border) !important;
    }
    
    /* Table cells */
    .stDataFrame td {
        color: var(--text-secondary) !important;
        border-bottom: 1px solid var(--glass-border) !important;
    }
    
    .stDataFrame tr:hover td {
        background: var(--glass-bg-hover) !important;
    }

    /* ===== CHECKBOX & RADIO ===== */
    .stCheckbox label,
    .stRadio label {
        color: var(--text-secondary) !important;
        transition: var(--transition-smooth);
    }
    
    .stCheckbox label:hover,
    .stRadio label:hover {
        color: var(--text-primary) !important;
    }
    
    .stCheckbox [data-testid="stCheckbox"] > div:first-child {
        background: var(--glass-bg) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 4px !important;
        transition: var(--transition-smooth);
    }
    
    .stCheckbox [data-testid="stCheckbox"] > div:first-child:hover {
        border-color: var(--accent-primary) !important;
    }
    
    .stCheckbox [data-testid="stCheckbox"][aria-checked="true"] > div:first-child {
        background: var(--accent-gradient) !important;
        border-color: transparent !important;
    }
    
    /* ===== DOWNLOAD BUTTON ===== */
    .stDownloadButton > button {
        background: var(--glass-bg) !important;
        border: 1px solid var(--glass-border) !important;
        color: var(--text-primary) !important;
        border-radius: var(--radius-md) !important;
        transition: var(--transition-smooth) !important;
    }
    
    .stDownloadButton > button:hover {
        background: var(--glass-bg-hover) !important;
        border-color: var(--accent-primary) !important;
        box-shadow: 0 0 15px rgba(0, 212, 255, 0.2) !important;
    }
    
    .stDownloadButton > button:active {
        transform: scale(0.98) !important;
    }

    /* ===== TYPOGRAPHY ===== */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary) !important;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    h1 {
        font-size: 2.5rem !important;
        background: var(--accent-gradient-full);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem !important;
    }
    
    h2 {
        font-size: 1.75rem !important;
        color: var(--text-primary) !important;
    }
    
    h3 {
        font-size: 1.25rem !important;
        color: var(--text-primary) !important;
    }
    
    p, span, label, div {
        color: var(--text-secondary);
    }
    
    a {
        color: var(--accent-primary) !important;
        text-decoration: none;
        transition: var(--transition-smooth);
    }
    
    a:hover {
        color: var(--accent-secondary) !important;
        text-decoration: underline;
    }
    
    /* Code blocks */
    code {
        background: var(--glass-bg) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: var(--radius-sm) !important;
        padding: 2px 6px !important;
        color: var(--accent-primary) !important;
        font-family: 'SF Mono', 'Fira Code', monospace !important;
    }
    
    pre {
        background: var(--glass-bg) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: var(--radius-md) !important;
        padding: 16px !important;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: var(--glass-border);
        margin: 2rem 0;
    }

    /* ===== SCROLLBAR STYLING ===== */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 4px;
        transition: var(--transition-smooth);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.3);
    }
    
    /* Firefox scrollbar */
    * {
        scrollbar-width: thin;
        scrollbar-color: rgba(255, 255, 255, 0.2) rgba(255, 255, 255, 0.05);
    }
    
    /* ===== ANIMATIONS ===== */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes fadeInLeft {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes fadeInRight {
        from { opacity: 0; transform: translateX(20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }
    
    @keyframes spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    
    @keyframes glow {
        0%, 100% { box-shadow: 0 0 5px rgba(0, 212, 255, 0.3); }
        50% { box-shadow: 0 0 20px rgba(0, 212, 255, 0.6); }
    }

    /* Animation classes */
    .animate-fade-in {
        animation: fadeIn 0.5s ease-out;
    }
    
    .animate-fade-in-up {
        animation: fadeInUp 0.5s ease-out;
    }
    
    .animate-fade-in-down {
        animation: fadeInDown 0.5s ease-out;
    }
    
    .animate-fade-in-left {
        animation: fadeInLeft 0.5s ease-out;
    }
    
    .animate-fade-in-right {
        animation: fadeInRight 0.5s ease-out;
    }
    
    .animate-pulse {
        animation: pulse 2s infinite;
    }
    
    .animate-spin {
        animation: spin 1s linear infinite;
    }
    
    .animate-bounce {
        animation: bounce 1s ease-in-out infinite;
    }
    
    .animate-glow {
        animation: glow 2s ease-in-out infinite;
    }
    
    /* Loading shimmer effect */
    .shimmer {
        background: linear-gradient(90deg, 
            rgba(255,255,255,0.05) 25%, 
            rgba(255,255,255,0.1) 50%, 
            rgba(255,255,255,0.05) 75%);
        background-size: 200% 100%;
        animation: shimmer 1.5s infinite;
    }
    
    /* Loading skeleton */
    .skeleton {
        background: var(--glass-bg);
        border-radius: var(--radius-md);
        position: relative;
        overflow: hidden;
    }
    
    .skeleton::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(90deg, 
            transparent, 
            rgba(255,255,255,0.1), 
            transparent);
        animation: shimmer 1.5s infinite;
    }

    /* ===== MICRO-INTERACTIONS ===== */
    .interactive {
        cursor: pointer;
        transition: var(--transition-smooth);
    }
    
    .interactive:hover {
        transform: scale(1.02);
    }
    
    .interactive:active {
        transform: scale(0.98);
    }
    
    /* Hover lift effect */
    .hover-lift {
        transition: var(--transition-smooth);
    }
    
    .hover-lift:hover {
        transform: translateY(-4px);
        box-shadow: var(--glass-shadow-hover);
    }
    
    /* Hover glow effect */
    .hover-glow {
        transition: var(--transition-smooth);
    }
    
    .hover-glow:hover {
        box-shadow: 0 0 30px rgba(0, 212, 255, 0.3);
    }
    
    /* Click ripple effect */
    .ripple {
        position: relative;
        overflow: hidden;
    }
    
    .ripple::after {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        background: rgba(255, 255, 255, 0.3);
        border-radius: 50%;
        transform: translate(-50%, -50%);
        transition: width 0.3s, height 0.3s;
    }
    
    .ripple:active::after {
        width: 200%;
        height: 200%;
    }
    
    /* ===== GLOW EFFECTS ===== */
    .glow-primary {
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
    }
    
    .glow-secondary {
        box-shadow: 0 0 20px rgba(123, 97, 255, 0.3);
    }
    
    .glow-success {
        box-shadow: 0 0 20px rgba(0, 230, 118, 0.3);
    }
    
    .glow-warning {
        box-shadow: 0 0 20px rgba(255, 214, 0, 0.3);
    }
    
    .glow-error {
        box-shadow: 0 0 20px rgba(255, 82, 82, 0.3);
    }

    /* ===== STATUS INDICATORS ===== */
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
    }
    
    .status-active {
        background: var(--success);
        box-shadow: 0 0 10px var(--success);
        animation: pulse 2s infinite;
    }
    
    .status-warning {
        background: var(--warning);
        box-shadow: 0 0 10px var(--warning);
    }
    
    .status-error {
        background: var(--error);
        box-shadow: 0 0 10px var(--error);
    }
    
    .status-info {
        background: var(--info);
        box-shadow: 0 0 10px var(--info);
    }
    
    /* ===== MEDIA STYLING ===== */
    video {
        border-radius: var(--radius-lg);
        box-shadow: var(--glass-shadow);
    }
    
    img {
        border-radius: var(--radius-md);
    }
    
    /* ===== TOOLTIP STYLING ===== */
    [data-testid="stTooltipIcon"] {
        color: var(--text-muted);
        transition: var(--transition-smooth);
    }
    
    [data-testid="stTooltipIcon"]:hover {
        color: var(--accent-primary);
    }
    
    /* ===== JSON VIEWER ===== */
    .stJson {
        background: var(--glass-bg) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: var(--radius-md) !important;
        padding: 16px !important;
    }

    /* ===== HIDE STREAMLIT BRANDING (Requirement 11.5) ===== */
    #MainMenu {visibility: hidden !important;}
    footer {visibility: hidden !important;}
    header {visibility: hidden !important;}
    
    /* Hide "Made with Streamlit" */
    .viewerBadge_container__1QSob {display: none !important;}
    .viewerBadge_link__1S137 {display: none !important;}
    
    /* Hide hamburger menu */
    [data-testid="stToolbar"] {display: none !important;}
    
    /* Hide deploy button */
    .stDeployButton {display: none !important;}
    
    /* Hide status widget */
    [data-testid="stStatusWidget"] {display: none !important;}
    
    /* ===== MULTISELECT ===== */
    .stMultiSelect > div > div {
        background: var(--glass-bg) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: var(--radius-md) !important;
    }
    
    .stMultiSelect > div > div:focus-within {
        border-color: var(--accent-primary) !important;
        box-shadow: 0 0 0 3px rgba(0, 212, 255, 0.15) !important;
    }
    
    .stMultiSelect [data-baseweb="tag"] {
        background: var(--accent-gradient) !important;
        border-radius: var(--radius-sm) !important;
    }
    
    /* ===== DATE INPUT ===== */
    .stDateInput > div > div > input {
        background: var(--glass-bg) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: var(--radius-md) !important;
        color: var(--text-primary) !important;
    }
    
    .stDateInput > div > div > input:focus {
        border-color: var(--accent-primary) !important;
        box-shadow: 0 0 0 3px rgba(0, 212, 255, 0.15) !important;
    }
    
    /* ===== TIME INPUT ===== */
    .stTimeInput > div > div > input {
        background: var(--glass-bg) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: var(--radius-md) !important;
        color: var(--text-primary) !important;
    }
    
    /* ===== COLOR PICKER ===== */
    .stColorPicker > div > div {
        background: var(--glass-bg) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: var(--radius-md) !important;
    }
    
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)



def glass_card(content: str, title: str = None, icon: str = None, animate: bool = True):
    """
    Render content in a glass card.
    
    Args:
        content: HTML content to display inside the card
        title: Optional title for the card
        icon: Optional emoji/icon to display with title
        animate: Whether to apply fade-in animation
    """
    title_html = f'<h3 style="margin-bottom: 16px;">{icon} {title}</h3>' if title else ''
    animation_class = "animate-fade-in" if animate else ""
    
    html = f"""
    <div class="glass-card {animation_class}">
        {title_html}
        {content}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def glass_metric(label: str, value: str, delta: str = None, icon: str = None, 
                 delta_color: str = "success"):
    """
    Render a glassmorphic metric.
    
    Args:
        label: Metric label
        value: Metric value
        delta: Optional delta/change value
        icon: Optional emoji/icon
        delta_color: Color for delta (success, warning, error)
    """
    delta_colors = {
        "success": "var(--success)",
        "warning": "var(--warning)",
        "error": "var(--error)",
        "neutral": "var(--text-muted)"
    }
    color = delta_colors.get(delta_color, delta_colors["success"])
    
    delta_html = f'<span style="color: {color}; font-size: 14px;">↑ {delta}</span>' if delta else ''
    icon_html = f'<span style="font-size: 24px; margin-right: 8px;">{icon}</span>' if icon else ''
    
    html = f"""
    <div class="glass-card hover-lift" style="text-align: center; padding: 20px;">
        <div style="color: var(--text-muted); font-size: 12px; text-transform: uppercase; 
                    letter-spacing: 1px; margin-bottom: 8px;">
            {icon_html}{label}
        </div>
        <div style="font-size: 32px; font-weight: 700; color: var(--text-primary); 
                    margin-bottom: 4px;">
            {value}
        </div>
        {delta_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def glass_progress(value: float, label: str = None, show_percentage: bool = True):
    """
    Render a glassmorphic progress bar.
    
    Args:
        value: Progress value between 0 and 1
        label: Optional label text
        show_percentage: Whether to show percentage value
    """
    percentage = int(value * 100)
    label_html = f'<span style="color: var(--text-secondary);">{label}</span>' if label else ''
    percentage_html = f'<span style="color: var(--accent-primary);">{percentage}%</span>' if show_percentage else ''
    
    html = f"""
    <div style="margin: 16px 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
            {label_html}
            {percentage_html}
        </div>
        <div style="background: var(--glass-bg); border-radius: 10px; height: 8px; 
                    overflow: hidden; border: 1px solid var(--glass-border);">
            <div style="background: var(--accent-gradient); width: {percentage}%; height: 100%; 
                        border-radius: 10px; transition: width 0.5s ease; position: relative;">
                <div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0;
                            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
                            animation: shimmer 2s infinite;"></div>
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def glass_badge(text: str, variant: str = "primary"):
    """
    Render a glassmorphic badge.
    
    Args:
        text: Badge text
        variant: Color variant (primary, secondary, success, warning, error)
    
    Returns:
        HTML string for the badge
    """
    colors = {
        "primary": "var(--accent-primary)",
        "secondary": "var(--accent-secondary)",
        "success": "var(--success)",
        "warning": "var(--warning)",
        "error": "var(--error)"
    }
    color = colors.get(variant, colors["primary"])
    
    html = f"""
    <span style="
        background: {color}20;
        color: {color};
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        border: 1px solid {color}40;
        display: inline-block;
    ">{text}</span>
    """
    return html


def render_logo():
    """Render the CinemaSense logo with gradient text."""
    html = """
    <div style="text-align: center; padding: 20px 0;" class="animate-fade-in">
        <h1 style="
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, #00D4FF 0%, #7B61FF 50%, #FF6B6B 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 8px;
        ">🎬 CinemaSense</h1>
        <p style="color: var(--text-muted); font-size: 14px; letter-spacing: 2px; 
                  text-transform: uppercase;">
            AI Video Intelligence Studio
        </p>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_status_indicator(status: str, label: str):
    """
    Render a status indicator with dot and label.
    
    Args:
        status: Status type (active, warning, error, info)
        label: Label text
    """
    status_class = {
        "active": "status-active",
        "warning": "status-warning",
        "error": "status-error",
        "info": "status-info"
    }.get(status, "status-active")
    
    html = f"""
    <div style="display: flex; align-items: center;">
        <span class="status-dot {status_class}"></span>
        <span style="color: var(--text-secondary);">{label}</span>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_feature_card(title: str, description: str, icon: str, status: str = "available"):
    """
    Render a feature card with icon, title, description, and status badge.
    
    Args:
        title: Feature title
        description: Feature description
        icon: Emoji/icon for the feature
        status: Status (available, coming_soon, beta)
    """
    status_variant = "success" if status == "available" else ("warning" if status == "beta" else "secondary")
    status_badge = glass_badge(status.replace("_", " ").title(), status_variant)
    
    html = f"""
    <div class="glass-card interactive hover-lift" style="height: 100%;">
        <div style="font-size: 32px; margin-bottom: 12px;">{icon}</div>
        <h3 style="margin-bottom: 8px; color: var(--text-primary);">{title}</h3>
        <p style="color: var(--text-secondary); font-size: 14px; margin-bottom: 12px;">{description}</p>
        {status_badge}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_timeline_item(time: str, title: str, description: str, emotion: str = None):
    """
    Render a timeline item with time, title, description, and optional emotion indicator.
    
    Args:
        time: Time string
        title: Item title
        description: Item description
        emotion: Optional emotion type for color coding
    """
    emotion_colors = {
        "Joy": "#00E676",
        "Tension": "#FF5252",
        "Calm": "#00D4FF",
        "Energy": "#FFD600",
        "Melancholy": "#7B61FF",
        "Mystery": "#9C27B0",
        "Neutral": "#888888"
    }
    color = emotion_colors.get(emotion, "#00D4FF")
    
    html = f"""
    <div style="display: flex; margin-bottom: 16px;" class="animate-fade-in-left">
        <div style="
            width: 12px;
            height: 12px;
            background: {color};
            border-radius: 50%;
            margin-right: 16px;
            margin-top: 4px;
            box-shadow: 0 0 10px {color};
            flex-shrink: 0;
        "></div>
        <div style="flex: 1;">
            <div style="color: var(--text-muted); font-size: 12px;">{time}</div>
            <div style="color: var(--text-primary); font-weight: 600;">{title}</div>
            <div style="color: var(--text-secondary); font-size: 14px;">{description}</div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_loading_skeleton(height: int = 100, width: str = "100%"):
    """
    Render a loading skeleton placeholder.
    
    Args:
        height: Height in pixels
        width: Width (can be percentage or pixels)
    """
    html = f"""
    <div class="skeleton" style="height: {height}px; width: {width};"></div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_loading_spinner(text: str = "Loading..."):
    """
    Render a custom loading spinner with text.
    
    Args:
        text: Loading text to display
    """
    html = f"""
    <div style="display: flex; flex-direction: column; align-items: center; 
                justify-content: center; padding: 40px;">
        <div style="
            width: 40px;
            height: 40px;
            border: 3px solid var(--glass-border);
            border-top-color: var(--accent-primary);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-bottom: 16px;
        "></div>
        <span style="color: var(--text-secondary); font-size: 14px;">{text}</span>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_stat_card(label: str, value: str, icon: str = None, trend: str = None, 
                     trend_direction: str = "up"):
    """
    Render a statistics card with optional trend indicator.
    
    Args:
        label: Stat label
        value: Stat value
        icon: Optional emoji/icon
        trend: Optional trend value (e.g., "+12%")
        trend_direction: Direction of trend (up, down, neutral)
    """
    icon_html = f'<span style="font-size: 20px; margin-right: 8px;">{icon}</span>' if icon else ''
    
    trend_colors = {
        "up": "var(--success)",
        "down": "var(--error)",
        "neutral": "var(--text-muted)"
    }
    trend_icons = {
        "up": "↑",
        "down": "↓",
        "neutral": "→"
    }
    
    trend_html = ""
    if trend:
        color = trend_colors.get(trend_direction, trend_colors["neutral"])
        icon_char = trend_icons.get(trend_direction, "→")
        trend_html = f'<span style="color: {color}; font-size: 12px; margin-left: 8px;">{icon_char} {trend}</span>'
    
    html = f"""
    <div class="glass-card hover-lift" style="padding: 16px;">
        <div style="display: flex; align-items: center; margin-bottom: 8px;">
            {icon_html}
            <span style="color: var(--text-muted); font-size: 12px; text-transform: uppercase; 
                        letter-spacing: 1px;">{label}</span>
        </div>
        <div style="display: flex; align-items: baseline;">
            <span style="font-size: 28px; font-weight: 700; color: var(--text-primary);">{value}</span>
            {trend_html}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_info_banner(message: str, variant: str = "info", icon: str = None):
    """
    Render an information banner.
    
    Args:
        message: Banner message
        variant: Color variant (info, success, warning, error)
        icon: Optional custom icon
    """
    colors = {
        "info": ("var(--info)", "var(--info-bg)", "ℹ️"),
        "success": ("var(--success)", "var(--success-bg)", "✅"),
        "warning": ("var(--warning)", "var(--warning-bg)", "⚠️"),
        "error": ("var(--error)", "var(--error-bg)", "❌")
    }
    
    color, bg, default_icon = colors.get(variant, colors["info"])
    display_icon = icon if icon else default_icon
    
    html = f"""
    <div style="
        background: {bg};
        border: 1px solid {color}40;
        border-left: 4px solid {color};
        border-radius: var(--radius-md);
        padding: 16px;
        display: flex;
        align-items: center;
        gap: 12px;
    " class="animate-fade-in">
        <span style="font-size: 20px;">{display_icon}</span>
        <span style="color: var(--text-primary);">{message}</span>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_section_header(title: str, subtitle: str = None, icon: str = None):
    """
    Render a section header with optional subtitle and icon.
    
    Args:
        title: Section title
        subtitle: Optional subtitle
        icon: Optional emoji/icon
    """
    icon_html = f'<span style="margin-right: 12px;">{icon}</span>' if icon else ''
    subtitle_html = f'<p style="color: var(--text-muted); font-size: 14px; margin-top: 4px;">{subtitle}</p>' if subtitle else ''
    
    html = f"""
    <div style="margin-bottom: 24px;" class="animate-fade-in">
        <h2 style="
            color: var(--text-primary);
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 0;
            display: flex;
            align-items: center;
        ">
            {icon_html}{title}
        </h2>
        {subtitle_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_divider(style: str = "solid"):
    """
    Render a styled divider.
    
    Args:
        style: Divider style (solid, gradient, dashed)
    """
    styles = {
        "solid": "background: var(--glass-border);",
        "gradient": "background: linear-gradient(90deg, transparent, var(--glass-border), transparent);",
        "dashed": "border-top: 1px dashed var(--glass-border); background: transparent;"
    }
    
    style_css = styles.get(style, styles["solid"])
    
    html = f"""
    <div style="height: 1px; margin: 24px 0; {style_css}"></div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_empty_state(title: str, description: str, icon: str = "📭", 
                       action_text: str = None, action_key: str = None):
    """
    Render an empty state placeholder.
    
    Args:
        title: Empty state title
        description: Description text
        icon: Emoji/icon to display
        action_text: Optional action button text
        action_key: Optional key for action button
    """
    html = f"""
    <div style="
        text-align: center;
        padding: 60px 20px;
        background: var(--glass-bg);
        border: 1px dashed var(--glass-border);
        border-radius: var(--radius-xl);
    " class="animate-fade-in">
        <div style="font-size: 48px; margin-bottom: 16px;">{icon}</div>
        <h3 style="color: var(--text-primary); margin-bottom: 8px;">{title}</h3>
        <p style="color: var(--text-muted); max-width: 400px; margin: 0 auto;">{description}</p>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
    
    if action_text and action_key:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            return st.button(action_text, key=action_key, type="primary")
    return False
