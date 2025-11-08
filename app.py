import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import linregress, ttest_ind, chi2_contingency
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings

warnings.filterwarnings('ignore')

# ============================================
# Page Configuration & Styling
# ============================================
st.set_page_config(
    page_title="Student Performance Prediction Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a classy, modern look
st.markdown("""
    <style>
    :root {
        --primary: #6366f1;
        --secondary: #8b5cf6;
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
        --bg-dark: #0f172a;
        --bg-card: #1e293b;
        --text-primary: #f1f5f9;
        --text-secondary: #cbd5e1;
        --border: #334155;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        color: var(--text-primary); /* <-- THIS ONE LINE FIXES ALL TEXT */
    }
    h3 {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
   }
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    
    .subtitle {
        color: var(--text-secondary);
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 12px -2px rgba(0, 0, 0, 0.4);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .section-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--primary);
    }
    
    .stat-box {
        background: var(--bg-card);
        border-left: 4px solid var(--primary);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        color: var(--text-primary) !important;
    }
    
    .stat-box strong {
        color: var(--text-primary) !important;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
    }
    
    .info-box {
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid var(--primary);
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================
# Title & Header
# ============================================
st.markdown('<div class="main-title"> Student Performance Prediction Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Comprehensive statistical analysis of student performance factors</div>', unsafe_allow_html=True)

# ============================================
# Data Generation Function
# ============================================
@st.cache_data
def generate_student_data(n_students=500):
    """Generate synthetic student performance dataset"""
    np.random.seed(42)
    
    # Generate realistic data with correlations
    study_hours = np.random.normal(25, 8, n_students)
    study_hours = np.clip(study_hours, 5, 50)  # Reasonable range
    
    attendance = np.random.beta(8, 2, n_students) * 100
    attendance = np.clip(attendance, 50, 100)
    
    # Past grades influence final score
    past_grades = np.random.normal(75, 12, n_students)
    past_grades = np.clip(past_grades, 40, 100)
    
    # Tutor groups (categorical)
    tutor_groups = np.random.choice(['Group A', 'Group B', 'Group C', 'Group D'], 
                                    n_students, p=[0.3, 0.25, 0.25, 0.2])
    
    # Final score depends on all factors with some noise
    # Strong positive correlation with study_hours and attendance
    # Adjusted coefficients to get more realistic scores (average around 60-70%)
    final_score = (
        15 +  # Base score (lowered)
        0.8 * study_hours +  # Study hours effect (reduced from 1.2)
        0.25 * attendance +  # Attendance effect (reduced from 0.4)
        0.35 * past_grades +  # Past performance effect
        np.random.normal(0, 10, n_students)  # Random noise (increased)
    )
    
    # Add tutor group effects
    tutor_effects = {'Group A': 5, 'Group B': 2, 'Group C': -2, 'Group D': -5}
    for i, group in enumerate(tutor_groups):
        final_score[i] += tutor_effects[group]
    
    final_score = np.clip(final_score, 0, 100)
    
    # Create pass/fail (pass if >= 50)
    pass_fail = ['Pass' if score >= 50 else 'Fail' for score in final_score]
    
    df = pd.DataFrame({
        'student_id': range(1, n_students + 1),
        'study_hours': np.round(study_hours, 1),
        'attendance': np.round(attendance, 1),
        'past_grades': np.round(past_grades, 1),
        'final_score': np.round(final_score, 1),
        'tutor_group': tutor_groups,
        'pass_fail': pass_fail
    })
    
    return df

# ============================================
# Data Loading Function (for uploaded files)
# ============================================
def load_uploaded_data(uploaded_file):
    """Load and validate uploaded CSV file"""
    try:
        df = pd.read_csv(uploaded_file)
        
        # Check for required columns (case-insensitive)
        required_cols_lower = ['study_hours', 'attendance', 'past_grades', 'final_score', 'tutor_group']
        df_cols_lower = [col.lower().strip() for col in df.columns]
        
        # Map columns (case-insensitive matching)
        col_mapping = {}
        for req_col in required_cols_lower:
            if req_col in df_cols_lower:
                idx = df_cols_lower.index(req_col)
                col_mapping[df.columns[idx]] = req_col
            else:
                # Try common variations
                variations = {
                    'study_hours': ['study_hours', 'studyhours', 'hours', 'study_hours_per_week'],
                    'attendance': ['attendance', 'attendance_percent', 'attendance_pct', 'attendance%'],
                    'past_grades': ['past_grades', 'pastgrades', 'previous_grades', 'past_score'],
                    'final_score': ['final_score', 'finalscore', 'final_grade', 'score', 'final_exam_score'],
                    'tutor_group': ['tutor_group', 'tutorgroup', 'group', 'tutor', 'section']
                }
                found = False
                for var in variations.get(req_col, []):
                    if var in df_cols_lower:
                        idx = df_cols_lower.index(var)
                        col_mapping[df.columns[idx]] = req_col
                        found = True
                        break
                if not found:
                    st.error(f"Required column '{req_col}' not found in uploaded file.")
                    st.info(f"Available columns: {', '.join(df.columns)}")
                    return None
        
        # Rename columns
        df = df.rename(columns=col_mapping)
        
        # Check if pass_fail exists, if not create it
        if 'pass_fail' not in df.columns:
            df['pass_fail'] = df['final_score'].apply(lambda x: 'Pass' if x >= 50 else 'Fail')
        
        # Ensure numeric columns are numeric
        numeric_cols = ['study_hours', 'attendance', 'past_grades', 'final_score']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows with missing critical data
        df = df.dropna(subset=numeric_cols + ['tutor_group']).reset_index(drop=True)
        
        # Add student_id if missing
        if 'student_id' not in df.columns:
            df.insert(0, 'student_id', range(1, len(df) + 1))
        
        return df
        
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

# ============================================
# Sidebar Controls
# ============================================
with st.sidebar:
    st.header("⚙️ Controls")
    
    # Data source selection
    data_source = st.radio(
        "Data Source",
        ["Generate Synthetic Data", "Upload CSV File"],
        index=0
    )
    
    if data_source == "Generate Synthetic Data":
        n_students = st.slider("Number of Students", 100, 1000, 500, 50)
        regenerate = st.button("🔄 Regenerate Data", use_container_width=True)
        uploaded_file = None
    else:
        uploaded_file = st.file_uploader(
            "Upload CSV File",
            type=['csv'],
            help="Upload a CSV file with columns: study_hours, attendance, past_grades, final_score, tutor_group"
        )
        regenerate = False
        n_students = 500  # Default, not used when file is uploaded
    
    st.markdown("---")
    st.header("📊 Analysis Sections")
    show_descriptive = st.checkbox("1. Descriptive Statistics", value=True)
    show_visualizations = st.checkbox("2. Visualizations", value=True)
    show_correlation = st.checkbox("3. Correlation & Regression", value=True)
    show_ttest = st.checkbox("4. T-Test Hypothesis Testing", value=True)
    show_chisquare = st.checkbox("5. Chi-Square Test", value=True)
    
    st.markdown("---")
    st.markdown("""
    <div class="info-box">
    <strong>📖 About</strong><br>
    This app demonstrates key statistical concepts:
    <ul style="font-size: 0.85rem; margin-top: 0.5rem;">
        <li>Descriptive Statistics</li>
        <li>Data Visualization</li>
        <li>Correlation & Regression</li>
        <li>Hypothesis Testing (t-test)</li>
        <li>Chi-Square Test</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# Load Data
# ============================================
if regenerate:
    st.cache_data.clear()

# Load data based on source
if data_source == "Upload CSV File" and uploaded_file is not None:
    df = load_uploaded_data(uploaded_file)
    if df is None:
        st.stop()  # Stop execution if file loading failed
    st.success(f"✅ Successfully loaded {len(df)} records from uploaded file!")
elif data_source == "Upload CSV File" and uploaded_file is None:
    st.info("👆 Please upload a CSV file to begin analysis, or switch to 'Generate Synthetic Data'.")
    st.stop()
else:
    df = generate_student_data(n_students)

# ============================================
# Overview Metrics
# ============================================
st.markdown("### 📈 Dataset Overview")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Students</div>
        <div class="metric-value">{len(df):,}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    avg_score = df['final_score'].mean()
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Avg Final Score</div>
        <div class="metric-value">{avg_score:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    avg_study = df['study_hours'].mean()
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Avg Study Hours</div>
        <div class="metric-value">{avg_study:.1f} hrs</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    avg_attendance = df['attendance'].mean()
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Avg Attendance</div>
        <div class="metric-value">{avg_attendance:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    pass_rate = (df['pass_fail'] == 'Pass').sum() / len(df) * 100
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Pass Rate</div>
        <div class="metric-value">{pass_rate:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

# Show data preview
with st.expander("📋 View Dataset", expanded=False):
    st.dataframe(df.head(100), use_container_width=True)

# ============================================
# 1. DESCRIPTIVE STATISTICS
# ============================================
if show_descriptive:
    st.markdown('<div class="section-header">1️⃣ Descriptive Statistics</div>', unsafe_allow_html=True)
    
    st.markdown("**Summary Statistics for Key Variables**")
    
    # Select variables for descriptive stats
    desc_vars = ['study_hours', 'attendance', 'past_grades', 'final_score']
    desc_df = df[desc_vars].describe()
    
    # Format the table nicely
    st.dataframe(desc_df.style.format("{:.2f}"), use_container_width=True)
    
    # Additional statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Study Hours**")
        st.markdown(f"""
        <div class="stat-box">
            <strong>Mean:</strong> {df['study_hours'].mean():.2f} hours<br>
            <strong>Median:</strong> {df['study_hours'].median():.2f} hours<br>
            <strong>Std Dev:</strong> {df['study_hours'].std():.2f} hours<br>
            <strong>Min:</strong> {df['study_hours'].min():.1f} hours<br>
            <strong>Max:</strong> {df['study_hours'].max():.1f} hours
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("**Final Score**")
        st.markdown(f"""
        <div class="stat-box">
            <strong>Mean:</strong> {df['final_score'].mean():.2f}%<br>
            <strong>Median:</strong> {df['final_score'].median():.2f}%<br>
            <strong>Std Dev:</strong> {df['final_score'].std():.2f}%<br>
            <strong>Min:</strong> {df['final_score'].min():.1f}%<br>
            <strong>Max:</strong> {df['final_score'].max():.1f}%
        </div>
        """, unsafe_allow_html=True)

# ============================================
# 2. VISUALIZATIONS
# ============================================
if show_visualizations:
    st.markdown('<div class="section-header">2️⃣ Data Visualizations</div>', unsafe_allow_html=True)
    
    # Set style
    sns.set_style("darkgrid")
    plt.rcParams['figure.facecolor'] = '#1e293b'
    plt.rcParams['axes.facecolor'] = '#1e293b'
    plt.rcParams['axes.labelcolor'] = '#f1f5f9'
    plt.rcParams['text.color'] = '#f1f5f9'
    plt.rcParams['xtick.color'] = '#cbd5e1'
    plt.rcParams['ytick.color'] = '#cbd5e1'
    
    # Histograms
    st.markdown("**Distribution of Final Scores**")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.hist(df['final_score'], bins=30, color='#6366f1', edgecolor='white', alpha=0.7)
    ax1.set_xlabel('Final Score (%)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution of Final Scores', fontsize=14, fontweight='bold')
    ax1.axvline(df['final_score'].mean(), color='#10b981', linestyle='--', linewidth=2, label=f'Mean: {df["final_score"].mean():.1f}%')
    ax1.legend()
    st.pyplot(fig1)
    
    # Scatter plots
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Final Score vs Study Hours**")
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.scatter(df['study_hours'], df['final_score'], alpha=0.6, color='#8b5cf6', s=50)
        ax2.set_xlabel('Study Hours', fontsize=11)
        ax2.set_ylabel('Final Score (%)', fontsize=11)
        ax2.set_title('Study Hours vs Final Score', fontsize=12, fontweight='bold')
        
        # Add trend line
        z = np.polyfit(df['study_hours'], df['final_score'], 1)
        p = np.poly1d(z)
        ax2.plot(df['study_hours'], p(df['study_hours']), "r--", alpha=0.8, linewidth=2, label='Trend Line')
        ax2.legend()
        st.pyplot(fig2)
    
    with col2:
        st.markdown("**Final Score vs Attendance**")
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        ax3.scatter(df['attendance'], df['final_score'], alpha=0.6, color='#10b981', s=50)
        ax3.set_xlabel('Attendance (%)', fontsize=11)
        ax3.set_ylabel('Final Score (%)', fontsize=11)
        ax3.set_title('Attendance vs Final Score', fontsize=12, fontweight='bold')
        
        # Add trend line
        z = np.polyfit(df['attendance'], df['final_score'], 1)
        p = np.poly1d(z)
        ax3.plot(df['attendance'], p(df['attendance']), "r--", alpha=0.8, linewidth=2, label='Trend Line')
        ax3.legend()
        st.pyplot(fig3)
    
    # Additional histogram for study hours
    st.markdown("**Distribution of Study Hours**")
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    ax4.hist(df['study_hours'], bins=25, color='#f59e0b', edgecolor='white', alpha=0.7)
    ax4.set_xlabel('Study Hours', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title('Distribution of Study Hours per Week', fontsize=14, fontweight='bold')
    ax4.axvline(df['study_hours'].mean(), color='#10b981', linestyle='--', linewidth=2, label=f'Mean: {df["study_hours"].mean():.1f} hrs')
    ax4.legend()
    st.pyplot(fig4)

# ============================================
# 3. CORRELATION & REGRESSION
# ============================================
if show_correlation:
    st.markdown('<div class="section-header">3️⃣ Correlation & Regression Analysis</div>', unsafe_allow_html=True)
    
    # Correlation Matrix
    st.markdown("**Correlation Matrix**")
    corr_vars = ['study_hours', 'attendance', 'past_grades', 'final_score']
    corr_matrix = df[corr_vars].corr()
    
    fig5, ax5 = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0, 
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax5)
    ax5.set_title('Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
    st.pyplot(fig5)
    
    # Linear Regression: Final Score vs Study Hours
    st.markdown("**Simple Linear Regression: Final Score ~ Study Hours**")
    
    slope, intercept, r_value, p_value, std_err = linregress(df['study_hours'], df['final_score'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="stat-box">
            <strong>Regression Results:</strong><br>
            <strong>Slope:</strong> {slope:.3f}<br>
            <strong>Intercept:</strong> {intercept:.3f}<br>
            <strong>R-squared:</strong> {r_value**2:.4f}<br>
            <strong>P-value:</strong> {p_value:.2e}<br>
            <strong>Std Error:</strong> {std_err:.3f}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"**Equation:** Final Score = {intercept:.2f} + {slope:.2f} × Study Hours")
    
    with col2:
        # Visualization of regression
        fig6, ax6 = plt.subplots(figsize=(8, 5))
        ax6.scatter(df['study_hours'], df['final_score'], alpha=0.5, color='#6366f1', s=30)
        ax6.plot(df['study_hours'], intercept + slope * df['study_hours'], 
                'r--', linewidth=2, label=f'Regression Line (R² = {r_value**2:.3f})')
        ax6.set_xlabel('Study Hours', fontsize=11)
        ax6.set_ylabel('Final Score (%)', fontsize=11)
        ax6.set_title('Linear Regression: Study Hours vs Final Score', fontsize=12, fontweight='bold')
        ax6.legend()
        st.pyplot(fig6)
    
    # Multiple Linear Regression
    st.markdown("**Multiple Linear Regression: Final Score ~ Study Hours + Attendance**")
    
    X = df[['study_hours', 'attendance']]
    X = sm.add_constant(X)  # Add intercept
    y = df['final_score']
    
    model = sm.OLS(y, X).fit()
    
    st.code(str(model.summary()), language=None)
    
    # Alternative using formula API
    st.markdown("**Multiple Linear Regression (with Past Grades): Final Score ~ Study Hours + Attendance + Past Grades**")
    
    model2 = smf.ols('final_score ~ study_hours + attendance + past_grades', data=df).fit()
    st.code(str(model2.summary()), language=None)

# ============================================
# 4. T-TEST HYPOTHESIS TESTING
# ============================================
if show_ttest:
    st.markdown('<div class="section-header">4️⃣ T-Test Hypothesis Testing</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="color: var(--text-primary);">
    <strong>Research Question:</strong> Do students with attendance >90% have statistically different mean scores 
    than students with attendance ≤90%?<br><br>
    
    <strong>Hypotheses:</strong><br>
    - H₀: μ₁ = μ₂ (No difference in mean scores)<br>
    - H₁: μ₁ ≠ μ₂ (There is a difference in mean scores)
    </div>
    """, unsafe_allow_html=True)
    
    # Split data
    high_attendance = df[df['attendance'] > 90]['final_score']
    low_attendance = df[df['attendance'] <= 90]['final_score']
    
    # Perform t-test
    t_stat, p_value = ttest_ind(high_attendance, low_attendance)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">High Attendance (>90%)</div>
            <div class="metric-value">{len(high_attendance)}</div>
            <div style="color: var(--text-secondary); margin-top: 0.5rem;">
                Mean: {high_attendance.mean():.2f}%<br>
                Std: {high_attendance.std():.2f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Low Attendance (≤90%)</div>
            <div class="metric-value">{len(low_attendance)}</div>
            <div style="color: var(--text-secondary); margin-top: 0.5rem;">
                Mean: {low_attendance.mean():.2f}%<br>
                Std: {low_attendance.std():.2f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        significance = "✅ Significant" if p_value < 0.05 else "❌ Not Significant"
        color = "#10b981" if p_value < 0.05 else "#ef4444"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">T-Test Results</div>
            <div class="metric-value" style="color: {color};">{significance}</div>
            <div style="color: var(--text-secondary); margin-top: 0.5rem;">
                t-statistic: {t_stat:.4f}<br>
                p-value: {p_value:.4f}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Display results
    st.markdown(f"""
    <div class="stat-box">
        <strong>T-Statistic:</strong> {t_stat:.4f}<br>
        <strong>P-Value:</strong> {p_value:.4f}<br>
        <strong>Conclusion:</strong> {'We reject H₀' if p_value < 0.05 else 'We fail to reject H₀'} 
        (α = 0.05). There is {'a significant' if p_value < 0.05 else 'no significant'} 
        difference in mean scores between the two groups.
    </div>
    """, unsafe_allow_html=True)
    
    # Visualization
    fig7, ax7 = plt.subplots(figsize=(10, 5))
    ax7.hist(high_attendance, bins=25, alpha=0.6, label=f'Attendance >90% (n={len(high_attendance)})', color='#10b981')
    ax7.hist(low_attendance, bins=25, alpha=0.6, label=f'Attendance ≤90% (n={len(low_attendance)})', color='#ef4444')
    ax7.axvline(high_attendance.mean(), color='#10b981', linestyle='--', linewidth=2, label=f'High Mean: {high_attendance.mean():.1f}%')
    ax7.axvline(low_attendance.mean(), color='#ef4444', linestyle='--', linewidth=2, label=f'Low Mean: {low_attendance.mean():.1f}%')
    ax7.set_xlabel('Final Score (%)', fontsize=12)
    ax7.set_ylabel('Frequency', fontsize=12)
    ax7.set_title('Distribution Comparison: High vs Low Attendance', fontsize=14, fontweight='bold')
    ax7.legend()
    st.pyplot(fig7)

# ============================================
# 5. CHI-SQUARE TEST
# ============================================
if show_chisquare:
    st.markdown('<div class="section-header">5️⃣ Chi-Square Test of Independence</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="color: var(--text-primary);">
    <strong>Research Question:</strong> Is there a relationship between tutor group and pass/fail outcome?<br><br>
    
    <strong>Hypotheses:</strong><br>
    - H₀: Tutor group and pass/fail are independent (no relationship)<br>
    - H₁: Tutor group and pass/fail are dependent (there is a relationship)
    </div>
    """, unsafe_allow_html=True)
    
    # Create contingency table
    contingency_table = pd.crosstab(df['tutor_group'], df['pass_fail'], margins=True)
    
    st.markdown("**Contingency Table**")
    st.dataframe(contingency_table, use_container_width=True)
    
    # Perform chi-square test
    contingency_table_no_margins = pd.crosstab(df['tutor_group'], df['pass_fail'])
    chi2, p_value, dof, expected = chi2_contingency(contingency_table_no_margins)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="stat-box">
            <strong>Chi-Square Test Results:</strong><br>
            <strong>Chi-square statistic:</strong> {chi2:.4f}<br>
            <strong>Degrees of freedom:</strong> {dof}<br>
            <strong>P-value:</strong> {p_value:.4f}<br>
            <strong>Expected frequencies:</strong>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        significance = "✅ Significant" if p_value < 0.05 else "❌ Not Significant"
        color = "#10b981" if p_value < 0.05 else "#ef4444"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Test Result</div>
            <div class="metric-value" style="color: {color}; font-size: 1.5rem;">{significance}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Display expected frequencies
    expected_df = pd.DataFrame(expected, 
                               index=contingency_table_no_margins.index,
                               columns=contingency_table_no_margins.columns)
    st.markdown("**Expected Frequencies (under independence)**")
    st.dataframe(expected_df.style.format("{:.2f}"), use_container_width=True)
    
    # Conclusion
    st.markdown(f"""
    <div class="stat-box">
        <strong>Conclusion:</strong> {'We reject H₀' if p_value < 0.05 else 'We fail to reject H₀'} 
        (α = 0.05). There is {'a significant' if p_value < 0.05 else 'no significant'} 
        relationship between tutor group and pass/fail outcome.
    </div>
    """, unsafe_allow_html=True)
    
    # Visualization
    fig8, ax8 = plt.subplots(figsize=(10, 5))
    contingency_table_no_margins.plot(kind='bar', ax=ax8, color=['#ef4444', '#10b981'], width=0.8)
    ax8.set_xlabel('Tutor Group', fontsize=12)
    ax8.set_ylabel('Number of Students', fontsize=12)
    ax8.set_title('Pass/Fail Distribution by Tutor Group', fontsize=14, fontweight='bold')
    ax8.legend(title='Outcome', labels=['Fail', 'Pass'])
    ax8.tick_params(axis='x', rotation=0)
    st.pyplot(fig8)

# ============================================
# Footer
# ============================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: var(--text-secondary); padding: 2rem 0;">
    <p>📊 Probability & Statistics Project | Student Performance Analysis</p>
    <p style="font-size: 0.85rem;">Generated with Streamlit • Statistical analysis powered by SciPy & Statsmodels</p>
</div>
""", unsafe_allow_html=True)

