# 📚 Student Performance Prediction Analyzer

A comprehensive, interactive Streamlit web application for statistical analysis of student performance data. This project demonstrates key probability and statistics concepts through real-world data analysis.

## ✨ Features

### 📊 Statistical Analyses Included:

1. **Descriptive Statistics**
   - Mean, median, standard deviation for all key variables
   - Summary statistics using `pandas.DataFrame.describe()`
   - Detailed breakdowns for study hours and final scores

2. **Data Visualizations**
   - Histograms showing distribution of final scores and study hours
   - Scatter plots with trend lines:
     - Final Score vs Study Hours
     - Final Score vs Attendance
   - Beautiful, modern dark-themed visualizations

3. **Correlation & Regression Analysis**
   - Correlation matrix heatmap for all numeric variables
   - Simple linear regression using `scipy.stats.linregress`
   - Multiple linear regression using `statsmodels`:
     - Final Score ~ Study Hours + Attendance
     - Final Score ~ Study Hours + Attendance + Past Grades
   - R-squared values, p-values, and regression equations

4. **Hypothesis Testing (T-Test)**
   - Independent samples t-test using `scipy.stats.ttest_ind`
   - Tests if students with >90% attendance have different mean scores
   - Visual comparison of distributions
   - Statistical significance testing (α = 0.05)

5. **Chi-Square Test**
   - Test of independence using `scipy.stats.chi2_contingency`
   - Examines relationship between tutor group and pass/fail outcome
   - Contingency tables with expected frequencies
   - Visual bar chart representation

### 🎨 UI Features:

- **Modern Dark Theme**: Beautiful gradient backgrounds and card-based layouts
- **Interactive Controls**: Sidebar with checkboxes to toggle each analysis section
- **Real-time Data Generation**: Slider to adjust dataset size (100-1000 students)
- **Key Metrics Dashboard**: Overview cards showing important statistics
- **Responsive Design**: Works beautifully on different screen sizes

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone or download this repository**

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**
   
   On Windows:
   ```bash
   venv\Scripts\activate
   ```
   
   On macOS/Linux:
   ```bash
   source venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

6. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - If it doesn't, navigate to the URL shown in the terminal

## 📁 Project Structure

```
.
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── venv/              # Virtual environment (created after setup)
```

## 📊 Dataset

### Option 1: Generate Synthetic Data
The application can generate synthetic student performance data with the following variables:

### Option 2: Upload Your Own CSV File
You can also upload your own CSV file! The app will automatically detect and map columns (case-insensitive). Required columns:
- `study_hours` (or variations: studyhours, hours, study_hours_per_week)
- `attendance` (or variations: attendance_percent, attendance_pct, attendance%)
- `past_grades` (or variations: pastgrades, previous_grades, past_score)
- `final_score` (or variations: finalscore, final_grade, score, final_exam_score)
- `tutor_group` (or variations: tutorgroup, group, tutor, section)

Optional: `pass_fail` (if not provided, will be auto-generated based on final_score ≥ 50)

### Data Variables:

- **student_id**: Unique identifier
- **study_hours**: Weekly study hours (5-50 hours)
- **attendance**: Class attendance percentage (50-100%)
- **past_grades**: Previous academic performance (40-100%)
- **final_score**: Final exam score (0-100%)
- **tutor_group**: Categorical variable (Group A, B, C, or D)
- **pass_fail**: Binary outcome (Pass if score ≥ 50%)

The data generation includes realistic correlations:
- Positive correlation between study hours and final score
- Positive correlation between attendance and final score
- Influence of past grades on final performance
- Tutor group effects on outcomes

## 📖 Statistical Concepts Covered

### Descriptive Statistics
- Measures of central tendency (mean, median)
- Measures of dispersion (standard deviation, range)
- Summary statistics tables

### Data Visualization
- Histograms for distribution analysis
- Scatter plots for relationship exploration
- Trend line fitting

### Correlation Analysis
- Pearson correlation coefficients
- Correlation matrix visualization
- Interpretation of correlation strength

### Regression Analysis
- Simple linear regression
- Multiple linear regression
- Model diagnostics (R², p-values, coefficients)
- Interpretation of regression results

### Hypothesis Testing
- **T-Test**: Comparing means of two independent groups
  - Null hypothesis (H₀) and alternative hypothesis (H₁)
  - Significance level (α = 0.05)
  - Interpretation of p-values
  
- **Chi-Square Test**: Testing independence of categorical variables
  - Contingency tables
  - Expected vs observed frequencies
  - Degrees of freedom
  - Test of independence

## 🎯 Learning Objectives

This project helps you understand:

1. How to perform comprehensive statistical analysis
2. The relationship between study habits and academic performance
3. Hypothesis testing procedures and interpretation
4. Data visualization best practices
5. Building interactive data analysis applications

## 🔧 Customization

### Data Source Options
- **Generate Synthetic Data**: Use the slider in the sidebar to generate datasets with 100-1000 students
- **Upload CSV File**: Click "Upload CSV File" in the sidebar to use your own dataset
  - The app automatically handles column name variations
  - Missing `pass_fail` column will be auto-generated
  - Invalid rows are automatically filtered out

### Adjusting Dataset Size
- When using synthetic data, use the slider to adjust the number of students (100-1000)
- Click "Regenerate Data" to create a new random dataset

### Toggling Analysis Sections
- Use the checkboxes in the sidebar to show/hide specific analysis sections
- This allows you to focus on specific statistical concepts

## 📝 Notes

- The synthetic data is randomly generated each time you click "Regenerate Data"
- Average final scores are now more realistic (typically 60-70% range) with proper variance
- All statistical tests use a significance level of α = 0.05
- The synthetic data includes realistic correlations and relationships
- Text readability has been optimized for the dark theme - all hypothesis testing sections are now clearly readable
- When uploading CSV files, ensure numeric columns contain valid numbers (text values will be converted or filtered)

## 🛠️ Technologies Used

- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib**: Static plotting
- **Seaborn**: Statistical data visualization
- **SciPy**: Statistical functions (t-test, chi-square)
- **Statsmodels**: Advanced statistical modeling

## 📚 Syllabus Coverage

This project covers:

✅ Descriptive Statistics  
✅ Data Visualization  
✅ Correlation Analysis  
✅ Linear Regression (Simple & Multiple)  
✅ Hypothesis Testing (T-Test)  
✅ Chi-Square Test of Independence  
✅ Statistical Interpretation & Reporting  

## 🤝 Contributing

Feel free to extend this project with:
- Additional statistical tests (ANOVA, Mann-Whitney U test, etc.)
- More sophisticated regression models (polynomial, logistic)
- Model diagnostics (residual plots, QQ plots)
- Export functionality for results
- Integration with real datasets

## 📄 License

This project is open source and available for educational purposes.

## 🙏 Acknowledgments

Built as a comprehensive probability and statistics project demonstrating real-world data analysis techniques.

---

**Happy Analyzing! 📊✨**

