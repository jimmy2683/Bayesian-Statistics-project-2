#!/bin/bash

# Project 2: Advanced MCMC Methods - Execution Script
# Bayesian Statistics Course - Jimmy2683
# This script runs the complete MCMC comparison analysis

set -e  # Exit on any error

echo "=========================================="
echo "Project 2: Advanced MCMC Methods Analysis"
echo "=========================================="
echo "Starting at: $(date)"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 is not installed or not in PATH"
    exit 1
fi

# Create output directories
echo "Setting up output directories..."
mkdir -p results
mkdir -p figures
mkdir -p logs

# Check required Python packages
echo "Checking Python dependencies..."
python3 -c "
import sys
required_packages = ['numpy', 'scipy', 'matplotlib', 'seaborn', 'pandas']
missing_packages = []

for package in required_packages:
    try:
        __import__(package)
        print(f'✓ {package}')
    except ImportError:
        missing_packages.append(package)
        print(f'✗ {package}')

if missing_packages:
    print(f'Missing packages: {missing_packages}')
    print('Install with: pip install ' + ' '.join(missing_packages))
    sys.exit(1)
else:
    print('All required packages are available')
"

if [ $? -ne 0 ]; then
    echo "Please install missing packages before running the analysis"
    exit 1
fi

# Validate that all required Python files exist
echo ""
echo "Checking required files..."
required_files=("main_analysis.py" "mh_mcmc.py" "gibbs_mcmc.py" "plots_mcmc.py" "bayes_factor.py")
for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $file"
    else
        echo "✗ $file (MISSING)"
        echo "Error: Required file $file not found"
        exit 1
    fi
done

# Check if data file exists
if [ -f "data.csv" ]; then
    echo "✓ data.csv"
else
    echo "⚠ data.csv not found - will use synthetic data"
fi

# Run the main analysis
echo ""
echo "=========================================="
echo "RUNNING MCMC ANALYSIS"
echo "=========================================="

# Execute main analysis with output logging
echo "Executing main_analysis.py..."
log_file="logs/analysis_$(date +%Y%m%d_%H%M%S).log"

# Run with better error handling
if python3 main_analysis.py 2>&1 | tee "$log_file"; then
    analysis_success=true
else
    analysis_success=false
fi

# Check if analysis completed successfully
if [ "$analysis_success" = true ]; then
    echo ""
    echo "=========================================="
    echo "ANALYSIS COMPLETED SUCCESSFULLY"
    echo "=========================================="
    
    # Move generated files to organized structure
    echo "Organizing output files..."
    
    # Move any generated CSV files to results (except input data.csv)
    if ls *.csv 1> /dev/null 2>&1; then
        for file in *.csv; do
            if [ "$file" != "data.csv" ]; then
                mv "$file" results/
            fi
        done
        echo "Summary CSV files moved to results/"
    fi
    
    # Move any generated figure files to figures directory
    if ls *.png 1> /dev/null 2>&1; then
        mv *.png figures/
        echo "Figure files (PNG) moved to figures/"
    fi
    
    if ls *.pdf 1> /dev/null 2>&1; then
        mv *.pdf figures/
        echo "PDF files moved to figures/"
    fi
    
    # Display summary of results
    echo ""
    echo "Output Summary:"
    echo "---------------"
    echo "Logs: $(ls logs/ | wc -l) files in logs/"
    
    if [ -d "results" ] && [ "$(ls -A results/)" ]; then
        echo "Results: $(ls results/ | wc -l) files in results/"
        echo "  - $(ls results/)"
    fi
    
    if [ -d "figures" ] && [ "$(ls -A figures/)" ]; then
        echo "Figures: $(ls figures/ | wc -l) files in figures/"
    fi
    
    # Generate quick summary report
    echo ""
    echo "Generating quick summary report..."
    
cat > results/analysis_summary.txt << EOF
Project 2: Advanced MCMC Methods Analysis Summary
================================================
Generated on: $(date)
Analysis Duration: Started at script execution

Files Generated:
- Execution logs in logs/
- Summary statistics in results/
- Diagnostic plots in figures/

Key Components Executed:
1. Metropolis-Hastings MCMC implementation
2. Gibbs Sampling implementation  
3. Convergence diagnostics
4. Posterior comparison across methods
5. Autocorrelation analysis
6. Summary statistics generation

Next Steps:
- Review logs/ for detailed execution output
- Examine results/ for numerical summaries
- Check figures/ for diagnostic plots
- Use report_template.md to structure final report
- Include results in academic writeup

For questions or issues, refer to the course materials or
consult the implementation files:
- mh_mcmc.py (Metropolis-Hastings)
- gibbs_mcmc.py (Gibbs Sampling)  
- plots_mcmc.py (Diagnostics & Plotting)
- main_analysis.py (Main execution)
EOF

    echo "Summary report saved to results/analysis_summary.txt"
    
    echo ""
    echo "=========================================="
    echo "ALL TASKS COMPLETED SUCCESSFULLY!"
    echo "=========================================="
    echo "Finished at: $(date)"
    echo ""
    echo "Next steps:"
    echo "1. Review the generated plots and results"
    echo "2. Use report_template.md to write your final report"
    echo "3. Include the numerical results in your analysis"
    echo "4. Discuss the comparison between MH and Gibbs methods"
    
else
    echo ""
    echo "=========================================="
    echo "ANALYSIS FAILED"
    echo "=========================================="
    echo "Error details have been logged to: $log_file"
    echo ""
    echo "Common issues and solutions:"
    echo "- Import errors: Check that all Python files are present"
    echo "- Missing dependencies: Run 'pip install numpy scipy matplotlib seaborn pandas'"
    echo "- Syntax errors: Check Python file syntax with 'python3 -m py_compile filename.py'"
    echo "- Memory issues: Reduce sample sizes in main_analysis.py"
    echo ""
    echo "Debug steps:"
    echo "1. Check the log file: cat $log_file"
    echo "2. Test individual components: python3 -c 'from mh_mcmc import MetropolisHastings'"
    echo "3. Validate file syntax: python3 -m py_compile main_analysis.py"
    exit 1
fi
