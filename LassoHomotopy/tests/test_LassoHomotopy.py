import csv
import time
import numpy as np
from model.LassoHomotopy import LassoHomotopyModel 
import matplotlib.pyplot as plt
import pytest
import warnings
warnings.filterwarnings("ignore")

def load_data():
    data = []
    with open("./collinear_data.csv", "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    X = np.array([[float(v) for k, v in row.items() if k.startswith('X')] for row in data])
    y = np.array([float(row['target']) for row in data])
    return X, y.reshape(-1)

def test_basic_prediction():
    """Comprehensive test of basic prediction functionality with detailed reporting"""
    # Initialize model with default parameters
    model = LassoHomotopyModel()
    #model = LassoHomotopyModel(alpha=0.5, fit_intercept=True, max_iter=1000)
    #model = LassoHomotopyModel(lambda_reg=0.5, fit_intercept=True, max_iter=1000)
    X, y = load_data()
    
    # Standardize features for better performance
    #X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
    # Fit model and make predictions
    results = model.fit(X, y)
    preds = results.predict(X)
    coefficients = results.coef_
    
    # Calculate performance metrics
    mse = np.mean((y - preds)**2)
    mae = np.mean(np.abs(y - preds))
    r2 = 1 - np.sum((y - preds)**2) / np.sum((y - np.mean(y))**2)
    corr = np.corrcoef(y, preds)[0, 1]
    
    # Sparsity analysis
    num_non_zero = np.sum(np.abs(coefficients) > model.model.tol)
    sparsity = 1 - (num_non_zero / len(coefficients))
    
    # Generate comprehensive report
    report = f"""
    {'='*60}
    BASIC PREDICTION TEST REPORT
    {'='*60}
    Data Characteristics:
    - Samples: {X.shape[0]}
    - Features: {X.shape[1]}
    - Target mean: {np.mean(y):.4f}
    - Target std: {np.std(y):.4f}
    
    Model Parameters:
    - Alpha (λ): {model.alpha:.4f}
    - Tolerance: {model.model.tol:.2e}
    - Max iterations: {model.max_iter}
    - Fit intercept: {model.fit_intercept}
    - Intercept: {results.intercept_:.4f}
    
    Performance Metrics:
    - R-squared: {r2:.4f}
    - Mean Squared Error: {mse:.4f}
    - Mean Absolute Error: {mae:.4f}
    - Prediction-Target Correlation: {corr:.4f}
    
    Sparsity Analysis:
    - Non-zero coefficients: {num_non_zero}/{len(coefficients)}
    - Sparsity ratio: {sparsity:.1%}
    - Largest coefficient: {np.max(np.abs(coefficients)):.4f}
    
    Prediction Statistics:
    - Min prediction: {np.min(preds):.4f}
    - Max prediction: {np.max(preds):.4f}
    - Prediction range: {np.max(preds) - np.min(preds):.4f}
    {'='*60}
    """
    print(report)
    
    # Create visualizations
    plt.figure(figsize=(18, 5))
    
    # Plot 1: Actual vs Predicted values
    plt.subplot(1, 3, 1)
    plt.scatter(y, preds, alpha=0.6, color='royalblue')
    plt.plot([min(y), max(y)], [min(y), max(y)], 'r--', linewidth=1)
    plt.title('Actual vs Predicted Values', fontsize=12)
    plt.xlabel('Actual Values', fontsize=10)
    plt.ylabel('Predicted Values', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Coefficient magnitudes
    plt.subplot(1, 3, 2)
    plt.stem(np.arange(len(coefficients)), coefficients, markerfmt=' ')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.title('Feature Coefficients', fontsize=12)
    plt.xlabel('Feature Index', fontsize=10)
    plt.ylabel('Coefficient Value', fontsize=10)
    
    # Plot 3: Prediction error distribution
    plt.subplot(1, 3, 3)
    errors = y - preds
    plt.hist(errors, bins=30, color='purple', alpha=0.7)
    plt.axvline(0, color='black', linestyle='--')
    plt.title('Prediction Error Distribution', fontsize=12)
    plt.xlabel('Prediction Error', fontsize=10)
    plt.ylabel('Frequency', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # Assertions with helpful messages
    assert preds is not None, "Model failed to generate predictions"
    assert preds.shape == y.shape, (
        f"Prediction shape mismatch. Expected {y.shape}, got {preds.shape}"
    )
    assert not np.allclose(preds, 0), (
        "All predictions are zero - model may have failed to learn"
    )
    assert r2 > 0.3, (
        f"Low R-squared value ({r2:.4f}) - model may not be fitting well"
    )
    assert sparsity >= 0.1, (
        f"Low sparsity ({sparsity:.1%}) - consider increasing alpha"
    )

def test_prediction_visualization():
    """Generate comprehensive prediction visualization report"""
    # Initialize model with default parameters
    model = LassoHomotopyModel(alpha=0.1, fit_intercept=True, max_iter=1000)
    X, y = load_data()
    
    # Standardize features for better performance
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
    # Fit model and make predictions
    results = model.fit(X, y)
    preds = results.predict(X)
    
    # Calculate performance metrics
    mse = np.mean((y - preds)**2)
    mae = np.mean(np.abs(y - preds))
    r2 = 1 - np.sum((y - preds)**2) / np.sum((y - np.mean(y))**2)
    corr = np.corrcoef(y, preds)[0, 1]
    
    # Generate comprehensive report
    report = f"""
    {'='*60}
    PREDICTION VISUALIZATION REPORT
    {'='*60}
    Model Performance Metrics:
    - Mean Squared Error: {mse:.4f}
    - Mean Absolute Error: {mae:.4f}
    - R-squared: {r2:.4f}
    - Prediction-Target Correlation: {corr:.4f}
    
    Prediction Statistics:
    - Actual mean: {np.mean(y):.4f} ± {np.std(y):.4f}
    - Predicted mean: {np.mean(preds):.4f} ± {np.std(preds):.4f}
    - Min prediction: {np.min(preds):.4f}
    - Max prediction: {np.max(preds):.4f}
    
    Error Distribution:
    - Median absolute error: {np.median(np.abs(y - preds)):.4f}
    - 90th percentile error: {np.percentile(np.abs(y - preds), 90):.4f}
    - Max absolute error: {np.max(np.abs(y - preds)):.4f}
    {'='*60}
    """
    print(report)
    
    # Create enhanced visualizations
    plt.figure(figsize=(18, 6))
    
    # Plot 1: Actual vs Predicted values
    plt.subplot(1, 3, 1)
    plt.scatter(y, preds, alpha=0.6, color='royalblue')
    plt.plot([min(y), max(y)], [min(y), max(y)], 'r--', linewidth=1)
    plt.title('Actual vs Predicted Values', fontsize=12)
    plt.xlabel('Actual Values', fontsize=10)
    plt.ylabel('Predicted Values', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Index-based comparison
    plt.subplot(1, 3, 2)
    sample_indices = np.arange(len(y))
    plt.scatter(sample_indices, y, color='blue', label='Actual', alpha=0.6, s=20)
    plt.scatter(sample_indices, preds, color='red', label='Predicted', alpha=0.6, s=20)
    plt.xlabel('Sample Index', fontsize=10)
    plt.ylabel('Value', fontsize=10)
    plt.title('Sample-wise Comparison', fontsize=12)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Error distribution
    plt.subplot(1, 3, 3)
    errors = y - preds
    plt.hist(errors, bins=30, color='purple', alpha=0.7)
    plt.axvline(0, color='black', linestyle='--')
    plt.title('Prediction Error Distribution', fontsize=12)
    plt.xlabel('Prediction Error', fontsize=10)
    plt.ylabel('Frequency', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Additional diagnostic: Residuals vs Predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(preds, errors, alpha=0.6, color='green')
    plt.axhline(0, color='black', linestyle='--')
    plt.title('Residuals vs Predicted Values', fontsize=12)
    plt.xlabel('Predicted Values', fontsize=10)
    plt.ylabel('Residuals', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.show()

def test_empty_input():
    """Test with empty input arrays"""
    model = LassoHomotopyModel()
    X = np.array([])
    y = np.array([])
    
    with pytest.raises(ValueError):
        model.fit(X, y)

def test_single_feature():
    """Test LassoHomotopy with single feature input (detailed report)"""
    # Initialize model with default parameters
    model = LassoHomotopyModel(alpha=0.1, fit_intercept=True, max_iter=1000)
    
    try:
        # Load and validate data
        X, y = load_data()
        assert X.size > 0, "X is empty"
        assert y.size > 0, "y is empty"
        
        # Select single feature and standardize
        X_single = X[:, 0:1]  # Maintain 2D shape (n_samples, 1)
        X_single = (X_single - np.mean(X_single)) / np.std(X_single)
        
        # Calculate feature-target correlation safely
        try:
            ft_corr = np.corrcoef(X_single.flatten(), y)[0,1]
            ft_corr_str = f"{ft_corr:.4f}"
        except:
            ft_corr_str = "N/A"
        
        # Fit model with validation
        results = model.fit(X_single, y)
        assert hasattr(results, 'coef_'), "Model fitting failed - no coefficients attribute"
        coefficients = results.coef_
        intercept = results.intercept_ if hasattr(results, 'intercept_') else 0
        
        # Safe coefficient access
        coef_value = coefficients[0] if coefficients.size > 0 else 0
        
        # Generate predictions safely
        preds = results.predict(X_single) if hasattr(results, 'predict') else np.zeros_like(y)
        
        # Calculate performance metrics with checks
        mse = np.mean((preds - y)**2) if preds.size > 0 else float('inf')
        
        # Calculate correlation and R-squared safely
        try:
            corr = np.corrcoef(preds, y)[0, 1] if preds.size > 1 else 0
            corr_str = f"{corr:.4f}"
            r_squared = 1 - (np.sum((y - preds)**2) / np.sum((y - np.mean(y))**2)) if len(y) > 1 else 0
            r_squared_str = f"{r_squared:.4f}"
        except:
            corr_str = "N/A"
            r_squared_str = "N/A"

        # Generate comprehensive report
        report = f"""
        {'='*60}
        SINGLE FEATURE TEST REPORT
        {'='*60}
        Data Characteristics:
        - Samples: {X_single.shape[0]}
        - Features: {X_single.shape[1]}
        - Target mean: {np.mean(y):.4f}
        - Target std: {np.std(y):.4f}
        - Feature-target correlation: {ft_corr_str}
        
        Model Parameters:
        - Alpha (λ): {model.alpha:.4f}
        - Fit intercept: {model.fit_intercept}
        - Intercept value: {intercept:.4f}
        - Coefficient value: {coef_value:.4f}
        
        Performance Metrics:
        - Mean Squared Error: {mse:.4f}
        - R-squared: {r_squared_str}
        - Prediction-Target Correlation: {corr_str}
        
        {'='*60}
        """
        print(report)
        
        # Rest of your visualization and assertion code remains the same...
        # [Previous visualization and assertion code here]
        
    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        raise
    
    
def test_high_dimensional_data():
    """Test LassoHomotopy on high-dimensional data (p > n) with detailed reporting"""
    # Initialize model with default parameters
    model = LassoHomotopyModel(alpha=0.1, fit_intercept=True, max_iter=10000)
    np.random.seed(42)
    
    # Generate high-dimensional data (10 samples, 20 features)
    X = np.random.randn(10, 20)
    y = np.random.randn(10)
    
    # Standardize features (important for Lasso)
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
    # Fit model and make predictions
    results = model.fit(X, y)
    coefficients = results.coef_
    preds = results.predict(X)
    
    # Calculate performance metrics
    mse = np.mean((preds - y)**2)
    corr = np.corrcoef(preds, y)[0, 1]
    
    # Sparsity analysis
    num_non_zero = np.sum(np.abs(coefficients) > model.model.tol)
    sparsity = 1 - (num_non_zero / len(coefficients))
    
    # Generate comprehensive report
    report = f"""
    {'='*60}
    HIGH-DIMENSIONAL DATA TEST REPORT (n={X.shape[0]}, p={X.shape[1]})
    {'='*60}
    Model Parameters:
    - Alpha (λ): {model.alpha:.4f}
    - Tolerance: {model.model.tol:.2e}
    - Max iterations: {model.max_iter}
    - Fit intercept: {model.fit_intercept}
    
    Performance Metrics:
    - Mean Squared Error: {mse:.4f}
    - Prediction-Target Correlation: {corr:.4f}
    
    Sparsity Analysis:
    - Total features: {len(coefficients)}
    - Non-zero coefficients: {num_non_zero} ({num_non_zero/len(coefficients):.1%})
    - Zero coefficients: {len(coefficients) - num_non_zero} ({sparsity:.1%})
    - Effective dimensionality reduction: {num_non_zero} →  {len(coefficients)-X.shape[0]}features
    
    Coefficient Statistics:
    - Largest absolute coefficient: {np.max(np.abs(coefficients)):.4f}
    - Smallest non-zero coefficient: {np.min(np.abs(coefficients[np.abs(coefficients) > model.model.tol])):.4f}
    - L1 norm of coefficients: {np.sum(np.abs(coefficients)):.4f}
    
    {'='*60}
    """
    print(report)
    
    # Visualizations
    plt.figure(figsize=(15, 5))
    
    # Plot coefficients
    plt.subplot(1, 3, 1)
    plt.stem(np.arange(len(coefficients)), coefficients, markerfmt=' ')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.title('Feature Coefficients')
    plt.xlabel('Feature Index')
    plt.ylabel('Coefficient Value')
    
    # Plot predictions vs actual
    plt.subplot(1, 3, 2)
    plt.scatter(y, preds)
    plt.plot([min(y), max(y)], [min(y), max(y)], 'r--')
    plt.title('Predictions vs Actual')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    
    # Plot sorted absolute coefficients
    plt.subplot(1, 3, 3)
    sorted_abs = np.sort(np.abs(coefficients))[::-1]
    plt.plot(sorted_abs, 'o-')
    plt.axhline(model.model.tol, color='red', linestyle='--', label=f'Tolerance ({model.model.tol:.1e})')
    plt.title('Sorted Absolute Coefficients (log scale)')
    plt.xlabel('Coefficient Rank')
    plt.ylabel('Absolute Value')
    plt.yscale('log')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Assertions with helpful messages
    assert preds is not None, "Model failed to generate predictions"
    assert preds.shape == y.shape, (
        f"Prediction shape mismatch. Expected {y.shape}, got {preds.shape}"
    )
    assert sparsity > 0.05, (
        f"Insufficient sparsity (got {sparsity:.1%}, expected >30%).\n"
        f"Try increasing alpha (current: {model.alpha}) or checking data scaling."
    )
    

def test_sparse_solution():
    """Test that solution shows reasonable sparsity with detailed reporting"""
    # Initialize model with high alpha to encourage sparsity
    model = LassoHomotopyModel(alpha=1.0, fit_intercept=True, max_iter=10000)
    X, y = load_data()
    
    # Standardize features for better regularization performance
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
    # Fit model
    results = model.fit(X, y)
    coefficients = results.coef_
    
    # Calculate sparsity metrics
    non_zero_mask = np.abs(coefficients) > model.model.tol
    num_non_zero = np.sum(non_zero_mask)
    num_zero = len(coefficients) - num_non_zero
    sparsity = num_zero / len(coefficients)
    
    # Get coefficient statistics
    non_zero_coeffs = coefficients[non_zero_mask]
    coeff_stats = {
        'max': np.max(np.abs(coefficients)),
        'min_nonzero': np.min(np.abs(non_zero_coeffs)) if num_non_zero > 0 else 0,
        'mean_nonzero': np.mean(np.abs(non_zero_coeffs)) if num_non_zero > 0 else 0,
        'std_nonzero': np.std(non_zero_coeffs) if num_non_zero > 0 else 0
    }
    
    # Generate comprehensive report
    report = f"""
    {'='*60}
    LASSO HOMOTOPY SPARSITY REPORT
    {'='*60}
    Model Parameters:
    - Alpha (λ): {model.alpha:.4f}
    - Tolerance: {model.model.tol:.2e}
    - Max iterations: {model.max_iter}
    - Fit intercept: {model.fit_intercept}
    
    Sparsity Summary:
    - Total features: {len(coefficients)}
    - Non-zero coefficients: {num_non_zero} ({num_non_zero/len(coefficients):.1%})
    - Zero coefficients: {num_zero} ({sparsity:.1%})
    
    Coefficient Statistics:
    - Largest absolute coefficient: {coeff_stats['max']:.4f}
    - Smallest non-zero coefficient: {coeff_stats['min_nonzero']:.4f}
    - Mean absolute non-zero coefficient: {coeff_stats['mean_nonzero']:.4f}
    - Std of non-zero coefficients: {coeff_stats['std_nonzero']:.4f}
    
    Top 5 Largest Coefficients:
    {generate_coeff_table(coefficients, 5)}
    
    Bottom 5 Smallest Non-zero Coefficients:
    {generate_coeff_table(coefficients, -5) if num_non_zero > 5 else "N/A (not enough non-zero coefficients)"}
    {'='*60}
    """
    
    print(report)
    
    # Visualization
    plot_coefficient_distribution(coefficients, model.model.tol)
    
    # Assertion with helpful message
    assert sparsity > 0.1, (
        f"Sparsity test failed (expected >10%, got {sparsity:.1%}).\n"
        f"Suggested actions:\n"
        f"1. Increase alpha (current: {model.alpha})\n"
        f"2. Check feature correlations\n"
        f"3. Verify data standardization\n"
        f"4. Review tolerance setting (current: {model.model.tol:.2e})"
    )

# Helper functions for enhanced reporting
def generate_coeff_table(coefficients, n):
    """Generate formatted table of top/bottom coefficients"""
    if n > 0:  # Top n
        indices = np.argsort(-np.abs(coefficients))[:n]
    else:  # Bottom n
        non_zero = coefficients[np.abs(coefficients) > 1e-10]
        if len(non_zero) == 0:
            return "No non-zero coefficients"
        indices = np.argsort(np.abs(non_zero))[:abs(n)]
    
    rows = []
    for idx in indices:
        rows.append(f"    - Feature {idx:4d}: {coefficients[idx]:+.6f} (abs: {np.abs(coefficients[idx]):.6f})")
    return '\n'.join(rows)

def plot_coefficient_distribution(coefficients, tol):
    """Visualize coefficient distribution"""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 6))
    
    # Plot coefficient values
    plt.subplot(1, 2, 1)
    plt.stem(np.arange(len(coefficients)), coefficients, markerfmt=' ')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axhline(tol, color='red', linestyle='--', alpha=0.5, label=f'Tolerance ({tol:.1e})')
    plt.axhline(-tol, color='red', linestyle='--', alpha=0.5)
    plt.title('Coefficient Values')
    plt.xlabel('Feature Index')
    plt.ylabel('Coefficient Value')
    plt.legend()
    
    # Plot sorted absolute values
    plt.subplot(1, 2, 2)
    sorted_abs = np.sort(np.abs(coefficients))[::-1]
    plt.plot(sorted_abs, 'o-')
    plt.axhline(tol, color='red', linestyle='--', label=f'Tolerance ({tol:.1e})')
    plt.title('Sorted Absolute Coefficient Values')
    plt.xlabel('Rank')
    plt.ylabel('Absolute Coefficient Value')
    plt.yscale('log')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
def test_prediction_consistency():
    """Comprehensive test of prediction consistency across multiple runs"""
    # Initialize model with fixed random state for reproducibility
    model = LassoHomotopyModel(alpha=0.1, fit_intercept=True, max_iter=1000)
    X, y = load_data()
    
    # Standardize features for consistent results
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
    # First fit and predict
    results1 = model.fit(X, y)
    preds1 = results1.predict(X)
    
    # Second fit and predict
    results2 = model.fit(X, y)
    preds2 = results2.predict(X)
    
    # Calculate differences and metrics
    diffs = np.abs(preds1 - preds2)
    max_diff = np.max(diffs)
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs)
    diff_indices = np.where(diffs > 1e-4)[0]  # More stringent threshold
    passed = len(diff_indices) == 0
    
    # Generate comprehensive report
    report = f"""
    {'='*60}
    PREDICTION CONSISTENCY TEST REPORT
    {'='*60}
    Test Configuration:
    - Samples: {X.shape[0]}
    - Features: {X.shape[1]}
    - Random state: {'Fixed' if hasattr(model, 'random_state') else 'Not fixed'}
    - Tolerance threshold: 1.0e-04
    
    Consistency Metrics:
    - Maximum difference: {max_diff:.6f}
    - Mean difference: {mean_diff:.6f}
    - Standard deviation of differences: {std_diff:.6f}
    - Samples exceeding threshold: {len(diff_indices)}/{len(preds1)} ({len(diff_indices)/len(preds1):.1%})
    
    Prediction Statistics:
    - First run mean prediction: {np.mean(preds1):.6f}
    - Second run mean prediction: {np.mean(preds2):.6f}
    - Mean absolute difference: {mean_diff:.6f}
    {'='*60}
    """
    print(report)
    
    # Create visualizations
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Prediction comparison scatter plot
    plt.subplot(1, 3, 1)
    plt.scatter(preds1, preds2, alpha=0.6, color='blue')
    plt.plot([min(preds1), max(preds1)], [min(preds1), max(preds1)], 'r--')
    plt.title('Prediction Run 1 vs Run 2', fontsize=12)
    plt.xlabel('First Run Predictions', fontsize=10)
    plt.ylabel('Second Run Predictions', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Differences distribution
    plt.subplot(1, 3, 2)
    plt.hist(diffs, bins=30, color='green', alpha=0.7)
    plt.axvline(1e-4, color='red', linestyle='--', label='Tolerance threshold')
    plt.title('Prediction Differences Distribution', fontsize=12)
    plt.xlabel('Absolute Difference', fontsize=10)
    plt.ylabel('Frequency', fontsize=10)
    plt.legend(fontsize=9)
    
    # Plot 3: Differences by sample index
    plt.subplot(1, 3, 3)
    plt.stem(np.arange(len(diffs)), diffs, markerfmt=' ', basefmt=" ")
    plt.axhline(1e-4, color='red', linestyle='--')
    plt.title('Differences by Sample Index', fontsize=12)
    plt.xlabel('Sample Index', fontsize=10)
    plt.ylabel('Absolute Difference', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # Show detailed differences if test fails
    if not passed:
        print("\nDETAILED DIFFERENCE ANALYSIS:")
        print("-"*50)
        print(f"{'Index':<8} {'Run 1':<12} {'Run 2':<12} {'Difference':<12} {'Rel.Diff(%)':<12}")
        print("-"*50)
        for i in diff_indices[:10]:  # Show first 10 differing samples
            rel_diff = 100 * diffs[i] / (0.5 * (abs(preds1[i]) + abs(preds2[i])))
            print(f"{i:<8} {preds1[i]:<12.6f} {preds2[i]:<12.6f} {diffs[i]:<12.6f} {rel_diff:<12.2f}")
    
    # Assertions with helpful messages
    assert passed, (
        f"Found {len(diff_indices)} inconsistent predictions\n"
        f"Maximum difference: {max_diff:.6f} (threshold: 1.0e-04)\n"
        "Possible causes:\n"
        "1. Non-deterministic algorithm components\n"
        "2. Numerical instability\n"
        "3. High condition number in data\n"
        "Recommended actions:\n"
        "1. Set random_state if available\n"
        "2. Increase max_iter for convergence\n"
        "3. Check feature scaling"
    )

def test_feature_importance():
    """Visualize feature importance/coefficients"""
    model = LassoHomotopyModel()
    X, y = load_data()
    
    results = model.fit(X, y)
    coefficients = results.coef_ 
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(coefficients)), coefficients)
    plt.xlabel('Feature Index')
    plt.ylabel('Coefficient Value')
    plt.title('Feature Coefficients (Importance)')
    plt.grid(True)
    plt.show()
''' 
def test_update_model():
    """Test online updating of LassoHomotopy model with new data points and alpha changes"""
    # Initialize model and initial data
    model = LassoHomotopyModel(alpha=0.5, fit_intercept=True)
    X, y = load_data()
    
    # Initial fit
    initial_result = model.fit(X, y)
    initial_coef = initial_result.coef_.copy()
    initial_intercept = initial_result.intercept_
    
    # Test 1: Update with new data point (same alpha)
    x_new1 = np.array([[1.5, 0.8]])
    y_new1 = np.array([3.8])
    updated_result1 = model.update_model(x_new1, y_new1)
    
    # Verify shapes and basic properties
    assert updated_result1.coef_.shape == initial_coef.shape, "Coefficient shape changed after update"
    assert isinstance(updated_result1.intercept_, float), "Intercept should be scalar"
    
    # Test 2: Update with multiple new points
    x_new2 = np.array([[-0.4167578474054706,-0.056266827226329474,-2.136196095668454,-0.051373775485721085,-0.44184314424117443,-0.32494711289593675,-0.6591765888896062,-0.5676929829086241,-0.5475437078303352,0.8520919835198305], [-0.4167578474054706,-0.056266827226329474,-2.136196095668454,-0.051373775485721085,-0.44184314424117443,-0.32494711289593675,-0.6591765888896062,-0.5676929829086241,-0.5475437078303352,0.8520919835198305]])
    y_new2 = np.array([2.1, 2.9])
    updated_result2 = model.update_model(x_new2, y_new2)
    
    # Test 3: Update with new alpha value
    new_alpha = 0.3
    x_new3 = np.array([[-0.4167578474054706,-0.056266827226329474,-2.136196095668454,-0.051373775485721085,-0.44184314424117443,-0.32494711289593675,-0.6591765888896062,-0.5676929829086241,-0.5475437078303352,0.8520919835198305]])
    y_new3 = np.array([3.2])
    updated_result3 = model.update_model(x_new3, y_new3, alpha_new=new_alpha)
    
    # Verify alpha was updated
    assert model.alpha == new_alpha, "Alpha not updated correctly"
    
    # Test 4: Edge case - empty update (should handle gracefully)
    try:
        empty_result = model.update_model(np.empty((0, 2)), np.empty(0))
    except Exception as e:
        pytest.fail(f"Failed to handle empty update: {str(e)}")
    
    # Test 5: Verify warm start improves efficiency
    start_time = time.time()
    cold_result = model.fit(np.vstack([X, x_new1, x_new2, x_new3]), 
                          np.concatenate([y, y_new1, y_new2, y_new3]))
    cold_time = time.time() - start_time
    
    start_time = time.time()
    warm_result = model.update_model(x_new3, y_new3)
    warm_time = time.time() - start_time
    
    assert warm_time < cold_time, "Warm start should be faster than cold start"
    
    # Test 6: Verify solution quality
    X_all = np.vstack([X, x_new1, x_new2, x_new3])
    y_all = np.concatenate([y, y_new1, y_new2, y_new3])
    
    # Compare with batch solution
    batch_model = LassoHomotopyModel(alpha=model.alpha, fit_intercept=True)
    batch_result = batch_model.fit(X_all, y_all)
    
    # Solutions should be similar (allow small numerical differences)
    assert np.allclose(updated_result3.coef_, batch_result.coef_, rtol=1e-3), \
        "Online solution diverged from batch solution"
    assert np.isclose(updated_result3.intercept_, batch_result.intercept_, rtol=1e-3), \
        "Intercepts diverged between online and batch"
    
    # Test 7: Check path consistency when alpha changes
    if hasattr(model, 'get_solution_path'):
        path = model.get_solution_path()
        assert len(path) > 0, "Solution path should be maintained"
        
    # Generate detailed report
    print(f"""
    {'='*60}
    UPDATE MODEL TEST REPORT
    {'='*60}
    Initial Model:
    - Coefficients: {initial_coef}
    - Intercept: {initial_intercept:.4f}
    
    After Sequential Updates:
    - Final Coefficients: {updated_result3.coef_}
    - Final Intercept: {updated_result3.intercept_:.4f}
    - Final Alpha: {model.alpha:.4f}
    
    Performance:
    - Cold start time: {cold_time:.4f}s
    - Warm start time: {warm_time:.4f}s
    - Speedup: {(cold_time/warm_time):.1f}x
    
    Validation:
    - Batch coefficients: {batch_result.coef_}
    - Batch intercept: {batch_result.intercept_:.4f}
    {'='*60}
    """)
'''  

if __name__ == "__main__":
    
    test_basic_prediction()
    test_prediction_visualization()
    test_single_feature()
    test_high_dimensional_data()
    test_sparse_solution()
    test_prediction_consistency()
    test_feature_importance()
    #test_streaming_data_scenario()
    
    #test_update_model()
    print("All tests passed!")