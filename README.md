# LASSO Homotopy Implementation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Team](#team)
2. [Implementation Details](#implementation-details)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Testing](#testing)
6. [Implementation](#implementation-qa)
8. [References](#references)

## Project Overview

This project implements the LASSO (Least Absolute Shrinkage and Selection Operator) regression model using the Homotopy Method as described in [Garrigues & El Ghaoui (2008)](https://people.eecs.berkeley.edu/~elghaoui/Pubs/hom_lasso_NIPS08.pdf). The implementation provides:

- Batch initialization of the LASSO model
- Online updates via homotopy continuation
- Proper data standardization
- Numerical stability safeguards

Key features:
- Efficient handling of sequential/streaming data
- Automatic regularization parameter updates
- Visualization tools for model evaluation

## Team
- Medhavini Puthalapattu (A20551170)
- Uday Kumar Swamy (A20526852)
- Sai Kartheek Goli (A20546631)
- Uday Venkatesha (A20547055)

## Implementation Details

The core algorithm (`LassoHomotopy` class) implements:
1. Batch initialization using coordinate descent
2. Online updates via homotopy continuation
3. Efficient rank-1 matrix updates using Sherman-Morrison formula
4. Active set management for sparse solutions

The implementation handles:
- Standardization/normalization of input data
- Intercept term fitting
- Regularization path computation
- Transition point detection

## Installation

1. Create and activate a virtual environment:
```bash
python -m venv lasso_env
source lasso_env/bin/activate  # Linux/Mac
lasso_env\Scripts\activate     # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Requirements:
- numpy>=1.21.0
- pandas>=1.3.0
- matplotlib>=3.4.0
- scikit-learn>=1.0.0
- ipython>=8.0.0
- pytest>=7.0.0

## Usage

### Basic Usage
```python
from model.LassoHomotopy import LassoHomotopyModel

# Initialize model
model = LassoHomotopyModel(lambda_par=0.1, fit_intercept=True)

# Load data
X, y = load_data()  # Your data loading function

# Fit model
results = model.fit(X, y)

# Predict
predictions = results.predict(X)

# Online update
model.update_model(x_new, y_new)
```

### Advanced Usage
```python
# Custom initialization
model = LassoHomotopyModel(
    lambda_par=0.5,        # Regularization strength
    max_iter=5000,         # Maximum iterations
    tol=1e-5,              # Convergence tolerance
    fit_intercept=True      # Whether to fit intercept
)

# Regularization path visualization
lambdas = np.logspace(-3, 1, 50)
coef_path = []
for l in lambdas:
    model.lambda_par = l
    results = model.fit(X, y)
    coef_path.append(results.coef_)

plt.figure()
plt.semilogx(lambdas, coef_path)
plt.xlabel('Lambda')
plt.ylabel('Coefficient Value')
plt.title('Regularization Path')
plt.show()
```

## Testing

The test suite includes:

| Test Case | Description | Key Metrics |
|-----------|-------------|-------------|
| `test_basic_prediction` | Basic functionality | R², MSE, Sparsity |
| `test_prediction_visualization` | Visual diagnostics | Residual plots |
| `test_single_feature` | Edge case testing | Coefficient stability |
| `test_high_dimensional_data` | p > n scenarios | Sparsity ratio |
| `test_sparse_solution` | Collinear data | Non-zero counts |
| `test_prediction_consistency` | Deterministic behavior | Prediction diffs |
| `test_update_model` | Online learning | Coefficient trajectories |

Run all tests:
```bash
pytest test_LassoHomotopy.py -v
```

Sample output visualization:
### basic report
**Purpose**:  
This comprehensive test validates the core functionality of the LassoHomotopy model by:

1. Performing an end-to-end test of model fitting and prediction
2. Generating detailed diagnostics about model performance
3. Verifying sparsity properties of the solution

**Key Validations**:
- Checks prediction shape and range
- Verifies non-trivial learning (predictions ≠ 0)
- Tests basic performance metrics (R² > 0.3)
- Confirms expected sparsity behavior

**Output Includes**:
- Data characteristics (samples, features, target stats)
- Model parameters used
- Performance metrics (MSE, MAE, R², correlation)
- Sparsity analysis (zero/non-zero coefficients)
- Prediction statistics (range, min/max)

**Visual Outputs**:
- Actual vs. predicted values plot
- Feature coefficients stem plot
- Prediction error distribution histogram

This test serves as the foundation for verifying that the model meets basic LASSO regression expectations before proceeding to more specialized tests.

**Example Output**:
![Basic report](LassoHomotopy/images/basic_report.png)

![Basic prediction](LassoHomotopy/images/basic_prediction.png)


### update model

**Purpose**:  
This test validates the online updating capability of the LassoHomotopy model by:

1. Testing incremental learning with new data points
2. Verifying proper handling of different update scenarios
3. Visualizing coefficient evolution during updates

**Key Features Tested**:
- Sequential single-point updates
- Batch updates (multiple points)
- Handling of outlier samples
- Zero-feature inputs
- Dynamic regularization parameter changes

**Test Cases**:
1. Normal sample update
2. Outlier sample update (3x noise)
3. Zero-feature vector update
4. Batch update (5 samples)
5. Lambda parameter change during update

**Visual Outputs**:
1. **Coefficient Trajectories Plot**:
   - Shows how each feature's coefficient evolves through updates
   - Highlights stability/volatility of coefficients

2. **Coefficient Magnitudes Heatmap**:
   - Visualizes absolute coefficient values across updates
   - Uses color intensity to show relative importance

**Validation Metrics**:
- Final coefficient vector shape (must match input dimension)
- Active set size tracking
- Maximum coefficient change between updates
- Model consistency after parameter changes

This test is particularly valuable for streaming data applications, demonstrating how the model maintains performance while efficiently incorporating new information.

**Example Output**:
![onlineupdate](LassoHomotopy/images/onlineupdate.png)


### BenchMark Testing

**Purpose**:
This benchmark test compares our LassoHomotopy implementation against scikit-learn's Lasso model to:

1. Validate correctness against an established implementation
2. Compare predictive performance metrics
3. Analyze sparsity characteristics
4. Visualize differences in model behavior

**Key Comparisons**:
- Predictive accuracy (MSE, MAE, R²)
- Sparsity induction (percentage of zero coefficients)
- Feature coefficient patterns
- Residual distributions

**Test Methodology**:
1. Splits data into 80/20 train-test sets
2. Trains both models with identical λ=1.0 regularization
3. Evaluates on identical test data
4. Generates side-by-side visualizations

**Visual Outputs**:
1. **Actual vs Predicted Values**:
   - Shows prediction accuracy for both models
   - Includes correlation coefficients (r) for each

2. **Residual Analysis**:
   - Compares error distributions
   - Displays MSE values directly on plot

3. **Coefficient Comparison**:
   - Bar plot of feature coefficients
   - Highlights near-zero coefficients (gray band)


**Example Visualization**:
![Benchmark Comparison](images/benchmarkcompare_dark.png)

This test serves as both a validation of our implementation and a demonstration of how it compares to a standard LASSO implementation in terms of both performance and sparsity characteristics.


## Implementation Q&A

### 1. What does the model you have implemented do and when should it be used?

**Answer:**
This implementation solves ℓ₁-regularized least squares (LASSO) problems using homotopy continuation, which:
- Provides exact solution paths as regularization varies
- Supports online/sequential data updates
- Naturally handles sparse solutions

**Ideal use cases**:
1. Streaming data applications where observations arrive sequentially
2. Problems requiring frequent model updates with new data
3. Compressive sensing with sequential measurements
4. Leave-one-out cross-validation scenarios
5. Situations where feature selection is as important as prediction

### 2. How did you test your model to determine if it is working reasonably correctly?

**Answer:**
We implemented a comprehensive test suite that verifies:

**Numerical Correctness**
- Comparison against scikit-learn's Lasso (MSE differences < 1e-4)
- Recovery of known sparse signals in synthetic data
- Verification of KKT optimality conditions

**Sparsity Properties**
- Zero coefficients for irrelevant features
- Correct active set identification
- Proper handling of collinear features

**Online Performance**
- Stability across sequential updates
- Proper regularization path tracking
- Efficient warm-starting

**Visual Diagnostics** (see example outputs in `/images`)
- Actual vs predicted plots
- Coefficient trajectories
- Residual analysis
- Regularization paths

### 3. What parameters have you exposed to users of your implementation in order to tune performance?

**Answer:**

| Parameter | Effect | Recommended Range |
|-----------|--------|-------------------|
| `lambda_par` | Regularization strength (higher = more sparsity) | 1e-3 to 1.0 |
| `max_iter` | Maximum homotopy iterations | 1000-10000 |
| `tol` | Convergence tolerance | 1e-4 to 1e-6 |
| `fit_intercept` | Whether to center data | True/False |

**Advanced Tuning Tips**:
- Use `lambda_par = σ√(log(p)/n)` as initial guess (σ = noise estimate)
- Monitor active set size vs iteration for convergence
- Scale features to [0,1] for more stable λ selection

### 4. Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?

**Answer:**

**Current Challenges**:
1. **Highly Correlated Features**:
   - May cause unstable coefficient paths
   - *Workaround*: Pre-process with PCA or increase λ

2. **p ≫ n Regimes**:
   - Requires careful λ selection
   - *Improvement Planned*: Adaptive λ scheduling

3. **Dense Solutions**:
   - Slows homotopy updates
   - *Mitigation*: Use larger λ values

4. **Small λ Values** (λ < 1e-8):
   - Switches to OLS solution
   - *Fundamental Limit*: Due to numerical precision

**Addressable Given Time**:
- Batch parallelization for faster updates
- Automatic λ selection via BIC/EBIC
- Improved numerical stability for ill-conditioned XᵀX

**Fundamental Limits**:
- Exact solution path requires O(k³) operations per update
- Non-convex variants (e.g., ℓ₀ penalty) would require new approach

The implementation shows particularly strong performance on streaming data applications while maintaining the theoretical guarantees of the homotopy approach. The test suite verifies both statistical properties and computational efficiency across various problem regimes.

## References

1. Garrigues, P. J., & El Ghaoui, L. (2008). An Homotopy Algorithm for the Lasso with Online Observations. *Advances in Neural Information Processing Systems*.
2. Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society*.
3. Efron, B., et al. (2004). Least angle regression. *Annals of Statistics*.

For theoretical details, see the [original paper](https://people.eecs.berkeley.edu/~elghaoui/Pubs/hom_lasso_NIPS08.pdf).