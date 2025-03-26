import numpy as np

class LassoHomotopy:
    """
    Complete implementation of RecLasso homotopy algorithm (Garrigues & El Ghaoui) with:
    - Batch initialization
    - Online updates via homotopy
    - Proper standardization
    - Numerical stability safeguards
    """
    
    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4, fit_intercept=True):
        self.alpha = alpha          # Regularization parameter (µ)
        self.max_iter = max_iter    # Maximum iterations for batch fitting
        self.tol = tol              # Convergence tolerance
        self.fit_intercept = fit_intercept
        self.coef_ = None           # Model coefficients (θ)
        self.intercept_ = 0.0       # Intercept term
        self.active_set = []        # Active feature set
        self.signs = []             # Signs of active coefficients (v)
        self.X_mean_ = None         # For standardization
        self.X_std_ = None          # For standardization
        self.y_mean_ = 0.0          # For centering
        self.inv_XtX = None         # For rank-1 updates
        self.X_ = None              # Storage for online updates
        self.y_ = None              # Storage for online updates

    def _standardize(self, X, y=None, fit=False):
        """Handle standardization with intercept"""
        if fit:
            if self.fit_intercept:
                self.X_mean_ = np.mean(X, axis=0)
                self.X_std_ = np.std(X, axis=0)
                self.X_std_[self.X_std_ == 0] = 1.0  # Handle constant features
                X_std = (X - self.X_mean_) / self.X_std_
                if y is not None:
                    self.y_mean_ = np.mean(y)
                    y_centered = y - self.y_mean_
                return X_std, y_centered
            return X, y
        else:
            if self.fit_intercept:
                return (X - self.X_mean_) / self.X_std_, y
            return X, y

    def fit(self, X, y):
        """Batch initialization of the model (θ(0, µ))"""
        # Standardize data
        X_std, y_centered = self._standardize(X, y, fit=True)
        
        n_samples, n_features = X_std.shape
        self.coef_ = np.zeros(n_features)
        self.X_ = X_std.copy()
        self.y_ = y_centered.copy()
        
        # Initialize with most correlated feature
        correlation = X_std.T @ y_centered / n_samples
        j_init = np.argmax(np.abs(correlation))
        self.active_set = [j_init]
        self.signs = [np.sign(correlation[j_init])]
        
        # Main fitting loop
        for _ in range(self.max_iter):
            X_active = X_std[:, self.active_set]
            sign_vector = np.array(self.signs)
            
            # Solve for active coefficients (Step 4 update)
            try:
                self.inv_XtX = np.linalg.inv(
                    X_active.T @ X_active + self.alpha * np.eye(len(self.active_set)))
                coef_active = self.inv_XtX @ (X_active.T @ y_centered - self.alpha * sign_vector)
            except np.linalg.LinAlgError:
                self.inv_XtX = np.linalg.pinv(
                    X_active.T @ X_active + self.alpha * np.eye(len(self.active_set)))
                coef_active = self.inv_XtX @ (X_active.T @ y_centered - self.alpha * sign_vector)
            
            # Update coefficients
            new_coef = np.zeros(n_features)
            new_coef[self.active_set] = coef_active
            delta = np.linalg.norm(new_coef - self.coef_)
            self.coef_ = new_coef
            
            # Update residual and correlations
            residual = y_centered - X_std @ self.coef_
            correlation = X_std.T @ residual / n_samples
            
            # Find features to add/remove
            to_add = [j for j in range(n_features) 
                     if j not in self.active_set 
                     and abs(correlation[j]) > self.alpha * (1 + self.tol)]
            
            to_remove = [idx for idx, j in enumerate(self.active_set)
                        if abs(self.coef_[j]) < self.tol]
            
            # Update active set
            for j in sorted(to_add, reverse=True):
                if j not in self.active_set:
                    self.active_set.append(j)
                    self.signs.append(np.sign(correlation[j]))
            
            for idx in sorted(to_remove, reverse=True):
                self.active_set.pop(idx)
                self.signs.pop(idx)
            
            # Check convergence
            if len(to_add) == 0 and len(to_remove) == 0 and delta < self.tol:
                break
        
        # Set intercept
        if self.fit_intercept:
            self.intercept_ = self.y_mean_ - np.dot(self.X_mean_ / self.X_std_, self.coef_)
        
        return self

    def update_model(self, x_new, y_new, alpha_new=None):
        """
        Online update per Algorithm 1:
        1. Update µ path (if alpha_new specified)
        2. Process new (x_{n+1}, y_{n+1}) with homotopy
        """
        if self.coef_ is None:
            raise ValueError("Model must be fitted before updating")
        
        # Standardize new sample using existing parameters
        x_new = np.asarray(x_new).reshape(-1)
        x_new_std = (x_new - self.X_mean_) / self.X_std_
        y_new_centered = y_new - self.y_mean_
        
        # Step 1: Update regularization if needed
        if alpha_new is not None:
            self._update_regularization(alpha_new)
        
        # Steps 2-5: Process new data point
        self._process_single_update(x_new_std, y_new_centered)
        
        return self

    def _update_regularization(self, new_alpha):
        """Step 1: Homotopy for µ changes (µ_n → µ_{n+1})"""
        if new_alpha == self.alpha:
            return
            
        # Create path from current alpha to new alpha
        alphas = np.linspace(self.alpha, new_alpha, num=20)
        
        for alpha in alphas:
            self.alpha = alpha  # Update current alpha
            
            if self.active_set:
                X_active = self.X_[:, self.active_set]
                sign_vector = np.array(self.signs)
                
                # Solve with current alpha (Step 4 update)
                coef_active = self._rank_one_update(X_active, self.y_, sign_vector)
                
                # Update coefficients
                new_coef = np.zeros(len(self.coef_))
                new_coef[self.active_set] = coef_active
                self.coef_ = new_coef
                
                # Check for changes in active set
                residual = self.y_ - self.X_ @ self.coef_
                correlation = self.X_.T @ residual / len(self.y_)
                
                to_remove = [idx for idx, j in enumerate(self.active_set)
                            if abs(self.coef_[j]) < self.tol]
                
                # Remove features that became zero
                for idx in sorted(to_remove, reverse=True):
                    self.active_set.pop(idx)
                    self.signs.pop(idx)
        
        self.alpha = new_alpha  # Ensure final alpha is set correctly

    def _process_single_update(self, x_new, y_new):
        """Steps 2-5: Process single (x_{n+1}, y_{n+1}) with homotopy (t: 0→1)"""
        # Step 2: Initialize with current active set
        active_set = self.active_set.copy()
        signs = self.signs.copy()
        x_new_active = x_new[active_set]
        
        # Initialize θ1
        X_active = self.X_[:, active_set]
        theta_active = self._rank_one_update(X_active, self.y_, signs)
        
        t = 0
        while t < 1:
            # Step 3: Compute all possible transition points
            transition_points = []
            
            # Case 1: Coefficients → 0
            for idx, j in enumerate(active_set):
                if abs(theta_active[idx]) < self.tol:
                    t_zero = -self.coef_[j] / (theta_active[idx] - self.coef_[j])
                    if 0 < t_zero <= 1:
                        transition_points.append((t_zero, j, 'zero'))
            
            # Case 2: New features enter
            residual = y_new - x_new @ self.coef_
            correlation = x_new * residual
            for j in range(len(self.coef_)):
                if j not in active_set:
                    c_j = correlation[j]
                    if abs(c_j) > self.alpha * (1 + self.tol):
                        t_enter = (self.alpha - abs(c_j)) / (self.alpha - abs(c_j)/t if t !=0 else 1)
                        if 0 < t_enter <= 1:
                            transition_points.append((t_enter, j, 'enter'))
            
            if not transition_points:
                break
                
            # Find next transition point
            t_next, j_event, event_type = min(transition_points, key=lambda x: x[0])
            t_next = min(t_next, 1)  # Don't exceed t=1
            
            # Step 4: Update to this transition point
            delta_t = t_next - t
            self.coef_[active_set] += delta_t * theta_active
            t = t_next
            
            # Handle event
            if event_type == 'zero':
                idx = active_set.index(j_event)
                active_set.pop(idx)
                signs.pop(idx)
            elif event_type == 'enter':
                active_set.append(j_event)
                signs.append(np.sign(correlation[j_event]))
            
            # Update θ1 for new active set
            if active_set:
                X_active = self.X_[:, active_set]
                theta_active = self._rank_one_update(X_active, self.y_, signs, 
                                                   x_new[active_set], y_new, t)
        
        # Step 5: Final update to t=1
        if active_set:
            delta_t = 1 - t
            self.coef_[active_set] += delta_t * theta_active
        
        # Update model state
        self.active_set = [j for j in range(len(self.coef_)) if abs(self.coef_[j]) > self.tol]
        self.signs = np.sign(self.coef_[self.active_set]).tolist()
        
        # Update stored data
        self.X_ = np.vstack([self.X_, x_new])
        self.y_ = np.append(self.y_, y_new)

    def _rank_one_update(self, X1, y, sign_vector, x_new1=None, y_new=None, t=0):
        """Efficient rank-1 update for (X₁ᵀX₁ + αI)⁻¹"""
        # Initialize inverse if needed
        if (self.inv_XtX is None or 
            self.inv_XtX.shape[0] != len(self.active_set)):
            self.inv_XtX = np.linalg.inv(
                X1.T @ X1 + self.alpha * np.eye(len(self.active_set)))
        
        # Sherman-Morrison update for new data
        if x_new1 is not None and y_new is not None:
            u = x_new1.reshape(-1, 1)
            denominator = 1 + u.T @ self.inv_XtX @ u
            if np.abs(denominator) > 1e-10:
                self.inv_XtX -= (self.inv_XtX @ u @ u.T @ self.inv_XtX) / denominator
        
        # Solve system
        rhs = X1.T @ y - self.alpha * np.array(sign_vector).reshape(-1, 1)
        return (self.inv_XtX @ rhs).flatten()

    def predict(self, X):
        """Predict with proper standardization"""
        if self.coef_ is None:
            raise ValueError("Model not fitted yet")
        X_std, _ = self._standardize(X)
        return X_std @ self.coef_ + self.intercept_
    
    
class LassoHomotopyModel:
    """User-friendly wrapper for LassoHomotopy"""
    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4, fit_intercept=True):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.model = None
        self.coef_ = None  
        self._is_fitted = False  # Track if model has been fitted

    def fit(self, X, y):
        if X.size == 0 or y.size == 0:
            raise ValueError("Input arrays cannot be empty")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y have incompatible shapes")
            
        self.model = LassoHomotopy(
            alpha=self.alpha,
            max_iter=self.max_iter,
            tol=self.tol,
            fit_intercept=self.fit_intercept
        )
        self.model.fit(X, y)
        self._is_fitted = True
        return LassoHomotopyResults(self.model, self.fit_intercept)
    
    def update_model(self, x_new, y_new, alpha_new=None):
        """
        Online update per Algorithm 1:
        1. Update µ path (if alpha_new specified)
        2. Process new (x_{n+1}, y_{n+1}) with homotopy
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before updating")
        
        # Validate input shapes
        x_new = np.asarray(x_new)
        y_new = np.asarray(y_new)
        
        if x_new.ndim == 1:
            x_new = x_new.reshape(1, -1)
        if self.coef_ is None:
            raise ValueError("Model coefficients not available")
        if x_new.shape[1] != self.coef_.shape[0]:
            raise ValueError(f"Expected {self.coef_.shape[0]} features, got {x_new.shape[1]}")
        if x_new.shape[0] != y_new.shape[0]:
            raise ValueError("x_new and y_new have incompatible shapes")
        
        # Update alpha if specified
        if alpha_new is not None:
            self.alpha = alpha_new
            self.model.alpha = alpha_new  # Update the underlying model's alpha
        
        # Update stored data
        self.X = np.vstack([self.X, x_new])
        self.y = np.concatenate([self.y, y_new])
        
        # Perform the homotopy update
        results = self.model.update_model(x_new, y_new)
        
        # Update our coefficients
        self.coef_ = results.coef_
        if self.fit_intercept:
            self.intercept_ = results.intercept_
        
        return results


class LassoHomotopyResults:
    """Results container with proper coefficient handling"""
    def __init__(self, model, fit_intercept):
        self.model = model
        self.fit_intercept = fit_intercept

    @property
    def coef_(self):
        if self.fit_intercept:
            return self.model.coef_[1:] if hasattr(self.model, 'coef_') else None
        return self.model.coef_ if hasattr(self.model, 'coef_') else None

    @property
    def intercept_(self):
        if self.fit_intercept:
            return self.model.intercept_ if hasattr(self.model, 'intercept_') else 0.0
        return 0.0

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not fitted yet")
        return self.model.predict(X)