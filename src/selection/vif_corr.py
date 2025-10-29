import numpy as np
import pandas as pd
from typing import List, Tuple
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

def vif_table(X: pd.DataFrame, threshold: float = 5.0) -> pd.DataFrame:
    """
    Calculate VIF for each feature and return as sorted DataFrame.
    
    Args:
        X: Feature DataFrame
        threshold: VIF threshold for flagging
        
    Returns:
        DataFrame with features and their VIF scores
    """
    # Drop any constant columns
    X = X.select_dtypes(include=[np.number]).copy()
    X = X.loc[:, X.std() > 0]
    
    if X.shape[0] <= X.shape[1] + 1:
        return pd.DataFrame(columns=["feature", "vif"])
        
    # Add constant term
    X_const = sm.add_constant(X)
    
    # Calculate VIF for each feature
    vif_data = []
    for i, col in enumerate(X.columns):
        try:
            vif = variance_inflation_factor(X_const.values, i+1)
            flag = "high_vif" if vif > threshold else ""
            vif_data.append({
                "feature": col,
                "vif": float(vif),
                "flag": flag
            })
        except Exception:
            continue
            
    vif_df = pd.DataFrame(vif_data)
    return vif_df.sort_values("vif", ascending=False)

def correlation_matrix(X: pd.DataFrame, threshold: float = 0.8) -> Tuple[pd.DataFrame, List[Tuple[str, str, float]]]:
    """
    Generate correlation matrix and identify highly correlated pairs.
    
    Args:
        X: Feature DataFrame
        threshold: Correlation threshold for flagging
        
    Returns:
        Tuple of (correlation matrix, list of high correlation pairs)
    """
    corr_matrix = X.corr()
    
    # Find highly correlated feature pairs
    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr = abs(corr_matrix.iloc[i, j])
            if corr > threshold:
                high_corr.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    float(corr)
                ))
                
    return corr_matrix, sorted(high_corr, key=lambda x: x[2], reverse=True)

def select_features_vif(X: pd.DataFrame, threshold: float = 5.0) -> List[str]:
    """
    Select features by iteratively removing highest VIF features.
    
    Args:
        X: Feature DataFrame
        threshold: VIF threshold for removal
        
    Returns:
        List of selected feature names
    """
    X = X.select_dtypes(include=[np.number]).copy()
    features = list(X.columns)
    
    while len(features) > 0:
        vif_stats = vif_table(X[features])
        if vif_stats.empty or vif_stats["vif"].max() <= threshold:
            break
        # Remove highest VIF feature
        worst_feature = vif_stats.iloc[0]["feature"]
        features.remove(worst_feature)
        
    return features