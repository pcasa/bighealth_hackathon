import numpy as np
import pandas as pd
from scipy import stats
import pickle
import json
import os
from datetime import datetime
import yaml
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import torch

class DataDriftDetector:
    def __init__(self, config_path='config/model_config.yaml'):
        """Initialize the data drift detector"""
        # Load configuration
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            self.config = config['data_drift_detection']
        
        self.reference_data = None
        self.reference_stats = None
        self.feature_importance = {}
        self.alert_thresholds = self.config['alert_thresholds']
        self.pca_model = None
        self.feature_columns = None
    
    def set_reference_data(self, reference_data, feature_columns=None):
        """Set the reference data to use for drift detection"""
        self.reference_data = reference_data
        
        if feature_columns is None:
            # Use all numeric columns
            feature_columns = reference_data.select_dtypes(include=['number']).columns.tolist()
        
        self.feature_columns = feature_columns
        
        # Calculate reference statistics
        self.reference_stats = self._calculate_statistics(reference_data[feature_columns])
        
        # Initialize PCA for multivariate drift detection
        if 'pca_reconstruction' in self.config['multivariate_methods']:
            self.pca_model = PCA(n_components=0.95)  # Capture 95% of variance
            self.pca_model.fit(reference_data[feature_columns])
        
        print(f"Reference data set with {len(feature_columns)} features")
    
    def detect_drift(self, current_data):
        """Detect drift between reference and current data"""
        if self.reference_stats is None:
            raise ValueError("Reference data not set. Call set_reference_data first.")
        
        # Check that all required columns are present
        missing_cols = set(self.feature_columns) - set(current_data.columns)
        if missing_cols:
            raise ValueError(f"Current data missing required columns: {missing_cols}")
        
        # Calculate current statistics
        current_stats = self._calculate_statistics(current_data[self.feature_columns])
        
        # Univariate drift detection
        univariate_results = {}
        
        if 'ks_test' in self.config['univariate_methods']:
            ks_results = self._kolmogorov_smirnov_test(current_data)
            univariate_results['ks_test'] = ks_results
        
        if 'chi_square' in self.config['univariate_methods']:
            chi2_results = self._chi_square_test(current_data)
            univariate_results['chi_square'] = chi2_results
        
        # Multivariate drift detection
        multivariate_results = {}
        
        if 'pca_reconstruction' in self.config['multivariate_methods']:
            pca_results = self._pca_reconstruction_error(current_data)
            multivariate_results['pca_reconstruction'] = pca_results
        
        if 'mmd' in self.config['multivariate_methods']:
            mmd_results = self._maximum_mean_discrepancy(current_data)
            multivariate_results['mmd'] = mmd_results
        
        # Calculate overall drift score
        drift_score = self._calculate_drift_score(univariate_results, multivariate_results)
        
        # Determine drift status based on threshold
        drift_detected = drift_score >= self.alert_thresholds['drift_score']
        
        # Calculate feature importance if drift detected
        feature_importance = {}
        if drift_detected:
            feature_importance = self._calculate_feature_importance(univariate_results)
        
        # Prepare results
        results = {
            'drift_score': drift_score,
            'drift_detected': drift_detected,
            'univariate_results': univariate_results,
            'multivariate_results': multivariate_results,
            'feature_importance': feature_importance,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return results
    
    def _calculate_statistics(self, data):
        """Calculate statistics for each feature"""
        stats_dict = {}
        
        for column in data.columns:
            stats_dict[column] = {
                'mean': data[column].mean(),
                'median': data[column].median(),
                'std': data[column].std(),
                'min': data[column].min(),
                'max': data[column].max(),
                'q1': data[column].quantile(0.25),
                'q3': data[column].quantile(0.75),
                'histogram': np.histogram(data[column].dropna(), bins=20)
            }
        
        return stats_dict
    
    def _kolmogorov_smirnov_test(self, current_data):
        """Perform Kolmogorov-Smirnov test for each feature"""
        ks_results = {}
        
        for column in self.feature_columns:
            # Get reference and current data
            reference_values = self.reference_data[column].dropna()
            current_values = current_data[column].dropna()
            
            # Skip if not enough data
            if len(reference_values) < 10 or len(current_values) < 10:
                ks_results[column] = {'statistic': np.nan, 'p_value': np.nan, 'drift': False}
                continue
            
            # Perform KS test
            statistic, p_value = stats.ks_2samp(reference_values, current_values)
            
            # Determine if drift detected (p-value < 0.05)
            drift = p_value < 0.05
            
            ks_results[column] = {
                'statistic': statistic,
                'p_value': p_value,
                'drift': drift
            }
        
        return ks_results
    
    def _chi_square_test(self, current_data):
        """Perform Chi-Square test for each feature using binned data"""
        chi2_results = {}
        
        for column in self.feature_columns:
            # Get reference and current histograms
            ref_hist, ref_bins = self.reference_stats[column]['histogram']
            
            # Create histogram for current data with same bins
            current_values = current_data[column].dropna()
            curr_hist, _ = np.histogram(current_values, bins=ref_bins)
            
            # Add small epsilon to avoid division by zero
            epsilon = 1e-10
            ref_hist = ref_hist + epsilon
            curr_hist = curr_hist + epsilon
            
            # Normalize histograms
            ref_hist = ref_hist / ref_hist.sum()
            curr_hist = curr_hist / curr_hist.sum()
            
            # Calculate chi-square statistic
            chi2_stat = np.sum((ref_hist - curr_hist)**2 / ref_hist)
            
            # Degrees of freedom is bins - 1
            dof = len(ref_hist) - 1
            
            # Calculate p-value
            p_value = 1 - stats.chi2.cdf(chi2_stat, dof)
            
            # Determine if drift detected (p-value < 0.05)
            drift = p_value < 0.05
            
            chi2_results[column] = {
                'statistic': chi2_stat,
                'p_value': p_value,
                'drift': drift
            }
        
        return chi2_results
    
    def _pca_reconstruction_error(self, current_data):
        """Calculate PCA reconstruction error as a measure of distribution shift"""
        if self.pca_model is None:
            return {'error': np.nan, 'drift': False}
        
        # Get feature data
        current_features = current_data[self.feature_columns]
        
        # Project data to PCA space and back
        projected = self.pca_model.transform(current_features)
        reconstructed = self.pca_model.inverse_transform(projected)
        
        # Calculate mean squared reconstruction error
        mse = np.mean((current_features.values - reconstructed)**2)
        
        # Compare to reference reconstruction error
        reference_projected = self.pca_model.transform(self.reference_data[self.feature_columns])
        reference_reconstructed = self.pca_model.inverse_transform(reference_projected)
        reference_mse = np.mean((self.reference_data[self.feature_columns].values - reference_reconstructed)**2)
        
        # Calculate error ratio
        error_ratio = mse / reference_mse if reference_mse > 0 else np.inf
        
        # Determine if drift detected (error_ratio > 1.5)
        drift = error_ratio > 1.5
        
        return {
            'current_error': mse,
            'reference_error': reference_mse,
            'error_ratio': error_ratio,
            'drift': drift
        }
    
    def _maximum_mean_discrepancy(self, current_data):
        """Calculate Maximum Mean Discrepancy between reference and current data"""
        # Simplified MMD implementation using Gaussian kernel
        def gaussian_kernel(x, y, sigma=1.0):
            """Gaussian kernel for MMD calculation"""
            norm = np.sum((x[:, np.newaxis, :] - y[np.newaxis, :, :])**2, axis=2)
            return np.exp(-norm / (2 * sigma**2))
        
        # Get feature data
        reference_features = self.reference_data[self.feature_columns].values
        current_features = current_data[self.feature_columns].values
        
        # Sample if datasets are large
        max_samples = 1000
        if len(reference_features) > max_samples:
            idx = np.random.choice(len(reference_features), max_samples, replace=False)
            reference_features = reference_features[idx]
        
        if len(current_features) > max_samples:
            idx = np.random.choice(len(current_features), max_samples, replace=False)
            current_features = current_features[idx]
        
        # Calculate kernel matrices
        k_xx = gaussian_kernel(reference_features, reference_features)
        k_yy = gaussian_kernel(current_features, current_features)
        k_xy = gaussian_kernel(reference_features, current_features)
        
        # Calculate MMD
        mmd = np.mean(k_xx) + np.mean(k_yy) - 2 * np.mean(k_xy)
        
        # Threshold for drift detection
        threshold = 0.1  # Typically determined empirically
        drift = mmd > threshold
        
        return {
            'mmd': mmd,
            'threshold': threshold,
            'drift': drift
        }
    
    def _calculate_drift_score(self, univariate_results, multivariate_results):
        """Calculate overall drift score"""
        # Count total drift detections
        univariate_drift_count = 0
        feature_count = 0
        
        # Count univariate drift detections
        for method, results in univariate_results.items():
            for feature, feature_result in results.items():
                if isinstance(feature_result, dict) and 'drift' in feature_result:
                    feature_count += 1
                    if feature_result['drift']:
                        univariate_drift_count += 1
        
        # Calculate univariate drift ratio
        univariate_drift_ratio = univariate_drift_count / feature_count if feature_count > 0 else 0
        
        # Count multivariate drift detections
        multivariate_drift_count = 0
        multivariate_method_count = 0
        
        for method, results in multivariate_results.items():
            if 'drift' in results:
                multivariate_method_count += 1
                if results['drift']:
                    multivariate_drift_count += 1
        
        # Calculate multivariate drift ratio
        multivariate_drift_ratio = multivariate_drift_count / multivariate_method_count if multivariate_method_count > 0 else 0
        
        # Combine univariate and multivariate results (weighted)
        # Give more weight to multivariate methods as they capture joint distributions
        drift_score = 0.4 * univariate_drift_ratio + 0.6 * multivariate_drift_ratio
        
        return drift_score
    
    def _calculate_feature_importance(self, univariate_results):
        """Calculate feature importance for drift"""
        feature_importance = {}
        
        # Combine results from different univariate methods
        for method, results in univariate_results.items():
            for feature, feature_result in results.items():
                if feature not in feature_importance:
                    feature_importance[feature] = 0.0
                
                if 'statistic' in feature_result:
                    # Normalize and add to importance
                    feature_importance[feature] += 0.5 * feature_result['statistic']
                
                if 'drift' in feature_result and feature_result['drift']:
                    # Add bonus for detected drift
                    feature_importance[feature] += 0.5
        
        # Normalize feature importance
        max_importance = max(feature_importance.values()) if feature_importance else 1.0
        feature_importance = {k: v / max_importance for k, v in feature_importance.items()}
        
        # Filter to important features based on threshold
        important_features = {k: v for k, v in feature_importance.items() 
                             if v >= self.alert_thresholds['feature_importance']}
        
        return important_features
    
    def save(self, filepath):
        """Save the drift detector state"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save PCA model if exists
        if self.pca_model is not None:
            with open(f"{filepath}_pca.pkl", 'wb') as f:
                pickle.dump(self.pca_model, f)
        
        # Save reference statistics
        with open(f"{filepath}_stats.json", 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_stats = {}
            for feature, stats in self.reference_stats.items():
                serializable_stats[feature] = {k: v for k, v in stats.items() if k != 'histogram'}
                
                # Handle histogram separately
                if 'histogram' in stats:
                    hist_vals, hist_bins = stats['histogram']
                    serializable_stats[feature]['histogram_vals'] = hist_vals.tolist()
                    serializable_stats[feature]['histogram_bins'] = hist_bins.tolist()
            
            json.dump(serializable_stats, f)
        
        # Save metadata
        metadata = {
            'feature_columns': self.feature_columns,
            'alert_thresholds': self.alert_thresholds,
            'univariate_methods': self.config['univariate_methods'],
            'multivariate_methods': self.config['multivariate_methods'],
            'reference_data_shape': self.reference_data.shape if self.reference_data is not None else None,
            'creation_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(f"{filepath}_metadata.json", 'w') as f:
            json.dump(metadata, f)
        
        print(f"Drift detector saved to {filepath}")
    
    def load(self, filepath):
        """Load the drift detector state"""
        # Load PCA model if exists
        pca_path = f"{filepath}_pca.pkl"
        if os.path.exists(pca_path):
            with open(pca_path, 'rb') as f:
                self.pca_model = pickle.load(f)
        
        # Load reference statistics
        with open(f"{filepath}_stats.json", 'r') as f:
            serialized_stats = json.load(f)
            
            # Convert back to proper format
            self.reference_stats = {}
            for feature, stats in serialized_stats.items():
                self.reference_stats[feature] = {k: v for k, v in stats.items() 
                                               if k not in ['histogram_vals', 'histogram_bins']}
                
                # Handle histogram separately
                if 'histogram_vals' in stats and 'histogram_bins' in stats:
                    hist_vals = np.array(stats['histogram_vals'])
                    hist_bins = np.array(stats['histogram_bins'])
                    self.reference_stats[feature]['histogram'] = (hist_vals, hist_bins)
        
        # Load metadata
        with open(f"{filepath}_metadata.json", 'r') as f:
            metadata = json.load(f)
            self.feature_columns = metadata['feature_columns']
            self.alert_thresholds = metadata['alert_thresholds']
        
        print(f"Drift detector loaded from {filepath}")
    
    def generate_drift_report(self, drift_results, output_filepath=None):
        """Generate a detailed drift report from drift detection results"""
        report = {
            "summary": {
                "drift_detected": drift_results['drift_detected'],
                "drift_score": drift_results['drift_score'],
                "timestamp": drift_results['timestamp'],
                "feature_count": len(self.feature_columns)
            },
            "feature_drift": {}
        }
        
        # Summarize univariate drift results
        for method, results in drift_results['univariate_results'].items():
            drifted_features = []
            stable_features = []
            
            for feature, feature_result in results.items():
                if 'drift' in feature_result:
                    if feature_result['drift']:
                        drifted_features.append(feature)
                    else:
                        stable_features.append(feature)
            
            report[f"{method}_summary"] = {
                "drifted_features_count": len(drifted_features),
                "stable_features_count": len(stable_features),
                "drifted_features": drifted_features,
                "stable_features": stable_features
            }
        
        # Add feature importance
        report["feature_importance"] = drift_results['feature_importance']
        
        # Add multivariate drift results
        report["multivariate_drift"] = drift_results['multivariate_results']
        
        # Generate detailed feature statistics
        for feature in self.feature_columns:
            if feature in self.reference_stats:
                report["feature_drift"][feature] = {
                    "reference": {
                        "mean": self.reference_stats[feature]['mean'],
                        "median": self.reference_stats[feature]['median'],
                        "std": self.reference_stats[feature]['std'],
                        "min": self.reference_stats[feature]['min'],
                        "max": self.reference_stats[feature]['max'],
                        "q1": self.reference_stats[feature]['q1'],
                        "q3": self.reference_stats[feature]['q3']
                    }
                }
                
                # Get feature drift status from univariate methods
                feature_drift_status = {}
                for method, results in drift_results['univariate_results'].items():
                    if feature in results:
                        feature_drift_status[method] = {
                            "drift": results[feature]['drift'],
                            "statistic": results[feature]['statistic'],
                            "p_value": results[feature]['p_value'] if 'p_value' in results[feature] else None
                        }
                
                report["feature_drift"][feature]["drift_status"] = feature_drift_status
        
        # Output report to file if path provided
        if output_filepath:
            os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
            with open(output_filepath, 'w') as f:
                json.dump(report, f, indent=2)
            
            # Also create a markdown version
            md_filepath = output_filepath.replace('.json', '.md')
            with open(md_filepath, 'w') as f:
                f.write(f"# Data Drift Report\n\n")
                f.write(f"**Generated on:** {report['summary']['timestamp']}\n\n")
                
                f.write("## Summary\n")
                f.write(f"- Drift detected: {report['summary']['drift_detected']}\n")
                f.write(f"- Drift score: {report['summary']['drift_score']:.4f}\n")
                f.write(f"- Features analyzed: {report['summary']['feature_count']}\n\n")
                
                if report['feature_importance']:
                    f.write("## Top Drifting Features\n")
                    for feature, importance in sorted(report['feature_importance'].items(), 
                                                     key=lambda x: x[1], reverse=True):
                        f.write(f"- {feature}: {importance:.4f}\n")
                    f.write("\n")
                
                for method in drift_results['univariate_methods']:
                    method_key = f"{method}_summary"
                    if method_key in report:
                        f.write(f"## {method.replace('_', ' ').title()} Results\n")
                        f.write(f"- Drifted features: {report[method_key]['drifted_features_count']}\n")
                        f.write(f"- Stable features: {report[method_key]['stable_features_count']}\n\n")
                        
                        if report[method_key]['drifted_features']:
                            f.write("### Drifted Features\n")
                            for feature in report[method_key]['drifted_features'][:10]:  # Show top 10
                                f.write(f"- {feature}\n")
                            
                            if len(report[method_key]['drifted_features']) > 10:
                                f.write(f"- ... and {len(report[method_key]['drifted_features']) - 10} more\n")
                            
                            f.write("\n")
                
                f.write("## Multivariate Drift Results\n")
                for method, results in report["multivariate_drift"].items():
                    f.write(f"### {method.replace('_', ' ').title()}\n")
                    for key, value in results.items():
                        if key != 'drift':
                            f.write(f"- {key}: {value}\n")
                    f.write(f"- Drift detected: {results.get('drift', False)}\n\n")
            
            print(f"Drift report saved to {output_filepath} and {md_filepath}")
        
        return report