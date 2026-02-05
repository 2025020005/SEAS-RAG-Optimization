import numpy as np

class SEASMetrics:
    """
    Evaluation metrics for the SEAS framework.
    Calculates Redundancy Removal (RR) and Information Retention Rate (IRR).
    """

    @staticmethod
    def calculate_rr(original_count, kept_count):
        """
        Calculate Redundancy Removal (RR) rate.
        
        Args:
            original_count (int): Total number of documents before deduplication.
            kept_count (int): Number of documents remaining after deduplication.
            
        Returns:
            float: RR percentage (0.0 to 100.0).
        """
        if original_count == 0:
            return 0.0
        return (1 - kept_count / original_count) * 100

    @staticmethod
    def calculate_irr(original_labels, kept_indices):
        """
        Calculate Information Retention Rate (IRR).
        IRR measures how many unique 'ground truth' events/facts are preserved.
        
        Args:
            original_labels (list or np.array): Ground truth labels (e.g., event IDs) for all original docs.
            kept_indices (list): Indices of documents kept by the algorithm.
            
        Returns:
            float: IRR percentage (0.0 to 100.0).
        """
        if len(original_labels) == 0:
            return 0.0
            
        # Unique events in the original dataset (Ground Truth)
        # Filter out noise labels (assuming -1 is noise) if applicable
        original_unique = set([l for l in original_labels if l != -1])
        
        if len(original_unique) == 0:
            return 100.0 # No information to lose
            
        # Unique events present in the kept documents
        kept_labels = [original_labels[i] for i in kept_indices]
        kept_unique = set([l for l in kept_labels if l != -1])
        
        # IRR = (Unique events retained) / (Total unique events originally present)
        return (len(kept_unique) / len(original_unique)) * 100