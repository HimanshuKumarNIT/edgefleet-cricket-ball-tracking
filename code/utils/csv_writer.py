"""
Utility for writing per-frame ball centroid annotations to CSV.
"""

import pandas as pd


class CSVWriter:
    def __init__(self, output_path):
        self.output_path = output_path
        self.records = []

    def add_record(self, frame_idx, x, y, visible):
        """
        Store a single frame annotation.
        """
        self.records.append({
            "frame": frame_idx,
            "x": x,
            "y": y,
            "visible": visible
        })

    def save(self):
        """
        Save all records to CSV file.
        """
        df = pd.DataFrame(self.records)
        df.to_csv(self.output_path, index=False)
