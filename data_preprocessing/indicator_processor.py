#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Indicator Processor for Agricultural Data

This module handles indicator renaming from Chinese to English abbreviations,
province name standardization, and zero value replacement for agricultural
research data processing.

Features:
- Comprehensive Chinese to English indicator mapping
- Province name standardization and abbreviation
- Zero value replacement with configurable values
- Batch processing support
- Statistical reporting

Author: Thesis Research Project
Date: 2025
"""

import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import numpy as np


class IndicatorProcessor:
    """
    Processor for standardizing indicators and handling data quality issues
    in agricultural research datasets.
    """
    
    def __init__(self):
        """Initialize with predefined mappings for indicators and provinces."""
        
        # Comprehensive indicator mapping (Chinese to English abbreviations)
        self.indicator_mapping = {
            "农业柴油用量": "ADF",  # Agricultural Diesel Fuel
            "农业生产总值": "AGV",  # Agricultural Gross Value
            "耕地面积（土地利用）": "CLA",  # Cultivated Land Area
            "耕地面积": "CLA",  # Alternative name
            "农村人口": "RP",  # Rural Population
            "农用机械总动力": "AMP",  # Agricultural Machinery Power
            "人均粮食产量": "APGC",  # Average Per Capita Grain Consumption
            "农业用水量": "AWU",  # Agricultural Water Usage
            "水土流失治理面积": "SLCA",  # Soil Loss Control Area
            "人口城镇化率": "UR",  # Urbanization Rate
            "农业碳排放": "ACE",  # Agricultural Carbon Emissions
            # Additional mappings
            "粮食产量": "GP",  # Grain Production
            "农业GDP": "AGDP",  # Agricultural GDP
            "化肥使用量": "FU",  # Fertilizer Usage
            "农药使用量": "PU",  # Pesticide Usage
            "农业从业人员": "AEP",  # Agricultural Employment Population
            "农业技术推广": "ATP",  # Agricultural Technology Promotion
            "农业投资": "AI",  # Agricultural Investment
            "农田水利": "FWC",  # Farmland Water Conservancy
            "农业机械化率": "AMR",  # Agricultural Mechanization Rate
            "有机农业面积": "OAA",  # Organic Agriculture Area
        }
        
        # Province mapping (Chinese to English abbreviations)
        self.province_mapping = {
            "北京": "BJ", "天津": "TJ", "河北": "HeB", "山西": "SX", "内蒙古": "NMG",
            "辽宁": "LN", "吉林": "JL", "黑龙江": "HLJ", "上海": "SH", "江苏": "JS",
            "浙江": "ZJ", "安徽": "AH", "福建": "FJ", "江西": "JX", "山东": "SD",
            "河南": "HN", "湖北": "HuB", "湖南": "HuN", "广东": "GD", "广西": "GX",
            "海南": "HaiN", "重庆": "CQ", "四川": "SC", "贵州": "GZ", "云南": "YN",
            "西藏": "XZ", "陕西": "SaX", "甘肃": "GS", "青海": "QH", "宁夏": "NX",
            "新疆": "XJ"
        }
        
        # Default replacement value for zeros
        self.default_zero_replacement = 0.01
        
        # Statistics tracking
        self.processing_stats = {
            'indicators_renamed': 0,
            'provinces_renamed': 0,
            'zeros_replaced': 0,
            'sheets_processed': 0
        }
    
    def add_indicator_mapping(self, chinese_name, english_abbr):
        """
        Add a new indicator mapping.
        
        Args:
            chinese_name: Chinese name of the indicator
            english_abbr: English abbreviation
        """
        self.indicator_mapping[chinese_name] = english_abbr
    
    def add_province_mapping(self, chinese_name, english_abbr):
        """
        Add a new province mapping.
        
        Args:
            chinese_name: Chinese name of the province
            english_abbr: English abbreviation
        """
        self.province_mapping[chinese_name] = english_abbr
    
    def rename_indicators_and_replace_zeros(self, file_path, output_path=None, 
                                          zero_replacement=None, process_provinces=True):
        """
        Process Excel file to rename indicators and replace zero values.
        
        Args:
            file_path: Path to input Excel file
            output_path: Path for output file (auto-generated if None)
            zero_replacement: Value to replace zeros with (default: 0.01)
            process_provinces: Whether to process province names in sheet names
            
        Returns:
            tuple: (output_path, processing_statistics)
        """
        if zero_replacement is None:
            zero_replacement = self.default_zero_replacement
        
        # Reset statistics
        self.processing_stats = {
            'indicators_renamed': 0,
            'provinces_renamed': 0,
            'zeros_replaced': 0,
            'sheets_processed': 0
        }
        
        try:
            # Read Excel file
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
            print(f"Successfully read file: {os.path.basename(file_path)}")
            print(f"File contains {len(sheet_names)} sheets: {', '.join(sheet_names)}")
            
            # Generate output path if not provided
            if output_path is None:
                base_name = os.path.splitext(file_path)[0]
                output_path = f"{base_name}_processed_indicators.xlsx"
            
            # Track sheet name mappings
            sheet_name_mapping = {}
            
            # Create Excel writer
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Process each sheet
                for sheet_name in sheet_names:
                    # Read original sheet
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                    
                    # Check if sheet has data
                    if df.empty:
                        print(f"Sheet '{sheet_name}' has no data, skipping")
                        sheet_name_mapping[sheet_name] = sheet_name
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                        continue
                    
                    # Determine new sheet name (process province names if enabled)
                    new_sheet_name = sheet_name
                    if process_provinces:
                        for province, abbr in self.province_mapping.items():
                            if province in sheet_name:
                                new_sheet_name = sheet_name.replace(province, abbr)
                                self.processing_stats['provinces_renamed'] += 1
                                break
                    
                    # Save sheet name mapping
                    sheet_name_mapping[sheet_name] = new_sheet_name
                    
                    # Process indicators if 'Indicator' column exists
                    if "指标" in df.columns:
                        # Create DataFrame for storing transformed data
                        new_df = df.copy()
                        
                        # Process each row to rename indicators
                        for index, row in df.iterrows():
                            indicator = row["指标"]
                            # Replace indicator name if it exists in mapping
                            if indicator in self.indicator_mapping:
                                new_df.at[index, "指标"] = self.indicator_mapping[indicator]
                                self.processing_stats['indicators_renamed'] += 1
                    else:
                        print(f"Sheet '{sheet_name}' doesn't have 'Indicator' column, keeping original")
                        new_df = df
                    
                    # Process province columns if they exist
                    for col in new_df.columns:
                        if isinstance(col, str) and col in ["省份", "省市", "地区", "城市"]:
                            # Process each value in this column
                            for index, value in new_df[col].items():
                                if isinstance(value, str) and value in self.province_mapping:
                                    new_df.at[index, col] = self.province_mapping[value]
                                    # Note: not incrementing counter here to avoid double-counting
                    
                    # Process zero values in numeric columns
                    numeric_columns = new_df.select_dtypes(include=['number']).columns
                    zero_mask = new_df[numeric_columns] == 0
                    sheet_zeros = zero_mask.sum().sum()
                    
                    if sheet_zeros > 0:
                        self.processing_stats['zeros_replaced'] += sheet_zeros
                        print(f"Sheet '{sheet_name}' found {sheet_zeros} zero values")
                        
                        # Replace zero values with specified replacement
                        for col in numeric_columns:
                            new_df[col] = new_df[col].replace(0, zero_replacement)
                    
                    # Save to new file
                    new_df.to_excel(writer, sheet_name=new_sheet_name, index=False)
                    print(f"Processed sheet '{sheet_name}' -> '{new_sheet_name}'")
                    self.processing_stats['sheets_processed'] += 1
            
            # Print processing summary
            print("\nIndicator name transformation statistics:")
            for cn, en in self.indicator_mapping.items():
                print(f"  {cn} -> {en}")
            
            print("\nSheet name transformation statistics:")
            for old_name, new_name in sheet_name_mapping.items():
                if old_name != new_name:
                    print(f"  {old_name} -> {new_name}")
            
            print(f"Total provinces renamed in sheet names: {self.processing_stats['provinces_renamed']}")
            
            if self.processing_stats['zeros_replaced'] > 0:
                print(f"\nTotal zero values found and replaced: {self.processing_stats['zeros_replaced']}")
                print(f"Replacement value used: {zero_replacement}")
            else:
                print("\nNo zero values found in any sheets")
            
            print(f"\nProcessing completed! File saved to: {output_path}")
            
            return output_path, self.processing_stats
            
        except Exception as e:
            error_msg = f"Error during processing: {e}"
            print(error_msg)
            raise Exception(error_msg)
    
    def batch_process_folder(self, folder_path, output_folder=None, **kwargs):
        """
        Process all Excel files in a folder.
        
        Args:
            folder_path: Input folder containing Excel files
            output_folder: Output folder (default: creates 'processed_indicators' subfolder)
            **kwargs: Additional arguments passed to rename_indicators_and_replace_zeros
            
        Returns:
            list: List of (output_path, statistics) tuples
        """
        if output_folder is None:
            output_folder = os.path.join(folder_path, "processed_indicators")
        
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Get all Excel files
        excel_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(('.xlsx', '.xls'))]
        
        if not excel_files:
            print("No Excel files found in the specified folder")
            return []
        
        results = []
        
        for file in excel_files:
            input_path = os.path.join(folder_path, file)
            file_name = os.path.splitext(file)[0]
            output_path = os.path.join(output_folder, f"{file_name}_processed.xlsx")
            
            try:
                result = self.rename_indicators_and_replace_zeros(
                    input_path, output_path, **kwargs
                )
                results.append(result)
                print(f"Successfully processed: {file}")
            except Exception as e:
                print(f"Failed to process {file}: {e}")
        
        return results
    
    def get_available_indicators(self):
        """
        Get list of available indicator mappings.
        
        Returns:
            dict: Current indicator mapping dictionary
        """
        return self.indicator_mapping.copy()
    
    def get_available_provinces(self):
        """
        Get list of available province mappings.
        
        Returns:
            dict: Current province mapping dictionary
        """
        return self.province_mapping.copy()
    
    def export_mappings(self, output_path):
        """
        Export current mappings to Excel file for reference.
        
        Args:
            output_path: Path to save the mappings file
        """
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Export indicator mappings
            indicator_df = pd.DataFrame([
                {"Chinese_Name": cn, "English_Abbreviation": en} 
                for cn, en in self.indicator_mapping.items()
            ])
            indicator_df.to_excel(writer, sheet_name='Indicator_Mappings', index=False)
            
            # Export province mappings
            province_df = pd.DataFrame([
                {"Chinese_Name": cn, "English_Abbreviation": en} 
                for cn, en in self.province_mapping.items()
            ])
            province_df.to_excel(writer, sheet_name='Province_Mappings', index=False)
        
        print(f"Mappings exported to: {output_path}")


def create_gui():
    """Create a graphical user interface for the indicator processor."""
    
    processor = IndicatorProcessor()
    
    def select_file_and_process():
        """Handle file selection and processing with options."""
        # Create file selection dialog
        root = tk.Tk()
        root.withdraw()
        
        # Let user select Excel file
        file_path = filedialog.askopenfilename(
            title="Select Excel File",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        
        if not file_path:
            print("No file selected, exiting")
            return
        
        try:
            # Ask for zero replacement value
            zero_replacement = tk.simpledialog.askfloat(
                "Zero Replacement",
                "Enter value to replace zeros with:",
                initialvalue=0.01,
                minvalue=0.0001,
                maxvalue=1.0
            )
            
            if zero_replacement is None:
                zero_replacement = 0.01
            
            # Ask if user wants to process province names
            process_provinces = messagebox.askyesno(
                "Province Processing",
                "Do you want to rename provinces in sheet names to English abbreviations?"
            )
            
            # Process the file
            output_path, stats = processor.rename_indicators_and_replace_zeros(
                file_path,
                zero_replacement=zero_replacement,
                process_provinces=process_provinces
            )
            
            # Show completion message with statistics
            message = (
                f"Processing completed!\n\n"
                f"Statistics:\n"
                f"• Sheets processed: {stats['sheets_processed']}\n"
                f"• Indicators renamed: {stats['indicators_renamed']}\n"
                f"• Provinces renamed: {stats['provinces_renamed']}\n"
                f"• Zero values replaced: {stats['zeros_replaced']}\n\n"
                f"Output saved to:\n{output_path}"
            )
            
            messagebox.showinfo("Processing Complete", message)
            
        except Exception as e:
            messagebox.showerror("Error", f"Processing failed:\n{str(e)}")
    
    def export_mappings():
        """Export current mappings to Excel file."""
        output_path = filedialog.asksaveasfilename(
            title="Save Mappings File",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        
        if output_path:
            try:
                processor.export_mappings(output_path)
                messagebox.showinfo("Export Complete", f"Mappings exported to:\n{output_path}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export mappings:\n{str(e)}")
    
    # Create main GUI window
    root = tk.Tk()
    root.title("Agricultural Indicator Processor")
    root.geometry("700x500")
    
    # Add description
    description = tk.Label(
        root,
        text="Agricultural Indicator Processor\n\n"
             "This tool standardizes agricultural research data by:\n"
             "• Converting Chinese indicator names to English abbreviations\n"
             "• Standardizing province names to English abbreviations\n"
             "• Replacing zero values with specified small values\n"
             "• Generating processing statistics and reports\n\n"
             "Features:\n"
             "• Comprehensive indicator and province mapping\n"
             "• Configurable zero value replacement\n"
             "• Batch processing support\n"
             "• Statistical reporting\n"
             "• Mapping export functionality\n\n"
             "Click 'Process File' to begin data processing or\n"
             "'Export Mappings' to view current translation tables.",
        justify=tk.LEFT,
        padx=20,
        pady=20
    )
    description.pack(expand=True)
    
    # Create button frame
    button_frame = tk.Frame(root)
    button_frame.pack(pady=20)
    
    # Add process button
    process_button = tk.Button(
        button_frame,
        text="Process File",
        command=select_file_and_process,
        padx=20,
        pady=10,
        font=("Arial", 12),
        bg="#4CAF50",
        fg="white"
    )
    process_button.pack(side=tk.LEFT, padx=10)
    
    # Add export mappings button
    export_button = tk.Button(
        button_frame,
        text="Export Mappings",
        command=export_mappings,
        padx=20,
        pady=10,
        font=("Arial", 12),
        bg="#2196F3",
        fg="white"
    )
    export_button.pack(side=tk.LEFT, padx=10)
    
    # Add exit button
    exit_button = tk.Button(
        button_frame,
        text="Exit",
        command=root.destroy,
        padx=20,
        pady=10,
        font=("Arial", 12)
    )
    exit_button.pack(side=tk.LEFT, padx=10)
    
    root.mainloop()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Command line usage
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        zero_replacement = float(sys.argv[3]) if len(sys.argv) > 3 else 0.01
        
        processor = IndicatorProcessor()
        output_path, stats = processor.rename_indicators_and_replace_zeros(
            input_file, output_file, zero_replacement=zero_replacement
        )
        
        print(f"Processing completed. Output: {output_path}")
        print(f"Statistics: {stats}")
    else:
        # GUI usage
        try:
            import tkinter.simpledialog
            create_gui()
        except ImportError:
            print("GUI requires tkinter.simpledialog. Running basic version...")
            # Fallback to basic processing
            processor = IndicatorProcessor()
            print("Available indicators:", list(processor.indicator_mapping.keys())[:5], "...")
            print("Available provinces:", list(processor.province_mapping.keys())[:5], "...")