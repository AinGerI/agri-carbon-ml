#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data Reorganizer for Agricultural Research

This module reorganizes Excel data by transforming indicator-based sheets into 
province-based organization. Each province gets its own sheet with all indicators
and year data in a structured format.

Features:
- Flexible year range configuration
- Province-based data organization
- Support for multiple indicators per province
- Automatic data validation and cleaning
- GUI interface for easy usage

Author: Thesis Research Project
Date: 2025
"""

import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
import os


class DataReorganizer:
    """
    Agricultural data reorganizer that transforms indicator-based Excel sheets
    into province-based organization for easier analysis.
    """
    
    def __init__(self):
        """Initialize with default settings and province mappings."""
        self.default_year_range = (2005, 2023)  # Default year range
        self.extended_year_range = (2000, 2024)  # Extended range for compatibility
        self.custom_year_range = None  # For user-specified ranges
        
    def detect_year_range(self, df):
        """
        Automatically detect year range from DataFrame columns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            tuple: (start_year, end_year) or None if no valid years found
        """
        years = []
        # Check all columns except the first (usually indicator names)
        for col in df.columns[1:]:
            try:
                # Try to extract year from column name
                year_str = str(col).strip()
                if '年' in year_str:
                    year_str = year_str.replace('年', '')
                year = int(year_str)
                if 1900 <= year <= 2100:
                    years.append(year)
            except (ValueError, TypeError):
                continue
        
        if years:
            return min(years), max(years)
        return None
    
    def filter_year_range(self, df, start_year=None, end_year=None, auto_detect=True):
        """
        Filter DataFrame to include only specified year range.
        
        Args:
            df: Input DataFrame
            start_year: Starting year (inclusive)
            end_year: Ending year (inclusive)
            auto_detect: If True, auto-detect years from column names
            
        Returns:
            pd.DataFrame: Filtered DataFrame with only specified years
        """
        if auto_detect:
            detected_range = self.detect_year_range(df)
            if detected_range and not start_year and not end_year:
                start_year, end_year = detected_range
        
        # Use default range if not specified
        if not start_year or not end_year:
            if self.custom_year_range:
                start_year, end_year = self.custom_year_range
            else:
                start_year, end_year = self.default_year_range
        
        # Filter columns to include only the year range
        year_columns = []
        indicator_column = df.columns[0]  # First column is usually indicators
        
        for col in df.columns[1:]:
            try:
                year_str = str(col).strip().replace('年', '')
                year = int(year_str)
                if start_year <= year <= end_year:
                    year_columns.append(col)
            except (ValueError, TypeError):
                continue
        
        # Return filtered DataFrame
        if year_columns:
            return df[[indicator_column] + year_columns]
        else:
            # If no valid year columns found, return original
            print(f"Warning: No valid year columns found in range {start_year}-{end_year}")
            return df
    
    def transform_excel_by_province(self, file_path, output_path=None, year_range=None, 
                                   filter_years=True, auto_detect_years=True):
        """
        Transform Excel file from indicator-based to province-based organization.
        
        Args:
            file_path: Path to input Excel file
            output_path: Path for output file (auto-generated if None)
            year_range: Tuple of (start_year, end_year) for filtering
            filter_years: Whether to filter data by year range
            auto_detect_years: Whether to auto-detect year range from data
            
        Returns:
            str: Path to the output file
        """
        print(f"Processing file: {os.path.basename(file_path)}")
        
        # Set custom year range if provided
        if year_range:
            self.custom_year_range = year_range
        
        try:
            # Read all sheet names
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
            print(f"File contains {len(sheet_names)} sheets: {', '.join(sheet_names)}")
            
            # Store all sheet data
            all_sheets_data = {}
            province_list = None
            
            # Read each sheet
            for sheet_name in sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                
                # Apply year filtering if requested
                if filter_years:
                    df = self.filter_year_range(
                        df, 
                        start_year=year_range[0] if year_range else None,
                        end_year=year_range[1] if year_range else None,
                        auto_detect=auto_detect_years
                    )
                
                # Assume first column contains province names
                if province_list is None:
                    province_list = df.iloc[:, 0].tolist()
                
                all_sheets_data[sheet_name] = df
                print(f"Read sheet '{sheet_name}', contains {len(df)} rows")
            
            if not province_list:
                raise ValueError("Unable to get province list")
            
            # Create DataFrame for each province containing all indicators
            province_data = {}
            
            for province in province_list:
                # Create empty DataFrame to store all indicator data for this province
                province_df = pd.DataFrame()
                
                # Extract data for this province from each sheet
                for sheet_name, sheet_df in all_sheets_data.items():
                    # Find rows for this province in current sheet
                    province_rows = sheet_df[sheet_df.iloc[:, 0] == province]
                    
                    if not province_rows.empty:
                        # Get data for this province (excluding province name column)
                        row_data = province_rows.iloc[0, 1:].reset_index(drop=True)
                        
                        # Get year column headers
                        year_columns = sheet_df.columns[1:].tolist()
                        
                        # Create temporary DataFrame
                        temp_df = pd.DataFrame(
                            [row_data.values], 
                            columns=year_columns
                        )
                        
                        # Add indicator name
                        temp_df.insert(0, 'Indicator', sheet_name)
                        
                        # Append to province DataFrame
                        province_df = pd.concat([province_df, temp_df], ignore_index=True)
                
                province_data[province] = province_df
            
            # Generate output file path
            if output_path is None:
                output_path = os.path.splitext(file_path)[0] + "_reorganized_by_province.xlsx"
            
            # Save results to new Excel file
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                for province, df in province_data.items():
                    # Ensure sheet name doesn't exceed Excel's 31 character limit
                    safe_name = str(province)
                    if len(safe_name) > 31:
                        safe_name = safe_name[:31]
                    
                    # Save to Excel
                    df.to_excel(writer, sheet_name=safe_name, index=False)
            
            print(f"\nTransformation completed! Results saved to: {output_path}")
            print(f"Each province's data saved in corresponding sheet")
            
            return output_path
            
        except Exception as e:
            error_msg = f"Error during processing: {e}"
            print(error_msg)
            raise Exception(error_msg)
    
    def batch_process_folder(self, folder_path, output_folder=None, **kwargs):
        """
        Process all Excel files in a folder.
        
        Args:
            folder_path: Input folder containing Excel files
            output_folder: Output folder (default: creates 'reorganized' subfolder)
            **kwargs: Additional arguments passed to transform_excel_by_province
            
        Returns:
            list: List of output file paths
        """
        if output_folder is None:
            output_folder = os.path.join(folder_path, "reorganized")
        
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Get all Excel files
        excel_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(('.xlsx', '.xls'))]
        
        if not excel_files:
            print("No Excel files found in the specified folder")
            return []
        
        output_files = []
        
        for file in excel_files:
            input_path = os.path.join(folder_path, file)
            file_name = os.path.splitext(file)[0]
            output_path = os.path.join(output_folder, f"{file_name}_reorganized.xlsx")
            
            try:
                result_path = self.transform_excel_by_province(
                    input_path, output_path, **kwargs
                )
                output_files.append(result_path)
                print(f"Successfully processed: {file}")
            except Exception as e:
                print(f"Failed to process {file}: {e}")
        
        return output_files


def create_gui():
    """Create a simple GUI for the data reorganizer."""
    
    def select_file_and_process():
        """Handle file selection and processing."""
        # Create file selection dialog
        root = tk.Tk()
        root.withdraw()  # Hide main window
        
        # Let user select Excel file containing multiple indicator sheets
        file_path = filedialog.askopenfilename(
            title="Select Excel File",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        
        if not file_path:
            print("No file selected, exiting")
            return
        
        try:
            reorganizer = DataReorganizer()
            
            # Ask user for year range preference
            year_choice = messagebox.askyesnocancel(
                "Year Range Selection",
                "Do you want to use the default year range (2005-2023)?\n\n"
                "Yes: Use default range (2005-2023)\n"
                "No: Use extended range (2000-2024)\n"
                "Cancel: Auto-detect from data"
            )
            
            if year_choice is True:
                year_range = (2005, 2023)
            elif year_choice is False:
                year_range = (2000, 2024)
            else:
                year_range = None  # Auto-detect
            
            # Process the file
            output_path = reorganizer.transform_excel_by_province(
                file_path, 
                year_range=year_range,
                filter_years=True,
                auto_detect_years=(year_range is None)
            )
            
            # Show completion message
            messagebox.showinfo(
                "Processing Complete", 
                f"Data reorganization completed!\n\nOutput saved to:\n{output_path}"
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Processing failed:\n{str(e)}")
    
    # Create and show GUI
    root = tk.Tk()
    root.title("Agricultural Data Reorganizer")
    root.geometry("600x400")
    
    # Add description
    description = tk.Label(
        root, 
        text="Agricultural Data Reorganizer\n\n"
             "This tool transforms Excel files from indicator-based organization\n"
             "to province-based organization for easier analysis.\n\n"
             "Features:\n"
             "• Transforms multiple indicator sheets into province-based sheets\n"
             "• Flexible year range configuration\n"
             "• Automatic data validation and cleaning\n"
             "• Support for both default and custom year ranges\n"
             "• Auto-detection of year ranges from data\n\n"
             "Click 'Select and Process File' to begin.",
        justify=tk.LEFT,
        padx=20,
        pady=20
    )
    description.pack(expand=True)
    
    # Add process button
    process_button = tk.Button(
        root, 
        text="Select and Process File", 
        command=select_file_and_process,
        padx=20, 
        pady=10,
        font=("Arial", 12)
    )
    process_button.pack(pady=10)
    
    # Add exit button
    exit_button = tk.Button(
        root, 
        text="Exit", 
        command=root.destroy, 
        padx=10, 
        pady=5
    )
    exit_button.pack(pady=5)
    
    root.mainloop()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Command line usage
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        
        reorganizer = DataReorganizer()
        
        # Parse additional arguments
        year_range = None
        if len(sys.argv) > 4:
            try:
                start_year = int(sys.argv[3])
                end_year = int(sys.argv[4])
                year_range = (start_year, end_year)
            except ValueError:
                print("Invalid year range provided, using auto-detection")
        
        output_path = reorganizer.transform_excel_by_province(
            input_file, output_file, year_range=year_range
        )
        print(f"Processing completed. Output: {output_path}")
    else:
        # GUI usage
        create_gui()