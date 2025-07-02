#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data Manipulation Tools for Agricultural Research

This module provides tools for flexible data manipulation including:
- Interactive indicator/column removal from Excel files
- Row filtering based on first column values
- Multi-sheet processing with structure preservation
- GUI interface for easy data manipulation
- Batch processing capabilities

Combines functionality from:
- 随心删指标数据.py

Author: Thesis Research Project
Date: 2025
"""

import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext
import os
import threading
from datetime import datetime


class DataManipulator:
    """
    Flexible data manipulation tool for Excel files.
    
    Supports removal of rows and columns across multiple worksheets.
    """
    
    def __init__(self):
        """Initialize with default settings."""
        self.current_file_path = None
        self.first_sheet_structure = None
        self.available_row_values = []
        self.available_columns = []
        self.processing_stats = {
            'sheets_processed': 0,
            'rows_removed_total': 0,
            'columns_removed_total': 0
        }
    
    def load_file_structure(self, file_path):
        """
        Load structure from the first worksheet of Excel file.
        
        Args:
            file_path: Path to Excel file
            
        Returns:
            dict: Structure information including row values and column headers
        """
        try:
            # Read first worksheet to get structure
            df_first_sheet = pd.read_excel(file_path, sheet_name=0)
            
            if df_first_sheet.empty:
                raise ValueError("First worksheet is empty")
            if df_first_sheet.shape[1] == 0:
                raise ValueError("First worksheet has no columns")
            
            # Get unique values from first column (row identifiers)
            self.available_row_values = []
            if df_first_sheet.shape[0] > 0:
                try:
                    first_col_values = pd.unique(df_first_sheet.iloc[:, 0].dropna()).tolist()
                    self.available_row_values = sorted([str(val) for val in first_col_values])
                except IndexError:
                    pass  # No accessible first column
            
            # Get column headers
            self.available_columns = [str(col) for col in df_first_sheet.columns.tolist()]
            
            self.current_file_path = file_path
            self.first_sheet_structure = {
                'row_values': self.available_row_values,
                'column_headers': self.available_columns,
                'shape': df_first_sheet.shape
            }
            
            return {
                'success': True,
                'row_values': self.available_row_values,
                'column_headers': self.available_columns,
                'sheet_shape': df_first_sheet.shape
            }
            
        except FileNotFoundError:
            return {'success': False, 'error': f"File not found: {file_path}"}
        except ValueError as ve:
            return {'success': False, 'error': f"Invalid Excel file format: {ve}"}
        except Exception as e:
            return {'success': False, 'error': f"Error loading file structure: {e}"}
    
    def process_excel_file(self, file_path, rows_to_remove=None, columns_to_remove=None, 
                          output_path=None, progress_callback=None):
        """
        Process Excel file by removing specified rows and columns.
        
        Args:
            file_path: Path to input Excel file
            rows_to_remove: List of row identifiers to remove (based on first column values)
            columns_to_remove: List of column headers to remove
            output_path: Output file path (auto-generated if None)
            progress_callback: Function to report progress (optional)
            
        Returns:
            dict: Processing results and statistics
        """
        if rows_to_remove is None:
            rows_to_remove = []
        if columns_to_remove is None:
            columns_to_remove = []
        if progress_callback is None:
            progress_callback = lambda msg: print(msg)
        
        # Validate inputs
        if not rows_to_remove and not columns_to_remove:
            return {
                'success': False, 
                'error': 'No rows or columns specified for removal'
            }
        
        # Generate output path if not provided
        if output_path is None:
            base_name = os.path.splitext(file_path)[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"{base_name}_processed_{timestamp}.xlsx"
        
        try:
            progress_callback(f"开始处理文件: {os.path.basename(file_path)}")
            
            # Read Excel file
            xls = pd.ExcelFile(file_path)
            sheet_names = xls.sheet_names
            
            # Reset statistics
            self.processing_stats = {
                'sheets_processed': 0,
                'rows_removed_total': 0,
                'columns_removed_total': 0
            }
            
            progress_callback(f"文件包含 {len(sheet_names)} 个工作表: {', '.join(sheet_names)}")
            
            # Process each worksheet
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                for i, sheet_name in enumerate(sheet_names):
                    progress_callback(f"处理工作表 {i+1}/{len(sheet_names)}: {sheet_name}")
                    
                    # Read worksheet
                    df = pd.read_excel(xls, sheet_name=sheet_name)
                    original_rows = len(df)
                    original_cols = len(df.columns)
                    processed_sheet = False
                    
                    # Remove rows based on first column values
                    rows_removed_this_sheet = 0
                    if rows_to_remove and not df.empty and df.shape[1] > 0:
                        first_col_name = df.columns[0]
                        try:
                            # Filter out rows where first column value is in removal list
                            df_filtered = df[~df[first_col_name].isin(rows_to_remove)]
                            rows_removed_this_sheet = original_rows - len(df_filtered)
                            if rows_removed_this_sheet > 0:
                                df = df_filtered
                                processed_sheet = True
                                progress_callback(f"  - 从 {sheet_name} 移除了 {rows_removed_this_sheet} 行")
                        except Exception as row_e:
                            progress_callback(f"  - 警告: 在工作表 '{sheet_name}' 删除行时出错: {row_e}")
                    
                    # Remove specified columns
                    cols_removed_this_sheet = 0
                    if columns_to_remove and not df.empty:
                        # Find columns that actually exist in this sheet
                        actual_cols_to_drop = [col for col in columns_to_remove if col in df.columns]
                        if actual_cols_to_drop:
                            try:
                                df = df.drop(columns=actual_cols_to_drop)
                                cols_removed_this_sheet = len(actual_cols_to_drop)
                                processed_sheet = True
                                progress_callback(f"  - 从 {sheet_name} 移除了 {cols_removed_this_sheet} 列")
                            except Exception as col_e:
                                progress_callback(f"  - 警告: 在工作表 '{sheet_name}' 删除列时出错: {col_e}")
                    
                    # Save processed sheet
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    
                    # Update statistics
                    if processed_sheet:
                        self.processing_stats['sheets_processed'] += 1
                        self.processing_stats['rows_removed_total'] += rows_removed_this_sheet
                        self.processing_stats['columns_removed_total'] += cols_removed_this_sheet
            
            progress_callback("处理完成!")
            
            return {
                'success': True,
                'output_path': output_path,
                'statistics': self.processing_stats.copy(),
                'message': f"处理完成。在 {self.processing_stats['sheets_processed']} 个工作表中进行了修改。"
            }
            
        except Exception as e:
            error_msg = f"文件处理过程中发生错误: {e}"
            progress_callback(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'statistics': self.processing_stats.copy()
            }
    
    def batch_process_folder(self, folder_path, rows_to_remove=None, columns_to_remove=None, 
                           output_folder=None, progress_callback=None):
        """
        Process all Excel files in a folder.
        
        Args:
            folder_path: Input folder containing Excel files
            rows_to_remove: List of row identifiers to remove
            columns_to_remove: List of column headers to remove
            output_folder: Output folder (default: creates 'processed' subfolder)
            progress_callback: Function to report progress
            
        Returns:
            dict: Batch processing results
        """
        if progress_callback is None:
            progress_callback = lambda msg: print(msg)
        
        if output_folder is None:
            output_folder = os.path.join(folder_path, "processed")
        
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Get all Excel files
        excel_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(('.xlsx', '.xls'))]
        
        if not excel_files:
            return {
                'success': False,
                'error': 'No Excel files found in the specified folder',
                'processed_files': []
            }
        
        progress_callback(f"找到 {len(excel_files)} 个Excel文件")
        
        processed_files = []
        failed_files = []
        
        for i, file in enumerate(excel_files):
            progress_callback(f"\n处理文件 {i+1}/{len(excel_files)}: {file}")
            
            input_path = os.path.join(folder_path, file)
            file_name = os.path.splitext(file)[0]
            output_path = os.path.join(output_folder, f"{file_name}_processed.xlsx")
            
            try:
                result = self.process_excel_file(
                    input_path, rows_to_remove, columns_to_remove, 
                    output_path, progress_callback
                )
                
                if result['success']:
                    processed_files.append({
                        'input_file': file,
                        'output_path': result['output_path'],
                        'statistics': result['statistics']
                    })
                    progress_callback(f"✓ 成功处理: {file}")
                else:
                    failed_files.append({
                        'file': file,
                        'error': result['error']
                    })
                    progress_callback(f"✗ 处理失败: {file} - {result['error']}")
                    
            except Exception as e:
                failed_files.append({
                    'file': file,
                    'error': str(e)
                })
                progress_callback(f"✗ 处理失败: {file} - {e}")
        
        return {
            'success': len(processed_files) > 0,
            'processed_files': processed_files,
            'failed_files': failed_files,
            'summary': f"成功处理 {len(processed_files)}/{len(excel_files)} 个文件"
        }
    
    def get_processing_statistics(self):
        """
        Get current processing statistics.
        
        Returns:
            dict: Processing statistics
        """
        return self.processing_stats.copy()
    
    def export_structure_info(self, output_path):
        """
        Export current file structure information to Excel.
        
        Args:
            output_path: Path to save structure information
            
        Returns:
            bool: Success status
        """
        if not self.first_sheet_structure:
            return False
        
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Export row values
                if self.available_row_values:
                    row_df = pd.DataFrame({
                        'Row_Values': self.available_row_values
                    })
                    row_df.to_excel(writer, sheet_name='Row_Values', index=False)
                
                # Export column headers
                if self.available_columns:
                    col_df = pd.DataFrame({
                        'Column_Headers': self.available_columns
                    })
                    col_df.to_excel(writer, sheet_name='Column_Headers', index=False)
                
                # Export summary info
                summary_df = pd.DataFrame({
                    'Property': ['File_Path', 'Sheet_Rows', 'Sheet_Columns', 'Available_Row_Values', 'Available_Columns'],
                    'Value': [
                        self.current_file_path or 'N/A',
                        self.first_sheet_structure['shape'][0],
                        self.first_sheet_structure['shape'][1],
                        len(self.available_row_values),
                        len(self.available_columns)
                    ]
                })
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            return True
            
        except Exception as e:
            print(f"Error exporting structure info: {e}")
            return False


class DataManipulatorGUI:
    """
    GUI application for data manipulation.
    """
    
    def __init__(self, master):
        self.master = master
        master.title("Excel 数据结构处理工具")
        master.geometry("800x700")
        
        # State variables
        self.file_path = tk.StringVar()
        self.status_var = tk.StringVar()
        self.manipulator = DataManipulator()
        
        # Create GUI elements
        self._create_widgets()
        
        # Set initial status
        self.set_status("就绪。请选择一个 Excel 文件。")
    
    def _create_widgets(self):
        """Create GUI widgets."""
        # File selection frame
        file_frame = ttk.LabelFrame(self.master, text="1. 选择 Excel 文件")
        file_frame.pack(padx=10, pady=5, fill="x")
        
        ttk.Button(file_frame, text="浏览...", command=self.browse_file).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Entry(file_frame, textvariable=self.file_path, width=70, state='readonly').pack(
            side=tk.LEFT, expand=True, fill="x", padx=5, pady=5)
        
        # Load structure button frame
        load_frame = ttk.Frame(self.master)
        load_frame.pack(padx=10, pady=5, fill="x")
        self.load_button = ttk.Button(
            load_frame, text="加载文件结构", 
            command=self.load_structure, state=tk.DISABLED
        )
        self.load_button.pack(pady=5)
        
        # Selection frame with two listboxes
        selection_main_frame = ttk.Frame(self.master)
        selection_main_frame.pack(padx=10, pady=5, fill="both", expand=True)
        
        # Row selection frame
        rows_frame = ttk.LabelFrame(selection_main_frame, text="2a. 选择要移除的行 (基于第一列值)")
        rows_frame.pack(side=tk.LEFT, padx=5, pady=5, fill="both", expand=True)
        
        row_list_frame = ttk.Frame(rows_frame)
        row_list_frame.pack(fill="both", expand=True, padx=5, pady=5)
        row_scrollbar = ttk.Scrollbar(row_list_frame, orient=tk.VERTICAL)
        self.row_listbox = tk.Listbox(
            row_list_frame, selectmode=tk.MULTIPLE, 
            yscrollcommand=row_scrollbar.set, height=15, exportselection=False
        )
        row_scrollbar.config(command=self.row_listbox.yview)
        row_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.row_listbox.pack(side=tk.LEFT, fill="both", expand=True)
        self.row_listbox.configure(state=tk.DISABLED)
        
        # Column selection frame
        cols_frame = ttk.LabelFrame(selection_main_frame, text="2b. 选择要移除的列")
        cols_frame.pack(side=tk.RIGHT, padx=5, pady=5, fill="both", expand=True)
        
        col_list_frame = ttk.Frame(cols_frame)
        col_list_frame.pack(fill="both", expand=True, padx=5, pady=5)
        col_scrollbar = ttk.Scrollbar(col_list_frame, orient=tk.VERTICAL)
        self.column_listbox = tk.Listbox(
            col_list_frame, selectmode=tk.MULTIPLE, 
            yscrollcommand=col_scrollbar.set, height=15, exportselection=False
        )
        col_scrollbar.config(command=self.column_listbox.yview)
        col_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.column_listbox.pack(side=tk.LEFT, fill="both", expand=True)
        self.column_listbox.configure(state=tk.DISABLED)
        
        # Processing button frame
        process_frame = ttk.Frame(self.master)
        process_frame.pack(padx=10, pady=10, fill="x")
        
        self.process_button = ttk.Button(
            process_frame, text="处理文件并保存", 
            command=self.start_processing, state=tk.DISABLED
        )
        self.process_button.pack(side=tk.LEFT, padx=5)
        
        self.batch_button = ttk.Button(
            process_frame, text="批量处理文件夹", 
            command=self.batch_process, state=tk.DISABLED
        )
        self.batch_button.pack(side=tk.LEFT, padx=5)
        
        self.export_button = ttk.Button(
            process_frame, text="导出结构信息", 
            command=self.export_structure, state=tk.DISABLED
        )
        self.export_button.pack(side=tk.LEFT, padx=5)
        
        # Log output area
        log_frame = ttk.LabelFrame(self.master, text="3. 处理日志")
        log_frame.pack(padx=10, pady=5, fill="both", expand=True)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=10)
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Status bar
        self.status_bar = ttk.Label(self.master, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def set_status(self, message):
        """Update status bar."""
        self.status_var.set(message)
        self.master.update_idletasks()
    
    def log_message(self, message):
        """Add message to log area."""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.master.update_idletasks()
    
    def browse_file(self):
        """Browse for Excel file."""
        f_path = filedialog.askopenfilename(
            title='选择 Excel 文件',
            filetypes=[('Excel 文件', '*.xlsx'), ('所有文件', '*.*')]
        )
        if f_path:
            self.file_path.set(f_path)
            self.load_button.config(state=tk.NORMAL)
            
            # Clear and disable listboxes and buttons
            self.row_listbox.delete(0, tk.END)
            self.row_listbox.configure(state=tk.DISABLED)
            self.column_listbox.delete(0, tk.END)
            self.column_listbox.configure(state=tk.DISABLED)
            self.process_button.config(state=tk.DISABLED)
            self.batch_button.config(state=tk.DISABLED)
            self.export_button.config(state=tk.DISABLED)
            
            self.set_status(f"已选择文件: {os.path.basename(f_path)}。请点击加载结构。")
        else:
            self.file_path.set("")
            self.load_button.config(state=tk.DISABLED)
            self.set_status("文件选择已取消。")
    
    def load_structure(self):
        """Load file structure from selected Excel file."""
        f_path = self.file_path.get()
        if not f_path:
            messagebox.showerror("错误", "请先选择一个 Excel 文件。")
            return
        
        # Clear existing lists
        self.row_listbox.delete(0, tk.END)
        self.column_listbox.delete(0, tk.END)
        
        # Disable listboxes and buttons during loading
        self.row_listbox.configure(state=tk.DISABLED)
        self.column_listbox.configure(state=tk.DISABLED)
        self.process_button.config(state=tk.DISABLED)
        self.batch_button.config(state=tk.DISABLED)
        self.export_button.config(state=tk.DISABLED)
        
        self.set_status("正在加载文件结构...")
        
        # Load structure
        result = self.manipulator.load_file_structure(f_path)
        
        if result['success']:
            # Populate row values listbox
            if result['row_values']:
                self.row_listbox.configure(state=tk.NORMAL)
                for val in result['row_values']:
                    self.row_listbox.insert(tk.END, val)
            
            # Populate column headers listbox
            if result['column_headers']:
                self.column_listbox.configure(state=tk.NORMAL)
                for header in result['column_headers']:
                    self.column_listbox.insert(tk.END, header)
            
            # Enable processing buttons if data was loaded
            if result['row_values'] or result['column_headers']:
                self.process_button.config(state=tk.NORMAL)
                self.batch_button.config(state=tk.NORMAL)
                self.export_button.config(state=tk.NORMAL)
                self.set_status("结构加载完成。请选择要移除的行或列。")
                self.log_message(f"成功加载文件结构:\n" +
                               f"- 行标识符: {len(result['row_values'])} 个\n" +
                               f"- 列标题: {len(result['column_headers'])} 个\n" +
                               f"- 工作表大小: {result['sheet_shape']}")
            else:
                self.set_status("未加载到任何结构信息。")
                self.log_message("警告: 未能从文件中提取结构信息。")
        else:
            messagebox.showerror("错误", f"加载文件结构失败:\n{result['error']}")
            self.set_status("加载结构时出错。")
            self.log_message(f"错误: {result['error']}")
    
    def start_processing(self):
        """Start processing selected file."""
        # Get selections
        selected_row_indices = self.row_listbox.curselection()
        rows_to_remove = [self.row_listbox.get(i) for i in selected_row_indices]
        
        selected_col_indices = self.column_listbox.curselection()
        columns_to_remove = [self.column_listbox.get(i) for i in selected_col_indices]
        
        # Check if at least one item is selected
        if not rows_to_remove and not columns_to_remove:
            messagebox.showwarning("警告", "请至少选择一个要移除的行或列。")
            return
        
        f_path = self.file_path.get()
        if not f_path:
            messagebox.showerror("错误", "文件路径丢失，请重新选择文件。")
            return
        
        # Disable controls during processing
        self._disable_controls()
        self.set_status("正在处理文件... 请稍候。")
        self.log_text.delete('1.0', tk.END)
        
        # Start processing thread
        thread = threading.Thread(
            target=self._process_file_thread,
            args=(f_path, rows_to_remove, columns_to_remove),
            daemon=True
        )
        thread.start()
    
    def batch_process(self):
        """Start batch processing of folder."""
        # Get selections
        selected_row_indices = self.row_listbox.curselection()
        rows_to_remove = [self.row_listbox.get(i) for i in selected_row_indices]
        
        selected_col_indices = self.column_listbox.curselection()
        columns_to_remove = [self.column_listbox.get(i) for i in selected_col_indices]
        
        # Check if at least one item is selected
        if not rows_to_remove and not columns_to_remove:
            messagebox.showwarning("警告", "请至少选择一个要移除的行或列。")
            return
        
        # Select folder
        folder_path = filedialog.askdirectory(title="选择包含Excel文件的文件夹")
        if not folder_path:
            return
        
        # Disable controls during processing
        self._disable_controls()
        self.set_status("正在批量处理文件夹... 请稍候。")
        self.log_text.delete('1.0', tk.END)
        
        # Start batch processing thread
        thread = threading.Thread(
            target=self._batch_process_thread,
            args=(folder_path, rows_to_remove, columns_to_remove),
            daemon=True
        )
        thread.start()
    
    def export_structure(self):
        """Export current structure information."""
        output_path = filedialog.asksaveasfilename(
            title="保存结构信息",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        
        if output_path:
            if self.manipulator.export_structure_info(output_path):
                messagebox.showinfo("导出完成", f"结构信息已导出到:\n{output_path}")
                self.log_message(f"结构信息已导出到: {output_path}")
            else:
                messagebox.showerror("导出失败", "无法导出结构信息。")
    
    def _disable_controls(self):
        """Disable controls during processing."""
        self.load_button.config(state=tk.DISABLED)
        self.row_listbox.config(state=tk.DISABLED)
        self.column_listbox.config(state=tk.DISABLED)
        self.process_button.config(state=tk.DISABLED)
        self.batch_button.config(state=tk.DISABLED)
        self.export_button.config(state=tk.DISABLED)
    
    def _enable_controls(self):
        """Re-enable controls after processing."""
        self.load_button.config(state=tk.NORMAL)
        self.row_listbox.config(state=tk.NORMAL)
        self.column_listbox.config(state=tk.NORMAL)
        self.process_button.config(state=tk.NORMAL)
        self.batch_button.config(state=tk.NORMAL)
        self.export_button.config(state=tk.NORMAL)
    
    def _process_file_thread(self, file_path, rows_to_remove, columns_to_remove):
        """Process file in background thread."""
        try:
            result = self.manipulator.process_excel_file(
                file_path, rows_to_remove, columns_to_remove, 
                progress_callback=self.log_message
            )
            
            if result['success']:
                message = (
                    f"处理完成！\n\n"
                    f"统计信息:\n"
                    f"• 处理的工作表: {result['statistics']['sheets_processed']}\n"
                    f"• 移除的行数: {result['statistics']['rows_removed_total']}\n"
                    f"• 移除的列数: {result['statistics']['columns_removed_total']}\n\n"
                    f"输出文件:\n{result['output_path']}"
                )
                self.master.after(0, lambda: self._processing_finished("Success", message))
            else:
                self.master.after(0, lambda: self._processing_finished("Error", f"处理失败:\n{result['error']}"))
                
        except Exception as e:
            error_message = f"文件处理过程中发生错误:\n{str(e)}"
            self.master.after(0, lambda: self._processing_finished("Error", error_message))
    
    def _batch_process_thread(self, folder_path, rows_to_remove, columns_to_remove):
        """Batch process folder in background thread."""
        try:
            result = self.manipulator.batch_process_folder(
                folder_path, rows_to_remove, columns_to_remove,
                progress_callback=self.log_message
            )
            
            if result['success']:
                message = (
                    f"批量处理完成！\n\n"
                    f"{result['summary']}\n\n"
                    f"成功处理的文件:\n"
                )
                for file_info in result['processed_files'][:5]:  # Show first 5
                    message += f"• {file_info['input_file']}\n"
                if len(result['processed_files']) > 5:
                    message += f"... 以及其他 {len(result['processed_files']) - 5} 个文件\n"
                
                if result['failed_files']:
                    message += f"\n失败的文件 ({len(result['failed_files'])}):\n"
                    for fail_info in result['failed_files'][:3]:  # Show first 3
                        message += f"• {fail_info['file']}: {fail_info['error'][:50]}...\n"
                
                self.master.after(0, lambda: self._processing_finished("Success", message))
            else:
                self.master.after(0, lambda: self._processing_finished("Error", f"批量处理失败:\n{result['error']}"))
                
        except Exception as e:
            error_message = f"批量处理过程中发生错误:\n{str(e)}"
            self.master.after(0, lambda: self._processing_finished("Error", error_message))
    
    def _processing_finished(self, status_type, message):
        """Handle processing completion."""
        # Re-enable controls
        self._enable_controls()
        
        if status_type == "Success":
            messagebox.showinfo("成功", message)
            self.set_status("处理完成。可以处理下一个文件。")
        else:
            messagebox.showerror("错误", message)
            self.set_status("处理失败。请检查错误信息。")


def create_gui():
    """Create and run the GUI application."""
    root = tk.Tk()
    app = DataManipulatorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Command line usage
        input_file = sys.argv[1]
        
        manipulator = DataManipulator()
        
        # Load structure
        structure_result = manipulator.load_file_structure(input_file)
        if not structure_result['success']:
            print(f"Error loading file structure: {structure_result['error']}")
            sys.exit(1)
        
        print(f"File structure loaded:")
        print(f"- Row values: {len(structure_result['row_values'])}")
        print(f"- Column headers: {len(structure_result['column_headers'])}")
        
        # Simple processing (remove nothing by default)
        if len(sys.argv) > 2:
            output_file = sys.argv[2]
        else:
            output_file = None
        
        result = manipulator.process_excel_file(
            input_file, rows_to_remove=[], columns_to_remove=[], 
            output_path=output_file
        )
        
        if result['success']:
            print(f"Processing completed. Output: {result['output_path']}")
            print(f"Statistics: {result['statistics']}")
        else:
            print(f"Processing failed: {result['error']}")
    else:
        # GUI usage
        create_gui()
