Overview
# Metadata Tables used in this ETL process
# •	Table to store expected table mappings
# •	Audit log table for tracking all loads
# •	Summary table for batch-level tracking
# •	Missing sheets tracking
# 1.	Script starts → Generates unique Batch ID
# 2.	Reads config.json → Gets database and email settings
# 3.	Queries ETL_TableConfig → Knows which 100 sheets to expect
# 4.	Opens Excel file → Finds only 50 sheets present
# 5.	Compares lists → Identifies 50 missing sheets
# 6.	Logs missing sheets → Records them in ETL_MissingSheets table
# 7.	Processes 50 available sheets → Loads data in parallel (10 at a time)
# 8.	For each sheet:
# a.	Reads data from Excel
# b.	Cleans and validates
# c.	Loads to staging table
# d.	Merges to target table
# e.	Logs results to ETL_LoadAuditLog
# 9.	Creates batch summary → Saves to ETL_BatchSummary
# 10.	Sends email → Notifies team about partial load and missing sheets


# 2. using ROWID/ctid
# SELECT name, value
# FROM (
#     SELECT name,
#            value,
#            ROW_NUMBER() OVER (PARTITION BY name ORDER BY ROWID DESC) rn
#     FROM data_table
# )
# WHERE rn = 1;
# 3.
# SELECT Email FROM Source_Users WHERE NOT REGEXP_LIKE(Email, '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}$');


# Complete Implementation Steps for Excel to SQL Server Data Loading
# Phase 1: Initial Database Setup
# Step 1: Create the Configuration and Audit Tables
# First, we need to create the database tables that will store our configuration and track all the data loads.
# 1.1 Create the Table Configuration Table
# This table stores information about which Excel sheet maps to which SQL Server table.
# -- Table to store sheet-to-table mapping configuration
CREATE TABLE dbo.ETL_TableConfig (
    ConfigID INT IDENTITY(1,1) PRIMARY KEY,
    SheetName NVARCHAR(255) NOT NULL,
    TargetTableName NVARCHAR(255) NOT NULL,
    TargetSchema NVARCHAR(50) DEFAULT 'dbo',
    IsActive BIT DEFAULT 1,
    IsMandatory BIT DEFAULT 1,
    ExpectedColumnCount INT,
    CreatedDate DATETIME DEFAULT GETDATE(),
    ModifiedDate DATETIME DEFAULT GETDATE(),
    CreatedBy NVARCHAR(100) DEFAULT SYSTEM_USER
);

# -- Create index for faster lookups
# CREATE INDEX IX_ETL_TableConfig_SheetName ON dbo.ETL_TableConfig(SheetName);
# Why we need this: This table tells our ETL process which Excel sheet should load into which SQL Server table. For example, the "Customer_Data" sheet should load into the "dbo.Customers" table.
# 1.2 Create the Load Audit Log Table
# This table records every single sheet load operation with complete details.
# -- Table to track every sheet load with detailed information
CREATE TABLE dbo.ETL_LoadAuditLog (
    LogID BIGINT IDENTITY(1,1) PRIMARY KEY,
    BatchID UNIQUEIDENTIFIER NOT NULL,
    FileName NVARCHAR(500) NOT NULL,
    SheetName NVARCHAR(255),
    TargetTableName NVARCHAR(255),
    TargetSchema NVARCHAR(50),
    RecordsReceived INT,
    RecordsInserted INT,
    RecordsFailed INT,
    LoadStatus NVARCHAR(50),
    ErrorMessage NVARCHAR(MAX),
    StartTime DATETIME,
    EndTime DATETIME,
    DurationSeconds AS DATEDIFF(SECOND, StartTime, EndTime),
    LoadedBy NVARCHAR(100) DEFAULT SYSTEM_USER,
    ServerName NVARCHAR(100) DEFAULT @@SERVERNAME
);

-- Create indexes for reporting
CREATE INDEX IX_ETL_LoadAuditLog_BatchID ON dbo.ETL_LoadAuditLog(BatchID);
CREATE INDEX IX_ETL_LoadAuditLog_LoadStatus ON dbo.ETL_LoadAuditLog(LoadStatus);
CREATE INDEX IX_ETL_LoadAuditLog_StartTime ON dbo.ETL_LoadAuditLog(StartTime);
Why we need this: Every time we load a sheet, we record how many records were loaded, whether it succeeded or failed, how long it took, and any error messages. This gives us a complete history of all our data loads.
1.3 Create the Batch Summary Table
This table stores high-level summary information for each complete ETL run.
-- Table to store overall batch summary
CREATE TABLE dbo.ETL_BatchSummary (
    BatchID UNIQUEIDENTIFIER PRIMARY KEY,
    FileName NVARCHAR(500) NOT NULL,
    TotalSheetsExpected INT,
    TotalSheetsReceived INT,
    TotalSheetsProcessed INT,
    TotalSheetsSuccessful INT,
    TotalSheetsFailed INT,
    TotalSheetsSkipped INT,
    TotalRecordsInserted BIGINT,
    BatchStatus NVARCHAR(50),
    AlertSent BIT DEFAULT 0,
    StartTime DATETIME,
    EndTime DATETIME,
    DurationMinutes AS DATEDIFF(MINUTE, StartTime, EndTime),
    ProcessedBy NVARCHAR(100) DEFAULT SYSTEM_USER
);

-- Create index for date-based queries
CREATE INDEX IX_ETL_BatchSummary_StartTime ON dbo.ETL_BatchSummary(StartTime);
Why we need this: When we finish processing an entire Excel file, we want to see the big picture - how many sheets were supposed to be there, how many we actually processed, total records loaded, and overall status.
1.4 Create the Missing Sheets Tracking Table
This table tracks which sheets we expected but didn't find in the Excel file.
-- Table to track missing sheets
CREATE TABLE dbo.ETL_MissingSheets (
    MissingID INT IDENTITY(1,1) PRIMARY KEY,
    BatchID UNIQUEIDENTIFIER NOT NULL,
    ExpectedSheetName NVARCHAR(255),
    TargetTableName NVARCHAR(255),
    IsMandatory BIT,
    DetectedTime DATETIME DEFAULT GETDATE()
);

-- Create index for batch queries
CREATE INDEX IX_ETL_MissingSheets_BatchID ON dbo.ETL_MissingSheets(BatchID);
Why we need this: When sheets are missing from the Excel file, we need to track which ones they are so we can alert the right people and know which data didn't get loaded.
________________________________________
Step 2: Populate the Configuration Table
Now we need to tell the system which Excel sheets map to which database tables.
-- Insert configuration for all expected sheets
-- This defines the mapping between Excel sheets and SQL Server tables

INSERT INTO dbo.ETL_TableConfig (SheetName, TargetTableName, TargetSchema, IsMandatory, ExpectedColumnCount)
VALUES 
    ('Customer_Data', 'Customers', 'dbo', 1, 15),
    ('Product_Catalog', 'Products', 'dbo', 1, 12),
    ('Sales_Transactions', 'Sales', 'dbo', 1, 20),
    ('Inventory_Details', 'Inventory', 'dbo', 1, 10),
    ('Employee_Info', 'Employees', 'dbo', 1, 18),
    ('Invoice_Data', 'Invoices', 'dbo', 1, 25),
    ('Warehouse_Stock', 'Warehouses', 'dbo', 0, 8),
    ('Supplier_List', 'Suppliers', 'dbo', 0, 10),
    -- Add all 100 sheet mappings here
    ('Vendor_Payments', 'VendorPayments', 'dbo', 1, 14);

-- Verify the configuration
SELECT * FROM dbo.ETL_TableConfig WHERE IsActive = 1;
Important Notes:
•	SheetName: The exact name of the sheet in the Excel file (case-sensitive)
•	TargetTableName: The SQL Server table where data should be loaded
•	IsMandatory: Set to 1 if this sheet MUST be present, 0 if it's optional
•	ExpectedColumnCount: Number of columns we expect (helps with validation)
________________________________________
Phase 2: Create Your Target Tables
Before loading data, you need to create the actual tables where data will be stored.
-- Example: Create target tables for your data

-- Customer table
CREATE TABLE dbo.Customers (
    CustomerID INT PRIMARY KEY,
    CustomerName NVARCHAR(255),
    Email NVARCHAR(255),
    Phone NVARCHAR(50),
    Address NVARCHAR(500),
    City NVARCHAR(100),
    State NVARCHAR(50),
    ZipCode NVARCHAR(20),
    Country NVARCHAR(100),
    CustomerType NVARCHAR(50),
    RegistrationDate DATE,
    LastPurchaseDate DATE,
    TotalPurchases DECIMAL(18,2),
    IsActive BIT,
    Notes NVARCHAR(MAX),
    LoadTimestamp DATETIME,
    BatchID UNIQUEIDENTIFIER
);

-- Product table
CREATE TABLE dbo.Products (
    ProductID INT PRIMARY KEY,
    ProductName NVARCHAR(255),
    Category NVARCHAR(100),
    SubCategory NVARCHAR(100),
    Brand NVARCHAR(100),
    UnitPrice DECIMAL(18,2),
    CostPrice DECIMAL(18,2),
    UnitOfMeasure NVARCHAR(50),
    StockQuantity INT,
    ReorderLevel INT,
    IsActive BIT,
    Description NVARCHAR(MAX),
    LoadTimestamp DATETIME,
    BatchID UNIQUEIDENTIFIER
);

-- Sales table
CREATE TABLE dbo.Sales (
    SaleID INT PRIMARY KEY,
    SaleDate DATE,
    CustomerID INT,
    ProductID INT,
    Quantity INT,
    UnitPrice DECIMAL(18,2),
    Discount DECIMAL(18,2),
    TaxAmount DECIMAL(18,2),
    TotalAmount DECIMAL(18,2),
    PaymentMethod NVARCHAR(50),
    PaymentStatus NVARCHAR(50),
    SalesRep NVARCHAR(100),
    Region NVARCHAR(100),
    StoreLocation NVARCHAR(100),
    OrderNumber NVARCHAR(100),
    InvoiceNumber NVARCHAR(100),
    ShippingAddress NVARCHAR(500),
    ShippingDate DATE,
    DeliveryDate DATE,
    OrderStatus NVARCHAR(50),
    LoadTimestamp DATETIME,
    BatchID UNIQUEIDENTIFIER
);

-- Create staging tables (temporary tables for data validation)
-- Staging tables have the same structure as target tables
-- but are used for initial data load and validation

CREATE TABLE dbo.Customers_Staging (
    CustomerID INT,
    CustomerName NVARCHAR(255),
    Email NVARCHAR(255),
    Phone NVARCHAR(50),
    Address NVARCHAR(500),
    City NVARCHAR(100),
    State NVARCHAR(50),
    ZipCode NVARCHAR(20),
    Country NVARCHAR(100),
    CustomerType NVARCHAR(50),
    RegistrationDate DATE,
    LastPurchaseDate DATE,
    TotalPurchases DECIMAL(18,2),
    IsActive BIT,
    Notes NVARCHAR(MAX),
    LoadTimestamp DATETIME,
    BatchID UNIQUEIDENTIFIER
);

-- Repeat for all other staging tables...
Why we need staging tables: We load data into staging tables first. This allows us to validate and clean the data before moving it to the final tables. If something goes wrong, the staging data can be reviewed and corrected without affecting the production tables.
________________________________________
Phase 3: Setting Up the Python ETL Script
Now let's create the Python script that will handle the entire ETL process.
Step 3: Install Required Python Libraries
First, install all the necessary Python packages:
# Open command prompt or terminal and run these commands:

pip install pandas
pip install openpyxl
pip install pyodbc
pip install sqlalchemy
What each library does:
•	pandas: Reads and manipulates Excel data
•	openpyxl: Engine for reading Excel files
•	pyodbc: Connects Python to SQL Server
•	sqlalchemy: Helps with database operations
________________________________________
Step 4: Create the Configuration File
Create a file called config.json with your database and email settings:
{
    "database": {
        "server": "your-sql-server-name",
        "database": "your-database-name",
        "username": "your-username",
        "password": "your-password",
        "connection_string": "DRIVER={ODBC Driver 17 for SQL Server};SERVER=your-sql-server-name;DATABASE=your-database-name;UID=your-username;PWD=your-password"
    },
    "email": {
        "server": "smtp.company.com",
        "port": 587,
        "username": "etl-notifications@company.com",
        "password": "your-email-password",
        "from_address": "etl-notifications@company.com",
        "spoc_list": [
            "manager@company.com",
            "data-team@company.com",
            "operations@company.com"
        ]
    },
    "file_paths": {
        "input_folder": "C:/ETL/Input",
        "archive_folder": "C:/ETL/Archive",
        "error_folder": "C:/ETL/Error"
    },
    "max_parallel_sheets": 10
}
Important: Replace the placeholder values with your actual server name, database name, credentials, and email settings.
________________________________________
Step 5: Create the Main ETL Python Script
Create a file called etl_main.py:
import pandas as pd
import pyodbc
import uuid
from datetime import datetime
import logging
import json
import os
from typing import Dict, List, Tuple
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from concurrent.futures import ThreadPoolExecutor, as_completed

class ExcelToSQLServerETL:
    """
    Main ETL class that handles the entire process of loading 
    Excel files into SQL Server tables
    """
    
    def __init__(self, config_file: str):
        """
        Initialize the ETL process
        
        Args:
            config_file: Path to the JSON configuration file
        """
        # Load configuration from JSON file
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        
        # Generate a unique batch ID for this ETL run
        self.batch_id = str(uuid.uuid4())
        
        # Setup logging
        self.setup_logging()
        
        self.logger.info("="*80)
        self.logger.info(f"ETL Process Started - Batch ID: {self.batch_id}")
        self.logger.info("="*80)
    
    def setup_logging(self):
        """
        Setup logging to file and console
        Creates a new log file for each run with timestamp
        """
        log_filename = f"ETL_Log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Log file created: {log_filename}")
    
    def get_database_connection(self):
        """
        Create and return a connection to SQL Server
        
        Returns:
            pyodbc connection object
        """
        try:
            conn = pyodbc.connect(self.config['database']['connection_string'])
            self.logger.info("Database connection established successfully")
            return conn
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {str(e)}")
            raise
    
    def get_expected_sheet_mappings(self) -> pd.DataFrame:
        """
        Retrieve the expected sheet-to-table mappings from the configuration table
        
        Returns:
            DataFrame with columns: SheetName, TargetTableName, TargetSchema, IsMandatory
        """
        self.logger.info("Retrieving expected sheet mappings from database...")
        
        query = """
        SELECT 
            SheetName, 
            TargetTableName, 
            TargetSchema, 
            IsMandatory, 
            ExpectedColumnCount
        FROM dbo.ETL_TableConfig
        WHERE IsActive = 1
        ORDER BY SheetName
        """
        
        conn = self.get_database_connection()
        expected_mappings = pd.read_sql(query, conn)
        conn.close()
        
        self.logger.info(f"Found {len(expected_mappings)} expected sheet mappings")
        return expected_mappings
    
    def detect_sheets_in_excel(self, excel_file: str) -> List[str]:
        """
        Detect all sheet names present in the Excel file
        
        Args:
            excel_file: Path to the Excel file
            
        Returns:
            List of sheet names
        """
        self.logger.info(f"Detecting sheets in Excel file: {excel_file}")
        
        try:
            # Open the Excel file
            excel_obj = pd.ExcelFile(excel_file)
            sheet_names = excel_obj.sheet_names
            
            self.logger.info(f"Detected {len(sheet_names)} sheets in the Excel file")
            
            # Log all detected sheet names
            for i, sheet in enumerate(sheet_names, 1):
                self.logger.info(f"  {i}. {sheet}")
            
            return sheet_names
            
        except Exception as e:
            self.logger.error(f"Error reading Excel file: {str(e)}")
            raise
    
    def compare_expected_vs_actual_sheets(self, excel_file: str) -> Tuple[Dict, List, List]:
        """
        Compare expected sheets (from config table) with actual sheets (in Excel file)
        
        Args:
            excel_file: Path to the Excel file
            
        Returns:
            Tuple of (valid_mappings_dict, missing_sheets_list, unexpected_sheets_list)
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("STEP: Comparing Expected vs Actual Sheets")
        self.logger.info("="*80)
        
        # Get expected mappings from database
        expected_mappings = self.get_expected_sheet_mappings()
        expected_sheet_names = set(expected_mappings['SheetName'].tolist())
        
        # Get actual sheets from Excel file
        actual_sheet_names = set(self.detect_sheets_in_excel(excel_file))
        
        # Find sheets that are in Excel file and match our configuration
        matching_sheets = expected_sheet_names.intersection(actual_sheet_names)
        
        # Find sheets that we expected but are missing from Excel file
        missing_sheets = expected_sheet_names - actual_sheet_names
        
        # Find sheets that are in Excel but not in our configuration
        unexpected_sheets = actual_sheet_names - expected_sheet_names
        
        # Create a dictionary of valid mappings (sheets that we can process)
        valid_mappings = {}
        for _, row in expected_mappings.iterrows():
            if row['SheetName'] in matching_sheets:
                valid_mappings[row['SheetName']] = {
                    'target_table': row['TargetTableName'],
                    'target_schema': row['TargetSchema'],
                    'is_mandatory': row['IsMandatory']
                }
        
        # Log summary
        self.logger.info("\nComparison Summary:")
        self.logger.info(f"  Expected sheets: {len(expected_sheet_names)}")
        self.logger.info(f"  Actual sheets in file: {len(actual_sheet_names)}")
        self.logger.info(f"  Matching sheets (will process): {len(matching_sheets)}")
        self.logger.info(f"  Missing sheets: {len(missing_sheets)}")
        self.logger.info(f"  Unexpected sheets: {len(unexpected_sheets)}")
        
        # Log missing sheets in detail
        if missing_sheets:
            self.logger.warning("\n⚠️  MISSING SHEETS ALERT:")
            mandatory_missing = expected_mappings[
                (expected_mappings['SheetName'].isin(missing_sheets)) & 
                (expected_mappings['IsMandatory'] == True)
            ]
            
            for sheet in sorted(missing_sheets):
                is_mandatory = sheet in mandatory_missing['SheetName'].values
                marker = "❌ CRITICAL" if is_mandatory else "⚠️  Optional"
                self.logger.warning(f"  {marker} - Missing: {sheet}")
        
        # Log unexpected sheets
        if unexpected_sheets:
            self.logger.info("\nUnexpected sheets found (not in configuration):")
            for sheet in sorted(unexpected_sheets):
                self.logger.info(f"  ℹ️  {sheet}")
        
        return valid_mappings, list(missing_sheets), list(unexpected_sheets)
    
    def read_excel_sheet(self, excel_file: str, sheet_name: str) -> pd.DataFrame:
        """
        Read a specific sheet from the Excel file into a pandas DataFrame
        
        Args:
            excel_file: Path to the Excel file
            sheet_name: Name of the sheet to read
            
        Returns:
            DataFrame containing the sheet data
        """
        try:
            self.logger.info(f"Reading sheet: {sheet_name}")
            
            # Read the Excel sheet
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            
            initial_row_count = len(df)
            self.logger.info(f"  Initial rows read: {initial_row_count}")
            
            # Data cleaning: Remove completely empty rows
            df = df.dropna(how='all')
            rows_after_cleaning = len(df)
            
            if rows_after_cleaning < initial_row_count:
                self.logger.info(f"  Removed {initial_row_count - rows_after_cleaning} empty rows")
            
            # Trim whitespace from string columns
            string_columns = df.select_dtypes(include=['object']).columns
            for col in string_columns:
                df[col] = df[col].astype(str).str.strip()
            
            self.logger.info(f"  Final rows to process: {rows_after_cleaning}")
            self.logger.info(f"  Columns: {len(df.columns)}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error reading sheet '{sheet_name}': {str(e)}")
            raise
    
    def load_sheet_to_staging(self, df: pd.DataFrame, staging_table: str, 
                             target_schema: str = 'dbo') -> int:
        """
        Load DataFrame data into a staging table
        
        Args:
            df: DataFrame containing the data
            staging_table: Name of the staging table
            target_schema: Schema name (default: dbo)
            
        Returns:
            Number of records loaded
        """
        try:
            self.logger.info(f"  Loading data to staging table: {target_schema}.{staging_table}")
            
            # Add audit columns
            df['LoadTimestamp'] = datetime.now()
            df['BatchID'] = self.batch_id
            
            # Create SQLAlchemy engine for bulk insert
            from sqlalchemy import create_engine
            engine_string = (
                f"mssql+pyodbc://{self.config['database']['username']}:"
                f"{self.config['database']['password']}@"
                f"{self.config['database']['server']}/"
                f"{self.config['database']['database']}?"
                f"driver=ODBC+Driver+17+for+SQL+Server"
            )
            engine = create_engine(engine_string)
            
            # Load data to staging table (replace existing data)
            df.to_sql(
                name=staging_table,
                con=engine,
                schema=target_schema,
                if_exists='replace',
                index=False,
                method='multi',
                chunksize=1000
            )
            
            records_loaded = len(df)
            self.logger.info(f"  ✓ Successfully loaded {records_loaded} records to staging")
            
            return records_loaded
            
        except Exception as e:
            self.logger.error(f"Error loading to staging table: {str(e)}")
            raise
    
    def merge_staging_to_target(self, staging_table: str, target_table: str,
                               target_schema: str = 'dbo') -> Dict:
        """
        Merge data from staging table to target table
        
        Args:
            staging_table: Name of the staging table
            target_table: Name of the target table
            target_schema: Schema name
            
        Returns:
            Dictionary with 'inserted' and 'failed' counts
        """
        try:
            self.logger.info(f"  Merging data from staging to target table: {target_schema}.{target_table}")
            
            conn = self.get_database_connection()
            cursor = conn.cursor()
            
            # Simple INSERT strategy (you can enhance this with MERGE for upserts)
            merge_query = f"""
            INSERT INTO {target_schema}.{target_table}
            SELECT * FROM {target_schema}.{staging_table}
            """
            
            cursor.execute(merge_query)
            inserted_count = cursor.rowcount
            
            conn.commit()
            cursor.close()
            conn.close()
            
            self.logger.info(f"  ✓ Successfully merged {inserted_count} records to target table")
            
            return {'inserted': inserted_count, 'failed': 0}
            
        except Exception as e:
            self.logger.error(f"Error during merge operation: {str(e)}")
            return {'inserted': 0, 'failed': 0}
    
    def load_single_sheet(self, excel_file: str, sheet_name: str,
                         target_table: str, target_schema: str = 'dbo') -> Dict:
        """
        Complete process to load a single sheet from Excel to SQL Server
        
        Args:
            excel_file: Path to the Excel file
            sheet_name: Name of the sheet to load
            target_table: Name of the target table
            target_schema: Schema name
            
        Returns:
            Dictionary with load statistics
        """
        start_time = datetime.now()
        
        # Initialize statistics dictionary
        stats = {
            'sheet_name': sheet_name,
            'target_table': target_table,
            'target_schema': target_schema,
            'records_received': 0,
            'records_inserted': 0,
            'records_failed': 0,
            'status': 'Failed',
            'error_message': None,
            'start_time': start_time,
            'end_time': None
        }
        
        try:
            self.logger.info("\n" + "-"*80)
            self.logger.info(f"Processing Sheet: {sheet_name}")
            self.logger.info("-"*80)
            
            # Step 1: Read data from Excel sheet
            df = self.read_excel_sheet(excel_file, sheet_name)
            stats['records_received'] = len(df)
            
            # Step 2: Load to staging table
            staging_table = f"{target_table}_Staging"
            self.load_sheet_to_staging(df, staging_table, target_schema)
            
            # Step 3: Merge from staging to target
            merge_result = self.merge_staging_to_target(
                staging_table, target_table, target_schema
            )
            
            stats['records_inserted'] = merge_result['inserted']
            stats['records_failed'] = merge_result['failed']
            
            # Determine final status
            if merge_result['failed'] == 0:
                stats['status'] = 'Success'
                self.logger.info(f"✓ Sheet '{sheet_name}' loaded successfully!")
            else:
                stats['status'] = 'Partial'
                self.logger.warning(f"⚠️  Sheet '{sheet_name}' loaded with some failures")
            
        except Exception as e:
            stats['status'] = 'Failed'
            stats['error_message'] = str(e)
            self.logger.error(f"❌ Failed to load sheet '{sheet_name}': {str(e)}")
        
        finally:
            stats['end_time'] = datetime.now()
            duration = (stats['end_time'] - stats['start_time']).total_seconds()
            self.logger.info(f"Duration: {duration:.2f} seconds")
        
        return stats
    
    def log_sheet_load_to_audit(self, file_name: str, stats: Dict):
        """
        Insert load statistics into the audit log table
        
        Args:
            file_name: Name of the Excel file
            stats: Dictionary containing load statistics
        """
        try:
            conn = self.get_database_connection()
            cursor = conn.cursor()
            
            insert_query = """
            INSERT INTO dbo.ETL_LoadAuditLog 
            (BatchID, FileName, SheetName, TargetTableName, TargetSchema,
             RecordsReceived, RecordsInserted, RecordsFailed, LoadStatus, 
             ErrorMessage, StartTime, EndTime)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            cursor.execute(insert_query, (
                self.batch_id,
                file_name,
                stats['sheet_name'],
                stats['target_table'],
                stats['target_schema'],
                stats['records_received'],
                stats['records_inserted'],
                stats['records_failed'],
                stats['status'],
                stats['error_message'],
                stats['start_time'],
                stats['end_time']
            ))
            
            conn.commit()
            cursor.close()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to log audit record: {str(e)}")
    
    def log_missing_sheets(self, missing_sheets: List[str], expected_mappings: pd.DataFrame):
        """
        Log missing sheets to the tracking table
        
        Args:
            missing_sheets: List of sheet names that are missing
            expected_mappings: DataFrame with expected mappings
        """
        if not missing_sheets:
            return
        
        try:
            self.logger.info("\nLogging missing sheets to database...")
            
            conn = self.get_database_connection()
            cursor = conn.cursor()
            
            for sheet in missing_sheets:
                mapping = expected_mappings[expected_mappings['SheetName'] == sheet]
                if not mapping.empty:
                    cursor.execute("""
                        INSERT INTO dbo.ETL_MissingSheets 
                        (BatchID, ExpectedSheetName, TargetTableName, IsMandatory)
                        VALUES (?, ?, ?, ?)
                    """, (
                        self.batch_id,
                        sheet,
                        mapping.iloc[0]['TargetTableName'],
                        int(mapping.iloc[0]['IsMandatory'])
                    ))
            
            conn.commit()
            cursor.close()
            conn.close()
            
            self.logger.info(f"Logged {len(missing_sheets)} missing sheets")
            
        except Exception as e:
            self.logger.error(f"Failed to log missing sheets: {str(e)}")
    
    def process_all_sheets_parallel(self, excel_file: str, valid_mappings: Dict) -> List[Dict]:
        """
        Process multiple sheets in parallel for better performance
        
        Args:
            excel_file: Path to the Excel file
            valid_mappings: Dictionary of sheet mappings to process
            
        Returns:
            List of load statistics for all sheets
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("STEP: Processing All Sheets (Parallel)")
        self.logger.info("="*80)
        
        load_results = []
        max_workers = self.config.get('max_parallel_sheets', 10)
        
        self.logger.info(f"Processing {len(valid_mappings)} sheets using {max_workers} parallel workers")
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all sheet processing tasks
            future_to_sheet = {
                executor.submit(
                    self.load_single_sheet,
                    excel_file,
                    sheet_name,
                    mapping['target_table'],
                    mapping['target_schema']
                ): sheet_name
                for sheet_name, mapping in valid_mappings.items()
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_sheet):
                sheet_name = future_to_sheet[future]
                try:
                    result = future.result()
                    load_results.append(result)
                    
                    # Log to audit table immediately after each sheet completes
                    self.log_sheet_load_to_audit(excel_file, result)
                    
                except Exception as e:
                    self.logger.error(f"Unexpected error processing sheet '{sheet_name}': {str(e)}")
        
        return load_results
    
    def create_batch_summary(self, file_name: str, expected_count: int,
                            received_count: int, load_results: List[Dict]):
        """
        Create and save batch summary to database
        
        Args:
            file_name: Name of the Excel file
            expected_count: Number of expected sheets
            received_count: Number of sheets received
            load_results: List of load statistics
        """
        try:
            self.logger.info("\n" + "="*80)
            self.logger.info("STEP: Creating Batch Summary")
            self.logger.info("="*80)
            
            # Calculate summary statistics
            successful = len([r for r in load_results if r['status'] == 'Success'])
            failed = len([r for r in load_results if r['status'] == 'Failed'])
            partial = len([r for r in load_results if r['status'] == 'Partial'])
            total_records = sum(r['records_inserted'] for r in load_results)
            
            # Determine overall batch status
            if received_count == expected_count and failed == 0:
                batch_status = 'Complete'
            elif failed > 0:
                batch_status = 'Partial'
            else:
                batch_status = 'Incomplete'
            
            # Log summary to console
            self.logger.info(f"\nBatch Summary:")
            self.logger.info(f"  Batch ID: {self.batch_id}")
            self.logger.info(f"  File Name: {file_name}")
            self.logger.info(f"  Expected Sheets: {expected_count}")
            self.logger.info(f"  Received Sheets: {received_count}")
            self.logger.info(f"  Processed Sheets: {len(load_results)}")
            self.logger.info(f"  Successful: {successful}")
            self.logger.info(f"  Failed: {failed}")
            self.logger.info(f"  Partial: {partial}")
            self.logger.info(f"  Total Records Inserted: {total_records:,}")
            self.logger.info(f"  Batch Status: {batch_status}")
            
            # Save to database
            conn = self.get_database_connection()
            cursor = conn.cursor()
            
            if load_results:
                batch_start = min(r['start_time'] for r in load_results)
                batch_end = max(r['end_time'] for r in load_results)
            else:
                batch_start = batch_end = datetime.now()
            
            cursor.execute("""
                INSERT INTO dbo.ETL_BatchSummary
                (BatchID, FileName, TotalSheetsExpected, TotalSheetsReceived,
                 TotalSheetsProcessed, TotalSheetsSuccessful, TotalSheetsFailed,
                 TotalRecordsInserted, BatchStatus, StartTime, EndTime)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self.batch_id,
                file_name,
                expected_count,
                received_count,
                len(load_results),
                successful,
                failed,
                total_records,
                batch_status,
                batch_start,
                batch_end
            ))
            
            conn.commit()
            cursor.close()
            conn.close()
            
            self.logger.info("✓ Batch summary saved to database")
            
        except Exception as e:
            self.logger.error(f"Failed to create batch summary: {str(e)}")
    
    def send_email_notification(self, file_name: str, load_results: List[Dict],
                                missing_sheets: List[str], expected_count: int):
        """
        Send email notification with load summary
        
        Args:
            file_name: Name of the Excel file
            load_results: List of load statistics
            missing_sheets: List of missing sheet names
            expected_count: Number of expected sheets
        """
        try:
            self.logger.info("\n" + "="*80)
            self.logger.info("STEP: Sending Email Notification")
            self.logger.info("="*80)
            
            smtp_config = self.config['email']
            
            # Determine email subject based on status
            if missing_sheets:
                subject = f"⚠️  ETL PARTIAL LOAD - {file_name}"
            else:
                subject = f"✓ ETL COMPLETE - {file_name}"
            
            # Generate HTML email body
            html_body = self.generate_email_html(
                file_name, load_results, missing_sheets, expected_count
            )
            
            # Setup email message
            msg = MIMEMultipart('alternative')
            msg['From'] = smtp_config['from_address']
            msg['To'] = ', '.join(smtp_config['spoc_list'])
            msg['Subject'] = subject
            
            msg.attach(MIMEText(html_body, 'html'))
            
            # Send email
            server = smtplib.SMTP(smtp_config['server'], smtp_config['port'])
            server.starttls()
            server.login(smtp_config['username'], smtp_config['password'])
            server.send_message(msg)
            server.quit()
            
            self.logger.info(f"✓ Email notification sent to: {', '.join(smtp_config['spoc_list'])}")
            
            # Update batch summary to mark alert as sent
            conn = self.get_database_connection()
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE dbo.ETL_BatchSummary SET AlertSent = 1 WHERE BatchID = ?",
                (self.batch_id,)
            )
            conn.commit()
            cursor.close()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to send email notification: {str(e)}")
    
    def generate_email_html(self, file_name: str, load_results: List[Dict],
                           missing_sheets: List[str], expected_count: int) -> str:
        """
        Generate HTML content for email notification
        
        Args:
            file_name: Name of the Excel file
            load_results: List of load statistics
            missing_sheets: List of missing sheet names
            expected_count: Number of expected sheets
            
        Returns:
            HTML string
        """
        successful = len([r for r in load_results if r['status'] == 'Success'])
        failed = len([r for r in load_results if r['status'] == 'Failed'])
        total_records = sum(r['records_inserted'] for r in load_results)
        
        # Determine header color based on status
        header_color = '#ff9800' if missing_sheets or failed > 0 else '#4CAF50'
        
        html = f"""
        <html>
        <head>
            <style>
                body {{ 
                    font-family: Arial, sans-serif; 
                    line-height: 1.6;
                    color: #333;
                }}
                .header {{ 
                    background-color: {header_color}; 
                    color: white; 
                    padding: 20px; 
                    border-radius: 5px 5px 0 0;
                }}
                .content {{
                    padding: 20px;
                    background-color: #f9f9f9;
                }}
                .stats {{ 
                    background-color: white; 
                    padding: 15px; 
                    border-radius: 5px; 
                    margin: 15px 0;
                    border-left: 4px solid #2196F3;
                }}
                .warning {{ 
                    background-color: #fff3cd;
                    border-left: 4px solid #ff9800;
                    padding: 15px;
                    margin: 15px 0;
                    border-radius: 5px;
                }}
                table {{ 
                    border-collapse: collapse; 
                    width: 100%; 
                    margin: 20px 0;
                    background-color: white;
                }}
                th, td {{ 
                    border: 1px solid #ddd; 
                    padding: 12px; 
                    text-align: left; 
                }}
                th {{ 
                    background-color: #2196F3; 
                    color: white;
                    font-weight: bold;
                }}
                tr:nth-child(even) {{
                    background-color: #f2f2f2;
                }}
                .success {{ color: #4CAF50; font-weight: bold; }}
                .failed {{ color: #f44336; font-weight: bold; }}
                .partial {{ color: #ff9800; font-weight: bold; }}
                .footer {{
                    margin-top: 30px;
                    padding: 15px;
                    background-color: #e9ecef;
                    border-radius: 5px;
                    font-size: 0.9em;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2>📊 ETL Load Summary Report</h2>
                <p><strong>File:</strong> {file_name}</p>
                <p><strong>Batch ID:</strong> {self.batch_id}</p>
                <p><strong>Timestamp:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="content">
                <div class="stats">
                    <h3>📈 Load Statistics</h3>
                    <ul>
                        <li><strong>Expected Sheets:</strong> {expected_count}</li>
                        <li><strong>Received Sheets:</strong> {len(load_results)}</li>
                        <li><strong>Successfully Loaded:</strong> <span class="success">{successful}</span></li>
                        <li><strong>Failed:</strong> <span class="failed">{failed}</span></li>
                        <li><strong>Total Records Inserted:</strong> {total_records:,}</li>
                    </ul>
                </div>
        """
        
        # Add missing sheets warning if applicable
        if missing_sheets:
            html += f"""
                <div class="warning">
                    <h3>⚠️  Missing Sheets Alert</h3>
                    <p><strong>{len(missing_sheets)} expected sheets were NOT found in the file:</strong></p>
                    <ul>
            """
            for sheet in sorted(missing_sheets):
                html += f"<li>{sheet}</li>"
            html += """
                    </ul>
                    <p><strong>Action Required:</strong> Please verify if these sheets should have been included in the Excel file.</p>
                </div>
            """
        
        # Add detailed results table
        html += """
                <h3>📋 Detailed Load Results</h3>
                <table>
                    <tr>
                        <th>Sheet Name</th>
                        <th>Target Table</th>
                        <th>Records Received</th>
                        <th>Records Inserted</th>
                        <th>Status</th>
                        <th>Duration (sec)</th>
                    </tr>
        """
        
        for result in sorted(load_results, key=lambda x: x['sheet_name']):
            status_class = 'success' if result['status'] == 'Success' else 'failed'
            duration = (result['end_time'] - result['start_time']).total_seconds()
            
            html += f"""
                    <tr>
                        <td>{result['sheet_name']}</td>
                        <td>{result['target_table']}</td>
                        <td>{result['records_received']:,}</td>
                        <td>{result['records_inserted']:,}</td>
                        <td class="{status_class}">{result['status']}</td>
                        <td>{duration:.2f}</td>
                    </tr>
            """
        
        html += """
                </table>
                
                <div class="footer">
                    <p><strong>Note:</strong> Detailed audit logs are available in the ETL_LoadAuditLog table.</p>
                    <p>For questions or issues, please contact the Data Engineering team.</p>
                    <p><em>This is an automated notification from the ETL system.</em></p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def execute_full_etl(self, excel_file: str):
        """
        Main method to execute the complete ETL process
        
        Args:
            excel_file: Path to the Excel file to process
            
        Returns:
            Dictionary with execution summary
        """
        overall_start = datetime.now()
        
        try:
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"Starting ETL Process")
            self.logger.info(f"Excel File: {excel_file}")
            self.logger.info(f"{'='*80}\n")
            
            # Step 1: Compare expected vs actual sheets
            valid_mappings, missing_sheets, unexpected_sheets = \
                self.compare_expected_vs_actual_sheets(excel_file)
            
            expected_mappings = self.get_expected_sheet_mappings()
            expected_count = len(expected_mappings)
            
            # Step 2: Log missing sheets
            self.log_missing_sheets(missing_sheets, expected_mappings)
            
            # Step 3: Process all valid sheets in parallel
            load_results = self.process_all_sheets_parallel(excel_file, valid_mappings)
            
            # Step 4: Create batch summary
            self.create_batch_summary(
                excel_file,
                expected_count,
                len(valid_mappings),
                load_results
            )
            
            # Step 5: Send email notification
            self.send_email_notification(
                excel_file,
                load_results,
                missing_sheets,
                expected_count
            )
            
            # Calculate total duration
            overall_end = datetime.now()
            duration_minutes = (overall_end - overall_start).total_seconds() / 60
            
            # Final summary
            self.logger.info("\n" + "="*80)
            self.logger.info("ETL PROCESS COMPLETED")
            self.logger.info("="*80)
            self.logger.info(f"Batch ID: {self.batch_id}")
            self.logger.info(f"Total Duration: {duration_minutes:.2f} minutes")
            self.logger.info(f"Sheets Processed: {len(load_results)}")
            self.logger.info(f"Status: {'SUCCESS' if not missing_sheets else 'PARTIAL - MISSING SHEETS'}")
            self.logger.info("="*80 + "\n")
            
            return {
                'status': 'Success' if not missing_sheets else 'Partial',
                'batch_id': self.batch_id,
                'sheets_processed': len(load_results),
                'missing_sheets': len(missing_sheets),
                'duration_minutes': duration_minutes
            }
            
        except Exception as e:
            self.logger.error(f"\n{'='*80}")
            self.logger.error("ETL PROCESS FAILED")
            self.logger.error(f"Error: {str(e)}")
            self.logger.error("="*80 + "\n")
            raise


# Main execution
if __name__ == "__main__":
    """
    Main entry point for the ETL process
    """
    print("\n" + "="*80)
    print("Excel to SQL Server ETL Process")
    print("="*80 + "\n")
    
    # Initialize ETL with configuration file
    etl = ExcelToSQLServerETL('config.json')
    
    # Specify the Excel file to process
    # You can modify this to accept command-line arguments or read from a folder
    excel_file_path = "C:/ETL/Input/Daily_Data_Load_20241230.xlsx"
    
    # Execute the complete ETL process
    result = etl.execute_full_etl(excel_file_path)
    
    # Print final result
    print("\n" + "="*80)
    print("EXECUTION SUMMARY")
    print("="*80)
    print(f"Batch ID: {result['batch_id']}")
    print(f"Status: {result['status']}")
    print(f"Sheets Processed: {result['sheets_processed']}")
    print(f"Missing Sheets: {result['missing_sheets']}")
    print(f"Duration: {result['duration_minutes']:.2f} minutes")
    print("="*80 + "\n")
________________________________________
Phase 4: Running the ETL Process
Step 6: Prepare Your Excel File
Make sure your Excel file has:
•	Sheets named exactly as configured in the ETL_TableConfig table
•	Column names that match your target table structure
•	Clean data (no formula errors, proper date formats, etc.)
Step 7: Run the ETL Script
Open command prompt or terminal and run:
python etl_main.py
The script will:
1.	Read the configuration
2.	Detect all sheets in the Excel file
3.	Compare with expected sheets
4.	Load each sheet to its target table
5.	Log all operations
6.	Send email notification
________________________________________
Phase 5: Monitoring and Troubleshooting
Step 8: Check the Audit Logs
-- View today's loads
SELECT 
    BatchID,
    FileName,
    SheetName,
    TargetTableName,
    RecordsInserted,
    LoadStatus,
    DurationSeconds,
    StartTime
FROM dbo.ETL_LoadAuditLog
WHERE CAST(StartTime AS DATE) = CAST(GETDATE() AS DATE)
ORDER BY StartTime DESC;

-- View batch summary
SELECT 
    BatchID,
    FileName,
    TotalSheetsExpected,
    TotalSheetsReceived,
    TotalSheetsSuccessful,
    TotalSheetsFailed,
    TotalRecordsInserted,
    BatchStatus,
    DurationMinutes
FROM dbo.ETL_BatchSummary
WHERE CAST(StartTime AS DATE) = CAST(GETDATE() AS DATE);

-- View missing sheets
SELECT 
    bs.FileName,
    ms.ExpectedSheetName,
    ms.TargetTableName,
    ms.IsMandatory,
    ms.DetectedTime
FROM dbo.ETL_MissingSheets ms
JOIN dbo.ETL_BatchSummary bs ON ms.BatchID = bs.BatchID
WHERE CAST(ms.DetectedTime AS DATE) = CAST(GETDATE() AS DATE);

-- View failed loads
SELECT 
    SheetName,
    TargetTableName,
    ErrorMessage,
    StartTime
FROM dbo.ETL_LoadAuditLog
WHERE LoadStatus = 'Failed'
AND CAST(StartTime AS DATE) = CAST(GETDATE() AS DATE);
________________________________________
Summary of the Complete Process
Here's what happens when you run the ETL:
1.	Script starts → Generates unique Batch ID
2.	Reads config.json → Gets database and email settings
3.	Queries ETL_TableConfig → Knows which 100 sheets to expect
4.	Opens Excel file → Finds only 50 sheets present
5.	Compares lists → Identifies 50 missing sheets
6.	Logs missing sheets → Records them in ETL_MissingSheets table
7.	Processes 50 available sheets → Loads data in parallel (10 at a time)
8.	For each sheet: 
o	Reads data from Excel
o	Cleans and validates
o	Loads to staging table
o	Merges to target table
o	Logs results to ETL_LoadAuditLog
9.	Creates batch summary → Saves to ETL_BatchSummary
10.	Sends email → Notifies team about partial load and missing sheets
________________________________________
This documentation provides a complete, human-readable guide to implementing the ETL solution. Each step is explained in simple terms with the actual code you need to execute.

