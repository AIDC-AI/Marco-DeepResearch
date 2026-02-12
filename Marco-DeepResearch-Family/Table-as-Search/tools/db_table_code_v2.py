# Copyright (C) 2026 AIDC-AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
DBTableCodeTool - A specialized tool for table operations using MongoDB

This tool provides a clean interface for table operations using MongoDB for storage,
with native support for concurrent writes and reads.

Features:
- CRUD operations on tabular data
- MongoDB-backed storage with atomic operations
- Error handling and debugging
- Data validation
- Batch operations support
- Concurrent write support

v2 版本特性：
1. 已删除 delete_records, delete_duplicates 函数，防止 agent 误删数据
2. create_table 添加共享计数器，任务完成期间只允许调用一次
"""

from typing import Any, Dict, List, Optional, Union
import logging
from datetime import datetime
import traceback
import threading
from pymongo import MongoClient
from pymongo.errors import PyMongoError
import json
import sys, ipdb

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GlobalCreateTableCounter:
    """
    线程安全的全局 create_table 调用计数器。
    
    用于在 main agent 和 sub-agents 之间共享 create_table 调用次数限制。
    所有使用同一个 GlobalCreateTableCounter 实例的工具会共享同一个计数器。
    """
    
    def __init__(self, limit: int = 1):
        """
        初始化全局 create_table 计数器。
        
        Args:
            limit: 最大 create_table 调用次数限制，默认为 1
        """
        self.limit = limit
        self.count = 0
        self._lock = threading.Lock()
    
    def try_increment(self) -> bool:
        """
        尝试增加计数。
        
        Returns:
            如果未超过限制，增加计数并返回 True；
            如果已达到限制，返回 False
        """
        with self._lock:
            if self.count >= self.limit:
                return False
            self.count += 1
            return True
    
    def get_count(self) -> int:
        """获取当前 create_table 调用计数"""
        with self._lock:
            return self.count
    
    def get_remaining(self) -> int:
        """获取剩余可调用次数"""
        with self._lock:
            return max(0, self.limit - self.count)
    
    def reset(self):
        """重置计数器"""
        with self._lock:
            self.count = 0
    
    def __repr__(self) -> str:
        return f"GlobalCreateTableCounter(count={self.count}, limit={self.limit})"


class DBTableCodeTool:
    """
    A tool for performing table operations using MongoDB.

    This tool provides methods for creating, reading, updating, and deleting
    records in MongoDB collections.
    """

    def __init__(self, connection_string: str = "mongodb://localhost:27017/", database_name: str = "tabular_memory", mode: str = "full"):
        """
        Initialize the DBTableCodeTool.

        Args:
            connection_string: MongoDB connection string
            database_name: Name of the database to use
            mode: Permission mode - "full" for all operations, "readonly" for read-only + add operations
        """
        self.client = MongoClient(connection_string)
        self.db = self.client[database_name]
        self.mode = mode
        
        # Create a metadata collection
        self.metadata_collection = self.db["_metadata"]
        self._initialize_metadata()

        logger.info(f"DBTableCodeTool initialized with database: {database_name}, mode: {self.mode}")

    def _initialize_metadata(self):
        """Initialize the metadata collection if it doesn't exist."""
        try:
            # Check if metadata collection exists
            if self.metadata_collection.count_documents({}) == 0:
                metadata = {
                    "_id": "metadata",
                    "tables": {},
                    "created_at": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat()
                }
                self.metadata_collection.insert_one(metadata)
                logger.info("Created metadata collection")
        except Exception as e:
            logger.warning(f"Could not initialize metadata: {e}")

    def _update_metadata(self, table_name: str, operation: str, columns: Optional[List[str]] = None):
        """Update the metadata with atomic operation."""
        try:
            # Use MongoDB's atomic $set operations
            update_doc = {
                "$set": {
                    f"tables.{table_name}.last_updated": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat()
                },
                "$push": {
                    f"tables.{table_name}.operations": {
                        "operation": operation,
                        "timestamp": datetime.now().isoformat()
                    }
                }
            }

            # Ensure table entry exists
            if columns:
                update_doc["$set"][f"tables.{table_name}.columns"] = columns
                if f"tables.{table_name}.created_at" not in update_doc["$set"]:
                    update_doc["$set"][f"tables.{table_name}.created_at"] = datetime.now().isoformat()

            self.metadata_collection.update_one(
                {"_id": "metadata"},
                update_doc,
                upsert=True
            )

        except Exception as e:
            logger.error(f"Failed to update metadata: {e}")

    def _read_metadata_safely(self) -> Dict[str, Any]:
        """Read metadata from MongoDB."""
        try:
            metadata = self.metadata_collection.find_one({"_id": "metadata"})
            if metadata is None:
                # Create new metadata if it doesn't exist
                self._initialize_metadata()
                metadata = self.metadata_collection.find_one({"_id": "metadata"})
            
            # Ensure required fields exist
            if "tables" not in metadata:
                metadata["tables"] = {}
            if "created_at" not in metadata:
                metadata["created_at"] = datetime.now().isoformat()
            if "last_updated" not in metadata:
                metadata["last_updated"] = datetime.now().isoformat()

            return metadata
        except Exception as e:
            logger.error(f"Failed to read metadata: {e}")
            return {
                "tables": {},
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat()
            }

    def _get_collection(self, table_name: str):
        """Get the MongoDB collection for a table."""
        return self.db[table_name]

    def _validate_table_name(self, table_name: str) -> bool:
        """Validate table name format."""
        if not table_name or not isinstance(table_name, str):
            return False

        # MongoDB collection names have restrictions
        # Check for valid characters
        if table_name.startswith("_") or table_name.startswith("system."):
            return False

        # Check for valid characters (alphanumeric, underscore, hyphen)
        valid_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.$")
        if not all(c in valid_chars for c in table_name):
            return False

        # Check length (MongoDB limit is 123 bytes, but we'll be conservative)
        if len(table_name) > 100:
            return False

        return True

    def _validate_record_data(self, record_data: Dict[str, Any]) -> bool:
        """Validate record data format."""
        if not isinstance(record_data, dict):
            return False

        if not record_data:  # Empty record
            return False

        # MongoDB needs _id to be unique, but we don't require it
        # Check that all keys are strings
        for key in record_data.keys():
            if not isinstance(key, str):
                return False

        return True

    def _check_permission(self, operation: str) -> bool:
        """Check if the current mode allows the operation."""
        if self.mode == "full":
            return True

        # readonly mode - only allow read operations and add operations
        readonly_allowed = ["filter_records", "count_records", "get_table_info", "add_records"]
        return operation in readonly_allowed

    def _handle_mongo_error(self, e: Exception, operation: str, table_name: str) -> str:
        """Handle MongoDB errors and provide meaningful messages."""
        error_msg = f"Error during {operation} on table '{table_name}': {str(e)}"
        logger.error(error_msg)

        # Provide specific guidance based on error type
        if "not found" in str(e) or "does not exist" in str(e):
            return f"Table '{table_name}' does not exist. Use create_table() first."
        elif "duplicate key" in str(e):
            return f"Duplicate key error in table '{table_name}'. Check your data for uniqueness conflicts."
        elif "connection" in str(e).lower() or "timeout" in str(e).lower():
            return f"Database connection error: {str(e)}"
        elif "unauthorized" in str(e).lower() or "authentication" in str(e).lower():
            return f"Database authentication error. Check your credentials."
        else:
            return error_msg

    def create_table(self, table_name: str, columns: List[str], task_id: str) -> str:
        """
        Create a new table with specified columns.

        Args:
            table_name: Name of the table
            columns: List of column names
            task_id: Task ID (typically instance_id) to prefix the collection name

        Returns:
            Success message or error message
        """
        # Collection 命名格式: {task_id}_{table_name}
        ######## TIAN LAN DEBUG: REMOVE THIS
        table_name = f'{task_id}_{table_name}' if task_id else table_name

        # Check permission
        if not self._check_permission("create_table"):
            return f"Operation 'create_table' is not allowed in {self.mode} mode. Only read operations and adding records are permitted."

        try:
            # Validate inputs
            if not self._validate_table_name(table_name):
                return f"Invalid table name: '{table_name}'. Use only alphanumeric characters, underscores, hyphens, dots, and dollar signs. Cannot start with '_' or 'system.'"

            if not columns or not isinstance(columns, list):
                return "Columns must be a non-empty list of strings."

            if not all(isinstance(col, str) for col in columns):
                return "All column names must be strings."

            # Check if table already exists (in metadata or physically in MongoDB)
            collection = self._get_collection(table_name)
            metadata = self._read_metadata_safely()
            
            # Check if collection exists in MongoDB
            collection_exists = table_name in self.db.list_collection_names()
            # Check if table exists in metadata
            table_in_metadata = table_name in metadata.get("tables", {})
            
            if collection_exists or table_in_metadata:
                # Table exists - drop it to create a fresh empty table
                logger.info(f"Table '{table_name}' already exists. Dropping it to create a new empty table.")
                try:
                    # Drop the MongoDB collection if it exists
                    if collection_exists:
                        collection.drop()
                        logger.info(f"Dropped MongoDB collection '{table_name}'")
                    
                    # Remove from metadata if it exists there
                    if table_in_metadata:
                        self.metadata_collection.update_one(
                            {"_id": "metadata"},
                            {"$unset": {f"tables.{table_name}": ""}}
                        )
                        logger.info(f"Removed '{table_name}' from metadata")
                except Exception as e:
                    logger.warning(f"Error dropping existing table '{table_name}': {e}")
                    # Continue anyway - try to create the new table

            # Create collection by inserting an empty document with columns info
            # We use a special document to store schema information
            schema_doc = {
                "_id": "__schema__",
                "columns": columns,
                "created_at": datetime.now().isoformat()
            }
            collection.insert_one(schema_doc)

            # Update metadata
            self._update_metadata(table_name, "create", columns)

            logger.info(f"Created table '{table_name}' with columns: {columns}")
            return f"Successfully created table '{table_name}' with {len(columns)} columns: {', '.join(columns)}"

        except Exception as e:
            print(traceback.print_exc())
            return self._handle_mongo_error(e, "create_table", table_name)

    def add_records(self, table_name: str, records_data: List[Dict[str, Any]]) -> str:
        """
        Add records to a table.
        
        For each record, checks if it already exists in the database using filter.
        If exists, skips insertion. If not exists, inserts the record.
        Returns detailed statistics about the insertion process.

        Args:
            table_name: Name of the table
            records_data: List of dictionaries with column names as keys

        Returns:
            Success message with detailed statistics or error message
        """
        try:
            # Validate inputs
            if not self._validate_table_name(table_name):
                return f"Invalid table name: '{table_name}'."

            if not records_data or not isinstance(records_data, list):
                return "Records data must be a non-empty list of dictionaries."

            if not all(self._validate_record_data(record) for record in records_data):
                return "All records must be non-empty dictionaries with string keys."

            collection = self._get_collection(table_name)
            
            # Check if table exists by checking for schema document
            schema_doc = collection.find_one({"_id": "__schema__"})
            if schema_doc is None:
                return f"Table '{table_name}' does not exist. Use create_table() first."

            # Get expected columns from schema (preserve order)
            expected_columns_list = schema_doc.get("columns", [])
            expected_columns = set(expected_columns_list)
            
            # Process all records: fill missing columns with None, remove extra columns
            processed_records = []
            for i, record in enumerate(records_data):
                processed_record = {}
                record_columns = set(record.keys())
                
                # Preserve _id if present (MongoDB special field, may not be in schema)
                if "_id" in record:
                    processed_record["_id"] = record["_id"]
                
                # Fill in all expected columns (in schema order)
                for col in expected_columns_list:
                    if col in record:
                        processed_record[col] = record[col]
                    else:
                        # Missing column - set to None (empty value)
                        processed_record[col] = None
                        logger.debug(f"Record {i} missing column '{col}', setting to None")
                
                # Remove extra columns that are not in the schema (except _id which is handled above)
                extra_columns = record_columns - expected_columns - {"_id"}
                if extra_columns:
                    logger.warning(f"Record {i} has unexpected columns: {', '.join(extra_columns)}. These will be ignored.")
                
                processed_records.append(processed_record)

            # Check each record for existence before insertion
            records_to_insert = []
            records_to_insert_original_indices = []  # Track original indices for summary display
            skipped_records = []
            column_mismatch_records = []  # Track records with column name mismatches
            total_records = len(processed_records)
            
            for i, record in enumerate(processed_records):
                # First check if there's a column mismatch (original data has columns not in schema)
                original_record = records_data[i]
                original_columns = set(original_record.keys()) - {"_id"}
                columns_not_in_schema = original_columns - expected_columns
                
                # Check if all original non-None values are being ignored due to column mismatch
                has_data_in_original = any(v is not None and v != "" for k, v in original_record.items() if k != "_id")
                has_data_in_processed = any(v is not None and v != "" for k, v in record.items() if k != "_id")
                
                if has_data_in_original and not has_data_in_processed and columns_not_in_schema:
                    # This means the original record has data, but after processing all fields became None
                    # because the column names don't match the schema
                    logger.warning(f"Record {i} has column name mismatch - original columns: {original_columns}, expected columns: {expected_columns}")
                    column_mismatch_records.append({
                        "index": i,
                        "original_record": original_record,
                        "original_columns": sorted(original_columns),
                        "expected_columns": sorted(expected_columns),
                        "mismatched_columns": sorted(columns_not_in_schema)
                    })
                    continue  # Skip to next record
                
                # Build filter query to check if record exists
                # Strategy: 
                # 1. First check by _id if present
                # 2. Otherwise, use partial fields from columns (in schema order) for flexible matching
                existing = None
                
                if "_id" in record and record["_id"] is not None:
                    # If _id is present and not None, use it for checking
                    # Make sure we don't match the schema document
                    if record["_id"] == "__schema__":
                        # Skip schema document, treat as new record
                        existing = None
                    else:
                        filter_query = {"_id": record["_id"]}
                        existing = collection.find_one(filter_query)
                else:
                    # Build filter_query using columns in schema order
                    # Use all non-None fields for accurate duplicate detection
                    filter_query = {"_id": {"$ne": "__schema__"}}
                    
                    # Collect all non-None fields in schema order
                    non_none_fields = []
                    for col in expected_columns_list:
                        if col in record and record[col] is not None:
                            non_none_fields.append(col)
                    
                    # Use all non-None fields for matching to ensure accurate duplicate detection
                    if len(non_none_fields) > 0:
                        for col in non_none_fields:
                            filter_query[col] = record[col]
                        
                        # Check if record exists with all non-None fields
                        existing = collection.find_one(filter_query)
                    else:
                        # All fields are None (valid case - might be intentional empty record)
                        logger.warning(f"Record {i} has all fields set to None, skipping insertion")
                        skipped_records.append({
                            "index": i,
                            "record": record,
                            "reason": "all_fields_none"
                        })
                        continue  # Skip to next record
                
                if existing is not None:
                    # Record exists, skip insertion
                    skipped_records.append({
                        "index": i,
                        "record": record,
                        "reason": "duplicate"
                    })
                    logger.debug(f"Record {i} already exists in database, skipping insertion")
                else:
                    # Record doesn't exist, add to insertion list
                    records_to_insert.append(record)
                    records_to_insert_original_indices.append(i)  # Track original index

            # Insert only new records
            inserted_count = 0
            if records_to_insert:
                try:
                    result = collection.insert_many(records_to_insert, ordered=False)
                    inserted_count = len(result.inserted_ids)
                except Exception as insert_error:
                    # Handle insertion errors (e.g., duplicate key errors)
                    logger.warning(f"Error during batch insertion: {insert_error}")
                    # Try inserting one by one to get better error reporting
                    inserted_count = 0
                    for record in records_to_insert:
                        try:
                            collection.insert_one(record)
                            inserted_count += 1
                        except Exception as e:
                            logger.warning(f"Failed to insert record: {e}")

            # Update metadata
            self._update_metadata(table_name, "add_records")

            # Get total count
            total_count = collection.count_documents({"_id": {"$ne": "__schema__"}})

            # Build detailed result message
            result_parts = [
                f"插入任务完成。",
                f"待插入数据总数: {total_records}",
                f"成功插入数据: {inserted_count}",
            ]
            
            # Handle column mismatch records (most critical issue)
            if column_mismatch_records:
                result_parts.append(f"\n❌ 发现 {len(column_mismatch_records)} 条数据的列名与表结构不匹配！")
                result_parts.append("=" * 60)
                result_parts.append("【重要】这些数据无法插入，因为字段名称与表结构定义的列名不一致。请重试！")
                result_parts.append(f"\n表 '{table_name}' 的标准列名（必须使用以下列名）：")
                result_parts.append(f"  {', '.join(expected_columns_list)}")
                
                # Show examples of mismatched records
                for idx, mismatch in enumerate(column_mismatch_records[:3], 1):  # Show first 3
                    result_parts.append(f"\n数据样本 {idx} (原始记录索引 {mismatch['index']}):")
                    result_parts.append(f"  实际使用的列名: {', '.join(mismatch['original_columns'])}")
                    result_parts.append(f"  表结构要求的列名: {', '.join(mismatch['expected_columns'])}")
                    result_parts.append(f"  不在表结构中的列名: {', '.join(mismatch['mismatched_columns'])}")
                    # Show a sample of the data
                    sample_data = {k: str(v)[:50] for k, v in list(mismatch['original_record'].items())[:3]}
                    result_parts.append(f"  数据示例: {sample_data}")
                
                if len(column_mismatch_records) > 3:
                    result_parts.append(f"  ... 还有 {len(column_mismatch_records) - 3} 条记录有相同问题")
                
                result_parts.append("\n💡 解决方案:")
                result_parts.append("  1. 修改数据，使用表结构中定义的标准列名")
                result_parts.append("  2. 或者使用 get_table_info 查看表结构，确认正确的列名并重试")
                result_parts.append("=" * 60)
            
            # Show other skipped records
            if skipped_records:
                # Separate skipped records by reason
                duplicates = [s for s in skipped_records if s.get("reason") == "duplicate"]
                all_none_records = [s for s in skipped_records if s.get("reason") == "all_fields_none"]
                
                result_parts.append(f"\n跳过数据总计: {len(skipped_records)} 条")
                
                if duplicates:
                    result_parts.append(f"  - 数据库中已存在: {len(duplicates)} 条")
                    result_parts.append("\n以下数据在数据库中已存在，已跳过插入（显示前 5 条）：")
                    for skipped in duplicates[:5]:
                        record_str = ", ".join([f"{k}={v}" for k, v in skipped["record"].items() if k != "_id" and v is not None][:3])
                        result_parts.append(f"  记录 {skipped['index']}: {record_str}")
                    if len(duplicates) > 5:
                        result_parts.append(f"  ... 还有 {len(duplicates) - 5} 条记录已跳过")
                    
                    result_parts.append(
                        "\n💡 提示: 如果这些已存在的记录需要更新某些字段信息，"
                        "可以使用 update_records 操作来补充信息。"
                    )
                
                if all_none_records:
                    result_parts.append(f"  - 所有字段为空: {len(all_none_records)} 条")
            
            # Show summary of inserted records
            if inserted_count > 0 and records_to_insert:
                result_parts.append(f"\n✅ 成功插入的数据概要（显示前 {min(10, inserted_count)} 条）：")
                for idx, (record_index, processed_record) in enumerate(zip(records_to_insert_original_indices[:10], records_to_insert[:10])):
                    # Use original record from records_data for display (before None-filling)
                    original_record = records_data[record_index]
                    
                    # Show key fields from original record (non-None fields, excluding _id)
                    # Filter out None, empty strings, and _id
                    record_fields = {k: v for k, v in original_record.items() 
                                   if k != "_id" and v is not None and v != ""}
                    
                    if record_fields:
                        # Get first 5 fields
                        fields_to_show = list(record_fields.items())[:5]
                        record_str = ", ".join([f"{k}={str(v)[:50]}" for k, v in fields_to_show])  # Truncate long values
                        if len(record_fields) > 5:
                            record_str += f" ... (共 {len(record_fields)} 个字段)"
                        result_parts.append(f"  记录 {idx+1}: {record_str}")
                    else:
                        result_parts.append(f"  记录 {idx+1}: (所有字段均为空)")
                if inserted_count > 10:
                    result_parts.append(f"  ... 还有 {inserted_count - 10} 条记录已插入")
            
            result_parts.append(f"\n当前表 '{table_name}' 总记录数: {total_count}")

            logger.info(f"Added {inserted_count} records to table '{table_name}', skipped {len(skipped_records)} existing records")
            return "\n".join(result_parts)

        except Exception as e:
            print(traceback.print_exc())
            return self._handle_mongo_error(e, "add_records", table_name)
            
    def filter_records(self, table_name: str, filter_string: Optional[str] = None, limit: int = 100) -> str:
        """
        Filter records from a table based on MongoDB query.

        Args:
            table_name: Name of the table
            filter_string: MongoDB query as JSON string (e.g., '{"location": "Florida"}')
                          If None, returns all records
            limit: Maximum number of records to return (default: 100)

        Returns:
            Filtered records as formatted string or error message
        """
        try:
            # Validate inputs
            if not self._validate_table_name(table_name):
                return f"Invalid table name: '{table_name}'."

            collection = self._get_collection(table_name)
            
            # Check if table exists
            schema_doc = collection.find_one({"_id": "__schema__"})
            if schema_doc is None:
                return f"Table '{table_name}' does not exist."

            # Build MongoDB query
            query = {"_id": {"$ne": "__schema__"}}  # Exclude schema document
            
            # Parse filter_string if provided
            if filter_string:
                try:
                    # Try to parse as JSON
                    if isinstance(filter_string, dict):
                        user_query = filter_string
                    else:
                        user_query = json.loads(filter_string)
                    
                    # Merge with base query
                    query.update(user_query)
                except (json.JSONDecodeError, TypeError) as e:
                    return f"Invalid filter_string format. Must be valid JSON. Error: {e}"

            # Execute query with limit
            results = list(collection.find(query).limit(limit))

            if not results:
                return "No records match the specified criteria."

            # Format output as a simple table-like representation
            result_lines = [f"Found {len(results)} matching records in table '{table_name}':\n"]
            
            # Get all columns from schema
            display_cols = schema_doc.get("columns", [])
            
            if display_cols:
                # Add header
                result_lines.append(" | ".join(display_cols))
                result_lines.append("-" * (len(" | ".join(display_cols))))
            
            # Add rows
            for record in results:
                values = [str(record.get(col, "")) for col in display_cols]
                result_lines.append(" | ".join(values))

            # Update metadata
            self._update_metadata(table_name, "filter_records")

            logger.info(f"Filtered {len(results)} records from table '{table_name}'")
            return "\n".join(result_lines)

        except Exception as e:
            print(traceback.print_exc())
            return self._handle_mongo_error(e, "filter_records", table_name)

    def count_records(self, table_name: str, filter_string: Optional[str] = None, count_non_null: bool = False) -> str:
        """
        Count records in a table based on MongoDB query.

        Args:
            table_name: Name of the table
            filter_string: MongoDB query as JSON string (e.g., '{"location": "Florida"}')
                          If None, counts all records
            count_non_null: If True, counts only records where all columns are non-null

        Returns:
            Count as string or error message
        """
        try:
            # Validate inputs
            if not self._validate_table_name(table_name):
                return f"Invalid table name: '{table_name}'."

            collection = self._get_collection(table_name)
            
            # Check if table exists
            schema_doc = collection.find_one({"_id": "__schema__"})
            if schema_doc is None:
                return f"Table '{table_name}' does not exist."

            # Build MongoDB query
            query = {"_id": {"$ne": "__schema__"}}  # Exclude schema document

            # If count_non_null is True, add condition to check all columns are non-null
            if count_non_null:
                columns = schema_doc.get("columns", [])
                for col in columns:
                    query[col] = {"$exists": True, "$ne": None}

            # Parse filter_string if provided
            if filter_string:
                try:
                    # Try to parse as JSON
                    if isinstance(filter_string, dict):
                        user_query = filter_string
                    else:
                        user_query = json.loads(filter_string)
                    
                    # Merge with base query
                    query.update(user_query)
                except (json.JSONDecodeError, TypeError) as e:
                    return f"Invalid filter_string format. Must be valid JSON. Error: {e}"

            count = collection.count_documents(query)
            
            result = f"Table '{table_name}' contains {count} record"
            if count != 1:
                result += "s"

            if count_non_null:
                result += " with all fields non-null"
            elif filter_string:
                result += " matching the specified conditions"

            # Update metadata
            self._update_metadata(table_name, "count_records")

            logger.info(f"Counted {count} records in table '{table_name}'")
            return result + "."

        except Exception as e:
            print(traceback.print_exc())
            return self._handle_mongo_error(e, "count_records", table_name)

    def update_record(self, table_name: str, record_id: Union[str, Any],
                     updates: Dict[str, Any], id_column: str = "id") -> str:
        """
        Update a record in a table by ID.

        Args:
            table_name: Name of the table
            record_id: ID of the record to update
            updates: Dictionary of column: new_value pairs
            id_column: Name of the ID column (default: "id")

        Returns:
            Success message or error message
        """
        # Check permission
        if not self._check_permission("update_record"):
            return f"Operation 'update_record' is not allowed in {self.mode} mode. Only read operations and adding records are permitted."

        try:
            # Validate inputs
            if not self._validate_table_name(table_name):
                return f"Invalid table name: '{table_name}'."

            if not self._validate_record_data(updates):
                return "Invalid update data. Must be a non-empty dictionary with string keys."

            collection = self._get_collection(table_name)
            
            # Check if table exists
            schema_doc = collection.find_one({"_id": "__schema__"})
            if schema_doc is None:
                return f"Table '{table_name}' does not exist."

            # Check if ID column exists
            columns = schema_doc.get("columns", [])
            if id_column not in columns:
                return f"ID column '{id_column}' not found in table '{table_name}'. Available columns: {', '.join(columns)}"

            # Build filter query
            mongo_filter = {id_column: record_id}

            # Validate update columns
            invalid_columns = set(updates.keys()) - set(columns)
            if invalid_columns:
                return f"Invalid columns for update: {', '.join(invalid_columns)}. Available columns: {', '.join(columns)}"

            # Build MongoDB update document
            update_doc = {"$set": updates}

            # Execute update
            result = collection.update_one(mongo_filter, update_doc)

            if result.matched_count == 0:
                return f"No record found with {id_column} = {record_id} in table '{table_name}'."

            # Update metadata
            self._update_metadata(table_name, "update_record")

            logger.info(f"Updated record {record_id} in table '{table_name}'")
            return f"Successfully updated record with {id_column} = {record_id} in table '{table_name}'. Updated columns: {', '.join(updates.keys())}"

        except Exception as e:
            print(traceback.print_exc())
            return self._handle_mongo_error(e, "update_record", table_name)

    def update_records(self, table_name: str, filter_query: Dict[str, Any],
                     updates: Dict[str, Any]) -> str:
        """
        Update records in a table.

        Args:
            table_name: Name of the table
            filter_query: Query to find records to update
            updates: Dictionary of column: new_value pairs

        Returns:
            Success message or error message
        """
        # Check permission
        if not self._check_permission("update_records"):
            return f"Operation 'update_records' is not allowed in {self.mode} mode. Only read operations and adding records are permitted."

        try:
            # Validate inputs
            if not self._validate_table_name(table_name):
                return f"Invalid table name: '{table_name}'."

            if not self._validate_record_data(updates):
                return "Invalid update data. Must be a non-empty dictionary with string keys."

            collection = self._get_collection(table_name)
            
            # Check if table exists
            schema_doc = collection.find_one({"_id": "__schema__"})
            if schema_doc is None:
                return f"Table '{table_name}' does not exist."

            # Build MongoDB filter query
            if filter_query:
                mongo_filter = filter_query.copy()
            else:
                mongo_filter = {}
            
            # Exclude schema document
            if "_id" in mongo_filter:
                # If _id is specified, make sure we're not targeting the schema
                if mongo_filter["_id"] == "__schema__":
                    return "Cannot update schema document."
            else:
                # If no _id filter, make sure we don't match schema
                mongo_filter["_id"] = {"$ne": "__schema__"}

            # Validate update columns
            invalid_columns = set(updates.keys()) - set(schema_doc.get("columns", []))
            if invalid_columns:
                return f"Invalid columns for update: {', '.join(invalid_columns)}. Available columns: {', '.join(schema_doc.get('columns', []))}"

            # Build MongoDB update document
            update_doc = {"$set": updates}

            # Execute update
            result = collection.update_many(mongo_filter, update_doc)

            if result.matched_count == 0:
                return f"No records found matching the filter in table '{table_name}'."

            # Update metadata
            self._update_metadata(table_name, "update_records")

            logger.info(f"Updated {result.modified_count} records in table '{table_name}'")
            return f"Successfully updated {result.modified_count} records in table '{table_name}'."

        except Exception as e:
            print(traceback.print_exc())
            return self._handle_mongo_error(e, "update_records", table_name)

    def list_tables(self) -> str:
        """
        List all tables in the database.

        Returns:
            List of tables with their information
        """
        try:
            # Read metadata
            metadata = self._read_metadata_safely()

            tables_info = []

            for table_name, table_meta in metadata["tables"].items():
                collection = self._get_collection(table_name)
                
                try:
                    # Get schema document to find columns
                    schema_doc = collection.find_one({"_id": "__schema__"})
                    record_count = collection.count_documents({"_id": {"$ne": "__schema__"}})
                    columns = schema_doc.get("columns", []) if schema_doc else "Unknown"
                except Exception as e:
                    logger.warning(f"Error reading table {table_name}: {e}")
                    record_count = "Error"
                    columns = "Unknown"

                tables_info.append({
                    "name": table_name,
                    "records": record_count,
                    "columns": columns,
                    "created_at": table_meta.get("created_at", "Unknown"),
                    "last_operation": table_meta.get("operations", [])[-1] if table_meta.get("operations") else None
                })

            if not tables_info:
                return "No tables found. Use create_table() to create a new table."

            # Format output
            result = f"Found {len(tables_info)} table(s):\n\n"

            for table_info in tables_info:
                result += f"Table: {table_info['name']}\n"
                result += f"  Records: {table_info['records']}\n"
                result += f"  Columns: {', '.join(table_info['columns']) if isinstance(table_info['columns'], list) else table_info['columns']}\n"
                result += f"  Created: {table_info['created_at']}\n"
                if table_info['last_operation']:
                    result += f"  Last operation: {table_info['last_operation']['operation']} at {table_info['last_operation']['timestamp']}\n"
                result += "\n"

            logger.info(f"Listed {len(tables_info)} tables")
            return result.rstrip()

        except Exception as e:
            print(traceback.print_exc())
            error_msg = f"Error listing tables: {e}"
            logger.error(error_msg)
            return error_msg

    def get_table_info(self, table_name: str) -> str:
        """
        Get detailed information about a specific table (with reduced output).

        Args:
            table_name: Name of the table

        Returns:
            Table information or error message
        """
        try:
            # Validate inputs
            if not self._validate_table_name(table_name):
                return f"Invalid table name: '{table_name}'."

            collection = self._get_collection(table_name)
            
            # Check if table exists
            schema_doc = collection.find_one({"_id": "__schema__"})
            if schema_doc is None:
                return f"Table '{table_name}' does not exist."

            # Get basic stats
            record_count = collection.count_documents({"_id": {"$ne": "__schema__"}})
            columns = schema_doc.get("columns", [])

            # Read metadata
            metadata = self._read_metadata_safely()
            table_meta = metadata["tables"].get(table_name, {})

            # Gather reduced statistics
            info = f"Table: {table_name}\n"
            info += f"Records: {record_count}\n"
            info += f"Columns: {len(columns)} ({', '.join(columns)})\n"

            # Metadata
            if table_meta:
                info += f"Created: {table_meta.get('created_at', 'Unknown')}\n"
                operations = table_meta.get('operations', [])
                if operations:
                    info += f"Last operation: {operations[-1]['operation']} ({len(operations)} total)\n"

            logger.info(f"Retrieved info for table '{table_name}'")
            return info

        except Exception as e:
            print(traceback.print_exc())
            return self._handle_mongo_error(e, "get_table_info", table_name)

    def delete_records(self, table_name: str, filter_string: Optional[str] = None) -> str:
        """
        Delete records from a table based on MongoDB filter query.
        Only allows deletion of completely empty records (all fields are None or empty).
        Records with any non-empty content will be protected from deletion.

        Args:
            table_name: Name of the table
            filter_string: MongoDB query as JSON string (e.g., '{"location": "Florida"}' or '{"year": {"$gte": 2015}}')
                          If None, deletes all records (use with caution)

        Returns:
            Success message with detailed statistics about deleted and protected records
        """
        # Check permission
        if not self._check_permission("delete_records"):
            return f"Operation 'delete_records' is not allowed in {self.mode} mode. Only read operations and adding records are permitted."

        try:
            # Validate inputs
            if not self._validate_table_name(table_name):
                return f"Invalid table name: '{table_name}'."

            collection = self._get_collection(table_name)
            
            # Check if table exists
            schema_doc = collection.find_one({"_id": "__schema__"})
            if schema_doc is None:
                return f"Table '{table_name}' does not exist."

            # Get schema columns
            columns = schema_doc.get("columns", [])

            # Build MongoDB filter query
            query = {"_id": {"$ne": "__schema__"}}  # Always exclude schema document
            
            # Parse filter_string if provided
            if filter_string:
                try:
                    # Try to parse as JSON
                    if isinstance(filter_string, dict):
                        user_query = filter_string
                    else:
                        user_query = json.loads(filter_string)
                    
                    # Merge with base query (user query takes precedence for _id if specified)
                    if "_id" in user_query:
                        # If user specified _id, make sure it's not the schema document
                        if user_query["_id"] == "__schema__":
                            return "Cannot delete schema document."
                        # Use user's _id filter directly
                        query["_id"] = user_query["_id"]
                        # Merge other fields
                        for key, value in user_query.items():
                            if key != "_id":
                                query[key] = value
                    else:
                        # No _id in user query, merge all fields
                        query.update(user_query)
                except (json.JSONDecodeError, TypeError) as e:
                    return f"Invalid filter_string format. Must be valid JSON. Error: {e}"
            else:
                # No filter provided - warn user but allow deletion of all records
                logger.warning(f"delete_records called without filter_string - will check all records in table '{table_name}'")

            # Fetch all matching records first
            matching_records = list(collection.find(query))

            if not matching_records:
                return f"No records found matching the filter in table '{table_name}'."

            # Helper function to check if a record is completely empty
            def is_record_empty(record: Dict[str, Any]) -> bool:
                """Check if a record is completely empty (all fields except _id are None or empty)."""
                for col in columns:
                    value = record.get(col)
                    # Check if value is not None and not empty string
                    if value is not None and value != "":
                        return False
                return True

            # Separate records into deletable (empty) and protected (non-empty)
            records_to_delete = []
            protected_records = []
            
            for record in matching_records:
                if is_record_empty(record):
                    records_to_delete.append(record)
                else:
                    protected_records.append(record)

            # Delete only empty records
            deleted_count = 0
            if records_to_delete:
                # Build query for empty records using their _id values
                empty_record_ids = [record["_id"] for record in records_to_delete]
                delete_query = {"_id": {"$in": empty_record_ids}}
                result = collection.delete_many(delete_query)
                deleted_count = result.deleted_count

            # Get total count after deletion
            total_after = collection.count_documents({"_id": {"$ne": "__schema__"}})

            # Update metadata
            self._update_metadata(table_name, "delete_records")

            logger.info(f"Deleted {deleted_count} empty records from table '{table_name}', protected {len(protected_records)} non-empty records")
            
            # Build detailed result message
            result_parts = [
                f"删除任务完成。",
                f"匹配的记录总数: {len(matching_records)}",
                f"成功删除的空记录数: {deleted_count}",
                f"因包含非空内容而禁止删除的记录数: {len(protected_records)}",
            ]

            # Show details of deleted records
            if deleted_count > 0:
                result_parts.append(f"\n✅ 成功删除的空记录（显示前 {min(5, deleted_count)} 条）：")
                for idx, record in enumerate(records_to_delete[:5], 1):
                    record_id = record.get("_id", "Unknown")
                    result_parts.append(f"  记录 {idx}: _id={record_id} (所有字段均为空)")
                if deleted_count > 5:
                    result_parts.append(f"  ... 还有 {deleted_count - 5} 条空记录已删除")

            # Show details of protected records
            if protected_records:
                result_parts.append(f"\n⚠️ 禁止删除的记录（包含非空内容，显示前 {min(5, len(protected_records))} 条）：")
                for idx, record in enumerate(protected_records[:5], 1):
                    record_id = record.get("_id", "Unknown")
                    # Show non-empty fields
                    non_empty_fields = []
                    for col in columns:
                        value = record.get(col)
                        if value is not None and value != "":
                            # Truncate long values for display
                            value_str = str(value)
                            if len(value_str) > 30:
                                value_str = value_str[:30] + "..."
                            non_empty_fields.append(f"{col}={value_str}")
                    
                    if non_empty_fields:
                        fields_str = ", ".join(non_empty_fields[:3])  # Show first 3 fields
                        if len(non_empty_fields) > 3:
                            fields_str += f" ... (共 {len(non_empty_fields)} 个非空字段)"
                        result_parts.append(f"  记录 {idx}: _id={record_id}, {fields_str}")
                    else:
                        result_parts.append(f"  记录 {idx}: _id={record_id} (包含非空内容)")
                
                if len(protected_records) > 5:
                    result_parts.append(f"  ... 还有 {len(protected_records) - 5} 条记录因包含非空内容而被保护")

            result_parts.append(f"\n当前表 '{table_name}' 总记录数: {total_after}")

            return "\n".join(result_parts)

        except Exception as e:
            print(traceback.print_exc())
            return self._handle_mongo_error(e, "delete_records", table_name)

# Tool interface for smolagents
from smolagents import Tool


class DBTableCodeToolInterface(Tool):
    """
    Smolagents interface for DBTableCodeTool.

    This provides the standard tool interface that agents can use
    to perform table operations with MongoDB backend.
    """

    name = "db_table_code"
    description = """A tool for performing table operations using MongoDB.

    Tables are stored as MongoDB collections and support concurrent writes.

    Available operations:
    - create_table: Create a new table with specified columns
    - add_records: Add records to a table (accepts list of dictionaries)
    - filter_records: Filter and view records from a table, using the **pymono usage**, **default show 100 records**
    - count_records: Count records matching conditions,  **pymono usage**
    - update_record: Update a specific record by ID,  **pymono usage**
    - update_records: Update multiple records matching a filter query,  **pymono usage**
    - delete_records: Delete error records matching a filter query,  **pymono usage**
    - get_table_info: Get basic information about a table

    MongoDB-specific features:
    - Native concurrent write support
    - Fast queries with indexes
    - Atomic operations
    - Flexible schema within column constraints

    Common usage patterns:
    1. create_table(table_name="crocodiles", column_names="species,location,year,source_url")
    2. add_records(table_name="crocodiles", records=[{'species': 'Crocodylus niloticus', 'location': 'Florida', 'year': 2012, 'source_url': 'https://example.com'}])
    3. filter_records(table_name="crocodiles", filter_string='{"location": "Florida"}', limit=100)
    4. count_records(table_name="crocodiles", filter_string='{"species": "Crocodylus niloticus"}')
    5. count_records(table_name="crocodiles", count_non_null=True)
    6. update_record(table_name="crocodiles", record_identifier="1", update_fields="{'location': 'Texas'}")
    7. update_records(table_name="crocodiles", filter_query="{'year': 2012}", update_fields="{'location': 'Texas'}")
    8. delete_records(table_name="crocodiles", filter_string='{"year": {"$lt": 2015}}')  # Delete all records where year < 2015
    """
    inputs = {
        "operation": {"type": "string", "description": "The operation to perform (create_table, add_records, filter_records, count_records, update_record, update_records, delete_records, get_table_info)"},
        "table_name": {"type": "string", "description": "Name of the table", "nullable": True},
        "column_names": {"type": "string", "description": "Column names for create_table, separated by commas (e.g., 'species,location,year,source_url')", "nullable": True},
        "records": {"type": "any", "description": "Records to add - should be a list of dictionaries, e.g., [{'name': 'item1', 'value': 100}, {'name': 'item2', 'value': 200}]", "nullable": True},
        "filter_string": {"type": "string", "description": "MongoDB query as JSON string for filter_records, count_records, and delete_records (e.g., '{\"location\": \"Florida\"}' or '{\"year\": {\"$gte\": 2015}}')", "nullable": True},
        "limit": {"type": "integer", "description": "Maximum number of records to return for filter_records (default: 100)", "nullable": True},
        "count_non_null": {"type": "boolean", "description": "For count_records only - if True, counts only records where all columns are non-null", "nullable": True},
        "record_identifier": {"type": "string", "description": "Value to identify the record for update_record operation", "nullable": True},
        "identifier_column": {"type": "string", "description": "Column name to identify records for update_record (default: 'id')", "nullable": True},
        "filter_query": {"type": "string", "description": "MongoDB-style filter query for update_records (JSON string)", "nullable": True},
        "update_fields": {"type": "string", "description": "Fields to update, format: JSON string or 'field1=new_value1,field2=new_value2'", "nullable": True}
    }
    output_type = "string"

    def __init__(self, connection_string: str = "mongodb://localhost:27017/", database_name: str = "tabular_memory", mode: str = "full", task_id: str = '0', create_table_counter: Optional[GlobalCreateTableCounter] = None):
        """
        初始化 DBTableCodeToolInterface。
        
        Args:
            connection_string: MongoDB 连接字符串
            database_name: 数据库名称
            mode: 权限模式 - "full" 允许所有操作，"readonly" 只允许读操作和添加记录
            task_id: 任务 ID，用于给 collection 名称添加前缀
            create_table_counter: 全局 create_table 计数器，用于限制 create_table 调用次数。
                                  如果提供，则所有共享此计数器的工具实例会共享调用次数限制。
                                  如果为 None，则不限制 create_table 调用次数。
        """
        super().__init__()
        self.db_table_tool = DBTableCodeTool(connection_string, database_name, mode)
        self.task_id = task_id
        self.create_table_counter = create_table_counter
    
    def _get_collection_name(self, table_name: str) -> str:
        """
        获取完整的 collection 名称，格式: {task_id}_{table_name}
        
        Args:
            table_name: 原始表名
            
        Returns:
            完整的 collection 名称
        """
        if not table_name:
            return table_name
        ######## TIAN LAN DEBUG: REMOVE THIS
        # 如果 table_name 已经包含 task_id 前缀，直接返回
        if self.task_id and table_name.startswith(f'{self.task_id}_'):
            return table_name
        # 否则添加 task_id 前缀
        return f'{self.task_id}_{table_name}' if self.task_id else table_name
        #return table_name

    def forward(self, operation: str, table_name: str = None, column_names: str = None, records: Any = None,
                filter_string: str = None, limit: int = 100, count_non_null: bool = False,
                record_identifier: str = None, identifier_column: str = None,
                filter_query: str = None, update_fields: str = None) -> str:
        """
        Execute a table operation.

        Args:
            operation: The operation to perform
            table_name: Name of the table
            column_names: Column names for create_table (comma-separated string)
            records: Records to add - should be a list of dictionaries
            filter_string: MongoDB query as JSON string for filter_records and count_records
            limit: Maximum number of records to return for filter_records (default: 100)
            count_non_null: For count_records only - if True, counts only records where all columns are non-null
            record_identifier: Value to identify record for update_record
            identifier_column: Column name to identify records for update_record
            filter_query: MongoDB filter query for update_records
            update_fields: Fields to update (formatted string or JSON)

        Returns:
            Result of the operation
        """
        try:
            if operation == "create_table":
                # 检查全局 create_table 计数器
                if self.create_table_counter is not None:
                    if not self.create_table_counter.try_increment():
                        return (
                            f"Error: create_table operation is not allowed. "
                            f"The create_table tool can only be called {self.create_table_counter.limit} time(s) during the entire task. "
                            f"You have already called create_table {self.create_table_counter.get_count()} time(s). "
                            f"Please use the existing table instead of creating a new one. "
                            f"If you need to modify the table structure, use update_record or update_records operations."
                        )
                
                # Parse column_names string into list
                columns = [col.strip() for col in column_names.split(',')] if column_names else []
                # create_table 内部会处理 task_id 前缀，所以这里直接传递原始 table_name
                return self.db_table_tool.create_table(table_name, columns, self.task_id)
            elif operation == "add_records":
                # Accept list of dictionaries or JSON string
                if isinstance(records, str):
                    import json
                    try:
                        records = json.loads(records)
                    except json.JSONDecodeError as e:
                        return f"Error: records parameter must be a valid JSON string or list. JSON parse error: {str(e)}"
                
                if isinstance(records, list):
                    # 使用带 task_id 前缀的 collection 名称
                    collection_name = self._get_collection_name(table_name)
                    return self.db_table_tool.add_records(collection_name, records)
                else:
                    return "Error: records parameter must be a list of dictionaries or valid JSON string"
            elif operation == "add_records_batch":
                # Legacy support - redirect to add_records
                if isinstance(records, str):
                    import json
                    try:
                        records = json.loads(records)
                    except json.JSONDecodeError as e:
                        return f"Error: records parameter must be a valid JSON string or list. JSON parse error: {str(e)}"
                
                if isinstance(records, list):
                    # 使用带 task_id 前缀的 collection 名称
                    collection_name = self._get_collection_name(table_name)
                    return self.db_table_tool.add_records(collection_name, records)
                else:
                    return "Error: records parameter must be a list of dictionaries or valid JSON string"
            elif operation == "filter_records":
                # 使用带 task_id 前缀的 collection 名称
                collection_name = self._get_collection_name(table_name)
                return self.db_table_tool.filter_records(collection_name, filter_string, limit)
            
            elif operation == "count_records":
                # 使用带 task_id 前缀的 collection 名称
                collection_name = self._get_collection_name(table_name)
                return self.db_table_tool.count_records(collection_name, filter_string, count_non_null)
            
            elif operation == "update_record":
                # Parse update_fields into a dictionary
                updates = {}
                if update_fields:
                    # If already a dict, use directly
                    if isinstance(update_fields, dict):
                        updates = update_fields
                    else:
                        # Try JSON first
                        try:
                            import json
                            if isinstance(update_fields, str) and update_fields.strip().startswith("{") and update_fields.strip().endswith("}"):
                                parsed = json.loads(update_fields)
                                if isinstance(parsed, dict):
                                    updates = parsed
                        except (json.JSONDecodeError, TypeError):
                            # Fall back to CSV format
                            import csv
                            for field in next(csv.reader([update_fields], skipinitialspace=True)):
                                if "=" not in field:
                                    continue
                                key, value = field.split("=", 1)
                                updates[key.strip()] = value.strip().strip("'\"")
                
                id_col = identifier_column or "id"
                # 使用带 task_id 前缀的 collection 名称
                collection_name = self._get_collection_name(table_name)
                return self.db_table_tool.update_record(collection_name, record_identifier, updates, id_col)
            
            elif operation == "update_records":
                # Parse filter_query and update_fields
                filter_dict = {}
                if filter_query:
                    try:
                        import json
                        filter_dict = json.loads(filter_query)
                        if not isinstance(filter_dict, dict):
                            filter_dict = {}
                    except (json.JSONDecodeError, TypeError):
                        filter_dict = {}
                
                # Parse update_fields
                updates = {}
                if update_fields:
                    # If already a dict, use directly
                    if isinstance(update_fields, dict):
                        updates = update_fields
                    else:
                        # Try JSON first
                        try:
                            import json
                            if isinstance(update_fields, str) and update_fields.strip().startswith("{") and update_fields.strip().endswith("}"):
                                parsed = json.loads(update_fields)
                                if isinstance(parsed, dict):
                                    updates = parsed
                        except (json.JSONDecodeError, TypeError):
                            # Fall back to CSV format
                            import csv
                            for field in next(csv.reader([update_fields], skipinitialspace=True)):
                                if "=" not in field:
                                    continue
                                key, value = field.split("=", 1)
                                updates[key.strip()] = value.strip().strip("'\"")
                
                # 使用带 task_id 前缀的 collection 名称
                collection_name = self._get_collection_name(table_name)
                return self.db_table_tool.update_records(collection_name, filter_dict, updates)
            
            #elif operation == "list_tables":
            #    # list_tables 不需要 table_name，直接返回
            #    return self.db_table_tool.list_tables()
            
            elif operation == "delete_records":
                # 使用带 task_id 前缀的 collection 名称
                collection_name = self._get_collection_name(table_name)
                return self.db_table_tool.delete_records(collection_name, filter_string)
            
            elif operation == "get_table_info":
                # 使用带 task_id 前缀的 collection 名称
                collection_name = self._get_collection_name(table_name)
                return self.db_table_tool.get_table_info(collection_name)
            
            else:
                return f"Unknown operation: {operation}. Available operations: create_table, add_records, filter_records, count_records, update_record, update_records, delete_records, get_table_info"

        except Exception as e:
            print(traceback.print_exc())
            error_msg = f"Error executing {operation} on table '{table_name}': {e}"
            logger.error(error_msg)
            return error_msg


if __name__ == "__main__":
    """
    Test suite for DBTableCodeTool
    Run this to test all operations
    """
    print("=" * 80)
    print("MongoDB Table Code Tool - Test Suite")
    print("=" * 80)
    print()
    
    # Initialize the tool with a test database
    # You should have MongoDB running on localhost:27017
    # Or adjust the connection string
    tool = DBTableCodeToolInterface(
        connection_string="mongodb://localhost:27017/",
        database_name="test_tabular_memory",
        mode="full"
    )
    
    # Clear previous test data if exists
    print("Clearing Previous Test Data")
    print("-" * 80)
    try:
        # Get the MongoDB database
        db = tool.db_table_tool.db
        # List all collections
        collections = db.list_collection_names()
        print(f"Found {len(collections)} collections to clean:")
        for collection_name in collections:
            print(f"  - {collection_name}")
            # Drop the collection
            db[collection_name].drop()
            print(f"    Dropped {collection_name}")
        print("Previous test data cleared successfully.")
    except Exception as e:
        print(f"Warning: Could not clear previous test data: {e}")
    print()

    ipdb.set_trace()
    
    # Test 1: Create table
    print("Test 1: Create Table")
    print("-" * 80)
    result = tool.forward(
        operation="create_table",
        table_name="animals",
        column_names="species,location,year,weight,source_url"
    )
    print(result)
    print()
    
    # Test 2: Add records
    print("Test 2: Add Records")
    print("-" * 80)
    records = [
        {
            "species": "Crocodylus niloticus",
            "location": "Florida",
            "year": 2012,
            "weight": 450.5,
            "source_url": "https://example.com/croc1"
        },
        {
            "species": "Crocodylus acutus",
            "location": "Texas",
            "year": 2015,
            "weight": 380.2,
            "source_url": "https://example.com/croc2"
        },
        {
            "species": "Crocodylus niloticus",
            "location": "Florida",
            "year": 2018,
            "weight": 520.0,
            "source_url": "https://example.com/croc3"
        },
        {
            "species": "Alligator mississippiensis",
            "location": "Louisiana",
            "year": 2020,
            "weight": 600.3,
            "source_url": "https://example.com/alligator1"
        }
    ]
    result = tool.forward(
        operation="add_records",
        table_name="animals",
        records=records
    )
    print(result)
    print()
    
    # Test 3: List tables
    print("Test 3: List Tables")
    print("-" * 80)
    result = tool.forward(operation="list_tables")
    print(result)
    print()
    
    # Test 4: Filter records (simple equality)
    print("Test 4: Filter Records - Simple Equality")
    print("-" * 80)
    result = tool.forward(
        operation="filter_records",
        table_name="animals",
        filter_string='{"location": "Florida"}'
    )
    print(result)
    print()
    
    # Test 5: Filter records (with complex query)
    print("Test 5: Filter Records - Complex Query")
    print("-" * 80)
    result = tool.forward(
        operation="filter_records",
        table_name="animals",
        filter_string='{"species": "Crocodylus niloticus", "year": {"$gte": 2015}}',
        limit=5
    )
    print(result)
    print()
    
    # Test 6: Filter records (with limit)
    print("Test 6: Filter Records - With Custom Limit")
    print("-" * 80)
    result = tool.forward(
        operation="filter_records",
        table_name="animals",
        filter_string='{"location": "Florida"}',
        limit=1
    )
    print(result)
    print()
    
    # Test 7: Filter all records (no filter)
    print("Test 7: Filter Records - Show All (No Filter)")
    print("-" * 80)
    result = tool.forward(
        operation="filter_records",
        table_name="animals",
        filter_string=None,
        limit=5
    )
    print(result)
    print()
    
    # Test 8: Count records
    print("Test 8: Count Records")
    print("-" * 80)
    result = tool.forward(
        operation="count_records",
        table_name="animals"
    )
    print(result)
    result = tool.forward(
        operation="count_records",
        table_name="animals",
        filter_string='{"location": "Florida"}'
    )
    print(result)
    
    # Test count with all fields non-null
    result = tool.forward(
        operation="count_records",
        table_name="animals",
        count_non_null=True
    )
    print(result)
    
    # Test count with both filter and non-null check
    result = tool.forward(
        operation="count_records",
        table_name="animals",
        filter_string='{"year": {"$gte": 2015}}',
        count_non_null=True
    )
    print(result)
    print()
    
    # Test 9: Get table info
    print("Test 9: Get Table Info")
    print("-" * 80)
    result = tool.forward(
        operation="get_table_info",
        table_name="animals"
    )
    print(result)
    print()
    
    # Test 10: Update record
    print("Test 10: Update Record (Single)")
    print("-" * 80)
    # First, let's see what we have
    result = tool.forward(
        operation="filter_records",
        table_name="animals",
        filter_string='{"species": "Crocodylus niloticus", "location": "Florida", "year": 2012}'
    )
    print("Before update:")
    print(result)
    print()
    
    # Update the weight
    result = tool.forward(
        operation="update_record",
        table_name="animals",
        record_identifier="Florida",
        identifier_column="location",
        update_fields='{"weight": 500.0, "year": 2013}'
    )
    print("Update result:")
    print(result)
    print()
    
    # Verify the update
    result = tool.forward(
        operation="filter_records",
        table_name="animals",
        filter_string='{"location": "Florida"}'
    )
    print("After update:")
    print(result)
    print()
    
    # Test 11: Update records (multiple)
    print("Test 11: Update Records (Multiple)")
    print("-" * 80)
    # Update all records with year >= 2018
    result = tool.forward(
        operation="update_records",
        table_name="animals",
        filter_query='{"year": {"$gte": 2018}}',
        update_fields='{"source_verified": true}'
    )
    print("Update result:")
    print(result)
    print()
    
    # Test 12: Create another table to test multiple tables
    print("Test 12: Create Another Table")
    print("-" * 80)
    result = tool.forward(
        operation="create_table",
        table_name="reptiles",
        column_names="name,type,habitat,conservation_status"
    )
    print(result)
    print()
    
    # Add data to the new table
    reptile_records = [
        {
            "name": "Komodo Dragon",
            "type": "Monitor Lizard",
            "habitat": "Indonesia",
            "conservation_status": "Endangered"
        },
        {
            "name": "Green Anaconda",
            "type": "Snake",
            "habitat": "South America",
            "conservation_status": "Least Concern"
        }
    ]
    result = tool.forward(
        operation="add_records",
        table_name="reptiles",
        records=reptile_records
    )
    print(result)
    print()
    
    # Test 13: List all tables
    print("Test 13: List All Tables")
    print("-" * 80)
    result = tool.forward(operation="list_tables")
    print(result)
    print()
    
    # Test 14: Test readonly mode
    print("Test 14: Test Readonly Mode")
    print("-" * 80)
    readonly_tool = DBTableCodeToolInterface(
        connection_string="mongodb://localhost:27017/",
        database_name="test_tabular_memory",
        mode="readonly"
    )
    
    # This should work - read operation
    result = readonly_tool.forward(
        operation="list_tables"
    )
    print("Readonly tool can list tables:")
    print(result)
    print()
    
    # This should work - add records is allowed
    result = readonly_tool.forward(
        operation="add_records",
        table_name="animals",
        records=[{
            "species": "Test Species",
            "location": "Test Location",
            "year": 2024,
            "weight": 100.0,
            "source_url": "https://test.com"
        }]
    )
    print("Readonly tool can add records:")
    print(result)
    print()
    
    # This should fail - create table not allowed
    result = readonly_tool.forward(
        operation="create_table",
        table_name="forbidden",
        column_names="col1,col2"
    )
    print("Readonly tool cannot create tables:")
    print(result)
    print()
    
    # Test 15: Test error handling - table doesn't exist
    print("Test 15: Error Handling - Table Doesn't Exist")
    print("-" * 80)
    result = tool.forward(
        operation="filter_records",
        table_name="nonexistent_table",
        filter_string='{"col": "value"}'
    )
    print(result)
    print()
    
    # Test 16: Test error handling - invalid operation
    print("Test 16: Error Handling - Invalid Operation")
    print("-" * 80)
    result = tool.forward(
        operation="invalid_operation",
        table_name="animals"
    )
    print(result)
    print()
    
    print("=" * 80)
    print("All tests completed!")
    print("=" * 80)
    
    # Cleanup: close the connection
    if hasattr(tool, 'db_table_tool'):
        try:
            # Drop test database
            print("\nCleaning up test database...")
            print("Note: To keep the data, comment out the cleanup section")
            # Uncomment the following line to automatically delete the test database
            # tool.db_table_tool.client.drop_database("test_tabular_memory")
            tool.db_table_tool.client.close()
            print("Test database closed.")
        except Exception as e:
            print(f"Error during cleanup: {e}")
    
    print("\nTest suite finished. Check the results above for any errors.")