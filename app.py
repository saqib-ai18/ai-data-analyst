import os
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")

ALLOWED_OPERATIONS = {
    "describe": ["mean", "median", "std", "min", "max", "count", "sum", "var", "skew", "kurt"],
    "groupby": ["mean", "sum", "count", "min", "max", "std", "median"],
    "filter": ["eq", "ne", "gt", "lt", "ge", "le", "isin", "between"],
    "sort": ["ascending", "descending"],
    "aggregate": ["mean", "sum", "count", "min", "max", "std", "median", "first", "last"],
    "calculate_metrics": ["sum", "mean", "median", "std", "count", "min", "max"],
    "time_series_aggregation": ["sum", "mean", "count", "min", "max", "median", "std", "pct_change"],
    "value_counts": [],
    "unique": [],
    "nunique": [],
    "corr": [],
    "cov": [],
    "head": [],
    "tail": [],
    "sample": [],
    "shape": [],
    "dtypes": [],
    "isnull_sum": [],
    "duplicates_sum": [],
    "data_quality_report": [],
}

CHART_TYPES = ["bar", "line", "scatter", "histogram", "box", "violin", "pie"]


def get_nvidia_client():
    if not NVIDIA_API_KEY:
        return None
    try:
        return OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY)
    except Exception:
        return None


def validate_column(df: pd.DataFrame, column: str) -> bool:
    return column in df.columns


def validate_columns(df: pd.DataFrame, columns: list) -> bool:
    return all(col in df.columns for col in columns)


def safe_execute_operation(df: pd.DataFrame, operation: dict) -> any:
    op_type = operation.get("operation")
    columns = operation.get("columns", [])
    params = operation.get("params", {})

    if not validate_columns(df, columns):
        return {"error": f"Invalid column(s): {columns}"}

    try:
        if op_type == "describe":
            return df[columns].describe(include="all").to_dict()

        elif op_type == "groupby":
            by = columns[0] if columns else None
            agg_cols = columns[1:] if len(columns) > 1 else df.select_dtypes(include="number").columns.tolist()
            agg_func = params.get("agg", "mean")
            if by and agg_cols:
                return df.groupby(by)[agg_cols].agg(agg_func).to_dict()
            return {"error": "groupby requires 'by' column and at least one aggregation column"}

        elif op_type == "filter":
            col = columns[0]
            condition = params.get("condition", "eq")
            value = params.get("value")
            if condition == "eq":
                return df[df[col] == value].to_dict(orient="records")
            elif condition == "ne":
                return df[df[col] != value].to_dict(orient="records")
            elif condition == "gt":
                return df[df[col] > value].to_dict(orient="records")
            elif condition == "lt":
                return df[df[col] < value].to_dict(orient="records")
            elif condition == "ge":
                return df[df[col] >= value].to_dict(orient="records")
            elif condition == "le":
                return df[df[col] <= value].to_dict(orient="records")
            elif condition == "isin":
                return df[df[col].isin(value)].to_dict(orient="records")
            elif condition == "between":
                return df[df[col].between(value[0], value[1])].to_dict(orient="records")
            return {"error": f"Unknown filter condition: {condition}"}

        elif op_type == "sort":
            ascending = params.get("ascending", True)
            return df.sort_values(by=columns, ascending=ascending).head(100).to_dict(orient="records")

        elif op_type == "aggregate":
            agg_func = params.get("agg", "mean")
            return df[columns].agg(agg_func).to_dict()

        elif op_type == "calculate_metrics":
            return calculate_sales_metrics(df, columns, params)

        elif op_type == "time_series_aggregation":
            return calculate_time_series_metrics(df, columns, params)

        elif op_type == "data_quality_report":
            return generate_data_quality_report(df, columns, params)

        elif op_type == "value_counts":
            col = columns[0]
            return df[col].value_counts().head(20).to_dict()

        elif op_type == "unique":
            col = columns[0]
            return df[col].unique().tolist()

        elif op_type == "nunique":
            return {col: df[col].nunique() for col in columns}

        elif op_type == "corr":
            numeric_df = df[columns].select_dtypes(include="number")
            return numeric_df.corr().to_dict()

        elif op_type == "cov":
            numeric_df = df[columns].select_dtypes(include="number")
            return numeric_df.cov().to_dict()

        elif op_type == "head":
            n = params.get("n", 5)
            return df[columns].head(n).to_dict(orient="records")

        elif op_type == "tail":
            n = params.get("n", 5)
            return df[columns].tail(n).to_dict(orient="records")

        elif op_type == "sample":
            n = params.get("n", 5)
            return df[columns].sample(n=min(n, len(df))).to_dict(orient="records")

        elif op_type == "shape":
            return {"rows": df.shape[0], "columns": df.shape[1]}

        elif op_type == "dtypes":
            return df[columns].dtypes.astype(str).to_dict()

        elif op_type == "isnull_sum":
            return df[columns].isnull().sum().to_dict()

        elif op_type == "duplicates_sum":
            return {"duplicate_rows": int(df.duplicated().sum())}

        return {"error": f"Unknown operation: {op_type}"}

    except Exception as e:
        return {"error": str(e)}


def calculate_sales_metrics(df: pd.DataFrame, columns: list, params: dict) -> dict:
    """Calculate specific sales metrics using actual dataset values."""
    metrics = params.get("metrics", [])
    group_by = params.get("group_by")
    
    if not metrics:
        return {"error": "No metrics specified. Use: total_revenue, total_units, avg_order_revenue, num_orders, median_revenue, std_revenue"}
    
    # Identify revenue and quantity columns
    revenue_col = None
    quantity_col = None
    
    for col in columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in ["revenue", "sales", "amount", "price", "total"]):
            if df[col].dtype in ["int64", "float64", "int32", "float32"]:
                revenue_col = col
        if any(kw in col_lower for kw in ["quantity", "units", "qty", "count", "volume"]):
            if df[col].dtype in ["int64", "float64", "int32", "float32"]:
                quantity_col = col
    
    # Fallback: first numeric column as revenue, second as quantity
    numeric_cols = [c for c in columns if df[c].dtype in ["int64", "float64", "int32", "float32"]]
    if not revenue_col and numeric_cols:
        revenue_col = numeric_cols[0]
    if not quantity_col and len(numeric_cols) > 1:
        quantity_col = numeric_cols[1]
    
    results = {}
    
    def compute_grouped(series, metric_name, func):
        if group_by and group_by in df.columns:
            grouped = df.groupby(group_by)[series.name].agg(func)
            results[f"{metric_name}_by_{group_by}"] = grouped.to_dict()
        else:
            results[metric_name] = series.agg(func)
    
    if "total_revenue" in metrics and revenue_col:
        compute_grouped(df[revenue_col], "total_revenue", "sum")
    
    if "total_units" in metrics and quantity_col:
        compute_grouped(df[quantity_col], "total_units", "sum")
    
    if "avg_order_revenue" in metrics and revenue_col:
        compute_grouped(df[revenue_col], "avg_order_revenue", "mean")
    
    if "num_orders" in metrics:
        if group_by and group_by in df.columns:
            grouped = df.groupby(group_by).size()
            results[f"num_orders_by_{group_by}"] = grouped.to_dict()
        else:
            results["num_orders"] = len(df)
    
    if "median_revenue" in metrics and revenue_col:
        compute_grouped(df[revenue_col], "median_revenue", "median")
    
    if "std_revenue" in metrics and revenue_col:
        compute_grouped(df[revenue_col], "std_revenue", "std")
    
    if "min_revenue" in metrics and revenue_col:
        compute_grouped(df[revenue_col], "min_revenue", "min")
    
    if "max_revenue" in metrics and revenue_col:
        compute_grouped(df[revenue_col], "max_revenue", "max")
    
    if not results:
        return {"error": "Could not compute metrics. Check that revenue/quantity columns exist and are numeric."}
    
    return results


def calculate_time_series_metrics(df: pd.DataFrame, columns: list, params: dict) -> dict:
    """Calculate time-series aggregated metrics (monthly, quarterly, yearly) from date column."""
    date_column = params.get("date_column")
    metrics = params.get("metrics", [])
    agg_func = params.get("agg", "sum")
    period = params.get("period", "monthly")  # monthly, quarterly, yearly
    date_range = params.get("date_range")  # optional: {"start": "2026-01", "end": "2026-12"}
    include_pct_change = params.get("include_pct_change", False)
    
    if not date_column:
        return {"error": "date_column parameter is required for time_series_aggregation"}
    
    if not metrics:
        return {"error": "No metrics specified."}
    
    if date_column not in df.columns:
        return {"error": f"Date column '{date_column}' not found in dataset"}
    
    # Validate metric columns exist and are numeric
    for metric_col in metrics:
        if metric_col not in df.columns:
            return {"error": f"Metric column '{metric_col}' not found in dataset"}
        if df[metric_col].dtype not in ["int64", "float64", "int32", "float32"]:
            return {"error": f"Metric column '{metric_col}' must be numeric"}
    
    # Validate period
    valid_periods = ["monthly", "quarterly", "yearly"]
    if period not in valid_periods:
        return {"error": f"Invalid period '{period}'. Must be one of: {valid_periods}"}
    
    # Validate aggregation function
    valid_aggs = ["sum", "mean", "count", "min", "max", "median", "std", "pct_change"]
    if agg_func not in valid_aggs:
        return {"error": f"Unsupported aggregation function: {agg_func}. Must be one of: {valid_aggs}"}
    
    try:
        df_work = df.copy()
        
        # Convert date column to datetime, coerce errors to NaT
        df_work[date_column] = pd.to_datetime(df_work[date_column], errors="coerce")
        
        # Check for invalid dates
        invalid_dates = df_work[date_column].isna().sum()
        if invalid_dates > 0:
            return {"error": f"Found {invalid_dates} invalid/missing dates in '{date_column}'. Cannot perform time series aggregation."}
        
        # Apply date range filter if provided
        if date_range:
            start = date_range.get("start")
            end = date_range.get("end")
            if start:
                start_dt = pd.to_datetime(start)
                df_work = df_work[df_work[date_column] >= start_dt]
            if end:
                end_dt = pd.to_datetime(end)
                df_work = df_work[df_work[date_column] <= end_dt]
            if len(df_work) == 0:
                return {"error": "No data remaining after applying date range filter."}
        
        # Create period columns based on aggregation level
        if period == "monthly":
            df_work["Period"] = df_work[date_column].dt.to_period("M").astype(str)
            df_work["Period_dt"] = df_work[date_column].dt.to_period("M").dt.to_timestamp()
            period_label = "Month"
        elif period == "quarterly":
            df_work["Period"] = df_work[date_column].dt.to_period("Q").astype(str)
            df_work["Period_dt"] = df_work[date_column].dt.to_period("Q").dt.to_timestamp()
            period_label = "Quarter"
        else:  # yearly
            df_work["Period"] = df_work[date_column].dt.to_period("Y").astype(str)
            df_work["Period_dt"] = df_work[date_column].dt.to_period("Y").dt.to_timestamp()
            period_label = "Year"
        
        results = {"period": period, "period_label": period_label}
        
        for metric_col in metrics:
            if agg_func == "pct_change":
                # For percentage change, we need to compute sum first then pct_change
                grouped = df_work.groupby(["Period", "Period_dt"])[metric_col].sum().reset_index()
                grouped = grouped.sort_values("Period_dt")
                grouped["pct_change"] = grouped[metric_col].pct_change() * 100
                # Replace first NaN with 0 or None
                grouped["pct_change"] = grouped["pct_change"].fillna(0)
                
                results[f"{metric_col}_by_{period}"] = grouped[["Period", "pct_change"]].to_dict(orient="records")
                results[f"{metric_col}_by_{period}_values"] = grouped["pct_change"].tolist()
                results[f"{metric_col}_by_{period}_labels"] = grouped["Period"].tolist()
                results[f"{metric_col}_base_values"] = grouped[metric_col].tolist()  # base values for reference
            else:
                if agg_func == "sum":
                    grouped = df_work.groupby(["Period", "Period_dt"])[metric_col].sum().reset_index()
                elif agg_func == "mean":
                    grouped = df_work.groupby(["Period", "Period_dt"])[metric_col].mean().reset_index()
                elif agg_func == "count":
                    grouped = df_work.groupby(["Period", "Period_dt"])[metric_col].count().reset_index()
                elif agg_func == "min":
                    grouped = df_work.groupby(["Period", "Period_dt"])[metric_col].min().reset_index()
                elif agg_func == "max":
                    grouped = df_work.groupby(["Period", "Period_dt"])[metric_col].max().reset_index()
                elif agg_func == "median":
                    grouped = df_work.groupby(["Period", "Period_dt"])[metric_col].median().reset_index()
                elif agg_func == "std":
                    grouped = df_work.groupby(["Period", "Period_dt"])[metric_col].std().reset_index()
                
                # Sort by datetime for chronological order
                grouped = grouped.sort_values("Period_dt")
                
                # Store results
                results[f"{metric_col}_by_{period}"] = grouped[["Period", metric_col]].to_dict(orient="records")
                results[f"{metric_col}_by_{period}_values"] = grouped[metric_col].tolist()
                results[f"{metric_col}_by_{period}_labels"] = grouped["Period"].tolist()
            
            # Optionally compute percentage change
            if include_pct_change and agg_func != "pct_change":
                grouped_sorted = grouped.sort_values("Period_dt")
                pct_changes = grouped_sorted[metric_col].pct_change() * 100
                pct_changes = pct_changes.fillna(0)
                results[f"{metric_col}_pct_change"] = pct_changes.tolist()
        
        # Overall summary
        results["period_count"] = len(grouped)
        results["date_range"] = {
            "start": grouped["Period"].iloc[0],
            "end": grouped["Period"].iloc[-1]
        }
        
        return results
    
    except Exception as e:
        return {"error": f"Time series aggregation failed: {str(e)}"}


def generate_data_quality_report(df: pd.DataFrame, columns: list, params: dict) -> dict:
    """Generate a comprehensive data quality report for the dataset."""
    try:
        # If no specific columns requested, analyze all columns
        if not columns:
            columns = df.columns.tolist()
        
        # Validate columns exist
        invalid_cols = [c for c in columns if c not in df.columns]
        if invalid_cols:
            return {"error": f"Invalid column(s): {invalid_cols}"}
        
        report = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "columns_analyzed": columns,
            "column_reports": {}
        }
        
        for col in columns:
            col_data = df[col]
            col_report = {
                "dtype": str(col_data.dtype),
                "non_null_count": int(col_data.notna().sum()),
                "null_count": int(col_data.isna().sum()),
                "null_percentage": round(col_data.isna().sum() / len(df) * 100, 2),
                "unique_count": int(col_data.nunique()),
            }
            
            # Numeric column analysis
            if col_data.dtype in ["int64", "float64", "int32", "float32"]:
                col_report.update({
                    "min": float(col_data.min()) if not col_data.isna().all() else None,
                    "max": float(col_data.max()) if not col_data.isna().all() else None,
                    "mean": float(col_data.mean()) if not col_data.isna().all() else None,
                    "median": float(col_data.median()) if not col_data.isna().all() else None,
                    "std": float(col_data.std()) if not col_data.isna().all() else None,
                    "zeros_count": int((col_data == 0).sum()),
                    "negative_count": int((col_data < 0).sum()),
                })
            
            # String/object column analysis
            elif col_data.dtype == "object":
                # Check if it might be a date column
                sample_vals = col_data.dropna().head(100)
                if len(sample_vals) > 0:
                    # Try to parse as dates
                    try:
                        parsed = pd.to_datetime(sample_vals, errors="coerce")
                        date_parse_rate = parsed.notna().sum() / len(sample_vals)
                        if date_parse_rate > 0.8:
                            col_report["likely_date_column"] = True
                            col_report["date_parse_rate"] = round(date_parse_rate, 2)
                    except Exception:
                        pass
                
                # Top values
                value_counts = col_data.value_counts().head(10)
                col_report["top_values"] = value_counts.to_dict()
                col_report["sample_values"] = sample_vals.tolist()[:5]
            
            # Check for potential issues
            issues = []
            if col_report["null_percentage"] > 50:
                issues.append(f"High null rate: {col_report['null_percentage']}%")
            if col_report["unique_count"] == 1:
                issues.append("Constant column (only one unique value)")
            if col_data.dtype in ["int64", "float64", "int32", "float32"]:
                if col_report.get("negative_count", 0) > 0 and "revenue" in col.lower():
                    issues.append("Negative values in revenue column")
            
            col_report["issues"] = issues
            report["column_reports"][col] = col_report
        
        # Dataset-level checks
        duplicate_rows = int(df.duplicated().sum())
        report["duplicate_rows"] = duplicate_rows
        report["duplicate_percentage"] = round(duplicate_rows / len(df) * 100, 2) if len(df) > 0 else 0
        
        # Memory usage
        report["memory_usage_kb"] = round(df.memory_usage(deep=True).sum() / 1024, 2)
        
        return report
    
    except Exception as e:
        return {"error": f"Data quality report generation failed: {str(e)}"}


def create_chart_from_time_series_data(metrics: dict, chart_config: dict) -> go.Figure:
    """Create a chart from time series aggregation results (monthly, quarterly, yearly)."""
    chart_type = chart_config.get("type", "line")
    x_col = chart_config.get("x")
    y_col = chart_config.get("y")
    title = chart_config.get("title", "")
    
    # Get period from metrics
    period = metrics.get("period", "monthly")
    period_label = metrics.get("period_label", "Month")
    
    # For time series data, x should be "Period", "Year_Month", "Month", "Quarter", or "Year"
    valid_x_cols = ["Period", "Year_Month", "Month", "Quarter", "Year"]
    if x_col not in valid_x_cols:
        return None
    
    # Find the metric data - try period-specific key first
    metric_key = f"{y_col}_by_{period}"
    if metric_key not in metrics:
        # Try alternative format with _values suffix
        metric_key = f"{y_col}_by_{period}_values"
        labels_key = f"{y_col}_by_{period}_labels"
        if metric_key not in metrics or labels_key not in metrics:
            return None
        values = metrics[metric_key]
        labels = metrics[labels_key]
    else:
        # Extract from the records format
        records = metrics[metric_key]
        if not records:
            return None
        labels = [r.get("Period", r.get("Year_Month", "")) for r in records]
        values = [r.get(y_col, 0) for r in records]
    
    if not labels or not values:
        return None
    
    # Create a dataframe for plotting
    chart_df = pd.DataFrame({"Period": labels, y_col: values})
    
    try:
        if chart_type == "line":
            fig = px.line(chart_df, x="Period", y=y_col, title=title, markers=True)
        elif chart_type == "bar":
            fig = px.bar(chart_df, x="Period", y=y_col, title=title)
        else:
            fig = px.line(chart_df, x="Period", y=y_col, title=title, markers=True)
        
        fig.update_layout(template="plotly_white", height=500)
        fig.update_xaxes(title_text=period_label)
        fig.update_yaxes(title_text=y_col.replace("_", " ").title())
        return fig
    except Exception:
        return None


def create_chart_from_monthly_data(metrics: dict, chart_config: dict) -> go.Figure:
    """Legacy function for backward compatibility - delegates to new function."""
    # Add period info for backward compatibility
    if "period" not in metrics:
        metrics["period"] = "monthly"
        metrics["period_label"] = "Month"
    return create_chart_from_time_series_data(metrics, chart_config)


def create_chart(df: pd.DataFrame, chart_config: dict, operation_results: list = None) -> go.Figure:
    chart_type = chart_config.get("type", "bar")
    x_col = chart_config.get("x")
    y_col = chart_config.get("y")
    color_col = chart_config.get("color")
    title = chart_config.get("title", "")

    # Check if this is a time series aggregation chart
    if operation_results:
        for op_result in operation_results:
            op = op_result.get("operation", {})
            if op.get("operation") == "time_series_aggregation":
                result = op_result.get("result", {})
                if "error" not in result:
                    valid_x_cols = ["Period", "Year_Month", "Month", "Quarter", "Year"]
                    if x_col in valid_x_cols:
                        fig = create_chart_from_time_series_data(result, chart_config)
                        if fig:
                            return fig

    if not validate_column(df, x_col):
        return None

    if y_col and not validate_column(df, y_col):
        return None

    if color_col and not validate_column(df, color_col):
        color_col = None

    try:
        if chart_type == "bar":
            fig = px.bar(df, x=x_col, y=y_col, color=color_col, title=title)
        elif chart_type == "line":
            fig = px.line(df, x=x_col, y=y_col, color=color_col, title=title)
        elif chart_type == "scatter":
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=title)
        elif chart_type == "histogram":
            fig = px.histogram(df, x=x_col, color=color_col, title=title)
        elif chart_type == "box":
            fig = px.box(df, x=x_col, y=y_col, color=color_col, title=title)
        elif chart_type == "violin":
            fig = px.violin(df, x=x_col, y=y_col, color=color_col, title=title)
        elif chart_type == "pie":
            fig = px.pie(df, names=x_col, values=y_col, title=title)
        else:
            return None

        fig.update_layout(template="plotly_white", height=500)
        return fig
    except Exception:
        return None


def get_system_prompt(df: pd.DataFrame) -> str:
    columns_info = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        null_count = int(df[col].isnull().sum())
        unique_count = int(df[col].nunique())
        sample_values = df[col].dropna().head(3).tolist()
        columns_info.append(f"- {col} ({dtype}): nulls={null_count}, unique={unique_count}, samples={sample_values}")

    columns_str = "\n".join(columns_info)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

    return f"""You are an AI Data Analyst. You help users analyze their dataset by generating structured JSON operations.

DATASET INFO:
- Rows: {len(df)}
- Columns: {len(df.columns)}
- Column Details:
{columns_str}
- Numeric Columns: {numeric_cols}
- Categorical Columns: {categorical_cols}
- Datetime Columns: {datetime_cols}

AVAILABLE OPERATIONS (respond ONLY with valid JSON):
{json.dumps(ALLOWED_OPERATIONS, indent=2)}

CHART TYPES: {CHART_TYPES}

SPECIAL OPERATION: calculate_metrics
- Use for sales/business metrics: total_revenue, total_units, avg_order_revenue, num_orders, median_revenue, std_revenue, min_revenue, max_revenue
- Provide "columns" list with relevant numeric columns (revenue, quantity, etc.)
- Provide "params" with "metrics" list and optional "group_by" column
- Example: {{"operation": "calculate_metrics", "columns": ["revenue", "quantity"], "params": {{"metrics": ["total_revenue", "total_units", "avg_order_revenue", "num_orders"], "group_by": "region"}}}}

SPECIAL OPERATION: time_series_aggregation
- Use for date-based time series analysis (monthly, quarterly, yearly revenue, units, orders, etc.)
- Provide "columns" list with the DATE column name (e.g., ["Order_Date"])
- Provide "params" with:
  - "date_column": the date column name (required)
  - "metrics": list of numeric columns to aggregate (e.g., ["Revenue_INR", "Units_Sold"])
  - "period": time period - "monthly", "quarterly", "yearly" (default: "monthly")
  - "agg": aggregation function - "sum", "mean", "count", "min", "max", "median", "std", "pct_change" (default: "sum")
  - "date_range": optional date range filter {{"start": "2026-01", "end": "2026-12"}}
  - "include_pct_change": optional boolean to include month-over-month % change (default: false)
- The operation automatically parses dates, extracts period, groups chronologically
- Example: {{"operation": "time_series_aggregation", "columns": ["Order_Date"], "params": {{"date_column": "Order_Date", "metrics": ["Revenue_INR", "Units_Sold"], "period": "monthly", "agg": "sum"}}}}

SPECIAL OPERATION: data_quality_report
- Use to generate a comprehensive data quality report for the dataset
- Provide "columns" list with column names to analyze (optional, defaults to all columns)
- Provide "params" as empty object {{}}
- Returns: null counts, unique counts, numeric stats, date detection, issues, duplicates, memory usage
- Example: {{"operation": "data_quality_report", "columns": ["Order_Date", "Revenue_INR", "Region"], "params": {{}}}}

RESPONSE FORMAT (JSON only):
{{
  "thought": "Your reasoning about what the user wants",
  "operations": [
    {{"operation": "calculate_metrics", "columns": ["revenue_col", "quantity_col"], "params": {{"metrics": ["total_revenue", "total_units", "avg_order_revenue", "num_orders"], "group_by": "optional_category_col"}}}},
    {{"operation": "time_series_aggregation", "columns": ["Order_Date"], "params": {{"date_column": "Order_Date", "metrics": ["Revenue_INR"], "period": "monthly", "agg": "sum"}}}},
    {{"operation": "groupby", "columns": ["category_col", "numeric_col"], "params": {{"agg": "sum"}}}}
  ],
  "chart": {{"type": "line", "x": "Period", "y": "Revenue_INR", "title": "Monthly Revenue Trend"}},
  "answer": "Natural language summary of findings. Reference the EXACT calculated values from the operations. Never invent or approximate numbers."
}}

RULES:
1. Only use operations from ALLOWED_OPERATIONS
2. Only reference columns that exist in the dataset
3. Never invent data or statistics - always use calculate_metrics, time_series_aggregation, or data_quality_report for exact values
4. For sales questions, USE calculate_metrics with appropriate metrics list
5. For time-series questions, USE time_series_aggregation with date_column, metrics, and period
6. For monthly trends, use period="monthly" and x="Period" in chart
7. For quarterly trends, use period="quarterly" and x="Period" in chart
8. For yearly trends, use period="yearly" and x="Period" in chart
9. For month-over-month change, use agg="pct_change" or include_pct_change=true
10. For date range filtering, include date_range in params
11. For data quality questions, USE data_quality_report
12. If user asks for total, use "sum" metric; for average, use "mean"; for count, use "count"
13. If user asks for grouped metrics, include "group_by" in params
14. In your "answer", reference the EXACT numbers returned by operations
15. If user asks for unsupported operation, explain in "answer" and return empty operations
16. For charts: use "line" for time trends (x=Period, y=metric), "bar" for categorical
17. Return ONLY valid JSON, no markdown, no extra text"""


def query_nvidia(prompt: str, df: pd.DataFrame) -> dict:
    client = get_nvidia_client()
    if not client:
        return {"error": "NVIDIA API key not configured. Set NVIDIA_API_KEY in .env file."}

    try:
        system_prompt = get_system_prompt(df)
        response = client.chat.completions.create(
            model=NVIDIA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=2000
        )
        content = response.choices[0].message.content.strip()
        return json.loads(content)
    except json.JSONDecodeError as e:
        return {"error": f"Failed to parse AI response: {str(e)}"}
    except Exception as e:
        return {"error": f"API error: {str(e)}"}


def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith((".xlsx", ".xls")):
            return pd.read_excel(uploaded_file)
        return None
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None


def display_dataset_info(df: pd.DataFrame):
    st.subheader("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", f"{len(df):,}")
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        st.metric("Missing Values", f"{df.isnull().sum().sum():,}")

    st.subheader("Column Information")
    info_df = pd.DataFrame({
        "Column": df.columns,
        "Type": [str(df[c].dtype) for c in df.columns],
        "Non-Null Count": [df[c].notna().sum() for c in df.columns],
        "Null Count": [df[c].isna().sum() for c in df.columns],
        "Unique Values": [df[c].nunique() for c in df.columns]
    })
    st.dataframe(info_df, use_container_width=True, hide_index=True)

    st.subheader("Preview (First 5 Rows)")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("Duplicate Rows")
    dup_count = int(df.duplicated().sum())
    st.metric("Duplicate Rows", dup_count)
    if dup_count > 0:
        st.dataframe(df[df.duplicated()].head(), use_container_width=True)


def main():
    st.title("📊 AI Data Analyst Chatbot")
    st.caption("Upload a dataset and ask questions in natural language")

    with st.sidebar:
        st.header("📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file",
            type=["csv", "xlsx", "xls"],
            help="Upload your dataset to begin analysis"
        )

        if uploaded_file:
            df = load_data(uploaded_file)
            if df is not None:
                st.session_state["df"] = df
                st.session_state["filename"] = uploaded_file.name
                st.success(f"Loaded: {uploaded_file.name}")
            else:
                st.error("Failed to load file")

        if "df" in st.session_state:
            st.divider()
            st.header("📋 Dataset Info")
            df = st.session_state["df"]
            st.write(f"**File:** {st.session_state.get('filename', 'Unknown')}")
            st.write(f"**Rows:** {len(df):,}")
            st.write(f"**Columns:** {len(df.columns)}")
            st.write(f"**Memory:** {df.memory_usage(deep=True).sum() / 1024:.1f} KB")

            with st.expander("Column Types"):
                for col in df.columns:
                    st.write(f"• {col}: {df[col].dtype}")

            if st.button("Clear Dataset", type="secondary"):
                for key in ["df", "filename", "messages"]:
                    st.session_state.pop(key, None)
                st.rerun()

    if "df" not in st.session_state:
        st.info("👈 Please upload a CSV or Excel file from the sidebar to get started")
        st.markdown("""
        ### Features
        - **Data Upload**: Support for CSV and Excel files
        - **Dataset Profiling**: View rows, columns, types, missing values, duplicates
        - **AI-Powered Analysis**: Ask questions in natural language
        - **Interactive Visualizations**: Bar, line, scatter, histogram, box, violin, pie charts
        - **Secure**: API key from environment variable, no code execution

        ### Example Questions
        - "What is the total revenue and total units sold?"
        - "Show me average order revenue and number of orders"
        - "Calculate sales metrics by region"
        - "What is the median and standard deviation of revenue?"
        - "Calculate total revenue for each month"
        - "Show monthly revenue in a line chart"
        - "Which month had the highest revenue?"
        - "Show monthly units sold"
        - "Compare monthly revenue across months"
        - "Are there any missing values in the dataset?"
        - "Create a bar chart of sales by region"
        """)
        return

    df = st.session_state["df"]

    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 Dataset", "📈 Visualize"])

    with tab1:
        if "messages" not in st.session_state:
            st.session_state["messages"] = []

        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "data" in msg:
                    if isinstance(msg["data"], pd.DataFrame):
                        st.dataframe(msg["data"], use_container_width=True)
                    elif isinstance(msg["data"], dict):
                        st.json(msg["data"])
                    elif isinstance(msg["data"], list):
                        st.dataframe(pd.DataFrame(msg["data"]), use_container_width=True)
                if "chart" in msg and msg["chart"] is not None:
                    st.plotly_chart(msg["chart"], use_container_width=True)

        user_question = st.chat_input("Ask a question about your data...")

        if user_question:
            st.session_state["messages"].append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    result = query_nvidia(user_question, df)

                if "error" in result:
                    st.error(result["error"])
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": f"Error: {result['error']}"
                    })
                else:
                    thought = result.get("thought", "")
                    operations = result.get("operations", [])
                    chart_config = result.get("chart")
                    answer = result.get("answer", "")

                    if thought:
                        st.caption(f"🤔 {thought}")

                    operation_results = []
                    for op in operations:
                        op_result = safe_execute_operation(df, op)
                        operation_results.append({"operation": op, "result": op_result})
                        if "error" not in op_result:
                            if op.get("operation") == "calculate_metrics":
                                display_metrics_table(op_result)
                            elif op.get("operation") == "time_series_aggregation":
                                display_time_series_table(op_result)
                            elif op.get("operation") == "data_quality_report":
                                display_data_quality_report(op_result)
                            elif isinstance(op_result, dict) and len(op_result) < 50:
                                st.json(op_result)
                            elif isinstance(op_result, list) and len(op_result) > 0:
                                st.dataframe(pd.DataFrame(op_result), use_container_width=True)

                    chart_fig = None
                    if chart_config:
                        chart_fig = create_chart(df, chart_config, operation_results)
                        if chart_fig:
                            st.plotly_chart(chart_fig, use_container_width=True)
                        else:
                            st.warning("Could not create chart with specified configuration")

                    if answer:
                        st.markdown(answer)

                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": answer or "Analysis complete",
                        "data": operation_results,
                        "chart": chart_fig
                    })

    with tab2:
        display_dataset_info(df)

    with tab3:
        st.subheader("Create Custom Visualization")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            chart_type = st.selectbox("Chart Type", CHART_TYPES)
        with col2:
            x_col = st.selectbox("X Axis", df.columns)
        with col3:
            y_options = [None] + df.select_dtypes(include="number").columns.tolist()
            y_col = st.selectbox("Y Axis (optional)", y_options)
        with col4:
            color_options = [None] + df.columns.tolist()
            color_col = st.selectbox("Color By (optional)", color_options)

        if st.button("Generate Chart", type="primary"):
            chart_config = {
                "type": chart_type,
                "x": x_col,
                "y": y_col,
                "color": color_col,
                "title": f"{chart_type.title()}: {x_col}" + (f" vs {y_col}" if y_col else "")
            }
            fig = create_chart(df, chart_config)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Could not generate chart. Check column types and selections.")


def display_metrics_table(metrics: dict):
    """Display calculated metrics in a clean Streamlit table."""
    rows = []
    for key, value in metrics.items():
        if isinstance(value, dict):
            for group, val in value.items():
                if isinstance(val, (int, np.integer)):
                    formatted_val = f"{int(val):,}"
                elif isinstance(val, (float, np.floating)):
                    formatted_val = f"{val:,.2f}"
                else:
                    formatted_val = str(val)
                rows.append({"Metric": key, "Group": str(group), "Value": formatted_val})
        else:
            if isinstance(value, (int, np.integer)):
                formatted_val = f"{int(value):,}"
            elif isinstance(value, (float, np.floating)):
                formatted_val = f"{value:,.2f}"
            else:
                formatted_val = str(value)
            rows.append({"Metric": key, "Group": "Overall", "Value": formatted_val})
    
    if rows:
        df_metrics = pd.DataFrame(rows)
        st.dataframe(df_metrics, use_container_width=True, hide_index=True)
        
        # Export button
        csv = df_metrics.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name="metrics_results.csv",
            mime="text/csv",
            key=f"export_metrics_{hash(str(metrics))}"
        )


def display_monthly_table(metrics: dict):
    """Display monthly aggregation results in a clean Streamlit table."""
    rows = []
    processed_metrics = set()
    
    for key, value in metrics.items():
        if key in ["month_count", "date_range"]:
            continue  # Skip summary fields, handle separately
        
        if isinstance(value, list) and len(value) > 0:
            # This is the detailed monthly data (e.g., Revenue_INR_by_month)
            if key.endswith("_by_month"):
                metric_name = key.replace("_by_month", "")
                if metric_name in processed_metrics:
                    continue
                processed_metrics.add(metric_name)
                for record in value:
                    if isinstance(record, dict):
                        month = record.get("Year_Month", "Unknown")
                        val = record.get(metric_name, 0)
                        if isinstance(val, (int, np.integer)):
                            formatted_val = f"{int(val):,}"
                        elif isinstance(val, (float, np.floating)):
                            formatted_val = f"{val:,.2f}"
                        else:
                            formatted_val = str(val)
                        rows.append({"Metric": metric_name, "Month": month, "Value": formatted_val})
            elif key.endswith("_by_month_values"):
                # Alternative format with separate labels/values arrays
                metric_name = key.replace("_by_month_values", "")
                if metric_name in processed_metrics:
                    continue
                processed_metrics.add(metric_name)
                labels_key = f"{metric_name}_by_month_labels"
                if labels_key in metrics:
                    labels = metrics[labels_key]
                    values = value
                    for month, val in zip(labels, values):
                        if isinstance(val, (int, np.integer)):
                            formatted_val = f"{int(val):,}"
                        elif isinstance(val, (float, np.floating)):
                            formatted_val = f"{val:,.2f}"
                        else:
                            formatted_val = str(val)
                        rows.append({"Metric": metric_name, "Month": month, "Value": formatted_val})
    
    if rows:
        df_monthly = pd.DataFrame(rows)
        st.dataframe(df_monthly, use_container_width=True, hide_index=True)
        
        # Export button
        csv = df_monthly.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name="monthly_results.csv",
            mime="text/csv",
            key=f"export_monthly_{hash(str(metrics))}"
        )
    
    # Display summary info
    if "month_count" in metrics:
        st.caption(f"📅 **Months analyzed:** {metrics['month_count']}")
    if "date_range" in metrics:
        dr = metrics["date_range"]
        st.caption(f"📅 **Date range:** {dr.get('start', 'N/A')} to {dr.get('end', 'N/A')}")


def display_time_series_table(metrics: dict):
    """Display time series aggregation results in a clean Streamlit table."""
    period = metrics.get("period", "monthly")
    period_label = metrics.get("period_label", "Month")
    rows = []
    processed_metrics = set()
    
    for key, value in metrics.items():
        if key in ["period", "period_label", "period_count", "date_range"]:
            continue  # Skip summary fields, handle separately
        
        if isinstance(value, list) and len(value) > 0:
            # This is the detailed time series data (e.g., Revenue_INR_by_monthly)
            if key.endswith(f"_by_{period}"):
                metric_name = key.replace(f"_by_{period}", "")
                if metric_name in processed_metrics:
                    continue
                processed_metrics.add(metric_name)
                for record in value:
                    if isinstance(record, dict):
                        period_val = record.get("Period", record.get("Year_Month", "Unknown"))
                        val = record.get(metric_name, 0)
                        if isinstance(val, (int, np.integer)):
                            formatted_val = f"{int(val):,}"
                        elif isinstance(val, (float, np.floating)):
                            formatted_val = f"{val:,.2f}"
                        else:
                            formatted_val = str(val)
                        rows.append({"Metric": metric_name, period_label: period_val, "Value": formatted_val})
            elif key.endswith(f"_by_{period}_values"):
                # Alternative format with separate labels/values arrays
                metric_name = key.replace(f"_by_{period}_values", "")
                if metric_name in processed_metrics:
                    continue
                processed_metrics.add(metric_name)
                labels_key = f"{metric_name}_by_{period}_labels"
                if labels_key in metrics:
                    labels = metrics[labels_key]
                    values = value
                    for period_val, val in zip(labels, values):
                        if isinstance(val, (int, np.integer)):
                            formatted_val = f"{int(val):,}"
                        elif isinstance(val, (float, np.floating)):
                            formatted_val = f"{val:,.2f}"
                        else:
                            formatted_val = str(val)
                        rows.append({"Metric": metric_name, period_label: period_val, "Value": formatted_val})
            
            # Display percentage change if available
            pct_change_key = f"{metric_name}_pct_change"
            if pct_change_key in metrics and metric_name in processed_metrics:
                # We already processed this metric, add pct change info
                pass
    
    if rows:
        df_ts = pd.DataFrame(rows)
        st.dataframe(df_ts, use_container_width=True, hide_index=True)
        
        # Export button
        csv = df_ts.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"{period}_results.csv",
            mime="text/csv",
            key=f"export_ts_{hash(str(metrics))}"
        )
    
    # Display percentage change if available
    for key, value in metrics.items():
        if key.endswith("_pct_change") and isinstance(value, list):
            metric_name = key.replace("_pct_change", "")
            labels_key = f"{metric_name}_by_{period}_labels"
            if labels_key in metrics:
                labels = metrics[labels_key]
                pct_data = []
                for period_val, pct in zip(labels, value):
                    pct_data.append({period_label: period_val, f"{metric_name} % Change": f"{pct:.2f}%"})
                if pct_data:
                    st.caption(f"**{metric_name} Period-over-Period Change:**")
                    pct_df = pd.DataFrame(pct_data)
                    st.dataframe(pct_df, use_container_width=True, hide_index=True)
                    
                    # Export button for percentage change
                    csv = pct_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download % Change as CSV",
                        data=csv,
                        file_name=f"{metric_name}_pct_change.csv",
                        mime="text/csv",
                        key=f"export_pct_{hash(str(metrics))}_{metric_name}"
                    )
    
    # Display summary info
    if "period_count" in metrics:
        st.caption(f"📅 **{period_label}s analyzed:** {metrics['period_count']}")
    if "date_range" in metrics:
        dr = metrics["date_range"]
        st.caption(f"📅 **Date range:** {dr.get('start', 'N/A')} to {dr.get('end', 'N/A')}")


def display_data_quality_report(report: dict):
    """Display data quality report in a clean Streamlit format."""
    if "error" in report:
        st.error(report["error"])
        return
    
    st.subheader("📋 Data Quality Report")
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", f"{report.get('total_rows', 0):,}")
    with col2:
        st.metric("Total Columns", report.get('total_columns', 0))
    with col3:
        st.metric("Duplicate Rows", f"{report.get('duplicate_rows', 0):,} ({report.get('duplicate_percentage', 0)}%)")
    with col4:
        st.metric("Memory Usage", f"{report.get('memory_usage_kb', 0):.1f} KB")
    
    # Column-level details
    col_reports = report.get("column_reports", {})
    if col_reports:
        st.markdown("### Column Analysis")
        
        for col_name, col_report in col_reports.items():
            with st.expander(f"**{col_name}** ({col_report.get('dtype', 'unknown')})"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Non-Null", f"{col_report.get('non_null_count', 0):,}")
                    st.metric("Null Count", f"{col_report.get('null_count', 0):,}")
                    st.metric("Null %", f"{col_report.get('null_percentage', 0)}%")
                with c2:
                    st.metric("Unique Values", f"{col_report.get('unique_count', 0):,}")
                    if "min" in col_report and col_report["min"] is not None:
                        st.metric("Min", f"{col_report['min']:,.2f}")
                        st.metric("Max", f"{col_report['max']:,.2f}")
                        st.metric("Mean", f"{col_report['mean']:,.2f}")
                        st.metric("Median", f"{col_report['median']:,.2f}")
                        st.metric("Std Dev", f"{col_report['std']:,.2f}")
                
                # Issues
                issues = col_report.get("issues", [])
                if issues:
                    st.warning("⚠️ **Issues Detected:**")
                    for issue in issues:
                        st.write(f"• {issue}")
                
                # Likely date column
                if col_report.get("likely_date_column"):
                    st.info(f"📅 Likely date column (parse rate: {col_report.get('date_parse_rate', 0)*100:.0f}%)")
                
                # Top values for categorical
                if "top_values" in col_report:
                    st.write("**Top Values:**")
                    top_vals_df = pd.DataFrame(list(col_report["top_values"].items()), columns=["Value", "Count"])
                    st.dataframe(top_vals_df, use_container_width=True, hide_index=True)
                    
                    # Export button for top values
                    csv = top_vals_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label=f"📥 Download {col_name} Top Values as CSV",
                        data=csv,
                        file_name=f"{col_name}_top_values.csv",
                        mime="text/csv",
                        key=f"export_top_vals_{col_name}_{hash(str(report))}"
                    )
        
        # Export full data quality report
        import json
        report_json = json.dumps(report, indent=2, default=str)
        st.download_button(
            label="📥 Download Full Data Quality Report (JSON)",
            data=report_json,
            file_name="data_quality_report.json",
            mime="application/json",
            key=f"export_dq_full_{hash(str(report))}"
        )


if __name__ == "__main__":
    main()