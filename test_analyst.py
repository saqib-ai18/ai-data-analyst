import pandas as pd
import numpy as np
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import (
    calculate_sales_metrics,
    calculate_time_series_metrics,
    safe_execute_operation,
    validate_column,
    validate_columns,
    ALLOWED_OPERATIONS,
)

def create_test_dataset():
    """Create a test dataset with known values for testing."""
    data = {
        'Order_ID': ['ORD001', 'ORD002', 'ORD003', 'ORD004', 'ORD005', 'ORD006'],
        'Order_Date': ['2026-01-01', '2026-01-15', '2026-02-01', '2026-02-15', '2026-03-01', '2026-03-15'],
        'Customer_ID': ['C001', 'C002', 'C003', 'C004', 'C005', 'C006'],
        'Region': ['North', 'South', 'North', 'East', 'West', 'South'],
        'Product': ['A', 'B', 'A', 'C', 'B', 'A'],
        'Category': ['Cat1', 'Cat2', 'Cat1', 'Cat3', 'Cat2', 'Cat1'],
        'Units_Sold': [10, 20, 15, 25, 30, 5],
        'Unit_Price_INR': [100, 200, 150, 300, 250, 100],
        'Discount_Pct': [0, 5, 10, 0, 15, 0],
        'Revenue_INR': [1000.0, 3800.0, 2025.0, 7500.0, 6375.0, 500.0],
        'Sales_Channel': ['Online', 'Partner', 'Online', 'Online', 'Partner', 'Online'],
        'Customer_Satisfaction': [5, 4, 5, 3, 4, 5]
    }
    return pd.DataFrame(data)

def create_multi_year_dataset():
    """Create a test dataset spanning multiple years."""
    data = {
        'Order_ID': [f'ORD{i:03d}' for i in range(1, 13)],
        'Order_Date': [
            '2025-01-01', '2025-02-01', '2025-03-01',
            '2026-01-01', '2026-02-01', '2026-03-01',
            '2027-01-01', '2027-02-01', '2027-03-01',
            '2026-04-01', '2026-05-01', '2026-06-01'
        ],
        'Revenue_INR': [1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0,
                        7000.0, 8000.0, 9000.0, 10000.0, 11000.0, 12000.0],
        'Units_Sold': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
    }
    return pd.DataFrame(data)

def create_invalid_date_dataset():
    """Create a dataset with invalid dates."""
    data = {
        'Order_ID': ['ORD001', 'ORD002', 'ORD003'],
        'Order_Date': ['2026-01-01', 'invalid-date', '2026-01-03'],
        'Revenue_INR': [1000.0, 2000.0, 3000.0]
    }
    return pd.DataFrame(data)

def create_missing_date_dataset():
    """Create a dataset with missing dates."""
    data = {
        'Order_ID': ['ORD001', 'ORD002', 'ORD003'],
        'Order_Date': ['2026-01-01', None, '2026-01-03'],
        'Revenue_INR': [1000.0, 2000.0, 3000.0]
    }
    return pd.DataFrame(data)

def create_empty_dataset():
    """Create an empty dataset."""
    return pd.DataFrame(columns=['Order_ID', 'Order_Date', 'Revenue_INR', 'Units_Sold'])

def test_calculate_sales_metrics_basic():
    """Test basic sales metrics calculation."""
    df = create_test_dataset()
    params = {'metrics': ['total_revenue', 'total_units', 'avg_order_revenue', 'num_orders']}
    result = calculate_sales_metrics(df, ['Revenue_INR', 'Units_Sold'], params)
    
    assert 'total_revenue' in result
    assert 'total_units' in result
    assert 'avg_order_revenue' in result
    assert 'num_orders' in result
    
    # Verify calculations
    assert result['total_revenue'] == df['Revenue_INR'].sum()
    assert result['total_units'] == df['Units_Sold'].sum()
    assert result['avg_order_revenue'] == df['Revenue_INR'].mean()
    assert result['num_orders'] == len(df)
    
    print("[PASS] test_calculate_sales_metrics_basic passed")

def test_calculate_sales_metrics_with_groupby():
    """Test sales metrics with grouping."""
    df = create_test_dataset()
    params = {'metrics': ['total_revenue', 'num_orders'], 'group_by': 'Region'}
    result = calculate_sales_metrics(df, ['Revenue_INR', 'Units_Sold'], params)
    
    assert 'total_revenue_by_Region' in result
    assert 'num_orders_by_Region' in result
    
    # Verify grouped calculations
    expected_revenue = df.groupby('Region')['Revenue_INR'].sum().to_dict()
    assert result['total_revenue_by_Region'] == expected_revenue
    
    print("[PASS] test_calculate_sales_metrics_with_groupby passed")

def test_calculate_sales_metrics_statistics():
    """Test statistical metrics (median, std, min, max)."""
    df = create_test_dataset()
    params = {'metrics': ['median_revenue', 'std_revenue', 'min_revenue', 'max_revenue']}
    result = calculate_sales_metrics(df, ['Revenue_INR'], params)
    
    assert 'median_revenue' in result
    assert 'std_revenue' in result
    assert 'min_revenue' in result
    assert 'max_revenue' in result
    
    assert result['median_revenue'] == df['Revenue_INR'].median()
    assert abs(result['std_revenue'] - df['Revenue_INR'].std()) < 0.01
    assert result['min_revenue'] == df['Revenue_INR'].min()
    assert result['max_revenue'] == df['Revenue_INR'].max()
    
    print("[PASS] test_calculate_sales_metrics_statistics passed")

def test_calculate_sales_metrics_error_cases():
    """Test error handling in sales metrics."""
    df = create_test_dataset()
    
    # No metrics specified - test via safe_execute_operation
    operation = {
        'operation': 'calculate_metrics',
        'columns': ['Revenue_INR'],
        'params': {}
    }
    result = safe_execute_operation(df, operation)
    assert 'error' in result
    
    # Invalid column - test via safe_execute_operation
    operation = {
        'operation': 'calculate_metrics',
        'columns': ['NonExistent'],
        'params': {'metrics': ['total_revenue']}
    }
    result = safe_execute_operation(df, operation)
    assert 'error' in result
    assert 'invalid' in result['error'].lower() or 'not found' in result['error'].lower()
    
    print("[PASS] test_calculate_sales_metrics_error_cases passed")

def test_calculate_monthly_metrics_basic():
    """Test basic monthly aggregation."""
    df = create_test_dataset()
    params = {'date_column': 'Order_Date', 'metrics': ['Revenue_INR', 'Units_Sold'], 'period': 'monthly', 'agg': 'sum'}
    result = calculate_time_series_metrics(df, ['Order_Date'], params)
    
    assert 'Revenue_INR_by_monthly' in result
    assert 'Units_Sold_by_monthly' in result
    assert 'period_count' in result
    assert 'date_range' in result
    
    # Should have 3 months (Jan, Feb, Mar 2026)
    assert result['period_count'] == 3
    assert result['date_range']['start'] == '2026-01'
    assert result['date_range']['end'] == '2026-03'
    
    # Verify monthly totals
    jan_revenue = df[df['Order_Date'].str.startswith('2026-01')]['Revenue_INR'].sum()
    feb_revenue = df[df['Order_Date'].str.startswith('2026-02')]['Revenue_INR'].sum()
    mar_revenue = df[df['Order_Date'].str.startswith('2026-03')]['Revenue_INR'].sum()
    
    monthly_data = {r['Period']: r['Revenue_INR'] for r in result['Revenue_INR_by_monthly']}
    assert monthly_data['2026-01'] == jan_revenue
    assert monthly_data['2026-02'] == feb_revenue
    assert monthly_data['2026-03'] == mar_revenue
    
    print("[PASS] test_calculate_monthly_metrics_basic passed")

def test_calculate_monthly_metrics_different_aggs():
    """Test monthly aggregation with different aggregation functions."""
    df = create_test_dataset()
    
    for agg in ['mean', 'count', 'min', 'max', 'median', 'std']:
        params = {'date_column': 'Order_Date', 'metrics': ['Revenue_INR'], 'period': 'monthly', 'agg': agg}
        result = calculate_time_series_metrics(df, ['Order_Date'], params)
        assert 'error' not in result, f"Failed for agg={agg}: {result}"
        assert 'Revenue_INR_by_monthly' in result
    
    print("[PASS] test_calculate_monthly_metrics_different_aggs passed")

def test_calculate_monthly_metrics_multi_year():
    """Test monthly aggregation with multi-year data (should not mix years)."""
    df = create_multi_year_dataset()
    params = {'date_column': 'Order_Date', 'metrics': ['Revenue_INR'], 'period': 'monthly', 'agg': 'sum'}
    result = calculate_time_series_metrics(df, ['Order_Date'], params)
    
    # Should have 9 months (Jan-Mar for 2025, 2026, 2027) + 3 more in 2026 = 12 months
    assert result['period_count'] == 12
    
    # Check that 2025-01 and 2026-01 are separate
    months = [r['Period'] for r in result['Revenue_INR_by_monthly']]
    assert '2025-01' in months
    assert '2026-01' in months
    assert '2027-01' in months
    
    print("[PASS] test_calculate_monthly_metrics_multi_year passed")

def test_calculate_monthly_metrics_invalid_dates():
    """Test monthly aggregation with invalid dates."""
    df = create_invalid_date_dataset()
    params = {'date_column': 'Order_Date', 'metrics': ['Revenue_INR'], 'period': 'monthly', 'agg': 'sum'}
    result = calculate_time_series_metrics(df, ['Order_Date'], params)
    
    assert 'error' in result
    assert 'invalid' in result['error'].lower() or 'missing' in result['error'].lower()
    
    print("[PASS] test_calculate_monthly_metrics_invalid_dates passed")

def test_calculate_monthly_metrics_missing_dates():
    """Test monthly aggregation with missing dates (NaN)."""
    df = create_missing_date_dataset()
    params = {'date_column': 'Order_Date', 'metrics': ['Revenue_INR'], 'period': 'monthly', 'agg': 'sum'}
    result = calculate_time_series_metrics(df, ['Order_Date'], params)
    
    assert 'error' in result
    assert 'invalid' in result['error'].lower() or 'missing' in result['error'].lower()
    
    print("[PASS] test_calculate_monthly_metrics_missing_dates passed")

def test_calculate_monthly_metrics_missing_column():
    """Test monthly aggregation with missing date column."""
    df = create_test_dataset()
    params = {'date_column': 'NonExistent', 'metrics': ['Revenue_INR'], 'period': 'monthly', 'agg': 'sum'}
    result = calculate_time_series_metrics(df, ['NonExistent'], params)
    
    assert 'error' in result
    assert 'not found' in result['error'].lower()
    
    print("[PASS] test_calculate_monthly_metrics_missing_column passed")

def test_calculate_monthly_metrics_non_numeric():
    """Test monthly aggregation with non-numeric metric column."""
    df = create_test_dataset()
    params = {'date_column': 'Order_Date', 'metrics': ['Region'], 'period': 'monthly', 'agg': 'sum'}
    result = calculate_time_series_metrics(df, ['Order_Date'], params)
    
    assert 'error' in result
    assert 'numeric' in result['error'].lower()
    
    print("[PASS] test_calculate_monthly_metrics_non_numeric passed")

def test_safe_execute_operation_calculate_metrics():
    """Test safe_execute_operation with calculate_metrics."""
    df = create_test_dataset()
    operation = {
        'operation': 'calculate_metrics',
        'columns': ['Revenue_INR', 'Units_Sold'],
        'params': {'metrics': ['total_revenue', 'total_units', 'avg_order_revenue', 'num_orders']}
    }
    result = safe_execute_operation(df, operation)
    
    assert 'error' not in result
    assert 'total_revenue' in result
    assert result['total_revenue'] == df['Revenue_INR'].sum()
    
    print("[PASS] test_safe_execute_operation_calculate_metrics passed")

def test_safe_execute_operation_monthly_aggregation():
    """Test safe_execute_operation with time_series_aggregation."""
    df = create_test_dataset()
    operation = {
        'operation': 'time_series_aggregation',
        'columns': ['Order_Date'],
        'params': {'date_column': 'Order_Date', 'metrics': ['Revenue_INR'], 'period': 'monthly', 'agg': 'sum'}
    }
    result = safe_execute_operation(df, operation)
    
    assert 'error' not in result
    assert 'Revenue_INR_by_monthly' in result
    
    print("[PASS] test_safe_execute_operation_monthly_aggregation passed")

def test_safe_execute_operation_groupby():
    """Test safe_execute_operation with groupby."""
    df = create_test_dataset()
    operation = {
        'operation': 'groupby',
        'columns': ['Region', 'Revenue_INR'],
        'params': {'agg': 'sum'}
    }
    result = safe_execute_operation(df, operation)
    
    assert 'error' not in result
    # Should have grouped by Region with sum of Revenue_INR
    
    print("[PASS] test_safe_execute_operation_groupby passed")

def test_safe_execute_operation_filter():
    """Test safe_execute_operation with filter."""
    df = create_test_dataset()
    operation = {
        'operation': 'filter',
        'columns': ['Region'],
        'params': {'condition': 'eq', 'value': 'North'}
    }
    result = safe_execute_operation(df, operation)
    
    assert 'error' not in result
    assert isinstance(result, list)
    assert all(r['Region'] == 'North' for r in result)
    
    print("[PASS] test_safe_execute_operation_filter passed")

def test_safe_execute_operation_unknown():
    """Test safe_execute_operation with unknown operation."""
    df = create_test_dataset()
    operation = {
        'operation': 'unknown_operation',
        'columns': ['Revenue_INR'],
        'params': {}
    }
    result = safe_execute_operation(df, operation)
    
    assert 'error' in result
    assert 'unknown' in result['error'].lower()
    
    print("[PASS] test_safe_execute_operation_unknown passed")

def test_validate_column():
    """Test column validation."""
    df = create_test_dataset()
    assert validate_column(df, 'Revenue_INR') == True
    assert validate_column(df, 'NonExistent') == False
    
    print("[PASS] test_validate_column passed")

def test_validate_columns():
    """Test multiple column validation."""
    df = create_test_dataset()
    assert validate_columns(df, ['Revenue_INR', 'Units_Sold']) == True
    assert validate_columns(df, ['Revenue_INR', 'NonExistent']) == False
    
    print("[PASS] test_validate_columns passed")

def test_allowed_operations_structure():
    """Test that ALLOWED_OPERATIONS has required operations."""
    required_ops = ['calculate_metrics', 'time_series_aggregation', 'groupby', 'filter', 'describe']
    for op in required_ops:
        assert op in ALLOWED_OPERATIONS, f"Missing operation: {op}"
    
    assert 'sum' in ALLOWED_OPERATIONS['time_series_aggregation']
    assert 'mean' in ALLOWED_OPERATIONS['time_series_aggregation']
    
    print("[PASS] test_allowed_operations_structure passed")

def test_baseline_sales_dataset():
    """Test with the actual sales dataset to verify baseline results."""
    df = pd.read_csv(r'C:\Users\Saqib\OneDrive\Desktop\ai-data-analyst\ai_data_analyst_sales_dataset.csv')
    
    # Test total revenue
    params = {'metrics': ['total_revenue']}
    result = calculate_sales_metrics(df, ['Revenue_INR'], params)
    assert abs(result['total_revenue'] - 4584510.0) < 0.01
    
    # Test total units
    params = {'metrics': ['total_units']}
    result = calculate_sales_metrics(df, ['Units_Sold'], params)
    assert result['total_units'] == 237
    
    # Test average revenue
    params = {'metrics': ['avg_order_revenue']}
    result = calculate_sales_metrics(df, ['Revenue_INR'], params)
    assert abs(result['avg_order_revenue'] - 76408.5) < 0.01
    
    # Test num_orders
    params = {'metrics': ['num_orders']}
    result = calculate_sales_metrics(df, ['Revenue_INR'], params)
    assert result['num_orders'] == 60
    
    print("[PASS] test_baseline_sales_dataset passed")

def run_all_tests():
    """Run all tests and report results."""
    tests = [
        test_calculate_sales_metrics_basic,
        test_calculate_sales_metrics_with_groupby,
        test_calculate_sales_metrics_statistics,
        test_calculate_sales_metrics_error_cases,
        test_calculate_monthly_metrics_basic,
        test_calculate_monthly_metrics_different_aggs,
        test_calculate_monthly_metrics_multi_year,
        test_calculate_monthly_metrics_invalid_dates,
        test_calculate_monthly_metrics_missing_dates,
        test_calculate_monthly_metrics_missing_column,
        test_calculate_monthly_metrics_non_numeric,
        test_safe_execute_operation_calculate_metrics,
        test_safe_execute_operation_monthly_aggregation,
        test_safe_execute_operation_groupby,
        test_safe_execute_operation_filter,
        test_safe_execute_operation_unknown,
        test_validate_column,
        test_validate_columns,
        test_allowed_operations_structure,
        test_baseline_sales_dataset,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__} FAILED: {e}")
            failed += 1
    
    print(f"\n{'='*50}")
    print(f"Tests passed: {passed}")
    print(f"Tests failed: {failed}")
    print(f"{'='*50}")
    
    if failed == 0:
        print("[ALL TESTS PASSED]")
    else:
        print("[SOME TESTS FAILED]")
    
    return failed == 0

if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)