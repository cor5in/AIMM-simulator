# Geon Kim 2024-11-21

import pandas as pd
import sqlite3
from datetime import datetime

class TimeSeriesDB:
    """시계열 데이터 저장소"""
    
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.initialize_tables()
        
    def initialize_tables(self):
        """테이블 초기화"""
        with self.conn:
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS traffic_data (
                    timestamp DATETIME,
                    cell_id INTEGER,
                    frequency INTEGER,
                    rb_usage REAL,
                    connected_ues INTEGER,
                    throughput REAL
                )
            ''')
            
    def store_traffic_data(self, cell_id, data):
        """트래픽 데이터 저장"""
        timestamp = datetime.now()
        
        with self.conn:
            for freq, metrics in data.items():
                self.conn.execute('''
                    INSERT INTO traffic_data
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    timestamp,
                    cell_id,
                    freq,
                    metrics['rb_usage'],
                    metrics['connected_ues'],
                    metrics['throughput']
                ))
                
    def get_cell_data(self, cell_id, start_time=None, end_time=None):
        """셀 데이터 조회"""
        query = "SELECT * FROM traffic_data WHERE cell_id = ?"
        params = [cell_id]
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
            
        return pd.read_sql_query(query, self.conn, params=params)