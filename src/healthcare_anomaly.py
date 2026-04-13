import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from collections import deque
import json
from datetime import datetime

class HealthcareAnomalyDetector:
    """Real-time patient vital signs anomaly detection"""
    
    def __init__(self, window_size=30, contamination=0.05):
        self.window_size = window_size
        self.scaler = StandardScaler()
        self.model = IsolationForest(contamination=contamination, random_state=42)
        
        # Patient data buffers
        self.patient_buffers = {}
        self.alert_threshold = 0.7
        self.alerts = []
    
    def normalize_vitals(self, vitals):
        """Normalize vital signs (age-aware thresholds)"""
        normalized = vitals.copy()
        
        # Age-appropriate thresholds
        age = vitals.get('age', 40)
        
        # Heart rate: target 60-100, adjust for age/activity
        normalized['heart_rate_norm'] = vitals['heart_rate'] / 100.0
        
        # Blood pressure: systolic 90-140, diastolic 60-90
        normalized['systolic_norm'] = vitals['systolic'] / 140.0
        normalized['diastolic_norm'] = vitals['diastolic'] / 90.0
        
        # Oxygen saturation: 95-100% is normal
        normalized['spo2_norm'] = 1.0 - ((100 - vitals['spo2']) / 5.0)
        
        # Temperature: 36.1-37.2°C normal
        normalized['temp_norm'] = abs(vitals['temperature'] - 36.8) / 1.0
        
        return normalized
    
    def ingest_patient_vitals(self, patient_id, vitals):
        """Ingest real-time patient vitals"""
        if patient_id not in self.patient_buffers:
            self.patient_buffers[patient_id] = deque(maxlen=self.window_size)
        
        normalized = self.normalize_vitals(vitals)
        vital_vector = np.array([
            normalized['heart_rate_norm'],
            normalized['systolic_norm'],
            normalized['diastolic_norm'],
            normalized['spo2_norm'],
            normalized['temp_norm']
        ]).reshape(1, -1)
        
        # Add to patient buffer
        self.patient_buffers[patient_id].append(vital_vector)
        
        # Detect anomaly if enough data
        if len(self.patient_buffers[patient_id]) >= 5:
            anomaly_score = self.detect_anomaly(patient_id)
            
            if anomaly_score and anomaly_score['is_anomaly']:
                self._trigger_alert(patient_id, vitals, anomaly_score)
            
            return anomaly_score
        
        return None
    
    def detect_anomaly(self, patient_id):
        """Detect anomalies using Isolation Forest"""
        if len(self.patient_buffers[patient_id]) < 5:
            return None
        
        buffer = list(self.patient_buffers[patient_id])
        X = np.vstack(buffer)
        
        # Train on patient's own history
        self.model.fit(X)
        
        # Score latest reading
        latest = buffer[-1]
        anomaly_score = self.model.score_samples(latest)[0]
        prediction = self.model.predict(latest)[0]
        
        return {
            'patient_id': patient_id,
            'anomaly_score': anomaly_score,  # -inf to 0
            'is_anomaly': prediction == -1,
            'severity': self._calculate_severity(anomaly_score),
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_severity(self, score):
        """Map anomaly score to severity level"""
        if score < -1.0:
            return 'CRITICAL'
        elif score < -0.5:
            return 'HIGH'
        elif score < -0.2:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _trigger_alert(self, patient_id, vitals, anomaly):
        """Trigger clinical alert"""
        alert = {
            'patient_id': patient_id,
            'alert_type': 'VITAL_SIGN_ANOMALY',
            'severity': anomaly['severity'],
            'vitals': vitals,
            'anomaly_details': anomaly,
            'timestamp': datetime.now().isoformat(),
            'recommended_action': self._recommend_action(anomaly['severity'])
        }
        
        self.alerts.append(alert)
        
        # Log alert
        print(f"\n🚨 ALERT: {patient_id}")
        print(f"   Severity: {alert['severity']}")
        print(f"   Action: {alert['recommended_action']}")
        
        return alert
    
    def _recommend_action(self, severity):
        """Recommend clinical action"""
        actions = {
            'CRITICAL': 'Call MD immediately, Prepare ICU bed',
            'HIGH': 'Notify MD on call, Continuous monitoring',
            'MEDIUM': 'Increase monitoring frequency',
            'LOW': 'Continue normal monitoring'
        }
        
        return actions.get(severity, 'Monitor')
    
    def get_patient_status(self, patient_id):
        """Get overall patient status"""
        if patient_id not in self.alerts:
            return {'status': 'NORMAL', 'recent_alerts': 0}
        
        recent_alerts = [a for a in self.alerts if a['patient_id'] == patient_id]
        
        return {
            'patient_id': patient_id,
            'status': 'CRITICAL' if any(a['severity'] == 'CRITICAL' for a in recent_alerts) else 'STABLE',
            'recent_alerts': len(recent_alerts),
            'last_alert': recent_alerts[0] if recent_alerts else None
        }

if __name__ == "__main__":
    detector = HealthcareAnomalyDetector()
    
    print("Simulating patient vital signs stream...\n")
    
    # Normal patient
    for i in range(20):
        vitals = {
            'heart_rate': np.random.normal(75, 5),
            'systolic': np.random.normal(120, 5),
            'diastolic': np.random.normal(80, 5),
            'spo2': np.random.normal(97, 1),
            'temperature': np.random.normal(36.8, 0.3),
            'age': 45
        }
        
        result = detector.ingest_patient_vitals('P001', vitals)
        
        if result and result['is_anomaly']:
            print(f"Anomaly detected: {result['severity']}")
    
    # Abnormal readings (fever + high HR)
    print("\nInjecting anomalous readings...\n")
    for i in range(5):
        vitals = {
            'heart_rate': np.random.normal(130, 5),  # High
            'systolic': np.random.normal(140, 5),
            'diastolic': np.random.normal(90, 5),
            'spo2': np.random.normal(92, 1),  # Low
            'temperature': np.random.normal(39.5, 0.5),  # Fever
            'age': 45
        }
        
        result = detector.ingest_patient_vitals('P001', vitals)
    
    print("\nPatient Status:")
    print(json.dumps(detector.get_patient_status('P001'), indent=2, default=str))
