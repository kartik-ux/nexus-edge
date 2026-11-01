# NEXUS EDGE

**An experimental edge AI system for sensor monitoring and anomaly detection.**

NEXUS EDGE explores how machine learning models can be used closer to the source of sensor data, rather than relying entirely on cloud-based processing.

The project currently includes a Python prototype for analysing sensor readings, detecting anomalies, monitoring changes in incoming data and managing model updates through a validation process.

The initial implementation uses synthetic sensor data representing vibration, temperature and humidity. The next stage is to connect physical sensors to an ESP32-S3 board and investigate running inference directly on the device.

### Current implementation

- Sensor feature extraction and data processing
- Isolation Forest-based anomaly detection
- Anomaly threshold calibration
- Data drift monitoring
- Controlled baseline adaptation
- Candidate model evaluation and promotion
- Model versioning and rollback
- Automated tests

### Development status

The Python prototype is implemented and tested. Physical sensor integration and embedded deployment are planned but have not yet been completed.

This is an ongoing engineering project, with development focused on reliability, practical deployment constraints and measurable performance.
