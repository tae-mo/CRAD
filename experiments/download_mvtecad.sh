mkdir mvtec
cd mvtec
# Download MVTec anomaly detection dataset
wget https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz
tar -xf mvtec_anomaly_detection.tar.xz
mv mvtec ../data/MVTec-AD/mvtec_anomaly_detection
rm mvtec_anomaly_detection.tar.xz