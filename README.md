# Integrating GPT and CNN for Climate-Resilient Crop Adaptation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A hybrid deep learning model combining Convolutional Neural Networks (CNN) and Generative Pre-trained Transformers (GPT) for precision agriculture in Tamil Nadu, India.

## 📄 Research Paper

This repository contains the implementation of the research paper:

**"Integrating GPT and CNN for Climate-Resilient Crop Adaptation: A Hybrid Model for Precision Agriculture"**

Published in: *Nexus in Agriculture Engineering (2025)*  
ISBN: 9789348596758  
Pages: 271-279

### Authors
- **Lingeswaran A.** - Department of Information Technology, St. Joseph's College of Engineering, Chennai, India
- **J.S. Melvin Fredrick** - Department of Information Technology, St. Joseph's College of Engineering, Chennai, India
- **R. Kabilesh Kumar** - Department of Information Technology, St. Joseph's College of Engineering, Chennai, India
- **G. Lathaselvi** - Department of Information Technology, St. Joseph's College of Engineering, Chennai, India
- **G. Annapoorani** - Department of CSE and IT, University College of Engineering, BIT campus, Anna University, Trichy, India

---

## 🎯 Overview

The agricultural sector in Tamil Nadu faces significant challenges including climate variability, soil health degradation, and lack of location-specific crop recommendations. This project introduces a **hybrid AI model** that combines:

- **EfficientNet-B3** (CNN) for high-resolution satellite image classification
- **GPT** (Generative Pre-trained Transformer) for natural language processing and recommendation generation

The system provides farmers with actionable, data-driven insights in their native language (Tamil) through Text-to-Speech technology.

---

## 🔬 Key Features

- **High-Resolution Land Classification**: Uses EfficientNet-B3 for accurate satellite imagery analysis
- **Real-Time Crop Recommendations**: Provides actionable insights based on weather patterns, soil health, and ROI analysis
- **Multilingual Support**: Generates recommendations in both Tamil and English
- **Text-to-Speech (TTS)**: Audio guidance in the farmer's native language for accessibility
- **Mobile-Friendly Interface**: Designed for easy access in rural areas
- **Low Computational Cost**: Optimized for real-time processing (1.2 seconds)

---

## 🏗️ System Architecture

### Data Sources
1. **Remote Sensing & Satellite Data**: Weather patterns and land characteristics analysis
2. **Soil Quality Data**: pH levels, fertility, NPK levels, and texture from government sources
3. **Infrared Spectral Indices**: NPK level calculations from remote sensing data
4. **Manual Observations**: Survey data, soil nutrient mapping, and historical agricultural performance

### Model Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           DATA COLLECTION                                │
│  (Satellite Images, Soil Parameters, Weather Data, Manual Observations) │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          DATA PRE-PROCESSING                             │
│            (Cleaning, Normalization, Key-Value Formatting)               │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     LAND TYPE CLASSIFICATION (CNN)                       │
│                          EfficientNet-B3                                 │
│    (Pattern Recognition, Soil Moisture, Vegetation Cover Analysis)      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     INSIGHT GENERATION (GPT)                             │
│       (Soil Health, Crop Efficiency, Weather Condition Analysis)        │
│                      Output: Structured JSON                             │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    NATURAL LANGUAGE PROCESSING                           │
│         (JSON to Human-Readable Text Conversion using GPT)              │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                  TRANSLATION & TEXT-TO-SPEECH (TTS)                      │
│            (English → Tamil Translation, Audio Generation)               │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    REAL-TIME CROP RECOMMENDATIONS                        │
│         (Optimal Crops, Planting Dates, Soil Improvements)              │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Performance Results

The hybrid model demonstrates significant improvements over existing approaches:

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | Processing Time (s) |
|-------|-------------|---------------|------------|--------------|---------------------|
| **Proposed Hybrid Model (EfficientNet-B3 + GPT)** | **94.7** | **93.8** | **93.2** | **93.5** | **1.2** |
| CNN-based Land Classification | 85.2 | 83.5 | 84.0 | 83.7 | 1.8 |
| Decision Tree-based Crop Recommendation | 78.4 | 76.2 | 77.5 | 76.8 | 1.8 |
| LSTM-based Weather Prediction | 81.0 | 79.8 | 81.2 | 80.5 | 2.4 |
| GAN-based Crop Yield Prediction | 88.5 | 87.1 | 87.0 | 87.5 | 3.5 |

### Key Performance Improvements
- **9.5% higher accuracy** than CNN-based models
- **6.7% higher precision** than GAN-based models
- **33% faster processing** than CNN-based models
- **66% faster processing** than GAN-based models

---

## 🛠️ Technologies Used

- **Deep Learning Framework**: EfficientNet-B3 (CNN architecture)
- **Natural Language Processing**: GPT (Generative Pre-trained Transformer)
- **Text-to-Speech**: TTS technology for audio output
- **Languages Supported**: Tamil and English
- **Data Format**: JSON for structured insights

---

## 📋 Recommendations Generated

The system provides comprehensive agricultural recommendations including:

1. **Soil Health Analysis**
   - pH level assessment
   - Nutrient deficiency identification
   - Fertility ratings

2. **Crop Selection**
   - Optimal crop types for current conditions
   - Alternative crop suggestions
   - Expected yield predictions

3. **Planting Guidelines**
   - Optimal planting dates
   - Seasonal considerations
   - Climate adaptation strategies

4. **Return on Investment (ROI)**
   - Cost-benefit analysis
   - Market trend considerations
   - Profitability predictions

---

## ⚠️ Limitations

1. **Data Availability Dependency**: Model accuracy depends on high-resolution satellite imagery and real-time weather data availability
2. **Computational Load**: Large-scale multimodal data processing may introduce latency on limited hardware
3. **Language Limitations**: Currently supports Tamil and English; may not perfectly translate local dialects or intricate agricultural terminologies

---

## 🚀 Future Work

- **Extended Language Support**: Additional regional dialects for greater accessibility
- **Extreme Climate Adaptation**: Reinforcement learning methods for droughts, floods, and pest management
- **Local Farming Practices Integration**: Custom crop rotation patterns and traditional methods
- **Mobile & Web Platform**: User-friendly application development
- **Cloud-Based Architecture**: Seamless data access and cross-platform functionality

---

## 📚 References

1. Ball, J. E., Anderson, D. T., & Chan, C. S. (2017). Comprehensive survey of deep learning in remote sensing. *Journal of Applied Remote Sensing*, 11(4), 042609.

2. Ferentinos, K. P. (2018). Deep learning models for plant disease detection and diagnosis. *Computers and Electronics in Agriculture*, 145, 311-318.

3. Kamilaris, A., & Prenafeta-Boldú, F. X. (2018). A review of the use of convolutional neural networks in agriculture. *The Journal of Agricultural Science*, 156(3), 312-322.

4. Kamilaris, A., & Prenafeta-Boldú, F. X. (2018). Deep learning in agriculture: A survey. *Computers and Electronics in Agriculture*, 147, 70-90.

5. Kattenborn, T., Leitloff, J., Schiefer, F., & Hinz, S. (2021). Review on Convolutional Neural Networks (CNN) in vegetation remote sensing. *ISPRS Journal of Photogrammetry and Remote Sensing*, 173, 24-49.

6. Ma, L., Liu, Y., Zhang, X., Ye, Y., Yin, G., & Johnson, B. A. (2019). Deep learning in remote sensing applications: A meta-analysis and review. *ISPRS Journal of Photogrammetry and Remote Sensing*, 152, 166-177.

7. Tan, M., & Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. *International Conference on Machine Learning*. PMLR.

8. Teimouri, N., Christiansen, M. P., Jørgensen, R. N., & Sørensen, C. G. (2019). A deep learning-based approach for crop classification using dual-polarimetric C-band radar data. *Computers and Electronics in Agriculture*, 164, 104887.

---

## 📄 Citation

If you use this work in your research, please cite:

```bibtex
@incollection{lingeswaran2025integrating,
  title={Integrating GPT and CNN for Climate-Resilient Crop Adaptation: A Hybrid Model for Precision Agriculture},
  author={Lingeswaran, A. and Fredrick, J.S. Melvin and Kumar, R. Kabilesh and Lathaselvi, G. and Annapoorani, G.},
  booktitle={Nexus in Agriculture Engineering},
  pages={271--279},
  year={2025},
  publisher={Today \& Tomorrow's Printers and Publishers},
  address={New Delhi, India},
  isbn={9789348596758}
}
```

---

## 📧 Contact

For questions or collaborations, please contact the authors through their respective institutions:
- St. Joseph's College of Engineering, Chennai, India
- University College of Engineering, BIT campus, Anna University, Trichy, India

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <i>Empowering Tamil Nadu farmers with AI-driven precision agriculture for climate resilience and sustainable productivity.</i>
</p>
