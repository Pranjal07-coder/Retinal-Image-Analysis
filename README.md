# Retinal-Image-Analysis  
By:-PRANJAL DESHMUKH 


Executive Summary & Project Scope
This project investigates deep learning methods for two distinct ophthalmic image analysis tasks:

Diabetic Retinopathy (DR) Severity Grading (Classification):

Task: 5-level ordinal grading of diabetic retinopathy severity (Grade 0: No DR, Grade 1: Mild NPDR, Grade 2: Moderate NPDR, Grade 3: Severe NPDR, Grade 4: Proliferative DR).
Active Benchmark Dataset: RetinaMNIST (MedMNIST v2 collection, derived from DeepDRiD challenge).
Architectures: Deep Convolutional Neural Networks (DenseNet121 and ResNet50) with transfer learning.
Explainability: Live Gradient-weighted Class Activation Mapping (Grad-CAM).
Retinal Blood Vessel Segmentation (Dense Pixel Classification):

Task: Binary semantic segmentation of retinal vasculature.
Dataset: DRIVE (Digital Retinal Images for Vessel Extraction, 40 images).
Architecture: Attention U-Net (Attention_U-Net_Vessel_Segmentation.ipynb and src/models/segmentation.py).
Note: Segmentation and classification are distinct tasks with separate clinical objectives and data pipelines; they are not conflated.
📊 Measured Benchmark Results vs. README Audit
1. Transparent Audit of Historical README Discrepancy
An initial commit (2159259, Oct 2025) previously listed placeholder figures (ResNet50: 68.5% accuracy; DenseNet121: 65.2% accuracy) and synthetic class distributions (95, 82, 88, 72, 63).

Audit Findings:

In the git commit history, those numbers were committed as pre-experiment "predicted metrics" before the real training pipeline was executed on CPU.
When the actual experiment ran on CPU (experiments/real_data_experiment.py, ~40 min per model), the actual measured test accuracy was 49.0% for ResNet50 and 52.5% for DenseNet121.
Literature Verification: In the official MedMNIST v2 benchmark publication (Nature Scientific Data, Yang et al., 2023), standard ResNet-18 achieves 49.3% accuracy (AUC 0.710) and ResNet-50 achieves 51.1% accuracy (AUC 0.701) on 224×224 RetinaMNIST.
Conclusion: The measured ~49.0% to 52.5% performance is authentic, fully reproducible, and directly in line with peer-reviewed literature for standard CNNs on 28×28 RetinaMNIST images.
2. Measured Held-Out Test Set Performance (400 Samples)
The RetinaMNIST held-out test split (400 images) was kept untouched during model training, tuning, and early stopping.

Model Backbone	Parameters	Raw Test Acc	Balanced Acc	Quadratic Weighted Kappa (QWK)	Macro F1	AUC-ROC (OvR)	Training Time (CPU)	Best Epoch
DenseNet121 (Recommended)	7.0M	52.5%	38.6%	0.511	0.387	0.727	~40.5 min	3 / 18
ResNet50	23.5M	49.0%	33.5%	0.423	0.333	0.735	~39.4 min	1 / 16
3. Critical Findings on Class Imbalance & Ordinal Evaluation
Class Imbalance in Test Set:
Grade 0 (No DR): 174 samples (43.5%)
Grade 1 (Mild DR): 46 samples (11.5%)
Grade 2 (Moderate DR): 92 samples (23.0%)
Grade 3 (Severe DR): 68 samples (17.0%)
Grade 4 (Proliferative DR): 20 samples (5.0%)
Majority-Class Baseline: A naive dummy classifier that always predicts Grade 0 achieves 43.5% raw accuracy, but has 20.0% balanced accuracy and 0.000 QWK.
Why QWK Matters: In diabetic retinopathy grading, predicting Grade 1 when the true label is Grade 0 is a minor error, whereas predicting Grade 0 when the true label is Grade 4 is catastrophic. Quadratic Weighted Kappa (QWK) penalizes misclassifications quadratically based on clinical severity distance.
Efficiency: DenseNet121 achieves significantly stronger ordinal agreement (QWK 0.511 vs 0.423) and macro F1 (0.387 vs 0.333) while utilizing 70% fewer parameters (7.0M vs 23.5M) than ResNet50.
🔍 Dataset Research & Governance Matrix
Before adopting or downloading any dataset, its clinical task, resolution, license terms, and disk footprint must be verified.

Dataset Name	Official Source / Host	Clinical Task	Number of Images	Resolution	Access & License	Disk Size	Project Status
RetinaMNIST	MedMNIST v2 / Zenodo (zenodo.org/records/10519652)	5-Class DR Severity (0–4)	1,600 (1,080 train / 120 val / 400 test)	28×28 native (cropped from ~1736×1824)	Open Access (CC BY 4.0)	~3.3 MB (.npz)	Integrated baseline
APTOS 2019	Kaggle / Aravind Eye Hospital	5-Class DR Severity (0–4)	3,662 labeled train, 1,928 test	High-Res (~1000×1000 to 3000×2000)	Kaggle competition terms (Login required)	~9.5 GB	Investigated (Requires approval)
EyePACS	Kaggle DR Detection (2015)	5-Class DR Severity (0–4)	88,702 images (Patient-level eye pairs)	Variable High-Res (up to 4000×3000)	Kaggle competition rules	~80 GB	Investigated (Requires approval & high compute)
IDRiD	IEEE DataPort / Grand Challenge (idrid.grand-challenge.org)	DR Grading (516) + Lesion Masks (81)	516 grading images (413 train / 103 test)	Very High-Res (4288×2848)	Open Access (CC BY 4.0, IEEE login)	~1.5 GB grading (~14 GB full)	Investigated (High clinical quality)
DRIVE	Grand Challenge (drive.grand-challenge.org)	Binary Blood Vessel Segmentation	40 images (20 train / 20 test)	565×584	Open for research	~30 MB	Integrated for segmentation
"Retinal Fundus Image 50K" (RFID50K)	Kaggle (Uploaded by Gautam Raj)	7-Class Multi-Disease Classification	~50,000 images	Variable High-Res	"Other" (Community aggregation)	~25–40 GB	Investigated; NOT 5-class DR grading
⚠️ Investigation of "Retinal Fundus Image 50K" (RFID50K)
The user-mentioned "Retinal Fundus Image 50K" was identified as the community dataset hosted on Kaggle by user Gautam Raj.

Task Incompatibility: It is structured into 7 disease classes (Cataract, Diabetic Retinopathy, Glaucoma, Hypertensive Retinopathy, Myopia, AMD, Normal Fundus), rather than 5-level diabetic retinopathy severity grading.
Provenance & Leakage Risk: It compiles images from multiple public datasets without patient identifiers, introducing severe risks of patient-level data leakage between arbitrary train/val/test splits.
Recommendation: It cannot be dropped into the 5-class DR severity pipeline without re-architecting the classification objective to multi-disease diagnosis.

RETINAL IMAGE ANALYSIS -- FAST SYSTEM SMOKE TEST
[Check 1/6] Python and PyTorch Environment...          [OK]
[Check 2/6] Verifying RetinaMNIST dataset archive...   [OK]
[Check 3/6] Verifying trained checkpoints...           [OK]
[Check 4/6] Running inference on synthetic image...    [OK]
[Check 5/6] Testing live Grad-CAM heatmaps...          [OK]
[Check 6/6] Verifying medical grading metrics...       [OK]
SMOKE TEST PASSED: 6/6 checks succeeded in ~0.82s!
3. Run the Unit Test Suite
Execute the 20 unit tests covering metrics, models, preprocessing, Grad-CAM, dataset integrity, and checkpoint loading:

python -m unittest discover tests
4. Launch the Local Streamlit Demonstration Dashboard
The dashboard runs strictly on your local machine (http://localhost:8501) and does not transmit data externally:

python -m streamlit run app.py
Dashboard Demonstration Workflow for College Evaluators:
Model Selection: Choose between DenseNet121 (7.0M params, higher QWK) and ResNet50 (23.5M params).
Input Mode:
RetinaMNIST Test Sample: Choose a known test-set sample (Grades 0 to 4) to verify model predictions against held-out ground truth.
Upload Local Image: Upload an authorized JPG or PNG fundus photograph (max 10 MB). Images are processed in RAM and never persisted to disk. An attribution field is available for presentation records.
Model Inference: Inspect predicted severity grade, highest softmax confidence, and the 5-class score distribution.
Live Explainability (Grad-CAM): View live convolutional feature activation heatmaps and overlays computed on the active image.
JSON Export: Download a structured prediction report containing timestamp, model metadata, all class scores, and non-clinical disclaimers.
Benchmark Tabs: Inspect held-out test confusion matrices, training history curves, dataset comparison tables, and research limitations.
🛠️ Reproducing or Running Experiments
To train or evaluate new models without overwriting historical baseline checkpoints:

# Run a new training experiment (outputs save to experiments/runs/run_<timestamp>/)
python experiments/real_data_experiment.py --epochs 20 --batch_size 32 --models DenseNet121 ResNet50

# Custom output directory
python experiments/real_data_experiment.py --output_dir experiments/runs/my_experiment --epochs 15
Note: CPU training takes ~35–45 minutes per model on 1,080 samples. If CUDA is available, set --device cuda.

🔬 Limitations & Research Caveats
Resolution & Micro-Lesion Physics: RetinaMNIST images are 28×28 pixels (center-cropped from ~1736×1824). Early diabetic retinopathy lesions (microaneurysms of 10–100 μm) become sub-pixel at this resolution. Models learn macro-retinal patterns rather than fine clinical microvascular lesions.
Clinical Generalization: Performance on RetinaMNIST does not establish performance on high-resolution clinical fundus photography, different camera hardware, or distinct patient populations.
Grad-CAM Limitations: Grad-CAM visualizes coarse feature activations (7×7 spatial grid). It does not represent causal medical reasoning or validated lesion boundaries.
