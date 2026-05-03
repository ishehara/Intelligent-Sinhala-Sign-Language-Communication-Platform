# CHAPTER 3: RESULTS AND DISCUSSION

**Project:** Intelligent Sinhala Sign Language Communication Platform (ISSLCP)
**Institution:** Sri Lanka Institute of Information Technology (SLIIT)
**Programme:** Bachelor of Science (Honours) in Information Technology
**Academic Year:** 2025 / 2026

---

## 3.1 Results

This section presents the results achieved across the four components of the Intelligent Sinhala Sign Language Communication Platform. Each component was developed and evaluated independently before integration into the unified platform.

The Environmental Sound Alert Module was trained on approximately 3,230 audio clips across seven dangerous sound categories using a three-block CNN with 13×40 MFCC feature inputs. The model trained for 76 epochs before early stopping triggered, achieving a final training accuracy of 95.86% and a best validation accuracy of 94.40% at epoch 61. Evaluation on the held-out test set of 448 samples produced a top-1 test accuracy of **92.19%**, with the three emergency siren classes (ambulance, fire truck, police siren) achieving the highest per-class performance due to their larger training representation (~833 clips each) and acoustically distinctive spectral signatures. The complete detection pipeline — from audio capture to mobile alert display — measured 2.7–2.9 seconds end-to-end over a local Wi-Fi connection. A confidence gate of 85% was found to be the optimal operating threshold, eliminating all background-noise false positives while retaining 97% of genuine hazard detections.

The SSL Reader was trained on the signVideo dataset comprising 8,472 videos across 383 Sinhala sign classes — the largest vocabulary addressed by any published Sinhala Sign Language recognition system. Four model architectures were systematically evaluated: a Multimodal BiLSTM (~47% validation accuracy), a Multimodal Transformer (~41%), a Hybrid LSTM-Transformer (~52%), and the final Multi-Stream Fusion Model (~76.3%). The Multi-Stream Fusion Model outperformed the best baseline by 24 percentage points while using approximately half the parameters (~2.52M vs ~5.2M), demonstrating that routing each modality (hand, face, pose) through a dedicated sub-network matched to its temporal characteristics is more effective than increasing model depth within a single shared architecture. Training converged at epoch 87, and evaluation on the held-out test set yielded a test accuracy of **80.1%**. All 45 system test cases across feature extraction, augmentation, model inference, and API endpoints passed without failure.

The Adaptive Learning System integrates a CNN sign classifier for 25 static Sinhala sign letters with a two-layer RL engine. The base `RLFeedbackAgent` uses a Q-table to adapt feedback granularity based on accumulated learner interaction history — providing detailed corrective cues to struggling learners and positive reinforcement to confident learners. The `AdaptiveLearningAgent` extends this with dynamic curriculum sequencing and spaced repetition scheduling derived from per-sign forgetting curve estimates, replacing fixed review intervals with individually predicted review priorities. The system operates in real time through a Flask REST API, with camera-based hand landmark detection handled by the MediaPipe pipeline and a 5-prediction smoothing buffer stabilising sign recognition output before RL-driven decisions are made.

The Two-Way Communication System delivers a fully offline, dual-view mobile interface for non-verbal communication between children with speaking disabilities and their caregivers. The parent view provides a Sinhala Unicode text-to-sign conversion engine, a categorised icon library of 40+ culturally adapted icons across five domains, and 36+ pre-built daily communication templates across six scenarios (mealtime, school, health, play, bedtime, emotions). The child view provides an emoji-style emotion selector covering five states with intensity gradations and colour-coded feedback, a quick-needs button panel, and a conversation history log. All sign images, icons, and templates are stored locally on-device, ensuring complete functionality without internet connectivity — a critical design requirement for rural deployment in Sri Lanka. All six communication scenario domains were verified as fully covered during functional testing.

---

## 3.2 Research Findings

### 3.2.1 Finding 1: MFCC + CNN Is Sufficient for Real-Time Dangerous Sound Classification

The Environmental Sound Alert experiment established that a compact 13 × 40 MFCC feature matrix combined with a three-block CNN (~293,000 parameters) is sufficient to classify seven dangerous sound categories at 92.19% test accuracy. Critically, the entire feature extraction step requires only 5–10 milliseconds on CPU, and end-to-end latency from audio capture to alert display measures 2.7–2.9 seconds — within the operational threshold for meaningful real-time warning. This validates the design hypothesis that a lightweight, server-deployable model is practical for this accessibility application, without requiring edge inference or specialised hardware.

The per-coefficient z-score normalisation was identified as a critical preprocessing step. Early experiments without normalisation showed significantly slower convergence because the zeroth MFCC coefficient's large magnitude (~[−300, 0]) dominated gradients and suppressed learning in higher-order coefficients. After applying normalisation, convergence stabilised and the model achieved the reported accuracy levels.

### 3.2.2 Finding 2: Modality-Specific Sub-Networks Substantially Outperform Single-Stream Architectures for Sign Language Recognition

The most significant finding of the SSL Reader research is that routing each input modality (hand gestures, facial expressions, body posture) through a dedicated sub-network matched to its temporal characteristics is more effective than increasing model depth or parameter count within a single shared architecture. The Multi-Stream Fusion Model (Model 4, ~2.52M parameters, 76.3% validation accuracy) outperformed the best single-stream baseline (Model 3, Hybrid LSTM-Transformer, ~5.2M parameters, 52% validation accuracy) by 24 percentage points, while using approximately half the parameters.

This is a significant result because it shows that the accuracy bottleneck in single-stream architectures is not insufficient capacity but architectural mismatch — sign language modalities have fundamentally different temporal statistics (rapid discrete hand shape transitions vs. slower continuous facial expression changes vs. context-dependent postural shifts) that cannot be optimally processed by a single shared filter bank or attention mechanism.

### 3.2.3 Finding 3: Transformer Architectures Are Ill-Suited to Low-Data Sign Language Tasks

Among the four SSL Reader architectures evaluated, the Multimodal Transformer (Model 2) achieved the lowest validation accuracy (~41%), despite being the second-largest model (~4.7M parameters). This finding is consistent with the established data-hunger of Transformer self-attention mechanisms. With approximately 22 training videos per class, the model lacked sufficient examples to learn reliable global attention patterns across 383 sign classes. In contrast, architectures with stronger inductive biases — the LSTM's sequential gating and the TCN's local dilated convolutions — generalised better under the same data constraint.

This finding has practical implications for future sign language recognition research in low-resource languages: inductive bias should be preferred over universal approximation when data is scarce.

### 3.2.4 Finding 4: Skeleton-Level Augmentation Effectively Mitigates Data Scarcity

The SSL Reader experiment confirmed that skeleton-level data augmentation is an effective strategy for training reliable deep models under severe data scarcity (~22 videos per class). The combination of seven stochastic operations (spatial rotation, spatial scaling, Gaussian noise, temporal shift, time warping, frame dropping, and time masking), applied at a global probability of 0.7 per sample, reduced the train-validation accuracy gap and improved generalisation. Controlled ablation experiments during development showed an approximately 8–12 percentage point drop in validation accuracy when augmentation was removed, directly attributing a substantial share of the model's generalisation capability to this strategy.

Horizontal flipping was explicitly excluded from the augmentation operations because Sinhala Sign Language distinguishes handedness: flipping a sign would change its meaning, producing semantically incorrect training samples that would degrade model quality.

### 3.2.5 Finding 5: Attention-Based Fusion Provides Interpretable Modality Weighting

The attention fusion module in the Multi-Stream Fusion Model produces per-sample stream weights (w_hand, w_face, w_pose) that are interpretable by design. Analysis of attention weights across the test set revealed linguistically consistent patterns:

- Signs with strong distinctive handshapes (e.g., number signs) showed high w_hand weights.
- Signs with grammatically required facial components (question markers, negations, emotional signs) showed elevated w_face weights.
- Signs involving pronounced body orientation or proxemic reference showed elevated w_pose weights.

This finding validates the architectural assumption that all three modalities contribute differentially to sign recognition — in agreement with sign language linguistic theory — and provides a mechanism for future work on computational understanding of SSL grammar.

### 3.2.6 Finding 6: Confidence Gating Is the Key Usability Engineering Decision for Sound Alert Systems

The threshold analysis for the Sound Alert Module demonstrates that the choice of confidence gate is the most impactful single engineering decision for an accessibility system of this type. At 60% confidence, three of ten background-only clips triggered false alerts — a rate that would render the application unusable in daily life. At 85%, all background clips were correctly suppressed while 97% of genuine hazard sounds were correctly detected. At 95%, a small number of true positives at greater distances were incorrectly rejected, reducing sensitivity without proportional benefit to precision.

This finding suggests that for safety-oriented accessibility applications, the confidence threshold should be calibrated specifically against the intended deployment acoustic environment rather than selected from general literature defaults.

### 3.2.7 Finding 7: RL-Based Personalisation Addresses the Static Lesson Limitation of Conventional SSL Learning Apps

The Adaptive Learning component demonstrates that integrating a Q-table-based reinforcement learning agent into a sign language learning app enables dynamic personalisation that conventional static-curriculum apps cannot provide. The RL agent's ability to track per-sign proficiency, adapt feedback granularity, and schedule spaced repetition reviews addresses the two primary failure modes of static SSL learning systems: learner frustration from excessive difficulty and learner disengagement from insufficient challenge.

The spaced repetition component models the forgetting curve per sign, ensuring that signs at risk of being forgotten receive review scheduling at optimal intervals — a functionality absent from all currently published Sinhala Sign Language learning platforms.

### 3.2.8 Finding 8: Offline-First Design Is Essential for Rural Accessibility in Sri Lanka

The Two-Way Communication System's decision to store all sign images, icon libraries, and phrase templates locally (offline-first architecture) reflects a critical finding about the deployment context. Smartphone internet connectivity in rural Sri Lanka is inconsistent, and a communication tool for children with speaking disabilities that relies on server connectivity would be unusable in exactly the communities that most need it. The local database approach ensures that the full communication functionality remains available regardless of network conditions, directly supporting the platform's equity objectives.

---

## 3.3 Discussion

### 3.3.1 Overall Platform Achievement

The Intelligent Sinhala Sign Language Communication Platform successfully realised four distinct, integrated components that collectively address the communication and safety challenges of the deaf and hearing-impaired community in Sri Lanka. Each component was technically validated against its objectives:

- The Sound Alert Module achieved 92.19% test accuracy on a 7-class dangerous sound classification problem, meeting the ≥90% accuracy objective and the real-time latency requirement (<3 seconds end-to-end).
- The SSL Reader achieved 80.1% test accuracy on 383 Sinhala sign classes — the largest vocabulary addressed by any published SSL recognition system — through a novel multi-stream architecture that outperformed all single-stream baselines by at least 24 percentage points.
- The Adaptive Learning System operationalised RL-based personalisation for SSL education, providing the first published implementation of reinforcement learning for adaptive Sinhala Sign Language tutoring.
- The Two-Way Communication System delivered a fully functional, offline-capable, family-centred visual communication interface for children with speaking disabilities, requiring no prior SSL knowledge from either communication partner.

### 3.3.2 Contextualisation of SSL Reader Accuracy

The 80.1% test accuracy of the SSL Reader must be interpreted in proper context. For a 383-class classification problem, random chance accuracy is approximately 0.26% (1/383). The system achieves accuracy approximately 308 times higher than chance, demonstrating clearly that the model is learning meaningful, sign-specific representations. The 24 percentage-point improvement over the best single-stream baseline further confirms that the architectural contribution — not merely the data or parameter count — is responsible for the performance gain.

Comparison with related work supports the result's positioning:

**Table 3.9: SSL Reader — Contextual Comparison with Prior Work**

| System | Language | Vocabulary | Accuracy |
|---|---|---|---|
| Kumara & Gunasekara (2019) | Sinhala SL | 25 signs | 85–95% (1 signer, controlled) |
| Jiang et al. (2021) | Chinese SL | 100+ classes | 85–90%+ (large corpus, many signers) |
| **This work — SSL Reader** | **Sinhala SL** | **383 classes** | **80.1%** (22 videos/class, limited signers) |

The lower absolute accuracy compared to systems trained on large corpora is directly attributable to data scarcity (~22 videos per class) rather than architectural limitations. Published multi-stream skeleton-based SLR systems with hundreds of training samples per class achieve 85–90%+. With similar data volume, the Multi-Stream Fusion architecture is expected to reach comparable accuracy.

### 3.3.3 Comparison of Sound Alert Results with Related Work

The Sound Alert Module's 92.19% test accuracy on a 7-class custom dataset compares favourably with published benchmarks:

**Table 3.10: Sound Alert — Comparison with Related Work**

| System | Dataset | Classes | Accuracy |
|---|---|---|---|
| Piczak (2015) — CNN | ESC-50 | 50 | 73.7% |
| Salamon & Bello (2017) — CNN | ESC-50 | 50 | 83.7% |
| Lim et al. (2017) — CNN | UrbanSound8K | 10 | 79.0% |
| Park & Han (2019) — CNN | Custom (horn detect) | 2 | 94.5% |
| **This work — Sound Alert** | **Custom (Sri Lanka context)** | **7** | **92.19%** |

The closest comparable prior work — Park and Han's 2-class binary horn detection system — achieved 94.5% on a simpler problem. This work achieves 92.19% on a more challenging 7-class problem covering both horn types and emergency sirens, calibrated specifically to the Sri Lankan urban acoustic environment.

### 3.3.4 Comparison of Adaptive Learning Results with Related Work

The Adaptive Learning System introduces RL-based personalisation to Sinhala Sign Language education — a combination that has no direct published precedent in the SSL context. Contextual comparison with the closest related systems situates this contribution:

**Table 3.11: Adaptive Learning — Comparison with Related Systems**

| System | Language | Personalisation Approach | Spaced Repetition | RL Component |
|---|---|---|---|---|
| SignAll / Ace ASL (2021) | American SL | Rule-based difficulty tiers | No | No |
| SignGuru — Paudyal et al. (2020) | American SL | Error-driven feedback, fixed curriculum | No | No |
| Leitner System (classical) | General | Box-based card scheduling | Yes (fixed intervals) | No |
| **This work — Adaptive Learning** | **Sinhala SL** | **Q-table RL agent, dynamic curriculum** | **Yes (ML-predicted intervals)** | **Yes** |

The two closest prior systems for sign language learning — Ace ASL (SignAll, 2021) and SignGuru (Paudyal et al., 2020) — both rely on hand-authored rule sets for feedback and difficulty adjustment. Neither system uses a learning agent that adapts its behaviour based on accumulated interaction history. SignGuru provides error-driven corrective feedback, which is conceptually aligned with this work's granular feedback mode, but its curriculum sequencing is static.

The classical Leitner spaced repetition system schedules reviews at predefined fixed intervals regardless of per-learner evidence. The Adaptive Learning System's spaced repetition component replaces fixed intervals with ML-predicted review priorities derived from each individual learner's forgetting curve, enabling personalisation at a granularity that rule-based scheduling cannot achieve.

No published SSL learning system for any sign language variety uses a reinforcement learning agent as the curriculum and feedback controller. This work is, to the best of the authors' knowledge, the first implementation of RL-based adaptive learning for any South Asian sign language variety, and the first to combine RL-driven curriculum selection with ML-predicted spaced repetition in a single mobile-deployed learning system.

---

### 3.3.5 Comparison of Two-Way Communication Results with Related Work

The Two-Way Communication System addresses a distinct sub-problem within assistive technology: enabling non-verbal, bidirectional communication between a child with a speaking disability and their family, adapted to the Sinhala language and Sri Lankan cultural context. Comparison with established systems in this space highlights both alignment with international best practice and the novel contributions of this work:

**Table 3.12: Two-Way Communication — Comparison with Related Systems**

| System | Target Users | Language | Offline Capable | Two-Way | Sign-Based Output | Cultural Localisation |
|---|---|---|---|---|---|---|
| Proloquo2Go (AssistiveWare) | Children with AAC needs | English | Yes | No (one-way) | No (icons/text) | Western |
| PECS (Picture Exchange) | Autism spectrum | Language-agnostic | Yes (physical) | Partial | No | General |
| LetMeTalk (Android AAC) | AAC users | Multi-language | Yes | No (one-way) | No | Limited |
| Cboard (Web AAC) | Children | Multi-language | Partial | No (one-way) | No | Configurable |
| **This work — Two-Way Comm** | **Children with speaking disabilities** | **Sinhala** | **Yes (fully offline)** | **Yes (bidirectional)** | **Yes (SSL images)** | **Sri Lankan** |

Proloquo2Go is the most widely used commercial AAC application globally and represents the state of the art in child-oriented symbol communication tools. However, it is designed for one-directional expression (child → caregiver) using a symbol vocabulary that is culturally adapted for Western contexts. It does not include sign language output, and it does not enable the caregiver to respond through the same interface.

PECS (Picture Exchange Communication System) is a well-validated behaviour-analytic approach to AAC but operates through physical picture cards — it is not a digital system and offers no caregiver response channel beyond verbal acknowledgement.

LetMeTalk and Cboard are open-source digital AAC alternatives with broad language support, but both are one-directional communication tools without sign language integration and without Sinhala cultural adaptation.

This work's Two-Way Communication System differentiates from all existing systems on three dimensions simultaneously: genuine bidirectionality (parent responds through the same interface using text-to-sign conversion), integration of Sinhala Sign Language image output directly within the communication interface, and full offline functionality with culturally adapted content developed specifically for Sri Lankan daily life scenarios. No published or commercially available AAC system combines all three of these properties for the Sinhala-speaking population.

---

### 3.3.6 Limitations

**Sound Alert Module:**
The training data was collected in a limited number of acoustic environments. In real deployment, greater acoustic diversity (different room acoustics, microphone qualities, wind noise, overlapping sounds) may reduce accuracy. The model performs single-label classification per 2.5-second clip; simultaneous overlapping sounds are classified by the dominant spectral component only. The current system requires a Wi-Fi-connected Flask server; cloud deployment is required for wider usability.

**SSL Reader:**
The primary limitation is data scarcity. With approximately 22 training videos per class, the model cannot fully learn the within-class variability of 383 distinct signs. The dataset currently contains recordings from a limited number of signers; multi-signer data would substantially improve robustness and generalisation. The system performs isolated sign recognition only — continuous signing recognition (detecting sign boundaries in unsegmented video) requires additional sequence modelling beyond the current architecture.

**Adaptive Learning System:**
The system faces a cold-start problem for new users who have no interaction history. Initial lesson selection is effectively random until the RL agent accumulates sufficient evidence to personalise. Camera quality and lighting conditions affect hand landmark detection accuracy, which propagates into sign recognition quality and, consequently, feedback reliability.

**Two-Way Communication System:**
The current sign image library covers a practical daily vocabulary but does not extend to specialised domains (medical, legal, technical). Sign images are 2D drawings rather than video-based demonstrations, which may reduce comprehension accuracy for complex dynamic signs.

### 3.3.7 Integrated System Considerations

The four components share a common React Native mobile frontend and communicate through Flask REST API endpoints, forming a cohesive platform rather than four isolated tools. The integration architecture enables future cross-component synergies: for example, the SSL Reader's recognition capability could supply sign input data to the Adaptive Learning System, enabling real-time sign assessment without separate camera processing pipelines. The Sound Alert Module's real-time monitoring can run concurrently with the Communication System, providing safety awareness during communication sessions.

The platform collectively addresses four dimensions of deaf accessibility in Sri Lanka that no single prior system has addressed simultaneously:
1. Safety awareness (Sound Alert)
2. Sign language recognition and communication (SSL Reader)
3. SSL education (Adaptive Learning)
4. Family-centred non-verbal communication for children (Two-Way Communication)

---

## 3.4 Summary of Each Student's Contribution

This section documents the specific technical contributions made by each member of the project group to the Intelligent Sinhala Sign Language Communication Platform.

---

### IT22325464 — Kodithuwakku M.A.S.S.H.
**Component: Environmental Sound Alert Module**

1. **Dataset Construction:** Assembled and curated the 7-class audio dataset from locally collected vehicle horn recordings and publicly available siren recordings. Made and justified the class-selection decisions — excluding motorcycle horn (acoustic overlap with car horn) and ambient traffic noise (continuous false-alert risk) — resulting in a focused 7-class dataset of approximately 3,230 clips.

2. **Audio Preprocessing Pipeline:** Designed and implemented the `AudioPreprocessor` class covering audio loading and mono resampling to 22,050 Hz, 40 dB silence trimming with short-clip fallback, fixed-length 2.5-second windowing, 13 × 40 MFCC feature extraction with Hann windowing and 128 Mel filterbanks, and per-coefficient z-score normalisation with training statistics saved to disk for consistent inference-time application.

3. **CNN Model Design and Training:** Designed the `SoundClassifierCNN` architecture — three convolutional blocks (32 → 64 → 128 filters, 3×3 kernels, BatchNorm, MaxPool, Dropout) followed by Dense(256) → Dense(128) → Dense(7, Softmax) — and implemented the full training configuration including per-class sample weights via `compute_class_weight`, EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint callbacks. Trained the model to **92.19% test accuracy** on the held-out 448-sample test set.

4. **Real-Time Inference Module:** Implemented the `RealtimeSoundDetector` class supporting both microphone-based continuous monitoring (using a 2.5-second recording loop) and file-based inference, with result formatting that includes class name, severity level, confidence score, and human-readable display title.

5. **Flask REST API:** Built `api_server.py` to host the trained model, supporting WAV and M4A audio uploads from the React Native client, lazy model loading with in-memory caching, and structured JSON prediction responses containing class, severity, confidence, and timestamp.

6. **Alert Tier System:** Designed and specified the three-tier severity alert framework: Tier 1 (full-screen red emergency overlay for ambulance, fire truck, police), Tier 2 (auto-dismissing banner for vehicle horns), and Tier 3 (silent background history log), including the class-to-tier mapping implemented in the React Native frontend.

7. **System Testing:** Conducted 18 documented test cases covering preprocessing unit tests, model inference validation, Flask API integration, end-to-end mobile pipeline testing, and real-world street environment testing.

---

### IT22304674 — Liyanage M.L.I.S.
**Component: SSL Reader (Smart Sinhala Sign Language Reader with Emotion Recognition)**

1. **Multi-Modal Feature Extraction Pipeline:** Designed and implemented the complete feature extraction pipeline using MediaPipe Tasks API v0.10.21, integrating three landmark detection models in parallel (Hand Landmarker: 126-d, Face Landmarker with blendshapes: 232-d, Pose Landmarker: 99-d) into a 457-dimensional combined feature vector per frame, with a feature caching system to eliminate redundant extraction during training.

2. **Root-Relative Normalisation:** Designed and implemented stream-specific coordinate normalisation — wrist-centred for hand streams, nose-tip-centred for face landmarks (with blendshape scalars preserved), and mid-shoulder-centred for pose — with verified consistency between training-time and inference-time code paths to prevent distribution mismatch.

3. **Skeleton-Level Augmentation Framework:** Designed and implemented the `SkeletonAugmenter` and `StreamSpecificAugmenter` classes with seven stochastic operations: spatial rotation, spatial scaling, Gaussian noise, temporal shift, time warping, frame dropping, and time masking. Explicitly excluded horizontal flipping from the operation set with documented justification based on sign language handedness semantics.

4. **Systematic Architecture Development:** Designed and implemented all four model architectures in PyTorch — `MultimodalLSTMModel`, `MultimodalTransformerModel`, `HybridModel`, and `MultiStreamFusionModel` — and conducted systematic comparative evaluation, identifying modality-specific processing as the critical architectural factor.

5. **Multi-Stream Fusion Model (Final Architecture):** Designed the novel Multi-Stream Fusion architecture, including TCN sub-networks for hand and face streams (dilations 1, 2, 4), a BiLSTM sub-network for the pose stream with adaptive average pooling, and an attention fusion module projecting all streams to 512 dimensions with softmax-normalised stream weights. The model produces dual outputs: classification logits and interpretable per-stream attention weights. Achieved **80.1% test accuracy** on 383 Sinhala sign classes — the largest vocabulary addressed by any published SSL recognition system.

6. **Training Methodology:** Implemented the `MediaPipeTrainer` class including cosine annealing with warm restarts (T₀=30, T_mult=2), Adam optimiser with weight decay, label smoothing cross-entropy (ε=0.1), early stopping, and best-checkpoint saving. Managed the full 120-epoch training run converging at epoch 87.

7. **Deployment and Integration:** Implemented the Flask REST API (`react_native_bridge.py`) with `/health`, `/labels`, `/predict_frame`, and `/predict_video` endpoints, CORS support, threading lock for thread-safe model inference, and structured JSON responses. Solved Sinhala Unicode rendering on live camera frames using PIL with the Nirmala.ttc font.

8. **Documentation:** Authored the component methodology documentation, research paper, and this thesis component's Results chapter.

---

### IT22315878 — Silva H.T.P.
**Component: Adaptive Sinhala Sign Language Learning System**

1. **Sign Recognition Model Integration:** Integrated the `sinhala_sign_language_classifier.keras` model (targeting 25 Sinhala sign letters) with a MediaPipe hand detection pipeline, including a 5-prediction smoothing buffer to stabilise recognition output before feeding into the RL agent. Implemented two hand detection backends (cvzone `HandDetector` and raw MediaPipe Hands) with a graceful fallback mechanism.

2. **RL Feedback Agent:** Designed and implemented the `RLFeedbackAgent` class using a Q-table reinforcement learning approach. The agent tracks per-learner, per-sign performance history and adapts feedback granularity — providing detailed corrective feedback (e.g., hand position adjustments) for struggling learners and positive reinforcement for confident learners — based on learned Q-values over interaction episodes.

3. **Enhanced Adaptive Learning Agent:** Developed the `AdaptiveLearningAgent` class extending the base RL agent with adaptive curriculum sequencing and spaced repetition scheduling. The agent maintains a proficiency state space (per-sign mastery level, practice frequency, time since last review) and selects actions (next sign, difficulty adjustment, review scheduling) to maximise a composite reward signal combining assessment accuracy, long-term retention, and learner engagement.

4. **Enhanced Preprocessing Pipeline:** Implemented the `EnhancedPreprocessor` class incorporating CLAHE contrast enhancement, white balance correction, and optional background reduction for improved hand detection robustness under varying lighting conditions.

5. **Video Sign Prediction Module:** Built the `VideoSignPredictor` class to extend recognition from static signs to dynamic/video-based sign sequences, expanding the system's coverage beyond the 25 static letter signs.

6. **Flask REST API (Adaptive Learning Backend):** Implemented `app.py` as the unified Flask backend for the adaptive learning system, exposing endpoints for sign recognition, RL feedback generation, lesson sequencing, and learner progress tracking, serving the React Native mobile frontend.

---

### IT22308870 — De Silva U.P.A.N.
**Component: Sinhala Two-Way Communication System for Children with Speaking Disabilities**

1. **Dual-View Interface Design:** Designed and implemented the dual-view mobile interface separating the parent's input view (text keyboard, pre-defined phrases, categorised icon library) from the child's expression view (sign image display, emotion selector, quick-needs buttons), creating an asymmetric but complementary interaction model suited to the communication needs of each party.

2. **Text-to-Sign Conversion Engine:** Implemented the Sinhala Unicode text-to-sign mapping pipeline covering word segmentation, grammar parsing for compound expressions, sign sequence generation, and fallback fingerspelling for words not in the sign library.

3. **Icon Library System:** Curated and implemented a culturally appropriate icon library of 40+ icons across five communication domains (food, school, emotions, health, play), with both visual browse and text search access modes and support for family-specific custom icon addition.

4. **Emotion Layer:** Implemented the five-state emotion expression system with intensity gradations (slightly / very) and colour-coded visual feedback (green = happy, blue = sad, red = angry, yellow = scared, grey = tired), providing an accessible, pre-linguistic emotion communication channel for non-verbal children.

5. **Offline-First Local Storage Architecture:** Designed and implemented the local database architecture storing all sign images, icon libraries, pre-built communication templates, and conversation history on-device, ensuring full functionality in the absence of internet connectivity — a design requirement for rural deployment in Sri Lanka.

6. **Conversation Manager:** Implemented the conversation history and context management system, including local storage, conversation threading, and predictive response suggestions based on communication patterns, enabling both parties to review interaction history and improving communication continuity across sessions.

7. **Pre-Built Communication Templates:** Authored and validated 36+ pre-built daily communication scenarios across six domains (mealtime, school/homework, health, play/activities, bedtime, emotions), incorporating culturally adapted content relevant to Sri Lankan family contexts.

---

*End of Chapter 3: Results and Discussion*
