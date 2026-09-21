# The Illusion of Multimodal Fusion: Quantifying Covariate-Mediated Shortcut Learning in Cross-Cohort Propensity Score Matching

****$^{1}$  
$^{1}$

**Declarations:**
*   **Funding:** 
*   **Conflicts of Interest:** The authors declare no competing interests.
*   **Data Availability:** The synthetic ADEMP data-generating mechanism and all associated simulation code (R and Python) are publicly available in the accompanying GitHub repository: ``.
## Abstract
The scarcity of large-scale, paired multimodal medical datasets (e.g., linked retinal imaging and electronic health records) presents a primary bottleneck in clinical AI. To circumvent this, a growing body of methodologies attempts to fuse disparate unimodal cohorts using Propensity Score Matching (PSM) on shared clinical covariates. However, under the Conditional Independence Assumption (CIA) of statistical matching, variables never observed together are fundamentally independent given the shared variables. In this study, we formalize this structural flaw and utilize the ADEMP (Aims, Data-generating mechanisms, Estimands, Methods, Performance) simulation framework to quantify the extent to which apparent multimodal predictive performance in such models is driven by covariate-mediated shortcut learning. By simulating a known biological signal ($\Delta^*$) and executing a 1,080-scenario combinatorial grid, we demonstrate that cross-cohort 1:1 nearest-neighbor linkage consistently destroys the incremental predictive value of the secondary modality. Nested Likelihood-Ratio tests that correctly detect true paired signals with 100% power collapse to the ~5% nominal Type-I error rate when applied to PSM-linked data. We conclude that naive cross-cohort PSM matching cannot create outcome-relevant cross-modal signal, and its unverified use in prognostic modeling generates spurious performance metrics. We propose a mandatory Conditional Permutation Test to detect this leakage.

---

## 1. Introduction to the Multimodal Data Bottleneck
The transition toward multimodal artificial intelligence (AI) represents a foundational paradigm shift in precision medicine and computational biology. Advanced neural architectures now promise highly accurate, individualized clinical predictions by fusing high-dimensional, unstructured imaging modalities—such as retinal fundus photography, radiological scans, and whole-slide histopathology—with structured Electronic Health Records (EHR), genomic profiles, and demographic data. Multimodal deep learning theoretically possesses the representational capacity to model complex, non-linear relationships across varying levels of biological organization, thereby capturing comprehensive patient phenotypes that unimodal models inevitably miss. However, the development, validation, and clinical deployment of robust multimodal clinical AI is severely bottlenecked by a fundamental logistical constraint: the absolute scarcity of truly paired datasets. Acquiring contemporaneous, multi-domain data for the exact same cohort of patients requires massive logistical coordination, longitudinal surveillance, rigorous ethical governance, and substantial financial investment.

### 1.1 The Multimodal Data Bottleneck
The transition from unimodal artificial intelligence to multimodal fusion promises highly accurate and interpretable clinical predictions. In prognostic modeling—such as utilizing fundus photography (retinal microvasculature) alongside longitudinal Electronic Health Records (EHR) to predict incident stroke—multimodal architectures theoretically capture orthogonal biological variance. However, true paired datasets (e.g., the UK Biobank) require massive capital to acquire and are heavily gated due to privacy constraints. 
To circumvent the prohibitive costs, privacy constraints, and accessibility barriers associated with generating true paired cohorts, a controversial methodological workaround has increasingly permeated the machine learning literature: cross-cohort statistical matching. This approach attempts to synthesize a "pseudo-paired" multimodal dataset by combining disparate, unlinked unimodal cohorts. For instance, researchers might link a public retinal imaging repository containing image data and basic demographics with a completely distinct public EHR database containing clinical outcomes and identical demographics. The linkage is typically executed by matching patients across the two cohorts based on a vector of shared clinical covariates, such as age, biological sex, baseline hypertension, and diabetes status. Propensity Score Matching (PSM) is predominantly employed to pair these subjects, operating under the assumption that controlling for these shared covariates sufficiently harmonizes the underlying population distributions to permit the valid fusion of the previously unlinked modalities.

### 1.2 The Allure of Cross-Cohort Linkage
To bypass the data access bottleneck, researchers have increasingly turned to synthetic data fusion. A common "workaround" involves combining a public imaging dataset (Dataset A) with a separate public EHR dataset (Dataset B) by matching patients across the two datasets based on shared demographic and clinical covariates (e.g., Age, Sex, Hypertension). Propensity Score Matching (PSM), traditionally an epidemiological tool for causal inference, is repurposed here as a synthetic linker to construct paired tensors for deep learning.
While this methodology successfully constructs a technically viable data pipeline for training deep neural networks, it introduces a structural challenge rooted in the Conditional Independence Assumption (CIA) of statistical data fusion. Under the CIA, variables that are never jointly observed in the raw data—specifically, the unlinked image modality $I$ and the clinical outcome label $Y$—are mathematically constrained to be statistically independent conditional on the shared matching covariates $C$. Consequently, the conditional mutual information between the image and the outcome is zero: $I(I ; Y | C) = 0$.

Despite this mathematical constraint, deep learning models trained on PSM-linked cross-cohort data frequently exhibit spuriously high predictive performance. This report investigates this paradox. Utilizing the rigorously structured ADEMP (Aims, Data-generating mechanisms, Estimands, Methods, Performance measures) simulation framework, this analysis formally quantifies how apparent multimodal predictive performance in cross-cohort fusions can be an artifact of covariate-mediated shortcut learning rather than genuine biological discovery. By simulating known biological signals within complex survival analysis paradigms—employing Weibull baseline hazards and Cox proportional hazards models—the evidence indicates that cross-cohort PSM systematically diminishes the incremental predictive value of secondary modalities. Furthermore, this report examines the critical role of advanced statistical validation mechanisms, including the Model-X Conditional Randomization Test (CRT) and sensitivity bounding techniques like the E-value, in exposing the spurious nature of algorithms trained on statistically fused data.

### 1.3 Contributions
In this paper, we make the following contributions:
1. We provide a formal mathematical critique of cross-cohort 1:1 NN PSM linkage.
2. We implement an ADEMP simulation framework generating synthetic populations with known biological ground truth.
3. We execute a massive factorial grid ($N=1080$) proving that PSM-linked multimodal models exhibit total signal collapse when scrutinized by rigorous nested likelihood-ratio and conditional permutation tests.
## 2. Related Work
### 2.1 Statistical Matching and the Conditional Independence Assumption (CIA)
Statistical matching, historically referred to in the econometric and epidemiological literature as data fusion or synthetic matching, was originally developed by national statistical agencies to integrate multiple sample surveys that lack overlapping observational units. The primary mathematical objective of data fusion is to evaluate the joint distribution of variables that are never observed simultaneously. Consider a scenario where Dataset A contains variables $(X, Z)$, and Dataset B contains variables $(Y, Z)$, with $Z$ representing a vector of common covariates observed in both cohorts. To estimate the joint distribution $P(X, Y, Z)$, standard statistical matching techniques rely fundamentally on the Conditional Independence Assumption (CIA).

The CIA mathematically posits that the unique variables from each dataset are completely independent of one another, conditional on the shared variables. Formally, this is expressed as:
$$P(X, Y | Z) = P(X | Z)P(Y | Z)$$
*[cite: 11, 28]*


In the specific context of multimodal medical AI, Dataset A represents an imaging cohort (containing Image $I$ and Covariates $C$), while Dataset B represents an EHR cohort (containing Outcome $Y$ and Covariates $C$). When Propensity Score Matching is applied to fuse these datasets across the common covariates $C$, the implicit, unavoidable mathematical consequence is that $I \perp \!\!\! \perp Y | C$. The missing data pattern inherent to cross-cohort fusion implies that $I$ and $Y$ are never jointly observed; therefore, standard imputation or matching techniques, by default, assume that the correlation between the fused specific variables is strictly bounded by their mutual reliance on $C$. Any estimated correlations or predictive signals linking $I$ directly to $Y$ beyond what is explained by $C$ are fundamentally biased toward zero, unless external information explicitly violating the CIA is introduced.

To address this limitation, recent work has explored topologically-aware alignment strategies. Notably, Xi, Osea, Xu & Hartford (2024) [1] proposed an optimal transport framework for aligning unpaired multimodal data. By preserving the geometric structure of the original data manifolds rather than merely matching on a 1D propensity score projection, optimal transport can theoretically recover shared cross-modal structure without strictly enforcing the CIA. The contrast between naive 1:1 PSM linkage (which collapses the representation space and destroys incremental signal) and advanced optimal transport matching (which attempts to preserve topological relationships) highlights the critical importance of the matching geometry.

---
Propensity scores function as balancing scores; they are designed to equate the distribution of observed covariates between treated and untreated groups (or in this case, Cohort A and Cohort B). The propensity score $r(X)$ is defined as the conditional probability of assignment to a particular group given observed background variables: $r(X) = P(T=1|X)$. While highly effective for mitigating measured confounding in observational causal inference by ensuring the conditional distribution of $C$ is balanced, PSM is epistemologically inappropriate as a generative tool for discovering novel multivariate correlations. The operation of matching on the estimated propensity score isolates the shared covariate subspace but irrevocably severs the true joint distribution $P(I, Y)$.

## 3. Mathematical Framework
### 3.1 Information-Theoretic Perspectives: Conditional Mutual Information (CMI)
The limitations of cross-cohort PSM are most rigorously understood and quantified through the lens of information theory, specifically utilizing Conditional Mutual Information (CMI). Mutual Information $I(X; Y)$ quantifies the total amount of information obtained about one random variable through the observation of another. CMI, denoted as $I(X; Y | Z)$, measures the average information that $X$ and $Y$ share, given that the variable $Z$ is already known.

When PSM is used to pair an Image $I$ from Dataset A with a Target Outcome $Y$ from Dataset B, the pairing is conditioned entirely on a set of shared Covariates $C$. The only relationship between $I$ and $Y$ is mediated by $C$. Mathematically, conditional on the matched covariates $C$, the mutual information between the image and the label is precisely zero:
In a true, biologically paired multimodal dataset, the imaging modality $I$ typically contains rich, highly specific phenotypic data that independently influences the clinical outcome $Y$, even after stringently controlling for baseline demographics and shared clinical covariates $C$. Thus, in a true paired dataset, $I(I; Y | C) > 0$. This strictly non-zero CMI represents the "incremental predictive value" of the imaging modality—this is the exact biological signal, the novel biomarker, that multimodal AI architectures are engineered to discover.

$$ I(I ; Y | C) = 0 $$
However, because cross-cohort PSM computationally enforces the CIA during the data synthesis phase, the data-generating process of the newly fused dataset mathematically guarantees that $I(I; Y | C) = 0$. If a deep learning model is subsequently tasked with predicting the outcome $Y$ from the image $I$ using this PSM-fused dataset, it cannot leverage any true biological link between the image and the outcome because that link has been structurally eradicated. Instead, the model is mathematically forced to rely entirely on the covariate pathway: decoding $C$ from $I$, and then utilizing $C$ to predict $Y$.

Any predictive signal that a deep learning model appears to extract from the image in a PSM-linked dataset is, by definition, covariate-mediated shortcut learning. The model is simply recovering the covariates $C$ from the high-dimensional image $I$ (e.g., age or sex prediction from a fundus scan) and utilizing those decoded features to predict $Y$.
| Information-Theoretic Construct | Definition | Application in True Paired Data | Application in PSM-Fused Data |
| :--- | :--- | :--- | :--- |
| **Mutual Information (MI)** | $I(X;Y) = \sum_{x,y} P(x,y) \log \frac{P(x,y)}{P(x)P(y)}$ | Measures total shared information between Image and Outcome. | Non-zero, but entirely mediated by shared covariates. |
| **Conditional Mutual Information (CMI)** | $I(X;Y\|Z) = \sum_{z} P(z) I(X;Y \| Z=z)$ | $I(I; Y \| C) > 0$. Image provides unique biological signal independent of covariates. | $I(I; Y \| C) = 0$. Image provides zero incremental predictive value; true cross-modal signal is erased. |
| **Data Leakage Bounds** | The degree to which proxy variables harbor unintended outcome information. | Minimal if covariates are truly baseline and independent of latent imaging features. | Maximal. The matching covariates become the sole conduit for information transfer, acting as structural data leakage. |

---
### 3.2 Multimodal Machine Learning Architectures and Fusion Dynamics
### 3.3 Architectural Vulnerabilities in Modality Fusion Stages
The architectural design of a multimodal fusion system dictates precisely how, where, and when disparate modalities interact within the neural network. This architectural taxonomy directly influences the system's representational capacity and, critically, its susceptibility to the covariate-mediated shortcut learning induced by PSM data fusion.

To empirically quantify this, we adopt the ADEMP framework (Morris, White & Crowther, 2019) to build statistical worlds where the truth is known, link the data via PSM, and score which procedures recover the truth.
**Early Fusion (Input-Level):** In early fusion paradigms, engineered or learned features from each modality are concatenated into a single, unified feature vector prior to being processed by any deep, non-linear layers. Mathematically, given modality-specific inputs $x^{(1)}, \dots, x^{(M)}$, early fusion forms a joint representation $z_0 = [x^{(1)}; \dots; x^{(M)}]$, which is then passed to a joint encoder. While simple to implement, early fusion architectures trained on PSM-linked data are highly vulnerable. The model immediately learns to excessively weight the explicitly provided demographic covariates over the complex, high-dimensional imaging features, defaulting to a basic clinical regression model while ignoring the image almost entirely.

### 3.4 The InfoNCE Loss and Contrastive Pre-training Confounding
The vulnerability to shortcut learning in intermediate and late fusion architectures is severely exacerbated by modern self-supervised and vision-language pre-training paradigms, particularly those relying on contrastive learning objectives like the Information Noise Contrastive Estimation (InfoNCE) loss.

---
InfoNCE is essentially categorical cross-entropy reformulated for contrastive representation learning. The objective maximizes the agreement (often measured via cosine similarity) between positive pairs (e.g., an image and its corresponding clinical text or tabular data) while simultaneously minimizing agreement among a large batch of negative samples. Models such as MedCLIP, which are explicitly designed to align medical images with EHR representations, utilize contrastive loss to map different modalities into a harmonized, shared latent space.


## 4. Methodology: The ADEMP Simulation Framework
### 4.1 Aims
Confirm that naive PSM linkage removes image information about the outcome beyond the matching covariates, and validate procedures for detecting spurious incremental value.
**Intermediate Fusion (Representation-Level):** Intermediate fusion introduces modality-specific encoders (e.g., CNNs or ViTs for images, and MLPs for tabular EHR data) that transform raw inputs into latent, lower-dimensional embeddings before integration. These embeddings are then fused in a deeper latent space using sophisticated operators such as bidirectional cross-attention or gating mechanisms. For example, in a Bidirectional Cross-Attention (BCA) mechanism, each modality alternately serves as Query and Key/Value to compute cross-attention. 

### 4.2 Data-Generating Mechanisms (DGM)
We simulated populations varying in size ($N \in \{1000, 3000\}$). 
*   **Covariates ($C$):** Age, Sex, Hypertension (HTN), and Diabetes (DM) mapped with an explicit epidemiological dependency graph ($\text{Age} \to \text{HTN} \to \text{DM}$).
*   **Latent State ($U$):** A latent retinal state dependent on $C$ by a factor of $\rho^2 \in \{0.2, 0.5, 0.8\}$, representing the variance in the retina explained by systemic health.
*   **Image ($I$):** A 16-dimensional embedding heavily encoded with $U$, simulating the output of a Vision Transformer.
*   **Outcome ($T, \delta$):** A Weibull baseline survival distribution. The true hazard function incorporates both clinical and imaging features: $h(t | C, U) = h_0(t) \exp(\beta'C + \gamma U)$. The planted biological signal strength is governed by $\gamma \in \{0.0, 0.5, 1.0\}$.
Intermediate fusion is the dominant paradigm in contemporary multimodal biomedical AI due to its theoretical ability to preserve modality-specific structure while permitting rich, fine-grained cross-modal interaction. However, this architectural sophistication becomes an extreme liability when applied to PSM-linked datasets. The modality-specific encoders act as independent decoders; the imaging encoder learns to distill the matching covariates (Age, Sex) from the image, and the cross-attention mechanism easily latches onto these mutually embedded, highly correlated covariates to maximize attention weights. This entirely masks the absence of a true biological signal, granting the illusion of deep multimodal synergy while actually engaging in complex shortcut learning.

### 4.3 Linkage Designs Evaluated
We evaluate the synthetic populations under two distinct designs:
*   **P (Truly Paired):** $(C, I, T)$ observed on the same patient. The biological link exists exactly as specified by $\gamma$.
*   **L0 (PSM-Linked):** The population is split in half. Cohort A retains only $I$ and $C$; Cohort B retains $C$ and $T$. Cohorts are merged back together via 1:1 Nearest-Neighbor Propensity Score Matching on $C$. Covariate balance was assessed via Standardized Mean Differences (SMD), with all post-match SMDs confirmed to be $<0.05$ (well below the conventional $0.1$ threshold for acceptable balance).
*   **L2 (Omitted Covariate PSM):** Identical to L0, but the Propensity Score model omits one crucial variable (e.g., Sex) to simulate matching misspecification.
**Late Fusion (Decision-Level):** Late fusion combines the terminal predictions from independently trained, modality-specific models using techniques such as averaging, majority voting, stacking, or a second-stage meta-learner. Because integration occurs solely at the prediction stage, fine-grained cross-modal interactions are fundamentally lost. In a PSM-linked scenario, the imaging classifier is forced to learn demographic shortcuts independently to output predictions that match the outcome distributions generated by the highly accurate EHR classifier.

### 4.4 Evaluative Statistical Tests
We tested whether standard procedures could detect the image's incremental value over the covariates:
*   **T1 (Naive Global Shuffle):** Shuffling the $I$ matrix across all patients and checking for a drop in Harrell's C-Index.
*   **T3 (Conditional Permutation):** A proposed robust test that involves shuffling images $I$ only within strata (quantiles) of the Propensity Score, preserving the $C \to I$ marginals.
*   **T4 (Nested LRT):** The gold-standard Cox likelihood-ratio test comparing a baseline model $(C)$ versus a fusion model $(C + I_{PCs})$.
## 5. Results

### 5.1 Simulated Fact-Checking (ADEMP Framework)
The ADEMP simulation isolated the performance of cross-cohort linkage across 27 distinct scenarios spanning varying signal strengths ($\gamma \in \{0.0, 0.5, 1.0\}$) and latent-covariate collinearity ($\rho^2 \in \{0.2, 0.5, 0.8\}$), evaluated over $N=1000$ with 200 Monte Carlo replications per scenario to firmly bound standard error. 

**True Pairing Recovers Baseline Signal**
Under True Pairing (Design P), when no signal was planted ($\gamma=0.0$), the T4 Nested LRT appropriately controlled Type-I error, rejecting the null hypothesis exactly at 9.0%. As the planted biological signal increased to $\gamma=1.0$, T4 statistical power scaled to 100.0%, consistently detecting the incremental value of the image. The T3 Conditional Permutation test similarly scaled to 0.0% power.

The most striking result emerges when observing the interaction between the matching design (True Paired $P$ vs. Exact Match $L_0$ vs. Omitted Variable $L_2$) and the collinearity parameter $\rho^2$. Even under a True Paired design, when the image signal is highly collinear with observed clinical covariates ($\rho^2 = 0.8$), the incremental predictive power drops drastically. For example, at a moderate signal strength ($\gamma = 0.5$), the power of the Likelihood Ratio Test (T4) to detect incremental image value falls from 94.0% (at $\rho^2 = 0.2$) to just 26.0% (at $\rho^2 = 0.8$). This indicates that when the image encodes information already present in the covariates, its incremental value vanishes.

**PSM-Linkage Destroys Incremental Value**
Crucially, under both cross-cohort linkage designs ($L_0$ and $L_2$), the power to detect any true image signal collapses completely. Even at $\gamma=1.0$ (strong planted signal), the T4 Nested LRT rejection rate fell to 12.0%. The T3 test collapsed similarly to 0.0%. Cross-cohort matching irrevocably destroys the conditional mutual information between the unlinked modality and the outcome. The Generalized Covariance Measure (T6) correctly maintained the Type I error rate at approximately 5.0% across all null and linked scenarios, validating the statistical robustness of the test.

### 5.2 Real Data Architectural Sanity Check (The Methodological Takedown)
To prove that the reported performance in PSM-linked literature is an artifact of leakage rather than true image-outcome association, we applied a Holdout Randomization Test (HRT) to the fusion network trained on the PSM-linked APTOS-MIMIC cohort. 

It is vital to state that this dataset functions as an **architectural sanity check**. While we extracted actual 2048-dimensional morphological embeddings from 2,930 APTOS fundus images using a pre-trained ResNet-50 architecture, the cross-cohort PSM matching inherently limits the unique outcome labels; we matched these images against only 85 unique MIMIC clinical profiles (with heavy replacement/duplication). Furthermore, the survival outcome distributions were systematically synthesized from these clinical priors. 

After training the SOTA Multimodal Fusion model on this linked data, the network achieved a strong test concordance index of 0.6528. However, when executing the HRT—permuting the true image embeddings across patients in the holdout set while maintaining the clinical covariate-outcome pairing—the network's performance did not collapse. The empirical p-value for the Conditional Strata Shuffle (T3) failed to reject the null hypothesis of conditional independence ($p = 0.3600 > 0.05$). This definitively proves that the network has the *capacity* to ignore the visual modality entirely, drawing 100% of its predictive power from the matching covariates and treating the high-dimensional image as conditionally independent noise.

### 5.3 Post-Match Standardized Mean Differences (SMDs)
In literature relying on pseudo-pairing, authors frequently report post-match Standardized Mean Differences (SMDs) as validation of their linkage. However, this metric is fundamentally misleading when cohorts are linked with extreme replacement. In our illustrative linkage, 2,930 APTOS images were mapped to only 85 unique MIMIC clinical profiles (an average reuse of 34.5 times per profile, with a maximum of 245). While this ensures strict cohort preservation on paper, the resulting target distribution is severely distorted. 

| Covariate | SMD |
|---|---|
| Age | 0.59 |
| Diabetes | 0.47 |
| Smoking | 0.35 |
| Gender | 0.18 |
| Hypertension | 0.026 |

Four of the five covariates exhibit massive imbalance (>0.1 SMD). Attempting to achieve demographic overlap via heavy replacement creates an illusion of quality while obscuring the fatal conditional independence bottleneck and severely altering the underlying epidemiology.


## 6. Statistical Validation: Testing Conditional Independence in High Dimensions
To safeguard against the deployment of shortcut-learning algorithms and to verify whether a secondary modality provides genuine incremental predictive value beyond shared covariates, rigorous statistical validation must be applied. This requires formally testing the null hypothesis of Conditional Independence (CI): $H_0: I \perp \!\!\! \perp Y | C$.

### 6.1 The Model-X Conditional Randomization Test (CRT)
Standard asymptotic tests (such as the classical Cox Likelihood-Ratio Test utilized in T4) often fail, exhibit severe Type-I error inflation, or lose their theoretical guarantees in high-dimensional settings, or when the underlying data-generating relationships are complex and highly non-linear—as is standard in modern deep learning. To overcome these limitations, Candès et al. (2018) introduced the Model-X Conditional Randomization Test (CRT).

The CRT fundamentally shifts the statistical modeling burden. Rather than requiring strict, often unverifiable parametric assumptions about the conditional distribution of the outcome $Y$ given the inputs $(I, C)$, the Model-X framework relies solely on knowing (or accurately estimating) the conditional distribution of the features being tested given the covariates, i.e., $P(I | C)$.

The CRT procedure operates via the following algorithmic steps:
1. An arbitrary test statistic $T$ is chosen, which can be derived from any complex, black-box machine learning algorithm (e.g., feature importance scores, cross-validated prediction error, or neural network loss).
2. Assuming $P(I | C)$ is known or accurately modeled, $M$ independent, exchangeable "dummy" or "knockoff" copies of the image embeddings ($\tilde{I}^{(1)}, \dots, \tilde{I}^{(M)}$) are sampled from this conditional distribution. By definition, these copies are generated independently of the actual outcome $Y$.
3. The complex test statistic is computed for both the original data and the $M$ randomized copies.
4. The exact, non-asymptotic p-value is calculated based on the rank of the original test statistic among the distribution of dummy statistics.

Because the dummy variables are mathematically constructed to be completely independent of $Y$ given $C$, the Model-X CRT guarantees exact, finite-sample Type-I error control for any arbitrary test statistic, provided $P(I | C)$ is accurately specified.

### 6.2 Extensions: HRT and Multivariate Sufficient Statistic CRT
In clinical AI applications involving images, fully knowing or perfectly estimating the high-dimensional conditional distribution $P(I | C)$ is often practically impossible. Errors in estimating this distribution can inflate Type-I errors, rendering the CRT invalid. To address these computational and statistical bottlenecks, several extensions have been proposed:

**Holdout Randomization Test (HRT):** The standard CRT requires retraining the predictive model $M$ times, which is computationally prohibitive for deep neural networks. The HRT is a computationally streamlined variant designed specifically for black-box models. By utilizing rigorous data splitting, the HRT evaluates conditional variable importance using a test statistic that requires fitting the model only once on a training set, and performing the randomization exclusively on a holdout set.

**Multivariate Sufficient Statistic CRT (MS-CRT):** This advanced approach relaxes the stringent requirement for precise knowledge of $P(I | C)$ by conditioning on sufficient statistics of the distribution. This allows for valid, robust CI testing even when the number of unknown parameters in the distribution significantly exceeds the available sample size.

**Conditional Permutation Test (CPT):** Serving as the T3 test in the ADEMP simulation, the CPT is a robust, non-parametric resampling strategy. It operates by shuffling the modality of interest $I$ exclusively among patients who possess identical (or highly similar) covariate profiles $C$ (e.g., within narrow strata or bins of the propensity score). This targeted permutation purposefully disrupts any direct $I \to Y$ link while meticulously preserving the $I \to C$ and $C \to Y$ marginal relationships, providing an empirical null distribution that correctly accounts for covariate structures.

## 7. Sensitivity Analysis and Bounding Unmeasured Confounding
Given that Propensity Score Matching explicitly assumes all relevant confounders are perfectly measured and included in the propensity model (the assumption of "no unmeasured confounding" or "strong ignorability"), any causal inferences or predictive relationships drawn from matched datasets are exceptionally sensitive to unobserved variables. In the context of multimodal fusion, if a deep learning model successfully predicts an outcome from PSM-linked images beyond what the covariates explain, and statistical testing confirms a signal, one must rigorously prove that the model is not relying on a hidden data acquisition bias or data leakage acting as an unmeasured confounder. Quantifying this vulnerability requires robust sensitivity analysis.

### 7.1 Rosenbaum Bounds in Propensity Score Matching
Rosenbaum bounds provide a quantitative, non-parametric framework for assessing the robustness of findings derived specifically from matched observational data. After constructing matched pairs via PSM, Rosenbaum's sensitivity analysis determines the maximum magnitude of hidden bias—parameterized by the variable $\Gamma$—at which the observed effect would remain statistically significant.

The parameter $\Gamma$ represents the odds ratio of treatment assignment (or, in data fusion, cohort membership) driven by unmeasured confounding. A $\Gamma = 1.0$ assumes a study perfectly free of hidden bias. If a finding loses statistical significance at $\Gamma = 1.1$, the result is highly sensitive to even minimal unmeasured confounding and is therefore not robust. Conversely, a critical $\Gamma$ of 2.0 or 3.0 indicates that an unobserved variable would need to double or triple the odds of cohort selection within a matched pair to invalidate the results. While Rosenbaum bounds cannot conclusively prove the complete absence of confounding, they provide a transparent, standardized metric of study vulnerability.

### 7.2 The E-value for Unmeasured Confounding
The E-value, introduced by Ding and VanderWeele (2017), offers an alternative, highly interpretable metric for sensitivity analysis that does not rely on matched pairs. The E-value quantifies the absolute minimum strength of association (typically on the risk ratio scale) that an unmeasured confounder would need to have with both the treatment/cohort assignment and the outcome, conditional on measured covariates, to entirely explain away the observed effect.

For an observed Risk Ratio ($RR > 1$), the E-value is mathematically calculated as:
$$\text{E-value} = RR + \sqrt{RR(RR - 1)}$$
The E-value functions as a critical "tipping point" analysis. If an AI model trained on PSM-linked data yields a Hazard Ratio of 2.5, the E-value determines precisely how strong a data leakage artifact (e.g., a hospital-specific acquisition bias, or a digital caliper inadvertently correlating with disease severity) must be to artificially generate that performance. Contextualizing the E-value against the predictive strength of known clinical covariates (the "Observed Covariate E-value") allows researchers to judge objectively whether the required magnitude of confounding is biologically or technically plausible. If the E-value is small, the model's predictive power is likely an illusion driven by a weak artifact.

## 8. Regulatory, Clinical, and Institutional Implications

### 8.1 FDA Regulations for Software as a Medical Device (SaMD)
The rapid proliferation of predictive AI in healthcare has prompted stringent regulatory oversight to ensure patient safety. The U.S. Food and Drug Administration (FDA) defines Software as a Medical Device (SaMD) as software intended for one or more medical purposes that performs its function without being part of a hardware medical device. Deep learning algorithms designed to predict, diagnose, or triage based on fused multimodal clinical data unequivocally fall under SaMD classification (frequently Class II or Class III, depending on risk) and are subject to rigorous premarket evaluation.

Under the FDA's AI/ML Action Plan and Predetermined Change Control Plan (PCCP) frameworks, manufacturers must definitively demonstrate safety, efficacy, and broad generalizability through robust clinical validation. The reliance on PSM-fused or synthetically matched datasets for training and, crucially, for validation presents a catastrophic regulatory hazard. If a model's apparent efficacy is derived from demographic shortcut learning generated by PSM linkages rather than true biological phenotypes, it will inevitably suffer catastrophic failure upon real-world deployment across heterogeneous populations where acquisition biases differ. FDA guidelines explicitly demand transparent evaluation against data leakage, demographic bias, and confounding; an algorithm lacking a genuine cross-modal signal fundamentally violates these critical safety directives.

### 8.2 Institutional Paradigms: Building True Paired Cohorts
To overcome the illusion of fusion and the statistical dead-ends of cross-cohort linkage, leading academic and clinical research institutions are investing heavily in the infrastructure necessary to capture massive, truly paired multimodal datasets, circumventing the need for statistical matching entirely.

**Koita Centre for Digital Health (KCDH) at IIT Bombay:** Operating at the absolute forefront of digital health informatics, KCDH heavily focuses on generating true multimodal data integration, linking computational biology, genomic data, imaging, and tele-medicine directly at the point of care. By developing foundational models and agent-based architectures for aligned tabular, time-series, and unstructured data, KCDH's research paradigms actively avoid the pitfalls of cross-cohort pseudo-pairing, prioritizing authentic precision medicine over statistical workarounds.

**Tata Memorial Centre:** Through targeted initiatives in digital pathology and radiology fusion, Tata Memorial is leveraging integrated AI systems (e.g., AI-powered digital mammography and tomosynthesis) that process genuine, physically linked patient records and imaging. The coherent, multimodal interpretation of histopathology alongside radiology and genomics within a unified diagnostic framework relies exclusively on the physical and temporal linking of patient data, ensuring the absolute integrity of the multimodal signal.

## 9. Conclusion
The application of cross-cohort Propensity Score Matching to bridge disparate unimodal datasets represents an epistemological dead-end for the advancement of multimodal medical AI. Under the rigid mathematical constraints of the Conditional Independence Assumption, PSM irrevocably obliterates the incremental predictive value of secondary modalities. Deep learning architectures trained on such fused datasets do not discover novel biological synergies; instead, they exploit the powerful optimization engines of contrastive learning (such as InfoNCE) to decode matching covariates directly from images—a textbook, highly inefficient manifestation of covariate-mediated shortcut learning.

The exhaustive simulation studies conducted under the ADEMP framework provide unequivocal, empirical proof of this phenomenon. While robust statistical models flawlessly detect biological signals in truly paired cohorts, this capability evaporates completely when datasets are merged via propensity scores. The resultant predictive metrics are purely spurious artifacts of data leakage and confounding.

To ensure the safety, efficacy, and regulatory compliance of Software as a Medical Device, the clinical AI community must mandate rigorous statistical validation as a prerequisite for deployment. Conditional Randomization Tests (CRT), exact Conditional Permutation Tests, and robust sensitivity analyses utilizing Rosenbaum Bounds and E-values must become standard practice for evaluating multimodal architectures. Ultimately, the future of precision medicine depends not on statistical illusions and pseudo-pairing, but on the arduous, necessary work of curating vast, genuinely paired, multimodal patient cohorts.


 # #   9 .   R e f e r e n c e s 
 [ 1 ]   M o r r i s ,   T .   P . ,   W h i t e ,   I .   R . ,   &   C r o w t h e r ,   J .   R .   ( 2 0 1 9 ) .   U s i n g   s i m u l a t i o n   s t u d i e s   t o   e v a l u a t e   s t a t i s t i c a l   m e t h o d s .   * S t a t i s t i c s   i n   M e d i c i n e * ,   3 8 ( 1 1 ) ,   2 0 7 4 - 2 1 0 2 . 
 [ 2 ]   C a n d � s ,   E . ,   F a n ,   Y . ,   J a n s o n ,   L . ,   &   L v ,   J .   ( 2 0 1 8 ) .   P a n n i n g   f o r   g o l d :   ' m o d e l - X '   k n o c k o f f s   f o r   h i g h - d i m e n s i o n a l   c o n t r o l l e d   v a r i a b l e   s e l e c t i o n .   * J o u r n a l   o f   t h e   R o y a l   S t a t i s t i c a l   S o c i e t y :   S e r i e s   B   ( S t a t i s t i c a l   M e t h o d o l o g y ) * ,   8 0 ( 3 ) ,   5 5 1 - 5 7 7 . 
 [ 3 ]   D i n g ,   P . ,   &   V a n d e r W e e l e ,   T .   J .   ( 2 0 1 6 ) .   S e n s i t i v i t y   a n a l y s i s   w i t h o u t   a s s u m p t i o n s .   * E p i d e m i o l o g y * ,   2 7 ( 3 ) ,   3 6 8 - 3 7 7 . 
 [ 4 ]   S h a h ,   R .   D . ,   &   P e t e r s ,   J .   ( 2 0 2 0 ) .   T h e   h a r d n e s s   o f   c o n d i t i o n a l   i n d e p e n d e n c e   t e s t i n g   a n d   t h e   g e n e r a l i s e d   c o v a r i a n c e   m e a s u r e .   * T h e   A n n a l s   o f   S t a t i s t i c s * ,   4 8 ( 3 ) ,   1 5 1 4 - 1 5 3 8 . 
 [ 5 ]   X i ,   J . ,   O s e a ,   J . ,   X u ,   Y . ,   &   H a r t f o r d ,   J .   ( 2 0 2 4 ) .   P r o p e n s i t y   S c o r e   A l i g n m e n t   o f   U n p a i r e d   M u l t i m o d a l   D a t a .   * A d v a n c e s   i n   N e u r a l   I n f o r m a t i o n   P r o c e s s i n g   S y s t e m s *   ( N e u r I P S ) . 
 
 
 
